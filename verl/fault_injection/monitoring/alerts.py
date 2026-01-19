"""Intelligent alerting system for fault injection events."""

import logging
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional, Union

from ..base import FaultLayer, FaultStatus

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Rule for triggering alerts."""

    name: str
    condition: str  # Type of condition: "threshold", "rate", "pattern", "custom"
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    cooldown_seconds: float = 300  # Prevent alert spam
    description: str = ""


@dataclass
class Alert:
    """Represents an alert."""

    rule_name: str
    severity: str
    title: str
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class AlertConfig:
    """Configuration for the alert manager."""

    enabled: bool = True
    rules: List[AlertRule] = field(default_factory=list)
    notification_channels: List[Dict[str, Any]] = field(default_factory=list)
    max_alerts_per_minute: int = 10
    alert_retention_hours: int = 24
    enable_auto_recovery: bool = True


class AlertManager:
    """Manages alerts for the fault injection system."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self._alerts: List[Alert] = []
        self._alert_history: List[Alert] = []
        self._rule_last_triggered: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._notification_handlers: Dict[str, Callable] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Register default notification handlers
        self._register_default_handlers()

        # Start if enabled
        if config.enabled:
            self.start()

    def start(self) -> None:
        """Start the alert manager."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Alert manager started")

    def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Alert manager stopped")

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        with self._lock:
            # Remove existing rule with same name
            self.config.rules = [r for r in self.config.rules if r.name != rule.name]
            self.config.rules.append(rule)
            logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            original_count = len(self.config.rules)
            self.config.rules = [r for r in self.config.rules if r.name != rule_name]
            return len(self.config.rules) < original_count

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        triggered_alerts = []

        with self._lock:
            rules = self.config.rules.copy()

        for rule in rules:
            if not rule.enabled:
                continue

            # Check cooldown
            last_triggered = self._rule_last_triggered.get(rule.name)
            if last_triggered:
                if datetime.now() - last_triggered < timedelta(seconds=rule.cooldown_seconds):
                    continue

            # Evaluate rule
            try:
                if self._evaluate_rule(rule, metrics):
                    alert = self._create_alert(rule, metrics)
                    triggered_alerts.append(alert)

                    # Update last triggered time
                    self._rule_last_triggered[rule.name] = datetime.now()

            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")

        # Process triggered alerts
        for alert in triggered_alerts:
            self._process_alert(alert)

        return triggered_alerts

    def acknowledge_alert(self, alert_index: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if 0 <= alert_index < len(self._alerts):
                alert = self._alerts[alert_index]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert acknowledged: {alert.rule_name} by {acknowledged_by}")
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unacknowledged) alerts."""
        with self._lock:
            return [a for a in self._alerts if not a.acknowledged]

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [a for a in self._alert_history if a.timestamp >= cutoff]

    def register_notification_handler(self, channel: str, handler: Callable) -> None:
        """Register a notification handler."""
        self._notification_handlers[channel] = handler

    def _register_default_handlers(self) -> None:
        """Register default notification handlers."""
        # Console logger
        def console_handler(alert: Alert) -> None:
            logger.warning(
                f"ALERT [{alert.severity.upper()}] {alert.title}: {alert.message}"
            )

        self.register_notification_handler("console", console_handler)

        # Email handler (if configured)
        email_config = self._get_email_config()
        if email_config:
            self.register_notification_handler("email", self._create_email_handler(email_config))

        # Webhook handler (if configured)
        webhook_config = self._get_webhook_config()
        if webhook_config:
            self.register_notification_handler("webhook", self._create_webhook_handler(webhook_config))

    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate a single rule."""
        condition = rule.condition
        params = rule.parameters

        if condition == "threshold":
            return self._evaluate_threshold(params, metrics)
        elif condition == "rate":
            return self._evaluate_rate(params, metrics)
        elif condition == "pattern":
            return self._evaluate_pattern(params, metrics)
        elif condition == "custom":
            # Custom evaluation function
            eval_func = params.get("function")
            if eval_func and callable(eval_func):
                return eval_func(metrics)
        else:
            logger.warning(f"Unknown alert condition type: {condition}")

        return False

    def _evaluate_threshold(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Evaluate threshold-based rule."""
        metric_name = params.get("metric")
        threshold = params.get("threshold")
        operator = params.get("operator", "greater")  # greater, less, equal

        if metric_name not in metrics:
            return False

        value = metrics[metric_name]

        if operator == "greater":
            return value > threshold
        elif operator == "less":
            return value < threshold
        elif operator == "equal":
            return value == threshold
        else:
            logger.warning(f"Unknown threshold operator: {operator}")
            return False

    def _evaluate_rate(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Evaluate rate-based rule."""
        # This would require historical data to calculate rates
        # For now, we'll use a simple threshold on recent faults
        recent_faults = metrics.get("recent_faults", 0)
        max_rate = params.get("max_faults_per_minute", 10)

        return recent_faults > max_rate

    def _evaluate_pattern(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Evaluate pattern-based rule."""
        pattern = params.get("pattern")
        metric_name = params.get("metric")

        if metric_name not in metrics:
            return False

        value = str(metrics[metric_name])

        # Simple pattern matching
        if pattern in value:
            return True

        # Regex pattern (if specified)
        import re
        regex_pattern = params.get("regex_pattern")
        if regex_pattern:
            return bool(re.search(regex_pattern, value))

        return False

    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """Create an alert from a triggered rule."""
        title = f"{rule.name} - {rule.severity.upper()}"
        message = rule.description or self._generate_alert_message(rule, metrics)

        return Alert(
            rule_name=rule.name,
            severity=rule.severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            details={
                "triggering_metrics": metrics,
                "rule_parameters": rule.parameters,
            },
        )

    def _generate_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate a descriptive alert message."""
        if rule.condition == "threshold":
            metric = rule.parameters.get("metric")
            threshold = rule.parameters.get("threshold")
            value = metrics.get(metric, "unknown")
            operator = rule.parameters.get("operator", "greater")
            return f"{metric} is {value} (threshold: {operator} {threshold})"

        elif rule.condition == "rate":
            recent = metrics.get("recent_faults", 0)
            max_rate = rule.parameters.get("max_faults_per_minute", 10)
            return f"{recent} faults in the last minute (max allowed: {max_rate})"

        else:
            return f"Alert rule '{rule.name}' triggered"

    def _process_alert(self, alert: Alert) -> None:
        """Process a triggered alert."""
        with self._lock:
            self._alerts.append(alert)
            self._alert_history.append(alert)

            # Limit alert history size
            if len(self._alert_history) > 10000:
                self._alert_history = self._alert_history[-5000:]

        logger.warning(f"Alert triggered: {alert.rule_name} - {alert.severity}")

        # Send notifications
        self._send_notifications(alert)

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for channel in self.config.notification_channels:
            if not channel.get("enabled", True):
                continue

            handler_name = channel.get("type")
            handler = self._notification_handlers.get(handler_name)

            if handler:
                try:
                    # Check severity filter
                    min_severity = channel.get("min_severity", "info")
                    if self._severity_to_level(alert.severity) >= self._severity_to_level(min_severity):
                        handler(alert, channel)
                except Exception as e:
                    logger.error(f"Notification handler {handler_name} failed: {e}")

    def _severity_to_level(self, severity: str) -> int:
        """Convert severity string to numeric level."""
        levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        return levels.get(severity, 1)

    def _get_email_config(self) -> Optional[Dict[str, Any]]:
        """Get email notification configuration."""
        for channel in self.config.notification_channels:
            if channel.get("type") == "email":
                return channel
        return None

    def _get_webhook_config(self) -> Optional[Dict[str, Any]]:
        """Get webhook notification configuration."""
        for channel in self.config.notification_channels:
            if channel.get("type") == "webhook":
                return channel
        return None

    def _create_email_handler(self, config: Dict[str, Any]) -> Callable:
        """Create email notification handler."""
        smtp_host = config.get("smtp_host", "localhost")
        smtp_port = config.get("smtp_port", 587)
        smtp_user = config.get("smtp_user")
        smtp_password = config.get("smtp_password")
        from_email = config.get("from_email", "alerts@fault-injection.local")
        to_emails = config.get("to_emails", [])

        def email_handler(alert: Alert, channel_config: Dict[str, Any]) -> None:
            try:
                # Create message
                msg = MIMEText(self._format_email_body(alert))
                msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
                msg["From"] = from_email
                msg["To"] = ", ".join(to_emails)

                # Send email
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    if smtp_user and smtp_password:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                    server.send_message(msg)

                logger.info(f"Email alert sent for: {alert.rule_name}")

            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")

        return email_handler

    def _create_webhook_handler(self, config: Dict[str, Any]) -> Callable:
        """Create webhook notification handler."""
        import requests

        webhook_url = config.get("url")
        headers = config.get("headers", {})
        timeout = config.get("timeout", 10)

        def webhook_handler(alert: Alert, channel_config: Dict[str, Any]) -> None:
            try:
                payload = {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "details": alert.details,
                }

                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )

                if response.status_code < 300:
                    logger.info(f"Webhook alert sent for: {alert.rule_name}")
                else:
                    logger.error(f"Webhook alert failed with status: {response.status_code}")

            except Exception as e:
                logger.error(f"Failed to send webhook alert: {e}")

        return webhook_handler

    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as email body."""
        body = f"""
Fault Injection Alert

Rule: {alert.rule_name}
Severity: {alert.severity.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.title}

{alert.message}

Details:
{json.dumps(alert.details, indent=2)}

This alert was generated by the fault injection monitoring system.
"""
        return body

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Alert manager monitoring started")

        while self._running:
            try:
                # Clean up old alerts
                self._cleanup_old_alerts()

                # Sleep for a minute
                time.sleep(60)

            except Exception as e:
                logger.error(f"Error in alert monitor loop: {e}")
                time.sleep(60)

    def _cleanup_old_alerts(self) -> None:
        """Clean up old acknowledged alerts."""
        cutoff = datetime.now() - timedelta(hours=self.config.alert_retention_hours)

        with self._lock:
            # Remove old acknowledged alerts
            self._alerts = [
                a for a in self._alerts
                if not a.acknowledged or a.timestamp >= cutoff
            ]

            # Remove old history
            self._alert_history = [
                a for a in self._alert_history
                if a.timestamp >= cutoff
            ]


# Predefined alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_fault_rate",
        condition="rate",
        parameters={"max_faults_per_minute": 10},
        severity="warning",
        description="High fault injection rate detected",
    ),
    AlertRule(
        name="system_health_critical",
        condition="threshold",
        parameters={"metric": "system_health", "threshold": 25, "operator": "less"},
        severity="critical",
        description="System health is critically low",
    ),
    AlertRule(
        name="low_recovery_success_rate",
        condition="threshold",
        parameters={"metric": "faults_success_rate", "threshold": 0.5, "operator": "less"},
        severity="error",
        description="Recovery success rate is below 50%",
    ),
    AlertRule(
        name="resource_exhaustion",
        condition="threshold",
        parameters={"metric": "avg_memory_percent", "threshold": 90, "operator": "greater"},
        severity="warning",
        description="High memory usage detected",
    ),
]