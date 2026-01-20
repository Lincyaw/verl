# Copyright 2026 Aoyang Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Integration between monitoring system and fault injection orchestrator."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..base import FaultResult

if TYPE_CHECKING:
    from ..orchestrator import FaultOrchestrator
from .alerts import DEFAULT_ALERT_RULES, AlertConfig, AlertManager
from .analyzer import FaultImpactAnalyzer
from .collector import MetricsCollector
from .dashboard import DashboardConfig, DashboardServer

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """Integrates monitoring components with the fault injection system."""

    def __init__(
        self,
        orchestrator: "FaultOrchestrator",
        enable_metrics: bool = True,
        enable_dashboard: bool = True,
        enable_alerts: bool = True,
        enable_analyzer: bool = True,
        dashboard_config: Optional[DashboardConfig] = None,
        alert_config: Optional[AlertConfig] = None,
    ):
        self.orchestrator = orchestrator
        self.enable_metrics = enable_metrics
        self.enable_dashboard = enable_dashboard
        self.enable_alerts = enable_alerts
        self.enable_analyzer = enable_analyzer

        # Initialize components
        self.metrics_collector: Optional[MetricsCollector] = None
        self.dashboard_server: Optional[DashboardServer] = None
        self.alert_manager: Optional[AlertManager] = None
        self.impact_analyzer: Optional[FaultImpactAnalyzer] = None

        if enable_metrics:
            self.metrics_collector = MetricsCollector()

        if enable_dashboard and dashboard_config:
            if not self.metrics_collector:
                raise ValueError("Metrics collector must be enabled for dashboard")
            self.dashboard_server = DashboardServer(self.metrics_collector, dashboard_config)

        if enable_alerts and alert_config:
            self.alert_manager = AlertManager(alert_config)
            # Add default rules if none specified
            if not alert_config.rules:
                for rule in DEFAULT_ALERT_RULES:
                    self.alert_manager.add_rule(rule)

        if enable_analyzer:
            self.impact_analyzer = FaultImpactAnalyzer()

        # Register callbacks with orchestrator
        self._register_callbacks()

    def start(self) -> None:
        """Start all monitoring components."""
        logger.info("Starting monitoring integration")

        if self.metrics_collector:
            self.metrics_collector.start()
            logger.info("Metrics collector started")

        if self.dashboard_server:
            self.dashboard_server.start()
            logger.info(f"Dashboard server started on port {self.dashboard_server.config.port}")

        if self.alert_manager:
            self.alert_manager.start()
            logger.info("Alert manager started")

    def stop(self) -> None:
        """Stop all monitoring components."""
        logger.info("Stopping monitoring integration")

        if self.dashboard_server:
            self.dashboard_server.stop()

        if self.alert_manager:
            self.alert_manager.stop()

        if self.metrics_collector:
            self.metrics_collector.stop()

    def _register_callbacks(self) -> None:
        """Register callbacks with the orchestrator."""
        # Register fault completion callback
        self.orchestrator.register_target_update_callback(self._on_target_update)

    def _on_fault_completed(self, fault_result: FaultResult) -> None:
        """Handle fault completion event."""
        logger.debug(f"Processing fault completion: {fault_result.fault_id}")

        # Record in metrics collector
        if self.metrics_collector:
            self.metrics_collector.record_fault(fault_result)

        # Analyze impact
        if self.impact_analyzer:
            analysis = self.impact_analyzer.analyze_fault_impact(fault_result)
            logger.info(
                f"Fault impact analysis: {fault_result.fault_id} - "
                f"Cascade risk: {analysis.cascade_risk}, "
                f"Affected services: {len(analysis.affected_services)}"
            )

            # Add to historical data
            self.impact_analyzer.add_historical_fault(fault_result)

        # Evaluate alerts
        if self.alert_manager and self.metrics_collector:
            metrics = self.metrics_collector.get_current_metrics()
            alerts = self.alert_manager.evaluate_rules(metrics)
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts from fault: {fault_result.fault_id}")

    def _on_target_update(self, targets: list) -> None:
        """Handle target update event."""
        # Could be used to update monitoring based on available targets
        logger.debug(f"Target update received: {len(targets)} targets available")

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get a summary of monitoring status and metrics."""
        summary = {
            "monitoring_enabled": {
                "metrics": self.enable_metrics,
                "dashboard": self.enable_dashboard,
                "alerts": self.enable_alerts,
                "analyzer": self.enable_analyzer,
            },
            "components_status": {},
            "current_metrics": None,
            "active_alerts": 0,
            "impact_summary": None,
        }

        # Check component status
        if self.metrics_collector:
            summary["components_status"]["metrics_collector"] = "running"
            summary["current_metrics"] = self.metrics_collector.get_current_metrics()
        else:
            summary["components_status"]["metrics_collector"] = "disabled"

        if self.dashboard_server:
            summary["components_status"]["dashboard_server"] = "running"
        else:
            summary["components_status"]["dashboard_server"] = "disabled"

        if self.alert_manager:
            summary["components_status"]["alert_manager"] = "running"
            summary["active_alerts"] = len(self.alert_manager.get_active_alerts())
        else:
            summary["components_status"]["alert_manager"] = "disabled"

        if self.impact_analyzer:
            summary["components_status"]["impact_analyzer"] = "running"
            summary["impact_summary"] = self.impact_analyzer.get_impact_summary(hours=1)
        else:
            summary["components_status"]["impact_analyzer"] = "disabled"

        return summary

    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL if dashboard is enabled."""
        if self.dashboard_server:
            config = self.dashboard_server.config
            return f"http://{config.host}:{config.port}"
        return None

    def acknowledge_alert(self, alert_index: int, user: str) -> bool:
        """Acknowledge an alert."""
        if self.alert_manager:
            return self.alert_manager.acknowledge_alert(alert_index, user)
        return False

    def add_custom_alert_rule(self, rule) -> None:
        """Add a custom alert rule."""
        if self.alert_manager:
            self.alert_manager.add_rule(rule)

    def get_impact_analysis(self, fault_result: FaultResult) -> Optional[dict[str, Any]]:
        """Get detailed impact analysis for a fault."""
        if self.impact_analyzer:
            analysis = self.impact_analyzer.analyze_fault_impact(fault_result)
            return {
                "analysis_id": analysis.analysis_id,
                "cascade_risk": analysis.cascade_risk,
                "affected_services": analysis.affected_services,
                "recommendations": analysis.recommendations,
                "confidence_score": analysis.confidence_score,
                "impact_metrics": {
                    "fault_count": analysis.impact_metrics.fault_count,
                    "failure_rate": analysis.impact_metrics.failure_rate,
                    "recovery_rate": analysis.impact_metrics.recovery_rate,
                    "cascade_probability": analysis.impact_metrics.cascade_probability,
                    "system_degradation": analysis.impact_metrics.system_degradation,
                },
            }
        return None


def create_monitoring_integration(
    orchestrator: "FaultOrchestrator",
    config: Optional[dict[str, Any]] = None,
) -> MonitoringIntegration:
    """Create a monitoring integration with the given configuration."""
    if config is None:
        config = {}

    # Dashboard configuration
    dashboard_config = None
    if config.get("dashboard", {}).get("enabled", True):
        dashboard_config = DashboardConfig(**config.get("dashboard", {}))

    # Alert configuration
    alert_config = None
    if config.get("alerts", {}).get("enabled", True):
        alert_config = AlertConfig(**config.get("alerts", {}))

    # Create integration
    integration = MonitoringIntegration(
        orchestrator=orchestrator,
        enable_metrics=config.get("metrics", {}).get("enabled", True),
        enable_dashboard=config.get("dashboard", {}).get("enabled", True),
        enable_alerts=config.get("alerts", {}).get("enabled", True),
        enable_analyzer=config.get("impact_analysis", {}).get("enabled", True),
        dashboard_config=dashboard_config,
        alert_config=alert_config,
    )

    return integration
