"""Unit tests for monitoring and observability components."""

import json
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from verl.fault_injection.base import FaultLayer, FaultStatus, FaultResult
from verl.fault_injection.monitoring.collector import (
    AggregatedMetrics,
    FaultMetrics,
    MetricsCollector,
    SystemMetrics,
)
from verl.fault_injection.monitoring.alerts import (
    Alert,
    AlertConfig,
    AlertManager,
    AlertRule,
    DEFAULT_ALERT_RULES,
)
from verl.fault_injection.monitoring.analyzer import (
    ComponentDependency,
    FaultImpactAnalyzer,
    ImpactAnalysisResult,
    ImpactMetrics,
)
from verl.fault_injection.monitoring.dashboard import DashboardConfig, DashboardServer


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(
            collection_interval=0.1,
            aggregation_window=1,
            max_history_size=100,
        )

    def tearDown(self):
        """Clean up after tests."""
        self.collector.stop()

    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector.collection_interval == 0.1
        assert self.collector.aggregation_window == 1
        assert self.collector.max_history_size == 100
        assert not self.collector._running

    def test_start_stop(self):
        """Test starting and stopping the collector."""
        self.collector.start()
        assert self.collector._running
        assert self.collector._collection_thread is not None

        self.collector.stop()
        assert not self.collector._running

    def test_record_fault(self):
        """Test recording fault metrics."""
        fault_result = FaultResult(
            fault_id="test_fault_1",
            fault_type="test_type",
            layer=FaultLayer.UI,
            status=FaultStatus.COMPLETED,
            start_time=time.time() - 1,
            end_time=time.time(),
            target_info={"target": "test"},
            error_message=None,
            recovery_time_ms=100,
        )

        self.collector.record_fault(fault_result)

        # Check fault was recorded
        assert len(self.collector._fault_metrics) == 1
        fault_metric = self.collector._fault_metrics[0]
        assert fault_metric.fault_id == "test_fault_1"
        assert fault_metric.status == FaultStatus.COMPLETED
        assert fault_metric.recovery_time_ms == 100

    def test_get_current_metrics_empty(self):
        """Test getting current metrics when empty."""
        metrics = self.collector.get_current_metrics()
        assert metrics["faults_total"] == 0
        assert metrics["faults_active"] == 0
        assert metrics["faults_success_rate"] == 0.0
        assert metrics["system_health"] == "unknown"

    def test_get_current_metrics_with_data(self):
        """Test getting current metrics with data."""
        # Add some fault metrics
        for i in range(5):
            fault_result = FaultResult(
                fault_id=f"test_fault_{i}",
                fault_type="test_type",
                layer=FaultLayer.UI,
                status=FaultStatus.COMPLETED,
                start_time=time.time() - 1,
                end_time=time.time(),
            )
            self.collector.record_fault(fault_result)

        metrics = self.collector.get_current_metrics()
        assert metrics["faults_total"] == 5
        assert metrics["faults_success_rate"] == 1.0  # All successful
        assert "by_layer" in metrics
        assert "by_type" in metrics

    def test_get_aggregated_metrics(self):
        """Test getting aggregated metrics."""
        # Add fault metrics
        for i in range(3):
            fault_result = FaultResult(
                fault_id=f"test_fault_{i}",
                fault_type="test_type",
                layer=FaultLayer.UI,
                status=FaultStatus.COMPLETED,
                start_time=time.time() - 1,
                end_time=time.time(),
                duration=1.0,
            )
            self.collector.record_fault(fault_result)

        aggregation = self.collector.get_aggregated_metrics(window_seconds=300)
        assert aggregation is not None
        assert aggregation.total_faults == 3
        assert aggregation.successful_faults == 3
        assert aggregation.avg_injection_duration_ms == 1000.0

    def test_system_health_calculation(self):
        """Test system health calculation."""
        # Mock system metrics
        system_metric = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=95.0,
            memory_percent=95.0,
            memory_available_mb=100.0,
        )
        self.collector._system_metrics.append(system_metric)

        health = self.collector._calculate_system_health()
        assert health == "critical"

    def test_callback_registration(self):
        """Test registering callbacks."""
        callback_called = False

        def test_callback(metric_type, metric):
            nonlocal callback_called
            callback_called = True
            assert metric_type == "fault"
            assert isinstance(metric, FaultMetrics)

        self.collector.register_callback(test_callback)

        fault_result = FaultResult(
            fault_id="test_fault",
            fault_type="test_type",
            layer=FaultLayer.UI,
            status=FaultStatus.COMPLETED,
            start_time=time.time() - 1,
            end_time=time.time(),
        )
        self.collector.record_fault(fault_result)

        assert callback_called


class TestAlertManager(unittest.TestCase):
    """Test cases for AlertManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AlertConfig(
            enabled=True,
            max_alerts_per_minute=10,
            alert_retention_hours=24,
        )
        self.manager = AlertManager(self.config)

    def tearDown(self):
        """Clean up after tests."""
        self.manager.stop()

    def test_initialization(self):
        """Test alert manager initialization."""
        assert self.manager.config.enabled is True
        assert len(self.manager._alerts) == 0
        assert len(self.manager._alert_history) == 0

    def test_add_remove_rule(self):
        """Test adding and removing alert rules."""
        rule = AlertRule(
            name="test_rule",
            condition="threshold",
            parameters={"metric": "faults_total", "threshold": 10},
            severity="warning",
        )

        self.manager.add_rule(rule)
        assert len(self.manager.config.rules) == 1
        assert self.manager.config.rules[0].name == "test_rule"

        removed = self.manager.remove_rule("test_rule")
        assert removed is True
        assert len(self.manager.config.rules) == 0

    def test_evaluate_threshold_rule(self):
        """Test evaluating threshold-based rules."""
        rule = AlertRule(
            name="high_faults",
            condition="threshold",
            parameters={"metric": "faults_total", "threshold": 5, "operator": "greater"},
            severity="warning",
        )

        # Test with metric above threshold
        metrics = {"faults_total": 10}
        triggered = self.manager._evaluate_rule(rule, metrics)
        assert triggered is True

        # Test with metric below threshold
        metrics = {"faults_total": 3}
        triggered = self.manager._evaluate_rule(rule, metrics)
        assert triggered is False

    def test_evaluate_rate_rule(self):
        """Test evaluating rate-based rules."""
        rule = AlertRule(
            name="high_fault_rate",
            condition="rate",
            parameters={"max_faults_per_minute": 5},
            severity="error",
        )

        # Test with high fault rate
        metrics = {"recent_faults": 10}
        triggered = self.manager._evaluate_rule(rule, metrics)
        assert triggered is True

        # Test with normal fault rate
        metrics = {"recent_faults": 3}
        triggered = self.manager._evaluate_rule(rule, metrics)
        assert triggered is False

    def test_alert_creation(self):
        """Test alert creation."""
        rule = AlertRule(
            name="test_rule",
            condition="threshold",
            parameters={"metric": "faults_total", "threshold": 5},
            severity="warning",
            description="Test alert",
        )

        metrics = {"faults_total": 10}
        alert = self.manager._create_alert(rule, metrics)

        assert alert.rule_name == "test_rule"
        assert alert.severity == "warning"
        assert "Test alert" in alert.message
        assert alert.timestamp is not None

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        # Create an alert
        alert = Alert(
            rule_name="test_rule",
            severity="warning",
            title="Test Alert",
            message="Test message",
            timestamp=datetime.now(),
        )

        self.manager._alerts.append(alert)
        assert alert.acknowledged is False

        # Acknowledge the alert
        acknowledged = self.manager.acknowledge_alert(0, "test_user")
        assert acknowledged is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None

    def test_notification_handlers(self):
        """Test notification handler registration."""
        handler_called = False

        def test_handler(alert: Alert) -> None:
            nonlocal handler_called
            handler_called = True

        self.manager.register_notification_handler("test", test_handler)
        assert "test" in self.manager._notification_handlers

        # Test console handler exists
        assert "console" in self.manager._notification_handlers

    def test_rule_cooldown(self):
        """Test alert rule cooldown."""
        rule = AlertRule(
            name="test_rule",
            condition="threshold",
            parameters={"metric": "faults_total", "threshold": 5},
            severity="warning",
            cooldown_seconds=60,
        )

        metrics = {"faults_total": 10}

        # First evaluation should trigger
        triggered1 = self.manager._evaluate_rule(rule, metrics)
        assert triggered1 is True

        # Record last triggered time
        self.manager._rule_last_triggered["test_rule"] = datetime.now()

        # Second evaluation within cooldown should not trigger
        triggered2 = self.manager._evaluate_rule(rule, metrics)
        assert triggered2 is False


class TestFaultImpactAnalyzer(unittest.TestCase):
    """Test cases for FaultImpactAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FaultImpactAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        assert len(self.analyzer._dependencies) > 0
        assert len(self.analyzer._layer_dependencies) > 0

    def test_add_dependency(self):
        """Test adding component dependencies."""
        self.analyzer.add_dependency(
            "test_component",
            depends_on=["base1", "base2"],
            dependents=["dep1", "dep2"],
            layer=FaultLayer.UI,
            criticality="high",
        )

        assert "test_component" in self.analyzer._dependencies
        dep = self.analyzer._dependencies["test_component"]
        assert dep.layer == FaultLayer.UI
        assert dep.criticality == "high"
        assert "base1" in dep.depends_on
        assert "dep1" in dep.dependents

    def test_analyze_fault_impact(self):
        """Test fault impact analysis."""
        fault_result = FaultResult(
            fault_id="test_fault_1",
            fault_type="ray_cluster_failure",
            layer=FaultLayer.ORCHESTRATION,
            status=FaultStatus.COMPLETED,
            start_time=time.time() - 2,
            end_time=time.time(),
            duration=2.0,
            target_info={"node": "test-node"},
            error_message="Simulated failure",
            recovery_time_ms=500,
        )

        analysis = self.analyzer.analyze_fault_impact(fault_result)

        assert analysis.fault_id == "test_fault_1"
        assert analysis.fault_type == "ray_cluster_failure"
        assert len(analysis.affected_services) > 0
        assert analysis.cascade_risk in ["low", "medium", "high", "critical"]
        assert len(analysis.recommendations) > 0
        assert 0.0 <= analysis.confidence_score <= 1.0

    def test_get_affected_components(self):
        """Test getting affected components for a fault."""
        fault_result = FaultResult(
            fault_id="test_fault",
            fault_type="hydra_config_error",
            layer=FaultLayer.UI,
            status=FaultStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time(),
        )

        affected = self.analyzer._get_affected_components(fault_result)

        assert "hydra_config" in affected
        assert FaultLayer.UI in {
            self.analyzer._dependencies[c].layer
            for c in affected
            if c in self.analyzer._dependencies
        }

    def test_calculate_cascade_probability(self):
        """Test cascade probability calculation."""
        fault_result = FaultResult(
            fault_id="test_fault",
            fault_type="ray_cluster_failure",
            layer=FaultLayer.ORCHESTRATION,
            status=FaultStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time(),
        )

        affected = {"ray_cluster", "gcs", "actor_system"}
        probability = self.analyzer._calculate_cascade_probability(fault_result, affected)

        assert 0.0 <= probability <= 1.0
        # Orchestration layer faults should have higher cascade probability
        assert probability > 0.1

    def test_calculate_system_degradation(self):
        """Test system degradation calculation."""
        fault_result = FaultResult(
            fault_id="test_fault",
            fault_type="engine_init_failure",
            layer=FaultLayer.ENGINE,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
        )

        affected = {"training_engine", "checkpoint_system"}
        degradation = self.analyzer._calculate_system_degradation(fault_result, affected)

        assert 0.0 <= degradation <= 1.0
        # Failed faults should have higher degradation
        assert degradation > 0.2

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        fault_result = FaultResult(
            fault_id="test_fault",
            fault_type="ray_cluster_failure",
            layer=FaultLayer.ORCHESTRATION,
            status=FaultStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time(),
        )

        # Create impact metrics with high cascade probability
        impact_metrics = ImpactMetrics()
        impact_metrics.cascade_probability = 0.8
        impact_metrics.system_degradation = 0.6
        impact_metrics.affected_components = {"ray_cluster", "gcs"}

        recommendations = self.analyzer._generate_recommendations(fault_result, impact_metrics)

        assert len(recommendations) > 0
        # Should include cascade-related recommendations
        assert any("cascade" in r for r in recommendations)
        assert any("degradation" in r for r in recommendations)

    def test_get_impact_summary(self):
        """Test getting impact summary."""
        # Add some historical faults
        for i in range(5):
            fault_result = FaultResult(
                fault_id=f"test_fault_{i}",
                fault_type="test_type",
                layer=FaultLayer.UI,
                status=FaultStatus.COMPLETED,
                start_time=time.time() - i,
                end_time=time.time(),
            )
            self.analyzer.add_historical_fault(fault_result)

        summary = self.analyzer.get_impact_summary(hours=24)

        assert summary["total_faults"] == 5
        assert "affected_layers" in summary
        assert "cascade_risk_distribution" in summary
        assert "recommendations" in summary


class TestDashboardServer(unittest.TestCase):
    """Test cases for DashboardServer."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics_collector = MetricsCollector()
        self.config = DashboardConfig(
            host="127.0.0.1",
            port=0,  # Use random port for testing
            update_interval=0.1,
        )
        self.dashboard = DashboardServer(self.metrics_collector, self.config)

    def tearDown(self):
        """Clean up after tests."""
        self.dashboard.stop()
        self.metrics_collector.stop()

    def test_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.metrics_collector == self.metrics_collector
        assert self.dashboard.config == self.config
        assert self.dashboard.app is not None

    def test_dashboard_routes(self):
        """Test dashboard API routes."""
        with self.dashboard.app.test_client() as client:
            # Test index route
            response = client.get("/")
            assert response.status_code == 200
            assert b"Fault Injection Monitor" in response.data

            # Test current metrics route
            response = client.get("/api/metrics/current")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "faults_total" in data

            # Test system health route
            response = client.get("/api/system/health")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "health_score" in data
            assert "status" in data

    def test_dashboard_with_faults(self):
        """Test dashboard with fault data."""
        # Add some fault metrics
        for i in range(3):
            fault_result = FaultResult(
                fault_id=f"test_fault_{i}",
                fault_type="test_type",
                layer=FaultLayer.UI,
                status=FaultStatus.COMPLETED,
                start_time=time.time() - 1,
                end_time=time.time(),
            )
            self.metrics_collector.record_fault(fault_result)

        with self.dashboard.app.test_client() as client:
            # Test faults by layer route
            response = client.get("/api/faults/by-layer")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "ui" in data
            assert data["ui"] == 3

            # Test faults by type route
            response = client.get("/api/faults/by-type")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "test_type" in data
            assert data["test_type"] == 3

            # Test fault history route
            response = client.get("/api/faults/history?limit=10")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data) == 3


class TestIntegration(unittest.TestCase):
    """Integration tests for monitoring components."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        # Create components
        collector = MetricsCollector()
        alert_config = AlertConfig(enabled=True)
        alert_manager = AlertManager(alert_config)
        analyzer = FaultImpactAnalyzer()

        # Add alert rule
        rule = AlertRule(
            name="high_fault_rate",
            condition="threshold",
            parameters={"metric": "faults_total", "threshold": 2},
            severity="warning",
        )
        alert_manager.add_rule(rule)

        # Start components
        collector.start()

        # Simulate fault injection
        fault_results = []
        for i in range(3):
            fault_result = FaultResult(
                fault_id=f"integration_test_fault_{i}",
                fault_type="ray_cluster_failure",
                layer=FaultLayer.ORCHESTRATION,
                status=FaultStatus.COMPLETED,
                start_time=time.time() - 1,
                end_time=time.time(),
                duration=1.0,
            )
            fault_results.append(fault_result)

            # Record in collector
            collector.record_fault(fault_result)

            # Add to analyzer history
            analyzer.add_historical_fault(fault_result)

        # Get current metrics
        metrics = collector.get_current_metrics()
        assert metrics["faults_total"] == 3

        # Evaluate alerts
        alerts = alert_manager.evaluate_rules(metrics)
        assert len(alerts) > 0  # Should trigger high_fault_rate rule

        # Analyze impact
        analysis = analyzer.analyze_fault_impact(fault_results[0])
        assert analysis.fault_id == "integration_test_fault_0"
        assert len(analysis.affected_services) > 0

        # Get impact summary
        summary = analyzer.get_impact_summary(hours=1)
        assert summary["total_faults"] == 3

        # Clean up
        collector.stop()
        alert_manager.stop()


if __name__ == "__main__":
    unittest.main()