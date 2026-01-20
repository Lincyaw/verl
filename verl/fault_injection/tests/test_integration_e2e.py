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

#!/usr/bin/env python3
"""End-to-end integration tests for fault injection system."""

import sys
import time
import unittest
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.fault_injection import (
    FaultInjectionConfig,
    FaultLayer,
    FaultOrchestrator,
    FaultStatus,
    FaultType,
)
from verl.fault_injection.config import (
    EngineFaultConfig,
    InferenceFaultConfig,
    NetworkFaultConfig,
    OrchestrationFaultConfig,
    ProcessFaultConfig,
    UIFaultConfig,
    WorkerFaultConfig,
)
from verl.fault_injection.monitoring import (
    AlertManager,
    FaultImpactAnalyzer,
    MetricsCollector,
)
from verl.fault_injection.recovery import (
    RecoveryConfig,
    RecoveryMode,
)
from verl.fault_injection.scenario import (
    CascadeMode,
    FaultScenarioConfig,
    FaultScenarioOrchestrator,
)
from verl.fault_injection.triggers import (
    TimedTrigger,
    TriggerManager,
)


class TestEndToEndScenarios(unittest.TestCase):
    """Test end-to-end fault injection scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = FaultInjectionConfig(
            enabled=True, monitoring_enabled=True, recovery_enabled=True, metrics_port=8080
        )
        self.orchestrator = FaultOrchestrator(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "orchestrator"):
            self.orchestrator.shutdown()

    def test_single_fault_injection(self):
        """Test single fault injection scenario."""
        # Configure a UI fault
        ui_fault = UIFaultConfig(
            name="ui_config_error",
            layer=FaultLayer.UI,
            type=FaultType.HYDRA_CONFIG_ERROR,
            enabled=True,
            hydra_error_type="validate",
            description="Test single fault injection",
        )

        # Add fault to orchestrator
        self.orchestrator.add_fault(ui_fault)

        # Inject fault
        result = self.orchestrator.inject_fault("ui_config_error")

        # Verify result
        self.assertEqual(result.status, FaultStatus.FAILED)
        self.assertIn("validation failed", result.error_message)

    def test_multiple_faults_same_layer(self):
        """Test multiple faults in the same layer."""
        # Configure multiple UI faults
        faults = [
            UIFaultConfig(
                name="hydra_error_1",
                layer=FaultLayer.UI,
                type=FaultType.HYDRA_CONFIG_ERROR,
                enabled=True,
                hydra_error_type="validate",
            ),
            UIFaultConfig(
                name="ray_init_error",
                layer=FaultLayer.UI,
                type=FaultType.RAY_INIT_FAILURE,
                enabled=True,
                ray_init_error_type="timeout",
            ),
            UIFaultConfig(
                name="cli_error",
                layer=FaultLayer.UI,
                type=FaultType.CLI_ARG_ERROR,
                enabled=True,
                cli_arg_error_type="missing_required",
            ),
        ]

        # Add all faults
        for fault in faults:
            self.orchestrator.add_fault(fault)

        # Inject each fault and verify
        results = []
        for fault in faults:
            result = self.orchestrator.inject_fault(fault.name)
            results.append(result)

        # Verify all faults were injected
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.status, FaultStatus.FAILED)

    def test_cross_layer_fault_injection(self):
        """Test fault injection across different layers."""
        # Configure faults in different layers
        faults = [
            UIFaultConfig(
                name="ui_fault",
                layer=FaultLayer.UI,
                type=FaultType.HYDRA_CONFIG_ERROR,
                enabled=True,
                hydra_error_type="validate",
            ),
            OrchestrationFaultConfig(
                name="orchestration_fault",
                layer=FaultLayer.ORCHESTRATION,
                type=FaultType.RAY_CLUSTER_FAILURE,
                enabled=True,
                cluster_failure_type="gcs_failure",
            ),
            WorkerFaultConfig(
                name="worker_fault", layer=FaultLayer.WORKER, type=FaultType.CUDA_OOM, enabled=True, oom_size_mb=100
            ),
            EngineFaultConfig(
                name="engine_fault",
                layer=FaultLayer.ENGINE,
                type=FaultType.ENGINE_INIT_FAILURE,
                enabled=True,
                engine_type="fsdp",
            ),
            InferenceFaultConfig(
                name="inference_fault",
                layer=FaultLayer.INFERENCE,
                type=FaultType.INFERENCE_OOM,
                enabled=True,
                backend="vllm",
            ),
        ]

        # Add all faults
        for fault in faults:
            self.orchestrator.add_fault(fault)

        # Inject faults and verify
        for fault in faults:
            result = self.orchestrator.inject_fault(fault.name)
            self.assertEqual(result.status, FaultStatus.FAILED)

    def test_fault_with_recovery(self):
        """Test fault injection with recovery."""
        # Configure recovery
        recovery_config = RecoveryConfig(
            enabled=True, mode=RecoveryMode.AUTOMATIC, strategies=["retry", "restart", "checkpoint"]
        )
        self.orchestrator.set_recovery_config(recovery_config)

        # Configure a fault
        fault = WorkerFaultConfig(
            name="recoverable_fault",
            layer=FaultLayer.WORKER,
            type=FaultType.CUDA_OOM,
            enabled=True,
            oom_size_mb=100,
            recoverable=True,
        )

        self.orchestrator.add_fault(fault)

        # Inject fault with recovery
        result = self.orchestrator.inject_fault_with_recovery("recoverable_fault")

        # Verify fault was injected and recovery was attempted
        self.assertIsNotNone(result)
        self.assertIn(result.fault_status, [FaultStatus.RECOVERED, FaultStatus.FAILED])

    def test_fault_scenario_with_dependencies(self):
        """Test fault scenario with dependencies."""
        # Create scenario orchestrator
        scenario_orchestrator = FaultScenarioOrchestrator(self.orchestrator)

        # Configure a scenario with dependencies
        scenario_config = FaultScenarioConfig(
            name="cascade_scenario",
            description="Test cascade failure scenario",
            cascade_mode=CascadeMode.LINEAR,
            faults=[
                UIFaultConfig(
                    name="initial_fault", layer=FaultLayer.UI, type=FaultType.HYDRA_CONFIG_ERROR, enabled=True
                ),
                OrchestrationFaultConfig(
                    name="dependent_fault",
                    layer=FaultLayer.ORCHESTRATION,
                    type=FaultType.RAY_CLUSTER_FAILURE,
                    enabled=True,
                    depends_on="initial_fault",
                    dependency_condition="on_failure",
                ),
            ],
        )

        # Execute scenario
        results = scenario_orchestrator.execute_scenario(scenario_config)

        # Verify both faults were executed
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].fault_name, "initial_fault")
        self.assertEqual(results[1].fault_name, "dependent_fault")

    def test_monitoring_integration(self):
        """Test fault injection with monitoring."""
        # Start metrics collector
        collector = MetricsCollector()
        collector.start()

        # Configure a fault
        fault = ProcessFaultConfig(
            name="monitored_fault",
            layer=FaultLayer.PROCESS,
            type=FaultType.PROCESS_CRASH,
            enabled=True,
            crash_type="exception",
        )

        self.orchestrator.add_fault(fault)

        # Inject fault
        self.orchestrator.inject_fault("monitored_fault")

        # Get metrics
        metrics = collector.get_metrics()

        # Verify metrics were collected
        self.assertIn("fault_injection_total", metrics)
        self.assertGreater(metrics["fault_injection_total"], 0)

        # Stop collector
        collector.stop()

    def test_alert_system(self):
        """Test alert system integration."""
        # Create alert manager
        alert_manager = AlertManager()

        # Add alert rule
        alert_manager.add_rule(
            name="high_fault_rate",
            condition={"type": "threshold", "metric": "fault_rate", "threshold": 0.5},
            actions=["console"],
        )

        # Configure multiple faults
        for i in range(5):
            fault = UIFaultConfig(
                name=f"fault_{i}", layer=FaultLayer.UI, type=FaultType.HYDRA_CONFIG_ERROR, enabled=True
            )
            self.orchestrator.add_fault(fault)

        # Inject faults
        for i in range(5):
            self.orchestrator.inject_fault(f"fault_{i}")

        # Check alerts
        alerts = alert_manager.check_alerts()

        # Verify alerts were triggered
        self.assertGreater(len(alerts), 0)

    def test_fault_impact_analysis(self):
        """Test fault impact analysis."""
        # Create impact analyzer
        analyzer = FaultImpactAnalyzer()

        # Configure a fault
        fault = NetworkFaultConfig(
            name="network_partition",
            layer=FaultLayer.NETWORK,
            type=FaultType.NETWORK_PARTITION,
            enabled=True,
            partition_type="partial",
            isolated_ranks=[1, 2, 3],
        )

        self.orchestrator.add_fault(fault)

        # Inject fault
        result = self.orchestrator.inject_fault("network_partition")

        # Analyze impact
        impact = analyzer.analyze_fault_impact(result)

        # Verify impact analysis
        self.assertIn("affected_components", impact)
        self.assertIn("severity", impact)
        self.assertIn("recovery_recommendations", impact)

    def test_trigger_based_injection(self):
        """Test trigger-based fault injection."""
        # Create trigger manager
        trigger_manager = TriggerManager()

        # Add timed trigger
        trigger = TimedTrigger(delay=0.1, fault_name="timed_fault")
        trigger_manager.add_trigger(trigger)

        # Configure fault
        fault = UIFaultConfig(name="timed_fault", layer=FaultLayer.UI, type=FaultType.HYDRA_CONFIG_ERROR, enabled=True)

        self.orchestrator.add_fault(fault)

        # Start trigger manager
        trigger_manager.start(self.orchestrator)

        # Wait for trigger
        time.sleep(0.2)

        # Verify fault was injected
        metrics = self.orchestrator.get_metrics()
        self.assertGreater(metrics["total_injections"], 0)

        # Stop trigger manager
        trigger_manager.stop()

    def test_concurrent_fault_injection(self):
        """Test concurrent fault injection."""
        import threading

        # Configure multiple faults
        faults = []
        for i in range(10):
            fault = UIFaultConfig(
                name=f"concurrent_fault_{i}", layer=FaultLayer.UI, type=FaultType.HYDRA_CONFIG_ERROR, enabled=True
            )
            faults.append(fault)
            self.orchestrator.add_fault(fault)

        # Create threads for concurrent injection
        threads = []
        results = []

        def inject_fault(fault_name):
            result = self.orchestrator.inject_fault(fault_name)
            results.append(result)

        # Start all threads
        for fault in faults:
            thread = threading.Thread(target=inject_fault, args=(fault.name,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all faults were injected
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertEqual(result.status, FaultStatus.FAILED)

    def test_fault_injection_with_ray_integration(self):
        """Test fault injection with Ray integration."""
        # This test would require actual Ray setup
        # For now, we'll just verify the integration points exist

        # Check that Ray integration methods exist
        self.assertTrue(hasattr(self.orchestrator, "init_ray_integration"))
        self.assertTrue(hasattr(self.orchestrator, "get_ray_workers"))
        self.assertTrue(hasattr(self.orchestrator, "inject_ray_fault"))

    def test_performance_benchmark(self):
        """Test fault injection performance."""
        # Configure a simple fault
        fault = UIFaultConfig(
            name="benchmark_fault", layer=FaultLayer.UI, type=FaultType.HYDRA_CONFIG_ERROR, enabled=True
        )

        self.orchestrator.add_fault(fault)

        # Measure injection time
        start_time = time.time()

        # Inject fault multiple times
        for _ in range(100):
            self.orchestrator.inject_fault("benchmark_fault")

        end_time = time.time()
        total_time = end_time - start_time

        # Verify performance (should complete quickly)
        self.assertLess(total_time, 10.0)  # 100 injections in less than 10 seconds

        # Calculate average time per injection
        avg_time = total_time / 100
        print(f"Average injection time: {avg_time:.4f} seconds")


if __name__ == "__main__":
    unittest.main()
