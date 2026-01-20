# Copyright 2026 Individual Contributor: Aoyang Fang
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


#!/usr/bin/env python3
"""
Tutorial: Getting Started with verl Fault Injection

This tutorial demonstrates the basic usage of the verl fault injection system.
"""

import sys
import time
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.fault_injection import FaultInjectionConfig, FaultOrchestrator
from verl.fault_injection.config import FaultLayer, FaultType, UIFaultConfig, WorkerFaultConfig


def tutorial_1_basic_fault_injection():
    """Tutorial 1: Basic fault injection"""
    print("\n=== Tutorial 1: Basic Fault Injection ===\n")

    # Step 1: Create fault injection configuration
    print("Step 1: Creating fault injection configuration...")
    config = FaultInjectionConfig(
        enabled=True,
        monitoring_enabled=True,
        recovery_enabled=False,  # Disable for basic tutorial
    )

    # Step 2: Initialize fault orchestrator
    print("Step 2: Initializing fault orchestrator...")
    orchestrator = FaultOrchestrator(config)

    # Step 3: Create a simple UI fault
    print("Step 3: Creating a UI fault...")
    ui_fault = UIFaultConfig(
        name="tutorial_hydra_error",
        layer=FaultLayer.UI,
        type=FaultType.HYDRA_CONFIG_ERROR,
        enabled=True,
        hydra_error_type="validate",
        description="Tutorial: Hydra configuration validation error",
    )

    # Step 4: Add fault to orchestrator
    print("Step 4: Adding fault to orchestrator...")
    orchestrator.add_fault(ui_fault)

    # Step 5: Inject fault
    print("Step 5: Injecting fault...")
    result = orchestrator.inject_fault("tutorial_hydra_error")

    # Step 6: Examine result
    print("\nInjection Result:")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.error_message}")
    print(f"  Timestamp: {result.timestamp}")

    # Cleanup
    orchestrator.shutdown()
    print("\nTutorial 1 completed!")


def tutorial_2_worker_layer_faults():
    """Tutorial 2: Worker layer fault injection"""
    print("\n=== Tutorial 2: Worker Layer Faults ===\n")

    # Initialize orchestrator
    config = FaultInjectionConfig(enabled=True)
    orchestrator = FaultOrchestrator(config)

    # Create multiple worker faults
    print("Creating worker layer faults...")

    # CUDA OOM fault
    cuda_oom = WorkerFaultConfig(
        name="tutorial_cuda_oom",
        layer=FaultLayer.WORKER,
        type=FaultType.CUDA_OOM,
        enabled=True,
        oom_size_mb=512,
        description="Tutorial: CUDA out of memory",
    )

    # Worker hang fault
    worker_hang = WorkerFaultConfig(
        name="tutorial_worker_hang",
        layer=FaultLayer.WORKER,
        type=FaultType.WORKER_HANG,
        enabled=True,
        hang_duration=2.0,  # 2 seconds
        hang_stage="forward",
        description="Tutorial: Worker hang during forward pass",
    )

    # Add faults
    orchestrator.add_fault(cuda_oom)
    orchestrator.add_fault(worker_hang)

    # Inject CUDA OOM
    print("\nInjecting CUDA OOM fault...")
    result1 = orchestrator.inject_fault("tutorial_cuda_oom")
    print(f"Result: {result1.status} - {result1.error_message}")

    # Inject worker hang
    print("\nInjecting worker hang fault...")
    start_time = time.time()
    result2 = orchestrator.inject_fault("tutorial_worker_hang")
    end_time = time.time()
    print(f"Result: {result2.status} - Took {end_time - start_time:.2f}s")

    # List all faults
    print(f"\nAvailable faults: {orchestrator.list_faults()}")

    orchestrator.shutdown()
    print("\nTutorial 2 completed!")


def tutorial_3_fault_recovery():
    """Tutorial 3: Fault injection with recovery"""
    print("\n=== Tutorial 3: Fault Recovery ===\n")

    from verl.fault_injection.recovery import RecoveryConfig, RecoveryMode

    # Configure recovery
    recovery_config = RecoveryConfig(
        enabled=True, mode=RecoveryMode.AUTOMATIC, strategies=["retry", "restart"], max_attempts=2, timeout=30.0
    )

    # Initialize with recovery
    config = FaultInjectionConfig(enabled=True, recovery_enabled=True)
    orchestrator = FaultOrchestrator(config)
    orchestrator.set_recovery_config(recovery_config)

    # Create recoverable fault
    print("Creating recoverable fault...")
    recoverable_fault = WorkerFaultConfig(
        name="tutorial_recoverable_oom",
        layer=FaultLayer.WORKER,
        type=FaultType.CUDA_OOM,
        enabled=True,
        oom_size_mb=256,
        recoverable=True,
        recovery_strategies=["retry"],
        description="Tutorial: Recoverable CUDA OOM",
    )

    orchestrator.add_fault(recoverable_fault)

    # Inject with recovery
    print("\nInjecting fault with recovery...")
    result = orchestrator.inject_fault_with_recovery("tutorial_recoverable_oom")

    print("\nRecovery Result:")
    print(f"  Fault Status: {result.fault_status}")
    print(f"  Recovery Status: {result.recovery_status}")
    print(f"  Attempts Made: {result.attempts_made}")
    print(f"  Recovery Message: {result.message}")

    orchestrator.shutdown()
    print("\nTutorial 3 completed!")


def tutorial_4_monitoring_and_alerts():
    """Tutorial 4: Monitoring and alerts"""
    print("\n=== Tutorial 4: Monitoring and Alerts ===\n")

    from verl.fault_injection.monitoring import AlertManager, MetricsCollector

    # Initialize monitoring
    print("Setting up monitoring...")
    collector = MetricsCollector(collection_interval=1)  # 1 second for demo
    collector.start()

    # Configure alerts
    alert_manager = AlertManager()
    alert_manager.add_rule(
        name="tutorial_high_fault_rate",
        condition={"type": "threshold", "metric": "fault_injection_total", "threshold": 3},
        actions=["console"],
        cooldown=5,  # 5 seconds for demo
    )

    # Initialize orchestrator
    config = FaultInjectionConfig(enabled=True, monitoring_enabled=True)
    orchestrator = FaultOrchestrator(config)

    # Create test faults
    for i in range(5):
        fault = UIFaultConfig(
            name=f"tutorial_monitored_fault_{i}",
            layer=FaultLayer.UI,
            type=FaultType.HYDRA_CONFIG_ERROR,
            enabled=True,
            description=f"Tutorial monitored fault {i}",
        )
        orchestrator.add_fault(fault)

    # Inject faults and monitor
    print("\nInjecting faults and monitoring...")
    for i in range(5):
        fault_name = f"tutorial_monitored_fault_{i}"
        print(f"\nInjecting {fault_name}...")
        orchestrator.inject_fault(fault_name)

        # Get current metrics
        metrics = collector.get_metrics()
        print(f"  Total injections: {metrics.get('fault_injection_total', 0)}")
        print(f"  Success rate: {metrics.get('fault_injection_success_rate', 0):.2%}")

        # Check alerts
        alerts = alert_manager.check_alerts()
        if alerts:
            print(f"  ALERTS: {[alert['name'] for alert in alerts]}")

        time.sleep(1)  # Wait for collection interval

    # Cleanup
    collector.stop()
    orchestrator.shutdown()
    print("\nTutorial 4 completed!")


def tutorial_5_scenario_based_injection():
    """Tutorial 5: Scenario-based fault injection"""
    print("\n=== Tutorial 5: Scenario-Based Injection ===\n")

    from verl.fault_injection.scenario import CascadeMode, FaultScenarioConfig, FaultScenarioOrchestrator

    # Initialize orchestrator
    config = FaultInjectionConfig(enabled=True)
    orchestrator = FaultOrchestrator(config)

    # Create scenario orchestrator
    scenario_orchestrator = FaultScenarioOrchestrator(orchestrator)

    # Create a simple cascade scenario
    scenario = FaultScenarioConfig(
        name="tutorial_cascade_scenario",
        description="Tutorial cascade failure scenario",
        cascade_mode=CascadeMode.LINEAR,
        faults=[
            UIFaultConfig(
                name="tutorial_initial_fault",
                layer=FaultLayer.UI,
                type=FaultType.HYDRA_CONFIG_ERROR,
                enabled=True,
                description="Initial fault in cascade",
            ),
            WorkerFaultConfig(
                name="tutorial_dependent_fault",
                layer=FaultLayer.WORKER,
                type=FaultType.WORKER_HANG,
                enabled=True,
                hang_duration=1.0,
                depends_on="tutorial_initial_fault",
                dependency_condition="on_failure",
                description="Dependent fault after initial failure",
            ),
        ],
    )

    # Execute scenario
    print("Executing cascade scenario...")
    results = scenario_orchestrator.execute_scenario(scenario)

    # Display results
    print("\nScenario Execution Results:")
    for i, result in enumerate(results):
        print(f"  Step {i + 1}: {result.fault_name}")
        print(f"    Status: {result.status}")
        print(f"    Timestamp: {result.timestamp}")
        if result.error_message:
            print(f"    Error: {result.error_message}")

    orchestrator.shutdown()
    print("\nTutorial 5 completed!")


def main():
    """Run all tutorials"""
    print("Welcome to verl Fault Injection Tutorials!")
    print("=" * 50)

    try:
        tutorial_1_basic_fault_injection()
        time.sleep(1)

        tutorial_2_worker_layer_faults()
        time.sleep(1)

        tutorial_3_fault_recovery()
        time.sleep(1)

        tutorial_4_monitoring_and_alerts()
        time.sleep(1)

        tutorial_5_scenario_based_injection()

        print("\n" + "=" * 50)
        print("All tutorials completed successfully!")
        print("\nNext steps:")
        print("1. Explore the example configurations in examples/")
        print("2. Read the API documentation in API_DOCUMENTATION.md")
        print("3. Try creating your own fault scenarios")
        print("4. Integrate with your verl training scripts")

    except Exception as e:
        print(f"\nError during tutorials: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
