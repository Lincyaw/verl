# verl Fault Injection System - Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Layer-Specific Examples](#layer-specific-examples)
4. [Advanced Scenarios](#advanced-scenarios)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

The fault injection system is included with verl. No additional installation is required.

### Quick Example

```python
from verl.fault_injection import FaultOrchestrator, FaultInjectionConfig
from verl.fault_injection.config import UIFaultConfig, FaultType, FaultLayer

# Initialize the system
config = FaultInjectionConfig(enabled=True)
orchestrator = FaultOrchestrator(config)

# Create a simple fault
fault = UIFaultConfig(
    name="example_fault",
    layer=FaultLayer.UI,
    type=FaultType.HYDRA_CONFIG_ERROR,
    enabled=True,
    hydra_error_type="validate"
)

# Add and inject
orchestrator.add_fault(fault)
result = orchestrator.inject_fault("example_fault")

print(f"Fault injection result: {result.status}")
print(f"Error message: {result.error_message}")
```

## Basic Usage

### 1. Creating Faults

#### UI Layer Faults

```python
# Hydra configuration error
hydra_fault = UIFaultConfig(
    name="hydra_validation_error",
    layer=FaultLayer.UI,
    type=FaultType.HYDRA_CONFIG_ERROR,
    enabled=True,
    hydra_error_type="validate",  # validate, merge, compose
    description="Simulate Hydra config validation failure"
)

# Ray initialization failure
ray_fault = UIFaultConfig(
    name="ray_init_timeout",
    layer=FaultLayer.UI,
    type=FaultType.RAY_INIT_FAILURE,
    enabled=True,
    ray_init_error_type="timeout",
    ray_init_timeout=30.0,
    description="Simulate Ray cluster initialization timeout"
)

# CLI argument error
cli_fault = UIFaultConfig(
    name="missing_model_path",
    layer=FaultLayer.UI,
    type=FaultType.CLI_ARG_ERROR,
    enabled=True,
    cli_arg_error_type="missing_required",
    missing_arg="--model-path",
    description="Simulate missing required CLI argument"
)
```

#### Worker Layer Faults

```python
# CUDA OOM fault
cuda_oom = WorkerFaultConfig(
    name="cuda_out_of_memory",
    layer=FaultLayer.WORKER,
    type=FaultType.CUDA_OOM,
    enabled=True,
    oom_size_mb=2048,
    description="Simulate CUDA out of memory error"
)

# FSDP sync failure
fsdp_sync = WorkerFaultConfig(
    name="fsdp_all_reduce_failure",
    layer=FaultLayer.WORKER,
    type=FaultType.FSDP_SYNC_FAILURE,
    enabled=True,
    sync_operation="all_reduce",
    world_size=8,
    rank=0,
    description="Simulate FSDP all_reduce synchronization failure"
)

# Worker crash
worker_crash = WorkerFaultConfig(
    name="worker_process_crash",
    layer=FaultLayer.WORKER,
    type=FaultType.WORKER_CRASH,
    enabled=True,
    crash_type="exception",  # exception, exit, signal
    description="Simulate worker process crash"
)
```

### 2. Managing Faults

```python
# Add multiple faults
orchestrator.add_fault(hydra_fault)
orchestrator.add_fault(cuda_oom)
orchestrator.add_fault(worker_crash)

# List all faults
fault_names = orchestrator.list_faults()
print(f"Available faults: {fault_names}")

# Enable/disable faults
orchestrator.disable_fault("hydra_validation_error")
orchestrator.enable_fault("hydra_validation_error")

# Remove a fault
orchestrator.remove_fault("missing_model_path")
```

### 3. Injecting Faults

```python
# Single fault injection
result = orchestrator.inject_fault("cuda_out_of_memory")
print(f"Status: {result.status}")
print(f"Message: {result.error_message}")

# Batch injection
results = orchestrator.inject_faults([
    "fsdp_all_reduce_failure",
    "worker_process_crash"
])

for name, result in results.items():
    print(f"{name}: {result.status}")
```

## Layer-Specific Examples

### UI Layer Examples

```python
# Simulate UI freeze during configuration
ui_freeze = UIFaultConfig(
    name="config_ui_freeze",
    layer=FaultLayer.UI,
    type=FaultType.UI_FREEZE,
    enabled=True,
    freeze_duration=10.0,  # Freeze for 10 seconds
    description="Simulate UI freeze during configuration loading"
)

# Environment variable error
env_fault = UIFaultConfig(
    name="missing_cuda_devices",
    layer=FaultLayer.UI,
    type=FaultType.ENV_VAR_ERROR,
    enabled=True,
    env_var_error_type="not_set",
    env_var_name="CUDA_VISIBLE_DEVICES",
    description="Simulate missing CUDA environment variable"
)
```

### Orchestration Layer Examples

```python
# Ray cluster failure
cluster_failure = OrchestrationFaultConfig(
    name="ray_cluster_gcs_failure",
    layer=FaultLayer.ORCHESTRATION,
    type=FaultType.RAY_CLUSTER_FAILURE,
    enabled=True,
    cluster_failure_type="gcs_failure",
    duration=30.0,
    description="Simulate Ray GCS failure"
)

# Actor crash
actor_crash = OrchestrationFaultConfig(
    name="actor_worker_crash",
    layer=FaultLayer.ORCHESTRATION,
    type=FaultType.ACTOR_CRASH,
    enabled=True,
    actor_crash_type="exception",
    actor_name_pattern="RolloutWorker*",
    description="Simulate Ray actor crash"
)

# Resource exhaustion
resource_exhaustion = OrchestrationFaultConfig(
    name="gpu_memory_exhaustion",
    layer=FaultLayer.ORCHESTRATION,
    type=FaultType.RESOURCE_EXHAUSTION,
    enabled=True,
    resource_type="memory",
    exhaustion_percentage=0.95,
    duration=60.0,
    description="Simulate GPU memory exhaustion"
)
```

### Engine Layer Examples

```python
# Checkpoint corruption
checkpoint_corrupt = EngineFaultConfig(
    name="model_checkpoint_corruption",
    layer=FaultLayer.ENGINE,
    type=FaultType.CHECKPOINT_CORRUPTION,
    enabled=True,
    corruption_type="random_bytes",
    checkpoint_path="/tmp/model_checkpoint.pt",
    description="Simulate model checkpoint corruption"
)

# NCCL failure
nccl_failure = EngineFaultConfig(
    name="nccl_all_reduce_timeout",
    layer=FaultLayer.ENGINE,
    type=FaultType.NCCL_FAILURE,
    enabled=True,
    nccl_failure_type="timeout",
    timeout_seconds=30.0,
    description="Simulate NCCL all_reduce timeout"
)

# Device mesh error
device_mesh_error = EngineFaultConfig(
    name="device_mesh_shape_error",
    layer=FaultLayer.ENGINE,
    type=FaultType.DEVICE_MESH_ERROR,
    enabled=True,
    mesh_error_type="invalid_shape",
    mesh_shape=[2, 2, 2],
    description="Simulate invalid device mesh configuration"
)
```

### Inference Layer Examples

```python
# Inference OOM (vLLM)
vllm_oom = InferenceFaultConfig(
    name="vllm_kv_cache_oom",
    layer=FaultLayer.INFERENCE,
    type=FaultType.INFERENCE_OOM,
    enabled=True,
    backend="vllm",
    oom_type="kv_cache",
    requested_memory_mb=4096,
    description="Simulate vLLM KV cache OOM"
)

# Scheduler deadlock
scheduler_deadlock = InferenceFaultConfig(
    name="request_queue_deadlock",
    layer=FaultLayer.INFERENCE,
    type=FaultType.SCHEDULER_DEADLOCK,
    enabled=True,
    deadlock_type="request_queue",
    max_queue_size=1000,
    description="Simulate request queue deadlock"
)

# Compilation failure
compilation_failure = InferenceFaultConfig(
    name="cuda_graph_compilation_failure",
    layer=FaultLayer.INFERENCE,
    type=FaultType.COMPILATION_FAILURE,
    enabled=True,
    compilation_failure_type="cuda_graph",
    description="Simulate CUDA graph compilation failure"
)
```

## Advanced Scenarios

### 1. Fault Scenarios with Dependencies

```python
from verl.fault_injection.scenario import FaultScenarioConfig, CascadeMode

# Create a cascading failure scenario
cascade_scenario = FaultScenarioConfig(
    name="training_cascade_failure",
    description="Simulate cascade failure during training",
    cascade_mode=CascadeMode.LINEAR,
    faults=[
        # Initial fault: UI configuration error
        UIFaultConfig(
            name="initial_config_error",
            layer=FaultLayer.UI,
            type=FaultType.HYDRA_CONFIG_ERROR,
            enabled=True,
            hydra_error_type="validate"
        ),
        # Dependent fault: Triggered after UI fault
        OrchestrationFaultConfig(
            name="ray_cluster_stress",
            layer=FaultLayer.ORCHESTRATION,
            type=FaultType.RESOURCE_EXHAUSTION,
            enabled=True,
            resource_type="memory",
            exhaustion_percentage=0.9,
            depends_on="initial_config_error",
            dependency_condition="on_failure"
        ),
        # Final fault: Worker crash due to resource exhaustion
        WorkerFaultConfig(
            name="worker_crash_due_to_resources",
            layer=FaultLayer.WORKER,
            type=FaultType.WORKER_CRASH,
            enabled=True,
            crash_type="exception",
            depends_on="ray_cluster_stress",
            dependency_condition="on_failure"
        )
    ]
)

# Execute scenario
scenario_orchestrator = FaultScenarioOrchestrator(fault_orchestrator)
results = scenario_orchestrator.execute_scenario(cascade_scenario)
```

### 2. Trigger-Based Fault Injection

```python
from verl.fault_injection.triggers import (
    TimedTrigger, CountTrigger, MetricThresholdTrigger
)

# Time-based trigger
timed_trigger = TimedTrigger(
    delay=300,  # 5 minutes
    fault_name="delayed_fault",
    recurring=True,
    interval=600  # Repeat every 10 minutes
)

# Count-based trigger
count_trigger = CountTrigger(
    count=100,
    fault_name="periodic_fault",
    reset_on_injection=True
)

# Metric-based trigger
metric_trigger = MetricThresholdTrigger(
    metric="gpu_memory_usage",
    threshold=0.95,
    comparison=">",
    fault_name="memory_pressure_fault",
    duration=60  # Must exceed threshold for 60 seconds
)

# Add triggers to manager
trigger_manager = TriggerManager()
trigger_manager.add_trigger(timed_trigger)
trigger_manager.add_trigger(count_trigger)
trigger_manager.add_trigger(metric_trigger)

# Start monitoring
trigger_manager.start(fault_orchestrator)
```

### 3. Recovery Strategies

```python
from verl.fault_injection.recovery import (
    RecoveryConfig, RecoveryMode, HierarchicalRecoveryManager
)

# Configure recovery
recovery_config = RecoveryConfig(
    enabled=True,
    mode=RecoveryMode.AUTOMATIC,
    strategies=["retry", "restart", "checkpoint", "degradation"],
    max_attempts=3,
    timeout=300.0,
    backoff_factor=2.0
)

# Create recovery manager
recovery_manager = HierarchicalRecoveryManager(recovery_config)

# Configure recoverable fault
recoverable_fault = WorkerFaultConfig(
    name="recoverable_cuda_oom",
    layer=FaultLayer.WORKER,
    type=FaultType.CUDA_OOM,
    enabled=True,
    oom_size_mb=1024,
    recoverable=True,
    recovery_strategies=["retry", "restart"],
    max_retries=2
)

# Inject with recovery
result = fault_orchestrator.inject_fault_with_recovery("recoverable_cuda_oom")
if result.status == "RECOVERED":
    print("Fault was successfully recovered!")
elif result.status == "FAILED":
    print("Recovery attempts failed, manual intervention required")
```

### 4. Monitoring and Alerting

```python
from verl.fault_injection.monitoring import (
    MetricsCollector, AlertManager, FaultImpactAnalyzer
)

# Start metrics collection
collector = MetricsCollector(
    collection_interval=10,  # Collect every 10 seconds
    export_metrics=True,
    metrics_port=8080
)
collector.start()

# Configure alerts
alert_manager = AlertManager()

# High fault rate alert
alert_manager.add_rule(
    name="high_fault_rate",
    condition={
        "type": "threshold",
        "metric": "fault_injection_rate",
        "threshold": 10,  # 10 faults per minute
        "window": 60  # Over 1 minute window
    },
    actions=["console", "webhook"],
    webhook_url="https://alerts.example.com/webhook",
    cooldown=300  # 5 minutes
)

# System health alert
alert_manager.add_rule(
    name="system_health_degraded",
    condition={
        "type": "threshold",
        "metric": "system_health_score",
        "threshold": 0.7,
        "comparison": "<"
    },
    actions=["console", "email"],
    email_recipients=["ops@example.com"]
)

# Impact analysis
analyzer = FaultImpactAnalyzer()
impact = analyzer.analyze_fault_impact(fault_result)
print(f"Fault impact severity: {impact['severity']}")
print(f"Recommended actions: {impact['recovery_recommendations']}")
```

## Integration Examples

### Integration with verl Training Script

```python
# main_ppo.py
import hydra
from verl.trainer.main_ppo import run_ppo
from verl.fault_injection import FaultOrchestrator, FaultInjectionConfig

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(cfg):
    # Initialize fault injection if configured
    fault_orchestrator = None
    if hasattr(cfg, 'fault_injection') and cfg.fault_injection.enabled:
        fault_config = FaultInjectionConfig.from_hydra(cfg.fault_injection)
        fault_orchestrator = FaultOrchestrator(fault_config)

        # Add configured faults
        for fault_cfg in cfg.fault_injection.faults:
            fault_orchestrator.add_fault(fault_cfg)

    # Run training with fault injection
    run_ppo(cfg, fault_orchestrator=fault_orchestrator)

if __name__ == "__main__":
    main()
```

### Custom Fault Injector

```python
from verl.fault_injection.base import BaseFaultInjector, BaseFaultConfig
from verl.fault_injection import FaultType, FaultLayer, FaultContext, FaultResult

@dataclass
class CustomFaultConfig(BaseFaultConfig):
    custom_parameter: str = "default_value"
    severity: int = 5

class CustomFaultInjector(BaseFaultInjector):
    def __init__(self, config: CustomFaultConfig):
        super().__init__(config)
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        # Custom injection logic
        if self.config.severity > 7:
            return FaultResult(
                fault_name=self.config.name,
                fault_type=FaultType.CUSTOM_FAULT,
                status=FaultStatus.FAILED,
                error_message=f"Custom fault injected with severity {self.config.severity}",
                timestamp=time.time()
            )
        else:
            return FaultResult(
                fault_name=self.config.name,
                fault_type=FaultType.CUSTOM_FAULT,
                status=FaultStatus.SUCCESS,
                message="Custom fault injected successfully",
                timestamp=time.time()
            )

    def recover(self, context: FaultContext) -> RecoveryResult:
        # Custom recovery logic
        return RecoveryResult(
            fault_name=self.config.name,
            strategy_name="custom_recovery",
            status=RecoveryStatus.SUCCESS,
            message="Custom recovery completed",
            timestamp=time.time()
        )

# Register custom injector
from verl.fault_injection import FaultInjectorRegistry
registry = FaultInjectorRegistry()
registry.register("custom_fault", CustomFaultInjector)
```

### Performance Testing

```python
from verl.fault_injection.performance import (
    set_performance_mode, PerformanceProfiler
)

# Set performance mode
set_performance_mode("BALANCED")

# Profile fault injection
with PerformanceProfiler() as profiler:
    # Inject multiple faults
    for i in range(100):
        fault_orchestrator.inject_fault(f"test_fault_{i}")

# Get performance metrics
metrics = profiler.get_metrics()
print(f"Average injection time: {metrics['avg_injection_time']}ms")
print(f"Total overhead: {metrics['total_overhead']}ms")
```

## Best Practices

### 1. Start Simple

- Begin with single faults in isolation
- Test each fault type before combining
- Use immediate triggers initially

### 2. Monitor Everything

- Always enable monitoring
- Set up alerts for critical metrics
- Track fault injection success rates

### 3. Use Meaningful Names

```python
# Good
fault = UIFaultConfig(
    name="hydra_missing_required_field_validation_error",
    ...
)

# Bad
fault = UIFaultConfig(
    name="fault1",
    ...
)
```

### 4. Document Your Faults

```python
fault = WorkerFaultConfig(
    name="fsdp_gradient_sync_timeout_rank0",
    layer=FaultLayer.WORKER,
    type=FaultType.FSDP_SYNC_FAILURE,
    enabled=True,
    description="""
    Simulates FSDP gradient synchronization timeout on rank 0.
    This typically occurs when:
    - Network latency is high
    - GPU memory is fragmented
    - NCCL operations are blocked
    Expected behavior: Training should stall or fail with timeout error
    """,
    sync_operation="all_reduce",
    world_size=8,
    rank=0,
    timeout=30.0
)
```

### 5. Test Recovery Mechanisms

```python
# Always test recovery
fault = WorkerFaultConfig(
    name="test_recovery_cuda_oom",
    layer=FaultLayer.WORKER,
    type=FaultType.CUDA_OOM,
    enabled=True,
    recoverable=True,
    recovery_strategies=["retry", "checkpoint"],
    max_retries=3,
    description="Test CUDA OOM recovery mechanisms"
)
```

### 6. Use Scenario Templates

```python
from verl.fault_injection.scenario import FaultScenarioTemplate

# Use built-in templates
network_chaos = FaultScenarioTemplate.NETWORK_CHAOS.value
results = scenario_orchestrator.execute_template(network_chaos)

# Create reusable templates
@dataclass
class MyTrainingFailureTemplate(FaultScenarioTemplate):
    def __init__(self):
        super().__init__(
            name="typical_training_failure",
            description="Common failure pattern during training",
            faults=[
                UIFaultConfig(...),
                WorkerFaultConfig(...),
                EngineFaultConfig(...)
            ],
            cascade_mode=CascadeMode.TREE
        )
```

## Troubleshooting

### Common Issues

#### Fault Not Triggering

```python
# Check if fault is enabled
status = fault_orchestrator.get_fault_status("my_fault")
print(f"Fault status: {status}")

# Check dependencies
fault_config = fault_orchestrator.get_fault_config("my_fault")
if hasattr(fault_config, 'depends_on'):
    print(f"Depends on: {fault_config.depends_on}")
    dep_status = fault_orchestrator.get_fault_status(fault_config.depends_on)
    print(f"Dependency status: {dep_status}")

# Check triggers
triggers = trigger_manager.get_triggers_for_fault("my_fault")
print(f"Active triggers: {triggers}")
```

#### High Memory Usage

```python
# Reduce monitoring frequency
config = FaultInjectionConfig(
    enabled=True,
    monitoring_enabled=True,
    metrics_collection_interval=60,  # Collect every minute
    max_stored_metrics=1000  # Limit stored metrics
)

# Use performance mode
set_performance_mode("MINIMAL")

# Clear old faults
fault_orchestrator.clear_injection_history()
```

#### Recovery Not Working

```python
# Check recovery configuration
recovery_config = fault_orchestrator.get_recovery_config()
print(f"Recovery enabled: {recovery_config.enabled}")
print(f"Recovery mode: {recovery_config.mode}")
print(f"Available strategies: {recovery_config.strategies}")

# Check fault recoverability
fault = fault_orchestrator.get_fault_config("my_fault")
print(f"Is recoverable: {getattr(fault, 'recoverable', False)}")
print(f"Recovery strategies: {getattr(fault, 'recovery_strategies', [])}")

# Test recovery manually
result = fault_orchestrator.test_recovery("my_fault")
print(f"Recovery test result: {result}")
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("verl.fault_injection")
logger.setLevel(logging.DEBUG)

# Enable fault injection debug mode
config = FaultInjectionConfig(
    enabled=True,
    debug_mode=True,
    verbose_logging=True,
    save_injection_traces=True
)

# View injection trace
trace = fault_orchestrator.get_injection_trace("my_fault")
print(json.dumps(trace, indent=2))
```

### Getting Help

```python
# Get system status
status = fault_orchestrator.get_system_status()
print(f"System health: {status['health_score']}")
print(f"Active faults: {status['active_faults']}")
print(f"Total injections: {status['total_injections']}")

# Get help for specific components
help_text = fault_orchestrator.get_help("recovery")
print(help_text)

# Generate diagnostic report
report = fault_orchestrator.generate_diagnostic_report()
with open("fault_injection_diagnostic.json", "w") as f:
    json.dump(report, f, indent=2)
```