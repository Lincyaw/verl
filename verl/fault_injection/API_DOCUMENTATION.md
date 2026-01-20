# verl Fault Injection System - API Documentation

## Overview

The verl fault injection system provides a comprehensive framework for injecting various types of faults into the verl training pipeline. It supports 5 layers of fault injection: UI, Orchestration, Worker, Engine, and Inference.

## Quick Start

```python
from verl.fault_injection import FaultOrchestrator, FaultInjectionConfig
from verl.fault_injection.config import UIFaultConfig, FaultType, FaultLayer

# Create configuration
config = FaultInjectionConfig(enabled=True)

# Create orchestrator
orchestrator = FaultOrchestrator(config)

# Define a fault
fault = UIFaultConfig(
    name="hydra_error",
    layer=FaultLayer.UI,
    type=FaultType.HYDRA_CONFIG_ERROR,
    enabled=True,
    hydra_error_type="validate"
)

# Add and inject fault
orchestrator.add_fault(fault)
result = orchestrator.inject_fault("hydra_error")
```

## Core Components

### FaultOrchestrator

The main entry point for fault injection.

```python
class FaultOrchestrator:
    def __init__(self, config: FaultInjectionConfig)
    def add_fault(self, fault_config: BaseFaultConfig) -> None
    def inject_fault(self, fault_name: str) -> FaultResult
    def inject_fault_with_recovery(self, fault_name: str) -> RecoveryResult
    def get_fault_status(self, fault_name: str) -> FaultStatus
    def list_faults(self) -> List[str]
    def remove_fault(self, fault_name: str) -> bool
    def enable_fault(self, fault_name: str) -> bool
    def disable_fault(self, fault_name: str) -> bool
```

### Configuration Classes

#### FaultInjectionConfig

Main configuration for the fault injection system.

```python
@dataclass
class FaultInjectionConfig:
    enabled: bool = False
    monitoring_enabled: bool = True
    recovery_enabled: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    max_concurrent_faults: int = 10
    default_timeout: float = 30.0
```

#### Layer-Specific Configurations

Each fault layer has its own configuration class:

- `UIFaultConfig` - UI layer faults
- `OrchestrationFaultConfig` - Ray orchestration faults
- `WorkerFaultConfig` - Worker process faults
- `EngineFaultConfig` - Training engine faults
- `InferenceFaultConfig` - Inference backend faults
- `ProcessFaultConfig` - System process faults
- `NetworkFaultConfig` - Network-related faults

### Fault Types

The system supports various fault types organized by layer:

#### UI Layer Faults
- `HYDRA_CONFIG_ERROR` - Hydra configuration errors
- `RAY_INIT_FAILURE` - Ray initialization failures
- `CLI_ARG_ERROR` - Command line argument errors
- `ENV_VAR_ERROR` - Environment variable errors
- `UI_FREEZE` - UI freezing
- `UI_CRASH` - UI crashes

#### Orchestration Layer Faults
- `RAY_CLUSTER_FAILURE` - Ray cluster failures
- `ACTOR_CRASH` - Ray actor crashes
- `RESOURCE_EXHAUSTION` - Resource exhaustion
- `TASK_SCHEDULING_FAILURE` - Task scheduling failures
- `RESOURCE_POOL_FAULT` - Resource pool faults
- `GCS_FAILURE` - Global Control Service failures
- `NETWORK_PARTITION` - Network partitions
- `PLACEMENT_GROUP_FAULT` - Placement group faults

#### Worker Layer Faults
- `FSDP_SYNC_FAILURE` - FSDP synchronization failures
- `FSDP_SHARDING_ERROR` - FSDP sharding errors
- `MEGATRON_SYNC_FAILURE` - Megatron synchronization failures
- `MEGATRON_PIPELINE_ERROR` - Megatron pipeline errors
- `GRADIENT_SYNC_TIMEOUT` - Gradient synchronization timeouts
- `CUDA_OOM` - CUDA out of memory
- `CUDA_MEMORY_FRAGMENTATION` - CUDA memory fragmentation
- `WORKER_CRASH` - Worker crashes
- `WORKER_HANG` - Worker hangs

#### Engine Layer Faults
- `ENGINE_INIT_FAILURE` - Engine initialization failures
- `ENGINE_HANG` - Engine hangs
- `CHECKPOINT_CORRUPTION` - Checkpoint corruption
- `NCCL_FAILURE` - NCCL communication failures
- `DEVICE_MESH_ERROR` - Device mesh errors
- `PRECISION_ERROR` - Precision conversion errors
- `DEVICE_MAP_ERROR` - Device mapping errors

#### Inference Layer Faults
- `INFERENCE_OOM` - Inference out of memory
- `SCHEDULER_DEADLOCK` - Scheduler deadlocks
- `COMPILATION_FAILURE` - Compilation failures

#### Process Layer Faults
- `PROCESS_CRASH` - Process crashes
- `PROCESS_HANG` - Process hangs
- `MEMORY_LEAK` - Memory leaks
- `FILE_DESCRIPTOR_EXHAUSTION` - File descriptor exhaustion

#### Network Layer Faults
- `NETWORK_LATENCY` - Network latency
- `NETWORK_PACKET_LOSS` - Packet loss
- `NETWORK_BANDWIDTH_THROTTLE` - Bandwidth throttling
- `NETWORK_PARTITION` - Network partitions
- `DNS_RESOLUTION_FAILURE` - DNS resolution failures
- `CONNECTION_REFUSED` - Connection refused
- `CONNECTION_TIMEOUT` - Connection timeouts
- `CONNECTION_RESET` - Connection resets

## Recovery System

### Recovery Configuration

```python
from verl.fault_injection.recovery import RecoveryConfig, RecoveryMode

recovery_config = RecoveryConfig(
    enabled=True,
    mode=RecoveryMode.AUTOMATIC,
    strategies=["retry", "restart", "checkpoint"],
    max_attempts=3,
    timeout=60.0
)
```

### Recovery Strategies

Available recovery strategies:

- `ignore` - Ignore the fault (no-op)
- `retry` - Retry the failed operation
- `restart` - Restart the affected component
- `process` - Process-level recovery
- `resource` - Resource exhaustion recovery
- `checkpoint` - Recovery from checkpoint
- `degradation` - Functional degradation
- `manual` - Manual intervention required
- `ray` - Ray-specific recovery
- `redeploy` - Full system redeployment

### Hierarchical Recovery

```python
from verl.fault_injection.recovery import HierarchicalRecoveryManager

recovery_manager = HierarchicalRecoveryManager(recovery_config)
recovery_result = recovery_manager.attempt_recovery(fault_result)
```

## Monitoring and Observability

### Metrics Collection

```python
from verl.fault_injection.monitoring import MetricsCollector

collector = MetricsCollector()
collector.start()

# Get metrics
metrics = collector.get_metrics()
print(f"Total faults: {metrics['fault_injection_total']}")
print(f"Success rate: {metrics['fault_injection_success_rate']}")
```

### Dashboard

The system provides a web dashboard accessible at `http://localhost:8080` when monitoring is enabled.

### Alerting

```python
from verl.fault_injection.monitoring import AlertManager

alert_manager = AlertManager()

# Add alert rule
alert_manager.add_rule(
    name="high_fault_rate",
    condition={"type": "threshold", "metric": "fault_rate", "threshold": 0.5},
    actions=["console", "email", "webhook"],
    cooldown=300  # 5 minutes
)
```

### Fault Impact Analysis

```python
from verl.fault_injection.monitoring import FaultImpactAnalyzer

analyzer = FaultImpactAnalyzer()
impact = analyzer.analyze_fault_impact(fault_result)

print(f"Affected components: {impact['affected_components']}")
print(f"Severity: {impact['severity']}")
print(f"Recovery recommendations: {impact['recovery_recommendations']}")
```

## Scenario-Based Fault Injection

### Fault Scenarios

```python
from verl.fault_injection.scenario import FaultScenarioConfig, CascadeMode

scenario = FaultScenarioConfig(
    name="training_disruption",
    description="Simulate training disruption",
    cascade_mode=CascadeMode.LINEAR,
    faults=[
        UIFaultConfig(...),
        WorkerFaultConfig(...),
        EngineFaultConfig(...)
    ]
)
```

### Triggers

```python
from verl.fault_injection.triggers import TimedTrigger, CountTrigger, MetricThresholdTrigger

# Time-based trigger
timed_trigger = TimedTrigger(
    delay=60,  # 60 seconds
    fault_name="delayed_fault"
)

# Count-based trigger
count_trigger = CountTrigger(
    count=5,
    fault_name="countdown_fault"
)

# Metric-based trigger
metric_trigger = MetricThresholdTrigger(
    metric="gpu_memory_usage",
    threshold=0.9,
    comparison=">",
    fault_name="memory_fault"
)
```

## Integration with verl

### Basic Integration

The fault injection system integrates seamlessly with verl's training pipeline:

```python
# In main_ppo.py
from verl.fault_injection import FaultOrchestrator, FaultInjectionConfig

# Initialize fault injection
config = FaultInjectionConfig.from_hydra(cfg.fault_injection)
orchestrator = FaultOrchestrator(config)

# Pass to trainer
trainer = RayPPOTrainer(
    config=cfg,
    fault_orchestrator=orchestrator,
    ...
)
```

### Hook Points

The system provides hook points in the training pipeline:

- `hook_rollout_generation` - Before rollout generation
- `hook_actor_update` - Before actor model update
- `hook_critic_update` - Before critic model update

### Configuration Files

Example configuration in YAML:

```yaml
# fault_injection.yaml
fault_injection:
  enabled: true
  monitoring_enabled: true
  recovery_enabled: true

  faults:
    - name: hydra_config_error
      layer: UI
      type: HYDRA_CONFIG_ERROR
      enabled: true
      hydra_error_type: validate

    - name: cuda_oom
      layer: WORKER
      type: CUDA_OOM
      enabled: true
      oom_size_mb: 1024

  recovery:
    enabled: true
    mode: AUTOMATIC
    strategies: ["retry", "restart", "checkpoint"]

  monitoring:
    port: 8080
    alerts:
      - name: high_fault_rate
        condition:
          type: threshold
          metric: fault_rate
          threshold: 0.5
        actions: ["console"]
```

## Performance Optimization

### Performance Modes

The system supports different performance modes:

- `DISABLED` - No fault injection overhead
- `MINIMAL` - Minimal overhead, basic faults only
- `BALANCED` - Balanced performance and features
- `AGGRESSIVE` - All features enabled, higher overhead

```python
from verl.fault_injection.performance import set_performance_mode

set_performance_mode("BALANCED")
```

### Caching

The system uses LRU caching to avoid repeated operations:

```python
from verl.fault_injection.performance import LRUCache

cache = LRUCache(max_size=1000, ttl=300)  # 5 minute TTL
```

### Asynchronous Injection

For high-performance scenarios, use asynchronous injection:

```python
from verl.fault_injection.async_injectors import AsyncBaseFaultInjector

class MyAsyncInjector(AsyncBaseFaultInjector):
    async def inject_async(self, context: FaultContext) -> FaultResult:
        # Async injection logic
        pass
```

## Best Practices

1. **Start Small**: Begin with simple faults and gradually increase complexity
2. **Monitor Impact**: Always enable monitoring to understand fault impact
3. **Use Recovery**: Enable recovery mechanisms for automatic fault handling
4. **Test Gradually**: Test faults in isolation before combining them
5. **Document Scenarios**: Document your fault scenarios for reproducibility
6. **Performance Testing**: Test performance impact in your environment

## Troubleshooting

### Common Issues

1. **Fault Not Injecting**
   - Check if fault injection is enabled
   - Verify fault configuration
   - Check fault dependencies

2. **Recovery Not Working**
   - Ensure recovery is enabled
   - Check recovery strategy configuration
   - Verify fault is marked as recoverable

3. **High Overhead**
   - Reduce monitoring frequency
   - Use performance mode
   - Enable caching

4. **Dashboard Not Accessible**
   - Check monitoring port configuration
   - Verify monitoring is enabled
   - Check firewall settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("verl.fault_injection").setLevel(logging.DEBUG)
```

## API Reference

For detailed API documentation, see the docstrings in the source code or generate documentation using:

```bash
cd docs && make html
```