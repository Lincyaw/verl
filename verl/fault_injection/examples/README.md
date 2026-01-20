# verl Fault Injection Examples

This directory contains example configurations and tutorials for the verl fault injection system.

## Files Overview

### Configuration Examples

- **[`comprehensive_example.yaml`](comprehensive_example.yaml)** - A comprehensive configuration file demonstrating all available fault types and system features
- **[`ui_faults.yaml`](ui_faults.yaml)** - UI layer fault injection examples
- **[`orchestration_faults.yaml`](orchestration_faults.yaml)** - Ray orchestration layer fault examples
- **[`worker_faults.yaml`](worker_faults.yaml)** - Worker process fault examples
- **[`engine_faults.yaml`](engine_faults.yaml)** - Training engine fault examples
- **[`inference_faults.yaml`](inference_faults.yaml)** - Inference backend fault examples
- **[`scenario_faults.yaml`](scenario_faults.yaml)** - Complex fault scenarios with dependencies
- **[`recovery_config.yaml`](recovery_config.yaml)** - Recovery mechanism configuration
- **[`monitoring_config.yaml`](monitoring_config.yaml)** - Monitoring and alerting configuration

### Tutorials

- **[`tutorial_getting_started.py`](tutorial_getting_started.py)** - Step-by-step tutorial for beginners
- **[`monitoring_demo.py`](monitoring_demo.py)** - Demonstrates monitoring and dashboard features

## Quick Start

### 1. Run the Getting Started Tutorial

```bash
python tutorial_getting_started.py
```

This will walk you through:
- Basic fault injection
- Worker layer faults
- Fault recovery
- Monitoring and alerts
- Scenario-based injection

### 2. Try a Simple Fault Injection

```python
from verl.fault_injection import FaultOrchestrator, FaultInjectionConfig
from verl.fault_injection.config import UIFaultConfig, FaultType, FaultLayer

# Initialize
config = FaultInjectionConfig(enabled=True)
orchestrator = FaultOrchestrator(config)

# Create a fault
fault = UIFaultConfig(
    name="hydra_error",
    layer=FaultLayer.UI,
    type=FaultType.HYDRA_CONFIG_ERROR,
    enabled=True,
    hydra_error_type="validate"
)

# Inject
orchestrator.add_fault(fault)
result = orchestrator.inject_fault("hydra_error")
print(f"Result: {result.status} - {result.error_message}")
```

### 3. Load Configuration from YAML

```python
import yaml
from verl.fault_injection import FaultInjectionConfig

# Load configuration
with open('comprehensive_example.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Create config
config = FaultInjectionConfig.from_dict(config_dict['fault_injection'])

# Use with orchestrator
orchestrator = FaultOrchestrator(config)
```

## Example Scenarios

### Training Disruption Scenario

Simulates common failures during training:
1. Configuration validation error
2. Ray cluster stress
3. Worker synchronization failure

```yaml
# Load from scenario_faults.yaml
scenario: training_disruption
```

### Network Chaos Scenario

Tests system resilience to network issues:
1. High latency
2. Packet loss
3. Network partition
4. Connection failures

```yaml
# Load from scenario_faults.yaml
scenario: network_chaos
```

### System Cascade Failure

Demonstrates cascade failure patterns:
1. Initial resource exhaustion
2. Triggered worker failures
3. Engine initialization failure
4. Inference degradation

```yaml
# Load from comprehensive_example.yaml
scenario: system_cascade_failure
```

## Customizing Examples

### Adding New Faults

1. Copy an existing fault configuration
2. Modify parameters as needed
3. Test the fault injection

### Creating Custom Scenarios

1. Define fault dependencies in `scenario_faults.yaml`
2. Set cascade mode (linear, tree, burst)
3. Configure trigger conditions

### Adjusting Recovery Settings

1. Edit `recovery_config.yaml`
2. Choose recovery strategies
3. Set retry limits and timeouts

## Monitoring Integration

### Enable Dashboard

```yaml
monitoring:
  dashboard_enabled: true
  port: 8080
```

Access at: http://localhost:8080

### Configure Alerts

```yaml
alerts:
  - name: "high_fault_rate"
    condition:
      type: "threshold"
      metric: "fault_injection_rate"
      threshold: 10
    actions: ["console", "email", "webhook"]
```

## Best Practices

1. **Start Simple**: Begin with single faults before creating complex scenarios
2. **Test Gradually**: Test each fault type in isolation first
3. **Monitor Impact**: Always enable monitoring to understand fault effects
4. **Document Changes**: Keep notes on which faults work best for your use case
5. **Use Recovery**: Enable recovery mechanisms for automatic fault handling

## Troubleshooting

### Fault Not Injecting

- Check if fault injection is enabled: `enabled: true`
- Verify fault is not disabled in configuration
- Check for dependency issues in scenarios

### High Memory Usage

- Reduce monitoring frequency
- Limit number of stored metrics
- Use performance mode: `mode: "MINIMAL"`

### Recovery Not Working

- Ensure recovery is enabled: `recovery_enabled: true`
- Check recovery strategies are configured
- Verify fault is marked as recoverable

## Next Steps

1. Read the [API Documentation](../API_DOCUMENTATION.md)
2. Check the [Usage Guide](../USAGE_GUIDE.md)
3. Explore the [test files](../tests/) for more examples
4. Integrate with your verl training scripts