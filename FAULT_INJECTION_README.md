# verl Fault Injection System

A comprehensive fault injection framework for distributed training systems, designed to generate rich log datasets for anomaly detection and root cause analysis.

## Overview

This system provides a command-line wrapper that injects faults into distributed training processes without requiring any code modifications. It supports 10 fault layers covering all aspects of distributed ML training.

## Architecture

The system uses a **hybrid injection architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fault Injection Wrapper                       │
│              python verl_fault_inject.py --scenario xxx.yaml     │
├─────────────────────────────────────────────────────────────────┤
│  1. Read fault configuration from YAML                           │
│  2. Classify faults: system-level vs code-level                  │
│  3. Set environment variable FAULT_INJECTION_CONFIG              │
│  4. Inject fault_injector.py into command (for code-level)       │
│  5. Start child process with modified command                    │
│  6. Execute system-level faults in parent process                │
├─────────────────────────────────────────────────────────────────┤
│                         Child Process                             │
│  python -c "import fault_injector; fault_injector.init(); ..."  │
├─────────────────────────────────────────────────────────────────┤
│                    fault_injector.py (in child)                  │
│  1. Read FAULT_INJECTION_CONFIG from environment                 │
│  2. Start background threads for each code-level fault           │
│  3. Wait for trigger conditions (time/probability)               │
│  4. Execute faults via monkey patching:                          │
│     - Patch torch.distributed for NCCL faults                    │
│     - Patch ray for Ray faults                                   │
│     - Allocate GPU memory for CUDA OOM                           │
│     - Inject NaN into gradients                                  │
│     - Sleep to trigger timeouts                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **System-level faults** (network, disk, process) run in the parent wrapper process and affect the entire system
- **Code-level faults** (NCCL, CUDA, Ray, gradients) run inside the child process via injected `fault_injector.py`
- This ensures faults actually affect the running training code, not just the wrapper process
- Faults trigger existing error handling paths in the training code, producing authentic logs

## Features

- **Zero Code Modification**: Wrap any training command without changing source code
- **10 Fault Layers**: Comprehensive coverage from hardware to application layer
- **Rich Logging**: JSONL format logs with timestamps and fault metadata
- **Flexible Triggers**: Time-based, probability-based, and conditional triggers
- **Rank-Specific Injection**: Target specific distributed training ranks
- **Cascade Scenarios**: Combine multiple faults to simulate realistic failures

## Supported Fault Types

### System-Level Faults (External Injection)
These faults are injected from the parent wrapper process and affect the system:

### 1. Network Layer
- `network_delay` - Add latency to network interface (tc netem)
- `network_loss` - Inject packet loss (tc netem)
- `network_partition` - Simulate network partition (iptables)
- `port_block` - Block specific TCP ports (iptables)

### 2. Process Layer
- `process_kill` - Send signals to terminate process

### 3. System Layer
- `disk_full` - Fill disk space
- `memory_pressure` - Allocate excessive memory

### 4. Data Layer
- `data_corrupt` - Corrupt training data files
- `data_missing` - Delete data files

### 5. Dependency Service Layer
- `hf_hub_unreachable` - Block HuggingFace Hub access (iptables)
- `wandb_disconnect` - Block Wandb connections (iptables)

### Code-Level Faults (Internal Injection)
These faults are injected into the child process via `fault_injector.py`:

### 6. Code Layer
- `code_sleep` - Inject delays to trigger timeouts
- `code_oom` - Allocate huge memory blocks
- `env_modify` - Modify environment variables

### 7. NCCL/Communication Layer
- `nccl_timeout` - Trigger NCCL collective timeouts
- `nccl_hang` - Hang NCCL barrier operations
- `nccl_mismatch` - Inject tensor shape mismatches

### 8. Ray Distributed Scheduling Layer
- `ray_actor_crash` - Crash Ray actors
- `ray_resource_exhausted` - Simulate resource exhaustion
- `ray_gcs_failure` - Simulate GCS connection failure

### 9. Deep Learning Framework Layer
- `cuda_oom` - Trigger CUDA out of memory
- `cuda_error` - Inject CUDA errors
- `gradient_nan` - Inject NaN into gradients
- `gradient_explode` - Trigger gradient explosion
- `checkpoint_corrupt` - Corrupt checkpoint files
- `checkpoint_missing` - Delete checkpoint files

### 10. vLLM Inference Engine Layer
- `vllm_init_fail` - Cause vLLM initialization failure
- `vllm_kv_cache_oom` - Trigger KV cache OOM
- `vllm_tokenizer_error` - Inject tokenizer errors

### 11. Timing/Synchronization Layer
- `rank_desync` - Desynchronize distributed ranks
- `barrier_timeout` - Trigger barrier timeouts
- `async_timeout` - Inject async operation timeouts
- `batch_size_mismatch` - Inject batch size mismatches
- `s3_timeout` - Inject S3 timeouts

## Installation

```bash
cd /path/to/verl
# The standalone script requires only Python 3.10+ and PyYAML
pip install pyyaml
```

## Usage

### List Available Fault Types

```bash
python verl_fault_inject.py --list
```

### Run Training with Fault Injection

```bash
python verl_fault_inject.py --scenario fault_scenarios/nccl_timeout.yaml -- python -m verl.trainer.main_ppo --config examples/ppo.yaml
```

### Custom Output Directory

```bash
python verl_fault_inject.py --scenario my_scenario.yaml --output ./my_logs -- python train.py
```

## Scenario Configuration

Scenarios are defined in YAML files. Here's an example:

```yaml
name: "nccl_timeout"
description: "Inject sleep to trigger NCCL timeout on rank 0"

faults:
  - type: nccl_timeout
    target_rank: 0
    trigger:
      type: timed
      after: 30s
    params:
      duration_sec: 120

log_output_dir: ./fault_logs/nccl_timeout
```

### Trigger Types

**Timed Trigger**:
```yaml
trigger:
  type: timed
  after: 60s  # Inject after 60 seconds
```

**Probability Trigger**:
```yaml
trigger:
  type: probability
  probability: 0.3  # 30% chance of injection
  after: 30s
```

### Rank Targeting

Target specific distributed training ranks:

```yaml
faults:
  - type: nccl_timeout
    target_rank: 0  # Only inject on rank 0
```

## Example Scenarios

### Simple NCCL Timeout
```bash
python verl_fault_inject.py --scenario fault_scenarios/nccl_communication_failure.yaml -- python train.py
```

### Cascade Failure
```bash
python verl_fault_inject.py --scenario fault_scenarios/multi_layer_cascade.yaml -- python train.py
```

### Comprehensive Random Faults
```bash
python verl_fault_inject.py --scenario fault_scenarios/comprehensive_random.yaml -- python train.py
```

## Log Format

Logs are saved in JSONL format with the following structure:

```json
{"ts": "2026-01-17T00:36:56.719681", "event": "start", "command": ["python", "train.py"], "scenario": "nccl_timeout"}
{"ts": "2026-01-17T00:36:56.721796", "event": "process_started", "pid": 12345}
{"ts": "2026-01-17T00:37:26.722000", "event": "fault_inject", "type": "nccl_timeout", "params": {"duration_sec": 120}}
{"ts": "2026-01-17T00:37:26.723247", "event": "output", "line": "[ERROR] NCCL timeout detected"}
{"ts": "2026-01-17T00:37:26.723376", "event": "exit", "code": 1}
```

## Creating Custom Scenarios

1. Create a YAML file in `fault_scenarios/`:

```yaml
name: "my_custom_scenario"
description: "Description of what this scenario tests"

faults:
  - type: cuda_oom
    trigger:
      type: timed
      after: 60s
    params:
      size_gb: 50

  - type: network_delay
    trigger:
      type: probability
      probability: 0.5
      after: 30s
    params:
      delay_ms: 200
      interface: eth0

log_output_dir: ./fault_logs/my_scenario
```

2. Run your scenario:

```bash
python verl_fault_inject.py --scenario fault_scenarios/my_custom_scenario.yaml -- python train.py
```

## Advanced Usage

### Combining Multiple Faults

Create cascade failures by combining faults with different timing:

```yaml
faults:
  # Stage 1: Network degradation
  - type: network_delay
    trigger:
      type: timed
      after: 20s
    params:
      delay_ms: 300

  # Stage 2: NCCL timeout
  - type: nccl_timeout
    trigger:
      type: timed
      after: 60s
    params:
      duration_sec: 120

  # Stage 3: Process crash
  - type: process_kill
    trigger:
      type: timed
      after: 90s
```

### Rank-Specific Scenarios

Test rank-specific failures:

```yaml
faults:
  - type: rank_desync
    target_rank: 0
    trigger:
      type: timed
      after: 30s
    params:
      duration_sec: 60
```

## Requirements

- Python 3.10+
- PyYAML
- Root/sudo access for network and system-level faults (tc, iptables)
- Optional: torch, ray, vllm (for framework-specific faults)

## Permissions

Some fault types require elevated privileges:

- **Network faults** (tc, iptables): Require root/sudo
- **Process faults**: Can kill child processes
- **System faults**: May require root for some operations
- **Code/Framework faults**: No special permissions needed

Run with sudo if needed:
```bash
sudo python verl_fault_inject.py --scenario scenario.yaml -- python train.py
```

## Troubleshooting

### "Command not found: tc"
Install iproute2: `sudo apt-get install iproute2`

### "Permission denied" for iptables
Run with sudo or configure capabilities

### Fault not triggering
- Check trigger timing and probability
- Verify rank targeting matches your setup
- Check logs for fault injection events

## Architecture

The system consists of:

1. **CLI Wrapper** (`verl_fault_inject.py`): Main entry point
2. **Executors**: Implement fault injection logic for each layer
3. **Orchestrator**: Manages subprocess and fault scheduling
4. **Config**: YAML-based scenario definitions
5. **Logger**: JSONL format log collection

## Contributing

To add new fault types:

1. Add enum to `FaultType` class
2. Create executor class implementing `BaseExecutor`
3. Update `_get_executor()` routing logic
4. Add to `list_faults()` display
5. Create example scenario YAML

## License

Apache 2.0
