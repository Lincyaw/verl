"""Configuration classes for fault injection system."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FaultLayer(Enum):
    """Fault injection layers in verl architecture."""

    UI = "ui"  # User interface layer
    ORCHESTRATION = "orchestration"  # Ray cluster orchestration
    WORKER = "worker"  # Worker processes (FSDP, Megatron)
    ENGINE = "engine"  # Training engines
    INFERENCE = "inference"  # Inference backends (vLLM, SGLang)


class FaultType(Enum):
    """Types of faults that can be injected."""

    # Process-level faults
    PROCESS_KILL = "process_kill"
    PROCESS_EXIT = "process_exit"
    PROCESS_HANG = "process_hang"

    # Network faults
    NETWORK_DELAY = "network_delay"
    NETWORK_LOSS = "network_loss"
    NETWORK_PARTITION = "network_partition"

    # Resource faults
    MEMORY_OOM = "memory_oom"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    FD_EXHAUSTION = "fd_exhaustion"

    # CUDA/GPU faults
    CUDA_OOM = "cuda_oom"
    CUDA_ERROR = "cuda_error"
    NCCL_FAILURE = "nccl_failure"

    # Configuration faults
    CONFIG_CORRUPTION = "config_corruption"
    ENV_VAR_MISSING = "env_var_missing"

    # UI layer faults
    HYDRA_CONFIG_ERROR = "hydra_config_error"
    RAY_INIT_FAILURE = "ray_init_failure"
    CLI_ARG_ERROR = "cli_arg_error"
    ENV_VAR_ERROR = "env_var_error"
    UI_FREEZE = "ui_freeze"
    UI_CRASH = "ui_crash"

    # Orchestration layer faults
    RAY_CLUSTER_FAILURE = "ray_cluster_failure"
    ACTOR_CRASH = "actor_crash"
    ACTOR_DEATH = "actor_death"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TASK_SCHEDULING_FAILURE = "task_scheduling_failure"
    RESOURCE_POOL_FAULT = "resource_pool_fault"
    GCS_FAILURE = "gcs_failure"
    NETWORK_PARTITION = "network_partition"
    PLACEMENT_GROUP_FAULT = "placement_group_fault"

    # Worker layer faults
    FSDP_SYNC_FAILURE = "fsdp_sync_failure"
    FSDP_SHARDING_ERROR = "fsdp_sharding_error"
    MEGATRON_SYNC_FAILURE = "megatron_sync_failure"
    MEGATRON_PIPELINE_ERROR = "megatron_pipeline_error"
    GRADIENT_SYNC_TIMEOUT = "gradient_sync_timeout"
    GRADIENT_NAN = "gradient_nan"
    PARAMETER_NAN = "parameter_nan"
    CUDA_OOM_WORKER = "cuda_oom_worker"
    CUDA_MEMORY_FRAGMENTATION = "cuda_memory_fragmentation"
    WORKER_CRASH = "worker_crash"
    WORKER_HANG = "worker_hang"

    # Engine faults
    ENGINE_INIT_FAILURE = "engine_init_failure"
    ENGINE_HANG = "engine_hang"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    NCCL_FAILURE = "nccl_failure"
    DEVICE_MESH_ERROR = "device_mesh_error"
    PRECISION_ERROR = "precision_error"
    DEVICE_MAP_ERROR = "device_map_error"

    # Inference faults
    INFERENCE_OOM = "inference_oom"
    SCHEDULER_DEADLOCK = "scheduler_deadlock"
    COMPILATION_FAILURE = "compilation_failure"


class FaultTrigger(Enum):
    """How faults are triggered."""

    IMMEDIATE = "immediate"  # Inject immediately when enabled
    TIMED = "timed"  # Inject after specified duration
    COUNT = "count"  # Inject after N operations
    PROBABILISTIC = "probabilistic"  # Inject with probability
    CONDITIONAL = "conditional"  # Inject when condition is met


class FaultTarget(Enum):
    """Target selection mode for faults."""

    ALL = "all"  # All workers/processes
    RANDOM = "random"  # Random selection
    RANK = "rank"  # Specific rank(s)
    HOST = "host"  # Specific host(s)
    PROCESS_TYPE = "process_type"  # By process type (actor, critic, etc)


@dataclass
class FaultTargetConfig:
    """Configuration for fault targets."""

    mode: FaultTarget = FaultTarget.RANDOM
    ranks: Optional[List[int]] = None
    hosts: Optional[List[str]] = None
    process_types: Optional[List[str]] = None
    count: int = 1


@dataclass
class FaultTriggerConfig:
    """Configuration for fault triggers."""

    type: FaultTrigger = FaultTrigger.IMMEDIATE
    delay_seconds: Optional[float] = None
    count_threshold: Optional[int] = None
    probability: Optional[float] = None
    condition: Optional[str] = None


@dataclass
class BaseFaultConfig:
    """Base configuration for all faults."""

    name: str
    layer: FaultLayer
    type: FaultType
    enabled: bool = True
    description: str = ""
    target: FaultTargetConfig = field(default_factory=FaultTargetConfig)
    trigger: FaultTriggerConfig = field(default_factory=FaultTriggerConfig)
    duration_seconds: Optional[float] = None  # How long the fault lasts
    recovery_seconds: Optional[float] = None  # Time to recover after fault
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessFaultConfig(BaseFaultConfig):
    """Configuration for process-level faults."""

    signal: int = 9  # Signal to send (for process_kill)
    exit_code: int = 1  # Exit code (for process_exit)
    hang_duration: Optional[float] = None  # Duration to hang (for process_hang)


@dataclass
class NetworkFaultConfig(BaseFaultConfig):
    """Configuration for network faults."""

    delay_ms: int = 100  # Network delay in milliseconds
    loss_rate: float = 0.1  # Packet loss rate (0.0-1.0)
    bandwidth_limit: Optional[int] = None  # Bandwidth limit in kbps
    target_ports: Optional[List[int]] = None  # Target ports


@dataclass
class ResourceFaultConfig(BaseFaultConfig):
    """Configuration for resource faults."""

    memory_mb: Optional[int] = None  # Memory to allocate
    leak_rate_mb_per_sec: Optional[float] = None  # Memory leak rate
    disk_path: Optional[str] = None  # Path to fill disk
    fd_count: Optional[int] = None  # Number of file descriptors


@dataclass
class CudaFaultConfig(BaseFaultConfig):
    """Configuration for CUDA/GPU faults."""

    memory_mb: int = 1024  # GPU memory to allocate
    error_type: str = "out_of_memory"  # Type of CUDA error
    nccl_rank: Optional[int] = None  # Specific NCCL rank to fail


@dataclass
class ConfigFaultConfig(BaseFaultConfig):
    """Configuration for configuration faults."""

    config_key: str = ""  # Key to corrupt
    config_value: Any = None  # Value to set
    env_var: str = ""  # Environment variable to unset


@dataclass
class DataFaultConfig(BaseFaultConfig):
    """Configuration for data faults."""

    checkpoint_path: Optional[str] = None  # Checkpoint to corrupt
    corruption_type: str = "random"  # Type of corruption
    nan_probability: float = 0.1  # Probability of NaN values
    target_tensors: Optional[List[str]] = None  # Specific tensors to corrupt


@dataclass
class EngineFaultConfig(BaseFaultConfig):
    """Configuration for engine faults."""

    engine_type: str = ""  # Engine type (fsdp, megatron, etc)
    init_error: str = ""  # Initialization error message
    hang_point: str = ""  # Where to hang (forward, backward, etc)

    # Checkpoint corruption
    corruption_type: str = "random_bytes"  # Type of corruption (random_bytes, truncate, delete)
    checkpoint_path: Optional[str] = None  # Path to checkpoint file

    # NCCL failure
    nccl_error_type: str = "timeout"  # Type of NCCL error (timeout, abort, crash)
    nccl_timeout_ms: int = 60000  # NCCL timeout in milliseconds

    # Device mesh errors
    mesh_shape: Optional[List[int]] = None  # Device mesh shape
    mesh_error_type: str = "invalid_shape"  # Type of mesh error

    # Precision errors
    precision_type: str = "fp16"  # Precision type (fp16, bf16, fp32)
    precision_error_type: str = "overflow"  # Type of precision error

    # Device mapping errors
    device_map_error: str = "missing_device"  # Type of device mapping error
    target_device: Optional[int] = None  # Target device ID


@dataclass
class InferenceFaultConfig(BaseFaultConfig):
    """Configuration for inference faults."""

    backend: str = ""  # Backend (vllm, sglang)
    kv_cache_size_mb: int = 1024  # KV cache size
    deadlock_type: str = ""  # Type of deadlock
    compilation_stage: str = ""  # Compilation stage to fail


@dataclass
class UIFaultConfig(BaseFaultConfig):
    """Configuration for UI layer faults."""

    # Hydra config errors
    hydra_error_type: str = ""  # Type of Hydra error (parse, validate, merge)
    hydra_config_path: Optional[str] = None  # Path to config file

    # Ray initialization errors
    ray_init_error: str = ""  # Ray initialization error message
    ray_address: Optional[str] = None  # Ray address to fail

    # CLI argument errors
    cli_arg_missing: Optional[str] = None  # Missing required argument
    cli_arg_invalid: Optional[str] = None  # Invalid argument value

    # Environment variable errors
    env_var_unset: Optional[str] = None  # Environment variable to unset
    env_var_invalid: Optional[str] = None  # Environment variable with invalid value
    env_var_value: Optional[str] = None  # Value to set for env var

    # UI freeze/crash
    freeze_duration_seconds: float = 30.0  # Duration to freeze
    crash_message: str = "Simulated UI crash"


@dataclass
class OrchestrationFaultConfig(BaseFaultConfig):
    """Configuration for orchestration layer faults."""

    # Ray cluster failures
    cluster_failure_type: str = ""  # Type of cluster failure (gcs, node, cluster)
    failure_duration: float = 30.0  # Duration of failure in seconds
    affected_nodes: Optional[List[str]] = None  # List of affected node IPs

    # Actor crashes/deaths
    actor_name: Optional[str] = None  # Name pattern of actor to crash
    actor_death_type: str = "exception"  # How to kill actor (exception, exit, kill)
    crash_delay: float = 0.0  # Delay before crashing in seconds

    # Resource exhaustion
    resource_type: str = "memory"  # Type of resource (memory, cpu, gpu)
    exhaustion_amount: Optional[int] = None  # Amount to exhaust (MB for memory)
    exhaustion_duration: float = 60.0  # How long to maintain exhaustion

    # Task scheduling failures
    task_name_pattern: Optional[str] = None  # Pattern of tasks to fail
    failure_rate: float = 1.0  # Rate of task failures (0.0-1.0)
    failure_type: str = "exception"  # Type of failure (exception, hang, lost)

    # Resource pool faults
    pool_name: Optional[str] = None  # Name of resource pool
    pool_operation: str = "acquire"  # Operation to fail (acquire, release)
    pool_block_duration: float = 30.0  # Duration to block pool operations

    # GCS failures
    gcs_failure_type: str = "disconnect"  # Type of GCS failure
    gcs_isolation_duration: float = 30.0  # Duration of GCS isolation

    # Network partitions
    partition_type: str = "partial"  # Type of partition (partial, total)
    isolated_ranks: Optional[List[int]] = None  # Ranks to isolate
    partition_duration: float = 30.0  # Duration of partition

    # Placement group faults
    placement_group_name: Optional[str] = None  # Name of placement group
    pg_fault_type: str = "creation_failure"  # Type of placement group fault


@dataclass
class WorkerFaultConfig(BaseFaultConfig):
    """Configuration for worker layer faults."""

    # FSDP faults
    fsdp_sync_type: str = "all_reduce"  # Type of FSDP sync (all_reduce, broadcast)
    fsdp_sharding_stage: str = "forward"  # Stage to inject sharding error

    # Megatron faults
    megatron_sync_op: str = "all_reduce"  # Megatron sync operation
    megatron_pipeline_stage: str = "forward"  # Pipeline stage to fail
    megatron_virtual_pipeline: Optional[int] = None  # Virtual pipeline rank

    # Gradient faults
    gradient_sync_timeout_ms: int = 30000  # Timeout for gradient sync
    gradient_nan_probability: float = 1.0  # Probability of NaN gradients

    # CUDA faults
    cuda_memory_mb: int = 1024  # Amount of CUDA memory to allocate
    cuda_fragment_size_mb: int = 64  # Size of memory fragments
    cuda_fragment_count: int = 100  # Number of fragments to create

    # Worker process faults
    worker_type: str = "actor"  # Type of worker (actor, critic, reward)
    crash_method: str = "exception"  # Method to crash (exception, exit, signal)
    hang_location: str = "forward"  # Where to hang (forward, backward, step)


# Union type for all fault configurations
FaultConfig = Union[
    ProcessFaultConfig,
    NetworkFaultConfig,
    ResourceFaultConfig,
    CudaFaultConfig,
    ConfigFaultConfig,
    DataFaultConfig,
    EngineFaultConfig,
    InferenceFaultConfig,
    UIFaultConfig,
    OrchestrationFaultConfig,
    WorkerFaultConfig,
    BaseFaultConfig,
]


@dataclass
class RecoveryStrategyConfig:
    """Configuration for recovery strategies."""

    name: str
    enabled: bool = True
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    timeout_seconds: float = 300.0


@dataclass
class RecoveryConfig:
    """Configuration for fault recovery system."""

    enabled: bool = False
    mode: str = "automatic"  # automatic, manual, semi_automatic
    strategies: List[RecoveryStrategyConfig] = field(default_factory=list)
    max_recovery_attempts: int = 3
    recovery_timeout_seconds: float = 300.0
    enable_preventive_recovery: bool = True
    degradation_threshold: float = 0.5  # Degrade when success rate below this


@dataclass
class FaultDependencyConfig:
    """Configuration for fault dependencies."""

    fault_name: str  # Name of the fault this depends on
    condition: str  # Condition type: "after", "on_success", "on_failure"
    delay_seconds: Optional[float] = None  # Delay after dependency


@dataclass
class FaultScenarioConfig:
    """Configuration for a fault scenario with dependencies."""

    name: str
    description: str = ""
    enabled: bool = True
    faults: List[FaultConfig] = field(default_factory=list)
    dependencies: List[FaultDependencyConfig] = field(default_factory=list)
    cascade_mode: str = "none"  # none, linear, tree, burst
    cascade_interval_seconds: float = 5.0  # Interval between cascaded faults
    max_cascade_depth: int = 3  # Maximum depth for tree cascade
    stop_on_failure: bool = False  # Stop scenario if a fault fails


@dataclass
class FaultScenarioTemplate:
    """Template for common fault scenarios."""

    name: str
    description: str
    category: str  # category: system, network, resource, training, inference
    faults: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    cascade_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability."""

    enabled: bool = True
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "collection_interval": 5.0,
        "aggregation_window": 60,
        "max_history_size": 10000,
        "enable_gpu_monitoring": True,
        "enable_ray_monitoring": True,
    })
    dashboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "update_interval": 2.0,
        "enable_cors": True,
        "max_datapoints": 1000,
    })
    alerts: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_alerts_per_minute": 10,
        "alert_retention_hours": 24,
        "enable_auto_recovery": True,
        "notification_channels": [
            {"type": "console", "enabled": True, "min_severity": "info"},
        ],
        "rules": [],  # Will use default rules if empty
    })
    impact_analysis: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "historical_window_hours": 24,
        "confidence_threshold": 0.7,
    })


@dataclass
class FaultInjectionConfig:
    """Main configuration for the fault injection system."""

    enabled: bool = False
    faults: List[FaultConfig] = field(default_factory=list)
    scenarios: List[FaultScenarioConfig] = field(default_factory=list)
    templates: List[FaultScenarioTemplate] = field(default_factory=list)
    global_cooldown_seconds: float = 60.0  # Cooldown between faults
    max_concurrent_faults: int = 1  # Maximum concurrent faults
    log_level: str = "INFO"
    output_dir: Optional[str] = None  # Directory for fault logs
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FaultInjectionConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # TODO: Implement proper deserialization
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        # TODO: Implement proper serialization
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
