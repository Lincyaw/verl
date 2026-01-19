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

    # Data faults
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    GRADIENT_NAN = "gradient_nan"
    PARAMETER_NAN = "parameter_nan"

    # Engine faults
    ENGINE_INIT_FAILURE = "engine_init_failure"
    ENGINE_HANG = "engine_hang"

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


@dataclass
class InferenceFaultConfig(BaseFaultConfig):
    """Configuration for inference faults."""

    backend: str = ""  # Backend (vllm, sglang)
    kv_cache_size_mb: int = 1024  # KV cache size
    deadlock_type: str = ""  # Type of deadlock
    compilation_stage: str = ""  # Compilation stage to fail


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
    BaseFaultConfig,
]


@dataclass
class FaultInjectionConfig:
    """Main configuration for the fault injection system."""

    enabled: bool = False
    faults: List[FaultConfig] = field(default_factory=list)
    global_cooldown_seconds: float = 60.0  # Cooldown between faults
    max_concurrent_faults: int = 1  # Maximum concurrent faults
    log_level: str = "INFO"
    output_dir: Optional[str] = None  # Directory for fault logs

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
