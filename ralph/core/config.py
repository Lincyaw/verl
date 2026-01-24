"""
Configuration classes for Ralph fault injection framework.

Contains TriggerType, StrategyType enums and TriggerConfig, FaultConfig dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TriggerType(Enum):
    """Types of fault injection triggers."""

    ONE_SHOT = "one_shot"  # Trigger exactly at specified step
    STEP_BASED = "step_based"  # Trigger within step range [start_step, end_step]
    PROBABILISTIC = "probabilistic"  # Trigger with given probability each step
    PERIODIC = "periodic"  # Trigger every N steps


class StrategyType(Enum):
    """Types of fault injection strategies."""

    # Common strategies (provided by Mixins)
    DELAY = "delay"
    CORRUPT_TENSOR = "corrupt_tensor"
    INJECT_NAN = "inject_nan"
    INJECT_INF = "inject_inf"
    RAISE_EXCEPTION = "raise_exception"
    SKIP = "skip"
    REPEAT = "repeat"
    MODIFY_RESULT = "modify_result"

    # L0 Ray-specific strategies
    OBJECT_LOST = "object_lost"
    PARTIAL_FAILURE = "partial_failure"

    # L1 Distributed-specific strategies
    DEADLOCK = "deadlock"

    # L2 verl-specific strategies
    REWARD_FLIP = "reward_flip"
    CONSTANT_REWARD = "constant_reward"

    # L0 Ray additional strategies
    STORE_FULL = "store_full"
    SILENT_DROP = "silent_drop"
    WORKER_DEATH = "worker_death"
    STRAGGLER = "straggler"
    SKIP_WORKER = "skip_worker"
    DUPLICATE_CALL = "duplicate_call"

    # L1 Distributed additional strategies
    CORRUPT_GATHERED = "corrupt_gathered"
    MISSING_RANK = "missing_rank"
    SHAPE_MISMATCH = "shape_mismatch"
    BARRIER_TIMEOUT = "barrier_timeout"
    BARRIER_SKIP = "barrier_skip"
    ASYNC_DESYNC = "async_desync"

    # Algorithm strategies (GAE)
    WRONG_ADVANTAGE = "wrong_advantage"
    ZERO_ADVANTAGE = "zero_advantage"
    INVERTED_ADVANTAGE = "inverted_advantage"
    SCALED_ADVANTAGE = "scaled_advantage"
    DELAYED_ADVANTAGE = "delayed_advantage"

    # Algorithm strategies (KL Penalty)
    WRONG_KL = "wrong_kl"
    ZERO_KL = "zero_kl"
    EXTREME_KL = "extreme_kl"
    NEGATIVE_KL = "negative_kl"

    # Inference strategies (Generate)
    GENERATION_TIMEOUT = "generation_timeout"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED_OUTPUT = "truncated_output"
    GARBAGE_OUTPUT = "garbage_output"

    # Worker strategies (ComputeValues)
    WRONG_VALUES = "wrong_values"
    CONSTANT_VALUES = "constant_values"
    NAN_VALUES = "nan_values"
    INVERTED_VALUES = "inverted_values"

    # L2 Checkpoint strategies
    FILE_NOT_FOUND = "file_not_found"
    CORRUPT_STATE_DICT = "corrupt_state_dict"
    PARTIAL_LOAD = "partial_load"

    # L2 UpdateActor strategies
    NAN_INPUT = "nan_input"
    EXPLODING_GRADIENTS = "exploding_gradients"
    VANISHING_GRADIENTS = "vanishing_gradients"
    SKIP_UPDATE = "skip_update"
    DOUBLE_UPDATE = "double_update"

    # Data Pipeline strategies
    DATA_MISMATCH = "data_mismatch"
    LOST_ITEMS = "lost_items"
    DUPLICATE_ITEMS = "duplicate_items"
    WRONG_ORDER = "wrong_order"
    UNEVEN_SPLIT = "uneven_split"
    LOST_CHUNKS = "lost_chunks"
    EMPTY_CHUNK = "empty_chunk"
    OVERLAPPING_CHUNKS = "overlapping_chunks"

    # GRPO strategies
    WRONG_GROUPING = "wrong_grouping"
    WRONG_NORMALIZATION = "wrong_normalization"
    SKIP_NORMALIZATION = "skip_normalization"
    SINGLE_SAMPLE_GROUPS = "single_sample_groups"

    # UpdateWeights strategies
    WEIGHT_MISMATCH = "weight_mismatch"
    PARTIAL_UPDATE = "partial_update"
    CORRUPT_WEIGHTS = "corrupt_weights"
    OLD_WEIGHTS = "old_weights"

    # Agent strategies (CallTool)
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_EXCEPTION = "tool_exception"
    WRONG_RESULT = "wrong_result"
    EMPTY_RESULT = "empty_result"
    MALFORMED_RESULT = "malformed_result"

    # Agent strategies (ToolParser)
    PARSE_FAILURE = "parse_failure"
    WRONG_TOOL_NAME = "wrong_tool_name"
    WRONG_ARGUMENTS = "wrong_arguments"
    EXTRA_TOOL_CALLS = "extra_tool_calls"
    MISSING_TOOL_CALLS = "missing_tool_calls"

    # Megatron Optimizer strategies
    GRADIENT_OVERFLOW = "gradient_overflow"
    NAN_PARAMS = "nan_params"
    WRONG_LR = "wrong_lr"

    # Parallelism strategies
    PP_STAGE_FAILURE = "pp_stage_failure"
    TP_DESYNC = "tp_desync"
    WRONG_MICRO_BATCH_ROUTING = "wrong_micro_batch_routing"
    ACTIVATION_CORRUPTION = "activation_corruption"

    # Optimizer Step strategies
    CORRUPTED_MOMENTUM = "corrupted_momentum"
    RESET_STATE = "reset_state"

    # LR Scheduler strategies
    LR_SPIKE = "lr_spike"
    LR_ZERO = "lr_zero"

    # L3 GPU Memory strategies
    MEMORY_PRESSURE = "memory_pressure"
    OOM_SIMULATION = "oom_simulation"
    MEMORY_FRAGMENTATION = "memory_fragmentation"
    MEMORY_LEAK = "memory_leak"


@dataclass
class TriggerConfig:
    """Configuration for fault injection trigger conditions."""

    type: TriggerType
    at_step: Optional[int] = None  # ONE_SHOT: trigger at this specific step
    start_step: Optional[int] = None  # STEP_BASED: start of step range
    end_step: Optional[int] = None  # STEP_BASED: end of step range
    probability: float = 1.0  # PROBABILISTIC: probability of triggering
    every_n_steps: Optional[int] = None  # PERIODIC: trigger every N steps

    def should_trigger(self, current_step: int, rng=None) -> bool:
        """
        Determine if fault should be triggered at the current step.

        Args:
            current_step: The current training step.
            rng: Optional random number generator for probabilistic triggers.
                 If None, uses Python's built-in random module.

        Returns:
            True if the fault should be triggered, False otherwise.
        """
        if self.type == TriggerType.ONE_SHOT:
            return current_step == self.at_step

        elif self.type == TriggerType.STEP_BASED:
            if self.start_step is None or self.end_step is None:
                return False
            return self.start_step <= current_step <= self.end_step

        elif self.type == TriggerType.PROBABILISTIC:
            import random

            r = rng if rng else random
            return r.random() < self.probability

        elif self.type == TriggerType.PERIODIC:
            if self.every_n_steps is None or self.every_n_steps <= 0:
                return False
            return current_step % self.every_n_steps == 0

        return False


@dataclass
class FaultConfig:
    """Configuration for a fault injection scenario."""

    id: str  # Unique identifier for this fault config
    strategy: StrategyType  # Type of fault injection strategy
    trigger: TriggerConfig  # Trigger conditions
    parameters: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific parameters
    enabled: bool = True  # Whether this fault is enabled
    severity: str = "medium"  # Severity level: low, medium, high, critical
    expected_behavior: str = ""  # Description of expected behavior when fault is triggered
    layer: str = ""  # Layer identifier (L0, L1, L2, L3)
    target: str = ""  # Target function/method name


@dataclass
class ExperimentConfig:
    """Configuration for the experiment metadata."""

    name: str = ""
    description: str = ""
    seed: Optional[int] = None
    version: str = "1.0"


@dataclass
class GlobalConfig:
    """Global configuration settings."""

    enabled: bool = True
    log_level: str = "INFO"
    output_dir: str = ""


@dataclass
class DataCollectionConfig:
    """Configuration for data collection streams."""

    stream_a_enabled: bool = True
    stream_a_output_path: str = ""
    stream_b_enabled: bool = True
    stream_b_output_path: str = ""


@dataclass
class RalphConfig:
    """Complete Ralph configuration loaded from YAML."""

    experiment: ExperimentConfig
    global_config: GlobalConfig
    data_collection: DataCollectionConfig
    scenarios: list  # List of FaultConfig objects
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Original parsed YAML


def _substitute_variables(obj: Any, variables: Dict[str, Any]) -> Any:
    """
    Recursively substitute ${var} references in config objects.

    Supports both ${key} and ${global.key} patterns for variable substitution.

    Args:
        obj: Object to process (dict, list, or scalar)
        variables: Dictionary of variable values for substitution

    Returns:
        Object with variables substituted

    Example:
        >>> vars = {"output_dir": "/data"}
        >>> _substitute_variables("${global.output_dir}/logs", vars)
        '/data/logs'
    """
    import re

    if isinstance(obj, str):
        # Handle ${global.key} or ${key} patterns
        pattern = r'\$\{(?:global\.)?(\w+)\}'

        def replacer(match):
            key = match.group(1)
            return str(variables.get(key, match.group(0)))

        return re.sub(pattern, replacer, obj)
    elif isinstance(obj, dict):
        return {k: _substitute_variables(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_variables(item, variables) for item in obj]
    else:
        return obj


def _parse_trigger_config(trigger_data: Dict[str, Any]) -> TriggerConfig:
    """
    Parse a trigger dictionary into a TriggerConfig object.

    Args:
        trigger_data: Dictionary containing trigger configuration

    Returns:
        TriggerConfig object

    Raises:
        ValueError: If trigger type is invalid
    """
    trigger_type_str = trigger_data.get("type", "one_shot")

    try:
        trigger_type = TriggerType(trigger_type_str)
    except ValueError:
        valid_types = [t.value for t in TriggerType]
        raise ValueError(
            f"Invalid trigger type '{trigger_type_str}'. "
            f"Valid types: {valid_types}"
        )

    return TriggerConfig(
        type=trigger_type,
        at_step=trigger_data.get("at_step"),
        start_step=trigger_data.get("start_step"),
        end_step=trigger_data.get("end_step"),
        probability=trigger_data.get("probability", 1.0),
        every_n_steps=trigger_data.get("every_n_steps"),
    )


def _parse_fault_config(
    scenario: Dict[str, Any],
    global_vars: Dict[str, Any]
) -> FaultConfig:
    """
    Parse a scenario dictionary into a FaultConfig object.

    Args:
        scenario: Dictionary containing scenario configuration
        global_vars: Global variables for substitution

    Returns:
        FaultConfig object

    Raises:
        ValueError: If required fields are missing or strategy is invalid
    """
    # Validate required fields
    required_fields = ["id", "layer", "target", "fault_type", "trigger", "expected_behavior"]
    missing = [f for f in required_fields if f not in scenario]
    if missing:
        raise ValueError(
            f"Scenario '{scenario.get('id', 'unknown')}' missing required fields: {missing}"
        )

    # Parse trigger config
    trigger_config = _parse_trigger_config(scenario["trigger"])

    # Parse strategy type
    strategy_str = scenario["fault_type"]
    try:
        strategy = StrategyType(strategy_str)
    except ValueError:
        # Try common aliases
        strategy_aliases = {
            "nccl_timeout": StrategyType.DELAY,
            "gradient_corruption": StrategyType.CORRUPT_TENSOR,
            "wrong_reward": StrategyType.CORRUPT_TENSOR,
            "io_error": StrategyType.RAISE_EXCEPTION,
            "memory_pressure": StrategyType.DELAY,  # Placeholder for L3 resource
        }
        if strategy_str in strategy_aliases:
            strategy = strategy_aliases[strategy_str]
        else:
            valid_strategies = [s.value for s in StrategyType]
            raise ValueError(
                f"Invalid fault_type '{strategy_str}'. "
                f"Valid types: {valid_strategies}"
            )

    # Parse parameters with variable substitution
    parameters = scenario.get("parameters", {})
    parameters = _substitute_variables(parameters, global_vars)

    return FaultConfig(
        id=scenario["id"],
        strategy=strategy,
        trigger=trigger_config,
        parameters=parameters,
        enabled=scenario.get("enabled", True),
        severity=scenario.get("severity", "medium"),
        expected_behavior=scenario.get("expected_behavior", ""),
        layer=scenario.get("layer", ""),
        target=scenario.get("target", ""),
    )


def load_yaml_config(yaml_path: str) -> RalphConfig:
    """
    Load and parse a Ralph YAML configuration file.

    This function parses a YAML configuration file and creates a RalphConfig
    object containing experiment metadata, global settings, data collection
    configuration, and fault injection scenarios.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        RalphConfig object containing all parsed configuration

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ImportError: If pyyaml is not installed
        ValueError: If the YAML format is invalid or required fields are missing

    Example:
        >>> config = load_yaml_config("ralph_config.yaml")
        >>> print(config.experiment.name)
        'nccl_timeout_resilience_test'
        >>> print(len(config.scenarios))
        5
    """
    import os

    # Import yaml
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for YAML config loading. "
            "Install with: pip install pyyaml"
        )

    # Check file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    # Load YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

    # Parse experiment section
    exp_data = data.get("experiment", {})
    experiment = ExperimentConfig(
        name=exp_data.get("name", ""),
        description=exp_data.get("description", ""),
        seed=exp_data.get("seed"),
        version=data.get("version", "1.0"),
    )

    # Parse global section
    global_data = data.get("global", {})
    global_config = GlobalConfig(
        enabled=global_data.get("enabled", True),
        log_level=global_data.get("log_level", "INFO"),
        output_dir=global_data.get("output_dir", ""),
    )

    # Create global variables for substitution
    global_vars = dict(global_data)

    # Parse data_collection section
    dc_data = data.get("data_collection", {})
    stream_a = dc_data.get("stream_a", {})
    stream_b = dc_data.get("stream_b", {})

    # Apply variable substitution to data collection paths
    stream_a_path = stream_a.get("output", {}).get("path", "")
    stream_b_path = stream_b.get("output", {}).get("path", "")
    stream_a_path = _substitute_variables(stream_a_path, global_vars)
    stream_b_path = _substitute_variables(stream_b_path, global_vars)

    data_collection = DataCollectionConfig(
        stream_a_enabled=stream_a.get("enabled", True),
        stream_a_output_path=stream_a_path,
        stream_b_enabled=stream_b.get("enabled", True),
        stream_b_output_path=stream_b_path,
    )

    # Parse scenarios section
    scenarios_data = data.get("scenarios", [])
    if not scenarios_data:
        raise ValueError(f"No scenarios found in {yaml_path}")

    scenarios = []
    for scenario in scenarios_data:
        fault_config = _parse_fault_config(scenario, global_vars)
        scenarios.append(fault_config)

    return RalphConfig(
        experiment=experiment,
        global_config=global_config,
        data_collection=data_collection,
        scenarios=scenarios,
        raw_data=data,
    )


def validate_yaml_config(yaml_path: str) -> tuple:
    """
    Validate a Ralph YAML configuration file without fully loading it.

    This is a lightweight validation that checks:
    - File exists and is readable
    - YAML syntax is valid
    - Required sections exist
    - Each scenario has required fields

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Tuple of (is_valid: bool, errors: List[str])

    Example:
        >>> is_valid, errors = validate_yaml_config("ralph_config.yaml")
        >>> if not is_valid:
        ...     print("Errors:", errors)
    """
    import os

    errors = []

    # Check file exists
    if not os.path.exists(yaml_path):
        return False, [f"File not found: {yaml_path}"]

    # Import yaml
    try:
        import yaml
    except ImportError:
        return False, ["pyyaml is not installed"]

    # Load YAML
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML syntax error: {e}"]

    if not data:
        return False, ["Empty or invalid YAML file"]

    # Check scenarios section
    if "scenarios" not in data:
        errors.append("Missing 'scenarios' section")
    else:
        scenarios = data.get("scenarios", [])
        if not scenarios:
            errors.append("'scenarios' section is empty")
        else:
            required_fields = ["id", "layer", "target", "fault_type", "trigger", "expected_behavior"]
            for i, scenario in enumerate(scenarios):
                missing = [f for f in required_fields if f not in scenario]
                if missing:
                    scenario_id = scenario.get("id", f"index {i}")
                    errors.append(f"Scenario '{scenario_id}' missing fields: {missing}")

    return len(errors) == 0, errors
