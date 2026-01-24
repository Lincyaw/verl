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
