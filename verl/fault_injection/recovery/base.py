"""Base classes and interfaces for fault recovery."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from ..base import FaultContext, FaultResult, FaultStatus

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """Status of a recovery operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecoveryMode(Enum):
    """Recovery execution modes."""

    AUTOMATIC = "automatic"  # Automatically execute recovery
    MANUAL = "manual"  # Require manual intervention
    SEMI_AUTOMATIC = "semi_automatic"  # Auto with manual approval for critical steps


class RecoveryPriority(Enum):
    """Priority levels for recovery strategies."""

    CRITICAL = 0  # Must recover immediately
    HIGH = 1  # Should recover quickly
    MEDIUM = 2  # Normal recovery priority
    LOW = 3  # Can be delayed


@dataclass
class RecoveryContext:
    """Context information for recovery operations."""

    fault_result: FaultResult
    fault_context: FaultContext
    attempt_count: int = 0
    max_attempts: int = 3
    recovery_start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    recovery_id: str
    status: RecoveryStatus
    strategy_name: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_recovery: Optional[str] = None  # Next recovery strategy to try

    @property
    def duration(self) -> Optional[float]:
        """Duration of the recovery operation."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class RecoveryDecision:
    """Decision made by the recovery engine."""

    strategy_name: str
    priority: RecoveryPriority
    mode: RecoveryMode
    confidence: float  # 0.0-1.0 confidence in this recovery
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)


class RecoveryStrategy(Protocol):
    """Protocol for recovery strategies."""

    @property
    def name(self) -> str:
        """Name of the recovery strategy."""
        ...

    @property
    def priority(self) -> RecoveryPriority:
        """Priority of this strategy."""
        ...

    def can_recover(self, context: RecoveryContext) -> bool:
        """Check if this strategy can recover from the fault."""
        ...

    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute the recovery strategy."""
        ...

    def estimate_impact(self, context: RecoveryContext) -> float:
        """Estimate the impact/cost of this recovery (0.0-1.0)."""
        ...


class BaseRecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.recovery_id = f"{self.name}_{id(self)}"

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the recovery strategy."""
        pass

    @property
    @abstractmethod
    def priority(self) -> RecoveryPriority:
        """Priority of this strategy."""
        pass

    @abstractmethod
    def can_recover(self, context: RecoveryContext) -> bool:
        """Check if this strategy can recover from the fault."""
        pass

    @abstractmethod
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute the recovery strategy."""
        pass

    def estimate_impact(self, context: RecoveryContext) -> float:
        """Estimate the impact/cost of this recovery (0.0-1.0)."""
        # Default implementation: medium impact
        return 0.5

    def pre_check(self, context: RecoveryContext) -> bool:
        """Pre-execution checks. Override if needed."""
        return True

    def post_check(self, context: RecoveryContext) -> bool:
        """Post-execution verification. Override if needed."""
        return True

    def _execute_with_checks(self, context: RecoveryContext) -> RecoveryResult:
        """Execute with pre and post checks."""
        start_time = time.time()

        try:
            # Pre-check
            if not self.pre_check(context):
                return RecoveryResult(
                    recovery_id=self.recovery_id,
                    status=RecoveryStatus.FAILED,
                    strategy_name=self.name,
                    start_time=start_time,
                    end_time=time.time(),
                    error=Exception("Pre-check failed"),
                    metadata={"reason": "Pre-check failed"},
                )

            # Execute recovery
            result = self.execute(context)
            result.recovery_id = self.recovery_id
            result.strategy_name = self.name

            # Post-check
            if not self.post_check(context):
                result.status = RecoveryStatus.PARTIAL_SUCCESS
                result.metadata["warning"] = "Post-check failed"

            return result

        except Exception as e:
            logger.error(f"Recovery strategy {self.name} failed: {e}")
            return RecoveryResult(
                recovery_id=self.recovery_id,
                status=RecoveryStatus.FAILED,
                strategy_name=self.name,
                start_time=start_time,
                end_time=time.time(),
                error=e,
                metadata={"error": str(e)},
            )


class RecoveryDecisionEngine(ABC):
    """Abstract base class for recovery decision engines."""

    def __init__(self, strategies: List[BaseRecoveryStrategy]):
        self.strategies = strategies
        self._strategy_map = {s.name: s for s in strategies}

    @abstractmethod
    def decide_recovery(
        self, context: RecoveryContext, available_strategies: List[str] = None
    ) -> RecoveryDecision:
        """Decide which recovery strategy to use."""
        pass

    def get_strategy(self, name: str) -> Optional[BaseRecoveryStrategy]:
        """Get a recovery strategy by name."""
        return self._strategy_map.get(name)

    def evaluate_strategies(self, context: RecoveryContext) -> List[RecoveryDecision]:
        """Evaluate all strategies and return ranked decisions."""
        decisions = []

        for strategy in self.strategies:
            if strategy.can_recover(context):
                decision = RecoveryDecision(
                    strategy_name=strategy.name,
                    priority=strategy.priority,
                    mode=RecoveryMode.AUTOMATIC,  # Default
                    confidence=self._calculate_confidence(strategy, context),
                    reason=f"Strategy {strategy.name} can recover from {context.fault_result.fault_id}",
                    parameters={},
                    alternatives=[],
                )
                decisions.append(decision)

        # Sort by priority and confidence
        decisions.sort(key=lambda d: (d.priority.value, -d.confidence))
        return decisions

    def _calculate_confidence(self, strategy: BaseRecoveryStrategy, context: RecoveryContext) -> float:
        """Calculate confidence score for a strategy."""
        # Base confidence from strategy capability
        base_confidence = 0.7 if strategy.can_recover(context) else 0.0

        # Adjust based on attempt count
        attempt_penalty = min(0.3, context.attempt_count * 0.1)

        # Adjust based on impact
        impact_factor = 1.0 - strategy.estimate_impact(context)

        return max(0.0, base_confidence - attempt_penalty) * impact_factor


class RecoveryStrategyRegistry:
    """Registry for recovery strategies."""

    _strategies: Dict[str, type[BaseRecoveryStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a recovery strategy."""

        def decorator(strategy_class: type[BaseRecoveryStrategy]):
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def create(cls, name: str, config: Dict[str, Any] = None) -> BaseRecoveryStrategy:
        """Create a recovery strategy from configuration."""
        strategy_class = cls._strategies.get(name)
        if strategy_class is None:
            raise ValueError(f"No recovery strategy registered: {name}")

        return strategy_class(config)

    @classmethod
    def get_registered_names(cls) -> List[str]:
        """Get all registered strategy names."""
        return list(cls._strategies.keys())