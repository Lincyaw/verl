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


"""Base classes and interfaces for fault injection."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .config import (
    BaseFaultConfig,
    FaultConfig,
    FaultLayer,
    FaultTarget,
    FaultTargetConfig,
    FaultTriggerConfig,
    FaultType,
)

logger = logging.getLogger(__name__)


class FaultStatus(Enum):
    """Status of a fault injection."""

    PENDING = "pending"
    INJECTING = "injecting"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class FaultContext:
    """Context information for fault injection."""

    worker_id: Optional[str] = None
    rank: Optional[int] = None
    host: Optional[str] = None
    process_type: Optional[str] = None
    layer: Optional[FaultLayer] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FaultResult:
    """Result of a fault injection operation."""

    fault_id: str
    status: FaultStatus
    start_time: float
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    metadata: dict[str, Any] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration of the fault injection."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None


class FaultTargetSelector(ABC):
    """Abstract base class for selecting fault targets."""

    def __init__(self, config: FaultTargetConfig):
        self.config = config

    @abstractmethod
    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        """Select targets for fault injection."""
        pass


class AllTargetSelector(FaultTargetSelector):
    """Select all available targets."""

    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        return available_targets


class RandomTargetSelector(FaultTargetSelector):
    """Randomly select targets."""

    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        import random

        count = min(self.config.count, len(available_targets))
        return random.sample(available_targets, count)


class RankTargetSelector(FaultTargetSelector):
    """Select targets by rank."""

    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        targets = []
        for target in available_targets:
            if target.rank in (self.config.ranks or []):
                targets.append(target)
        return targets


class HostTargetSelector(FaultTargetSelector):
    """Select targets by host."""

    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        targets = []
        for target in available_targets:
            if target.host in (self.config.hosts or []):
                targets.append(target)
        return targets


class ProcessTypeTargetSelector(FaultTargetSelector):
    """Select targets by process type."""

    def select_targets(self, available_targets: list[FaultContext]) -> list[FaultContext]:
        targets = []
        for target in available_targets:
            if target.process_type in (self.config.process_types or []):
                targets.append(target)
        return targets


class BaseFaultTrigger(ABC):
    """Abstract base class for fault triggers."""

    def __init__(self, config: FaultTriggerConfig):
        self.config = config
        self._triggered = False
        self._trigger_count = 0
        self._start_time = time.time()

    @abstractmethod
    def should_trigger(self, context: FaultContext) -> bool:
        """Check if the fault should be triggered."""
        pass

    def reset(self) -> None:
        """Reset the trigger state."""
        self._triggered = False
        self._trigger_count = 0
        self._start_time = time.time()


class ImmediateTrigger(BaseFaultTrigger):
    """Trigger fault immediately."""

    def should_trigger(self, context: FaultContext) -> bool:
        if not self._triggered:
            self._triggered = True
            return True
        return False


class TimedTrigger(BaseFaultTrigger):
    """Trigger fault after a delay."""

    def should_trigger(self, context: FaultContext) -> bool:
        if self.config.delay_seconds is None:
            return False

        elapsed = time.time() - self._start_time
        if elapsed >= self.config.delay_seconds and not self._triggered:
            self._triggered = True
            return True
        return False


class CountTrigger(BaseFaultTrigger):
    """Trigger fault after N operations."""

    def __init__(self, config: FaultTriggerConfig):
        super().__init__(config)
        self._count = 0

    def should_trigger(self, context: FaultContext) -> bool:
        self._count += 1
        threshold = self.config.count_threshold or 1

        if self._count >= threshold and not self._triggered:
            self._triggered = True
            return True
        return False


class ProbabilisticTrigger(BaseFaultTrigger):
    """Trigger fault with a probability."""

    def should_trigger(self, context: FaultContext) -> bool:
        import random

        probability = self.config.probability or 0.0

        if random.random() < probability and not self._triggered:
            self._triggered = True
            return True
        return False


class BaseFaultInjector(ABC):
    """Abstract base class for all fault injectors."""

    def __init__(self, config: BaseFaultConfig):
        self.config = config
        self.fault_id = f"{config.name}_{id(self)}"
        self.status = FaultStatus.PENDING
        self._lock = threading.Lock()
        self._trigger = self._create_trigger()
        self._result: Optional[FaultResult] = None

    def _create_trigger(self) -> BaseFaultTrigger:
        """Create trigger based on configuration."""
        trigger_type = self.config.trigger.type

        if trigger_type.value == "immediate":
            return ImmediateTrigger(self.config.trigger)
        elif trigger_type.value == "timed":
            return TimedTrigger(self.config.trigger)
        elif trigger_type.value == "count":
            return CountTrigger(self.config.trigger)
        elif trigger_type.value == "probabilistic":
            return ProbabilisticTrigger(self.config.trigger)
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")

    @abstractmethod
    def inject(self, context: FaultContext) -> FaultResult:
        """Inject the fault. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def recover(self, context: FaultContext) -> None:
        """Recover from the fault. Must be implemented by subclasses."""
        pass

    def should_inject(self, context: FaultContext) -> bool:
        """Check if fault should be injected based on trigger."""
        return self._trigger.should_trigger(context)

    def execute(self, context: FaultContext) -> FaultResult:
        """Execute the fault injection."""
        with self._lock:
            if self.status != FaultStatus.PENDING:
                logger.warning(f"Fault {self.fault_id} already executed")
                return self._result

            self.status = FaultStatus.INJECTING
            start_time = time.time()

            try:
                logger.info(f"Injecting fault: {self.config.name} in layer: {self.config.layer.value}")
                result = self.inject(context)
                self.status = FaultStatus.COMPLETED
                self._result = FaultResult(
                    fault_id=self.fault_id,
                    status=FaultStatus.COMPLETED,
                    start_time=start_time,
                    end_time=time.time(),
                    metadata=result.metadata if result else {},
                )
            except Exception as e:
                logger.error(f"Failed to inject fault {self.config.name}: {e}")
                self.status = FaultStatus.FAILED
                self._result = FaultResult(
                    fault_id=self.fault_id,
                    status=FaultStatus.FAILED,
                    start_time=start_time,
                    end_time=time.time(),
                    error=e,
                    metadata={"error": str(e)},
                )

            return self._result

    def execute_recovery(self, context: FaultContext) -> None:
        """Execute fault recovery."""
        with self._lock:
            if self.status not in [FaultStatus.COMPLETED, FaultStatus.FAILED]:
                logger.warning(f"Cannot recover fault {self.fault_id} in status {self.status}")
                return

            try:
                logger.info(f"Recovering from fault: {self.config.name}")
                self.recover(context)
                self.status = FaultStatus.RECOVERED
            except Exception as e:
                logger.error(f"Failed to recover from fault {self.config.name}: {e}")

    def get_result(self) -> Optional[FaultResult]:
        """Get the result of fault injection."""
        return self._result


class FaultInjectorRegistry:
    """Registry for fault injectors."""

    _injectors: dict[FaultType, type[BaseFaultInjector]] = {}

    @classmethod
    def register(cls, fault_type: FaultType):
        """Decorator to register a fault injector."""

        def decorator(injector_class: type[BaseFaultInjector]):
            cls._injectors[fault_type] = injector_class
            return injector_class

        return decorator

    @classmethod
    def create(cls, config: FaultConfig) -> BaseFaultInjector:
        """Create a fault injector from configuration."""
        injector_class = cls._injectors.get(config.type)
        if injector_class is None:
            raise ValueError(f"No injector registered for fault type: {config.type}")

        return injector_class(config)

    @classmethod
    def get_registered_types(cls) -> list[FaultType]:
        """Get all registered fault types."""
        return list(cls._injectors.keys())


# Target selector factory
TARGET_SELECTORS = {
    FaultTarget.ALL: AllTargetSelector,
    FaultTarget.RANDOM: RandomTargetSelector,
    FaultTarget.RANK: RankTargetSelector,
    FaultTarget.HOST: HostTargetSelector,
    FaultTarget.PROCESS_TYPE: ProcessTypeTargetSelector,
}


def create_target_selector(config: FaultTargetConfig) -> FaultTargetSelector:
    """Create a target selector from configuration."""
    selector_class = TARGET_SELECTORS.get(config.mode)
    if selector_class is None:
        raise ValueError(f"Unknown target selector: {config.mode}")

    return selector_class(config)
