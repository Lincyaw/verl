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


"""Trigger-based fault injection system."""

import asyncio
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import FaultTrigger, FaultTriggerConfig

logger = logging.getLogger(__name__)


class TriggerEventType(Enum):
    """Types of trigger events."""

    METRIC_THRESHOLD = "metric_threshold"
    LOG_PATTERN = "log_pattern"
    TIME_SCHEDULE = "time_schedule"
    MANUAL = "manual"
    API_CALL = "api_call"
    SYSTEM_EVENT = "system_event"


@dataclass
class TriggerEvent:
    """Represents a trigger event."""

    event_type: TriggerEventType
    source: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class BaseTrigger(ABC):
    """Base class for fault triggers."""

    def __init__(self, config: FaultTriggerConfig):
        self.config = config
        self.enabled = True
        self.trigger_count = 0
        self.last_trigger_time = 0.0

    @abstractmethod
    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Check if trigger condition is met."""
        pass

    def reset(self) -> None:
        """Reset trigger state."""
        self.trigger_count = 0
        self.last_trigger_time = 0.0


class ImmediateTrigger(BaseTrigger):
    """Trigger that fires immediately."""

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Always trigger immediately."""
        return self.enabled


class TimedTrigger(BaseTrigger):
    """Trigger that fires after a delay."""

    def __init__(self, config: FaultTriggerConfig):
        super().__init__(config)
        self.start_time = time.time()

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger after specified delay."""
        if not self.enabled:
            return False

        elapsed = time.time() - self.start_time
        delay = self.config.delay_seconds or 0

        return elapsed >= delay


class CountTrigger(BaseTrigger):
    """Trigger that fires after N operations."""

    def __init__(self, config: FaultTriggerConfig):
        super().__init__(config)
        self.operation_count = 0

    def increment(self) -> None:
        """Increment operation counter."""
        self.operation_count += 1

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger after N operations."""
        if not self.enabled:
            return False

        threshold = self.config.count_threshold or 1
        return self.operation_count >= threshold


class ProbabilisticTrigger(BaseTrigger):
    """Trigger that fires with a probability."""

    def __init__(self, config: FaultTriggerConfig):
        super().__init__(config)
        self.rng = __import__("random").Random()

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger with specified probability."""
        if not self.enabled:
            return False

        probability = self.config.probability or 0.0
        return self.rng.random() < probability


class ConditionalTrigger(BaseTrigger):
    """Trigger that fires when a condition is met."""

    def __init__(self, config: FaultTriggerConfig):
        super().__init__(config)
        self.condition_func = self._parse_condition(config.condition or "")

    def _parse_condition(self, condition: str) -> Callable[[dict[str, Any]], bool]:
        """Parse condition string into a function."""
        # Simple condition parser - in real implementation would be more sophisticated
        if not condition:
            return lambda ctx: False

        # Support simple conditions like "metric > value"
        match = re.match(r"(\w+)\s*([><=]+)\s*([\d.]+)", condition)
        if match:
            metric, operator, value = match.groups()
            value = float(value)

            def condition_func(context: dict[str, Any]) -> bool:
                metric_value = context.get("metrics", {}).get(metric)
                if metric_value is None:
                    return False

                if operator == ">":
                    return metric_value > value
                elif operator == "<":
                    return metric_value < value
                elif operator == ">=":
                    return metric_value >= value
                elif operator == "<=":
                    return metric_value <= value
                elif operator == "==":
                    return metric_value == value
                elif operator == "!=":
                    return metric_value != value

                return False

            return condition_func

        # Return a function that always returns False for unknown conditions
        return lambda ctx: False

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger when condition is met."""
        if not self.enabled:
            return False

        return self.condition_func(context)


class MetricThresholdTrigger(BaseTrigger):
    """Trigger based on metric thresholds."""

    def __init__(self, config: FaultTriggerConfig, metric_name: str, threshold: float):
        super().__init__(config)
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = config.condition or ">"

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger when metric crosses threshold."""
        if not self.enabled:
            return False

        metrics = context.get("metrics", {})
        value = metrics.get(self.metric_name)

        if value is None:
            return False

        if self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold

        return False


class LogPatternTrigger(BaseTrigger):
    """Trigger based on log patterns."""

    def __init__(self, config: FaultTriggerConfig, pattern: str, count: int = 1):
        super().__init__(config)
        self.pattern = re.compile(pattern)
        self.target_count = count
        self.match_count = 0
        self.log_monitor = None

    def start_monitoring(self, log_source: str | Callable[[], list[str]]) -> None:
        """Start monitoring logs for pattern matches."""
        if isinstance(log_source, str):
            # Monitor file
            self.log_monitor = threading.Thread(target=self._monitor_log_file, args=(log_source,), daemon=True)
            self.log_monitor.start()
        else:
            # Monitor function
            self.log_monitor = threading.Thread(target=self._monitor_log_function, args=(log_source,), daemon=True)
            self.log_monitor.start()

    def _monitor_log_file(self, log_file: str) -> None:
        """Monitor log file for pattern matches."""
        try:
            with open(log_file) as f:
                # Start from end of file
                f.seek(0, 2)

                while self.enabled:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue

                    if self.pattern.search(line):
                        self.match_count += 1
                        logger.info(f"Log pattern matched: {self.pattern.pattern}")
        except Exception as e:
            logger.error(f"Error monitoring log file: {e}")

    def _monitor_log_function(self, log_func: Callable[[], list[str]]) -> None:
        """Monitor logs from function."""
        last_count = 0

        while self.enabled:
            try:
                logs = log_func()
                new_logs = logs[last_count:]

                for log in new_logs:
                    if self.pattern.search(log):
                        self.match_count += 1
                        logger.info(f"Log pattern matched: {self.pattern.pattern}")

                last_count = len(logs)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error monitoring logs: {e}")
                time.sleep(1)

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger when pattern is matched enough times."""
        if not self.enabled:
            return False

        return self.match_count >= self.target_count


class TimeScheduleTrigger(BaseTrigger):
    """Trigger based on time schedule."""

    def __init__(self, config: FaultTriggerConfig, schedule: str):
        super().__init__(config)
        self.schedule = schedule
        self.cron = __import__("croniter").croniter(schedule) if "croniter" in __import__("sys").modules else None

    async def check_trigger(self, context: dict[str, Any]) -> bool:
        """Trigger based on cron schedule."""
        if not self.enabled or not self.cron:
            return False

        # Check if current time matches schedule
        now = time.time()
        return self.cron.get_next(float) <= now


class TriggerManager:
    """Manages multiple fault triggers."""

    def __init__(self):
        self.triggers: dict[str, BaseTrigger] = {}
        self.trigger_history: list[TriggerEvent] = []
        self.event_handlers: dict[TriggerEventType, list[Callable]] = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None

    def register_trigger(self, fault_name: str, trigger: BaseTrigger) -> None:
        """Register a trigger for a fault."""
        self.triggers[fault_name] = trigger

    def create_trigger(self, config: FaultTriggerConfig) -> BaseTrigger:
        """Create trigger based on configuration."""

        if config.type == FaultTrigger.IMMEDIATE:
            return ImmediateTrigger(config)
        elif config.type == FaultTrigger.TIMED:
            return TimedTrigger(config)
        elif config.type == FaultTrigger.COUNT:
            return CountTrigger(config)
        elif config.type == FaultTrigger.PROBABILISTIC:
            return ProbabilisticTrigger(config)
        elif config.type == FaultTrigger.CONDITIONAL:
            return ConditionalTrigger(config)
        else:
            raise ValueError(f"Unknown trigger type: {config.type}")

    def add_event_handler(self, event_type: TriggerEventType, handler: Callable) -> None:
        """Add event handler for trigger events."""
        self.event_handlers[event_type].append(handler)

    def emit_event(self, event: TriggerEvent) -> None:
        """Emit a trigger event."""
        self.trigger_history.append(event)

        # Call handlers
        for handler in self.event_handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def check_triggers(self, context: dict[str, Any]) -> list[str]:
        """Check all triggers and return list of triggered faults."""
        triggered = []

        for fault_name, trigger in self.triggers.items():
            try:
                if await trigger.check_trigger(context):
                    triggered.append(fault_name)

                    # Emit trigger event
                    event = TriggerEvent(
                        event_type=TriggerEventType.MANUAL,
                        source="trigger_manager",
                        data={"fault_name": fault_name, "trigger_type": str(type(trigger))},
                    )
                    self.emit_event(event)
            except Exception as e:
                logger.error(f"Error checking trigger for {fault_name}: {e}")

        return triggered

    def start_monitoring(self, context_provider: Optional[Callable[[], dict[str, Any]]] = None) -> None:
        """Start continuous trigger monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(context_provider or (lambda: {}),), daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop trigger monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitor_loop(self, context_provider: Callable[[], dict[str, Any]]) -> None:
        """Main monitoring loop."""
        logger.info("Starting trigger monitoring loop")

        while self.monitoring:
            try:
                context = context_provider()
                asyncio.run(self.check_triggers(context))
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in trigger monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error

    def get_trigger_history(self, fault_name: Optional[str] = None) -> list[TriggerEvent]:
        """Get trigger history."""
        if fault_name:
            return [e for e in self.trigger_history if e.data.get("fault_name") == fault_name]
        return self.trigger_history.copy()

    def reset_trigger(self, fault_name: str) -> None:
        """Reset a specific trigger."""
        if fault_name in self.triggers:
            self.triggers[fault_name].reset()

    def reset_all_triggers(self) -> None:
        """Reset all triggers."""
        for trigger in self.triggers.values():
            trigger.reset()


# Integration with fault injection orchestrator
class TriggerBasedFaultInjector:
    """Integrates trigger-based injection with the fault injection system."""

    def __init__(self, trigger_manager: TriggerManager):
        self.trigger_manager = trigger_manager
        self.injection_enabled = True

    def enable_injection(self) -> None:
        """Enable fault injection."""
        self.injection_enabled = True

    def disable_injection(self) -> None:
        """Disable fault injection."""
        self.injection_enabled = False

    async def check_and_inject(self, context: dict[str, Any]) -> list[str]:
        """Check triggers and inject faults."""
        if not self.injection_enabled:
            return []

        triggered_faults = await self.trigger_manager.check_triggers(context)

        # Here you would integrate with your fault injection orchestrator
        # to actually inject the faults
        injected = []
        for fault_name in triggered_faults:
            try:
                # This would call your fault injection orchestrator
                # await self.fault_orchestrator.inject_fault_by_name(fault_name)
                injected.append(fault_name)
                logger.info(f"Injected fault: {fault_name}")
            except Exception as e:
                logger.error(f"Failed to inject fault {fault_name}: {e}")

        return injected
