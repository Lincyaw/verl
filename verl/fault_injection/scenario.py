# Copyright 2026 Aoyang Fang
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
# ==============================================================================

"""Scenario-based fault injection with dependencies and cascading."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .config import (
    BaseFaultConfig,
    FaultConfig,
    FaultDependencyConfig,
    FaultScenarioConfig,
    FaultScenarioTemplate,
    FaultType,
)
from .orchestrator import FaultOrchestrator

logger = logging.getLogger(__name__)


class CascadeMode(Enum):
    """Modes for fault cascading."""

    NONE = "none"
    LINEAR = "linear"  # One fault triggers next in sequence
    TREE = "tree"  # Fault triggers multiple children
    BURST = "burst"  # All faults trigger at once


class DependencyCondition(Enum):
    """Conditions for fault dependencies."""

    AFTER = "after"  # Inject after dependency completes
    ON_SUCCESS = "on_success"  # Inject only if dependency succeeds
    ON_FAILURE = "on_failure"  # Inject only if dependency fails


@dataclass
class FaultExecutionState:
    """State tracking for fault execution."""

    name: str
    config: FaultConfig
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)


@dataclass
class ScenarioExecutionContext:
    """Context for scenario execution."""

    scenario_name: str
    execution_graph: dict[str, FaultExecutionState] = field(default_factory=dict)
    ready_queue: deque = field(default_factory=deque)
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    cascade_depth: int = 0


class FaultScenarioOrchestrator:
    """Orchestrates fault scenarios with dependencies and cascading."""

    def __init__(self, fault_orchestrator: FaultOrchestrator):
        self.fault_orchestrator = fault_orchestrator
        self.active_scenarios: dict[str, ScenarioExecutionContext] = {}
        self.templates: dict[str, FaultScenarioTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default fault scenario templates."""

        # System failure cascade template
        self.templates["system_failure_cascade"] = FaultScenarioTemplate(
            name="system_failure_cascade",
            description="Simulate cascading system failures",
            category="system",
            faults=[
                {
                    "name": "memory_pressure",
                    "type": FaultType.MEMORY_OOM,
                    "layer": "orchestration",
                    "memory_mb": 4096,
                    "target": {"mode": "random", "count": 2},
                },
                {
                    "name": "network_partition",
                    "type": FaultType.NETWORK_PARTITION,
                    "layer": "orchestration",
                    "partition_type": "partial",
                    "partition_duration": 30.0,
                    "target": {"mode": "rank", "ranks": [0, 1]},
                },
                {
                    "name": "worker_crash",
                    "type": FaultType.WORKER_CRASH,
                    "layer": "worker",
                    "worker_type": "actor",
                    "crash_method": "exception",
                    "target": {"mode": "random", "count": 1},
                },
            ],
            dependencies=[
                {"fault_name": "network_partition", "condition": "on_failure", "delay_seconds": 5.0},
                {"fault_name": "worker_crash", "condition": "after", "delay_seconds": 10.0},
            ],
            cascade_config={"mode": "tree", "interval": 5.0, "max_depth": 3},
        )

        # Training disruption template
        self.templates["training_disruption"] = FaultScenarioTemplate(
            name="training_disruption",
            description="Simulate common training disruptions",
            category="training",
            faults=[
                {
                    "name": "gradient_nan",
                    "type": FaultType.GRADIENT_NAN,
                    "layer": "worker",
                    "gradient_nan_probability": 1.0,
                    "target": {"mode": "all"},
                },
                {
                    "name": "checkpoint_corruption",
                    "type": FaultType.CHECKPOINT_CORRUPTION,
                    "layer": "engine",
                    "corruption_type": "random_bytes",
                    "target": {"mode": "random", "count": 1},
                },
                {
                    "name": "nccl_timeout",
                    "type": FaultType.NCCL_FAILURE,
                    "layer": "engine",
                    "nccl_error_type": "timeout",
                    "nccl_timeout_ms": 30000,
                    "target": {"mode": "random", "count": 2},
                },
            ],
            dependencies=[{"fault_name": "checkpoint_corruption", "condition": "on_success", "delay_seconds": 60.0}],
            cascade_config={"mode": "linear", "interval": 30.0},
        )

        # Inference degradation template
        self.templates["inference_degradation"] = FaultScenarioTemplate(
            name="inference_degradation",
            description="Simulate inference service degradation",
            category="inference",
            faults=[
                {
                    "name": "inference_oom",
                    "type": FaultType.INFERENCE_OOM,
                    "layer": "inference",
                    "backend": "vllm",
                    "kv_cache_size_mb": 2048,
                    "target": {"mode": "random", "count": 1},
                },
                {
                    "name": "scheduler_deadlock",
                    "type": FaultType.SCHEDULER_DEADLOCK,
                    "layer": "inference",
                    "deadlock_type": "request_queue",
                    "target": {"mode": "all"},
                },
                {
                    "name": "compilation_failure",
                    "type": FaultType.COMPILATION_FAILURE,
                    "layer": "inference",
                    "compilation_stage": "graph_capture",
                    "target": {"mode": "random", "count": 1},
                },
            ],
            dependencies=[],
            cascade_config={"mode": "burst", "interval": 0.0},
        )

    def create_scenario_from_template(
        self, template_name: str, scenario_name: Optional[str] = None, parameters: Optional[dict[str, Any]] = None
    ) -> FaultScenarioConfig:
        """Create a scenario from a template."""

        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.templates[template_name]
        scenario_name = scenario_name or f"{template_name}_{len(self.active_scenarios)}"
        parameters = parameters or {}

        # Convert template faults to FaultConfig objects
        faults = []
        for fault_dict in template.faults:
            fault_dict = fault_dict.copy()
            fault_name = fault_dict.pop("name")

            # Apply parameters
            for key, value in parameters.items():
                if key in fault_dict:
                    fault_dict[key] = value

            # Create appropriate FaultConfig based on type
            fault_type = fault_dict.pop("type")
            layer = fault_dict.pop("layer")

            # This would need proper factory method in real implementation
            config = BaseFaultConfig(name=f"{scenario_name}_{fault_name}", layer=layer, type=fault_type, **fault_dict)
            faults.append(config)

        # Convert dependencies
        dependencies = []
        for dep_dict in template.dependencies:
            dependencies.append(FaultDependencyConfig(**dep_dict))

        return FaultScenarioConfig(
            name=scenario_name,
            description=template.description,
            faults=faults,
            dependencies=dependencies,
            cascade_mode=template.cascade_config.get("mode", "none"),
            cascade_interval_seconds=template.cascade_config.get("interval", 5.0),
            max_cascade_depth=template.cascade_config.get("max_depth", 3),
        )

    def build_execution_graph(self, scenario: FaultScenarioConfig) -> dict[str, FaultExecutionState]:
        """Build execution graph with dependency relationships."""

        graph = {}

        # Create states for all faults
        for fault in scenario.faults:
            graph[fault.name] = FaultExecutionState(name=fault.name, config=fault)

        # Build dependency relationships
        for dep in scenario.dependencies:
            if dep.fault_name in graph:
                graph[dep.fault_name].dependents.add(dep.fault_name)

                # Find the fault that has this dependency
                for fault in scenario.faults:
                    if any(d.fault_name == dep.fault_name for d in scenario.dependencies):
                        graph[fault.name].dependencies.add(dep.fault_name)

        return graph

    async def execute_scenario(self, scenario: FaultScenarioConfig) -> bool:
        """Execute a fault scenario with dependencies."""

        logger.info(f"Starting scenario execution: {scenario.name}")

        # Build execution graph
        execution_graph = self.build_execution_graph(scenario)

        # Create execution context
        context = ScenarioExecutionContext(scenario_name=scenario.name, execution_graph=execution_graph)

        self.active_scenarios[scenario.name] = context

        try:
            # Find root faults (no dependencies)
            ready_faults = [name for name, state in execution_graph.items() if not state.dependencies]
            context.ready_queue.extend(ready_faults)

            # Execute faults based on cascade mode
            if scenario.cascade_mode == "burst":
                success = await self._execute_burst_mode(context, scenario)
            elif scenario.cascade_mode == "linear":
                success = await self._execute_linear_mode(context, scenario)
            elif scenario.cascade_mode == "tree":
                success = await self._execute_tree_mode(context, scenario)
            else:
                success = await self._execute_sequential_mode(context, scenario)

            logger.info(f"Scenario {scenario.name} completed with success: {success}")
            return success

        except Exception as e:
            logger.error(f"Scenario {scenario.name} failed: {e}")
            return False
        finally:
            # Cleanup
            if scenario.name in self.active_scenarios:
                del self.active_scenarios[scenario.name]

    async def _execute_burst_mode(self, context: ScenarioExecutionContext, scenario: FaultScenarioConfig) -> bool:
        """Execute all faults simultaneously."""

        tasks = []
        for fault_name in context.execution_graph:
            task = asyncio.create_task(self._execute_fault_with_deps(fault_name, context, scenario))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if any fault failed when stop_on_failure is True
        if scenario.stop_on_failure:
            return all(not isinstance(r, Exception) for r in results)

        return True

    async def _execute_linear_mode(self, context: ScenarioExecutionContext, scenario: FaultScenarioConfig) -> bool:
        """Execute faults in sequence."""

        while context.ready_queue:
            fault_name = context.ready_queue.popleft()

            try:
                success = await self._execute_fault_with_deps(fault_name, context, scenario)

                if not success and scenario.stop_on_failure:
                    return False

                # Add dependents to ready queue
                for dependent in context.execution_graph[fault_name].dependents:
                    # Check if all dependencies are completed
                    dep_state = context.execution_graph[dependent]
                    if all(d in context.completed for d in dep_state.dependencies):
                        context.ready_queue.append(dependent)

                # Wait before next fault
                if scenario.cascade_interval_seconds > 0:
                    await asyncio.sleep(scenario.cascade_interval_seconds)

            except Exception as e:
                logger.error(f"Fault {fault_name} failed: {e}")
                if scenario.stop_on_failure:
                    return False

        return True

    async def _execute_tree_mode(self, context: ScenarioExecutionContext, scenario: FaultScenarioConfig) -> bool:
        """Execute faults in tree cascade mode."""

        current_level = list(context.ready_queue)
        depth = 0

        while current_level and depth < scenario.max_cascade_depth:
            next_level = []

            # Execute current level
            tasks = []
            for fault_name in current_level:
                task = asyncio.create_task(self._execute_fault_with_deps(fault_name, context, scenario))
                tasks.append((fault_name, task))

            # Wait for completion
            for fault_name, task in tasks:
                try:
                    success = await task

                    if not success and scenario.stop_on_failure:
                        return False

                    # Add children to next level
                    for dependent in context.execution_graph[fault_name].dependents:
                        if dependent not in next_level:
                            next_level.append(dependent)

                except Exception as e:
                    logger.error(f"Fault {fault_name} failed: {e}")
                    if scenario.stop_on_failure:
                        return False

            current_level = next_level
            depth += 1

            if scenario.cascade_interval_seconds > 0:
                await asyncio.sleep(scenario.cascade_interval_seconds)

        return True

    async def _execute_sequential_mode(self, context: ScenarioExecutionContext, scenario: FaultScenarioConfig) -> bool:
        """Execute faults sequentially without cascading."""

        for fault_name in context.execution_graph:
            try:
                await self._execute_fault_with_deps(fault_name, context, scenario)
            except Exception as e:
                logger.error(f"Fault {fault_name} failed: {e}")
                if scenario.stop_on_failure:
                    return False

        return True

    async def _execute_fault_with_deps(
        self, fault_name: str, context: ScenarioExecutionContext, scenario: FaultScenarioConfig
    ) -> bool:
        """Execute a single fault considering its dependencies."""

        state = context.execution_graph[fault_name]

        # Check dependency conditions
        for dep_name in state.dependencies:
            dep_state = context.execution_graph[dep_name]

            # Find dependency configuration
            dep_config = next((d for d in scenario.dependencies if d.fault_name == dep_name), None)

            if not dep_config:
                continue

            # Check condition
            if dep_config.condition == "on_success":
                if dep_state.status != "completed" or dep_name in context.failed:
                    logger.info(f"Skipping {fault_name} due to dependency failure")
                    return True
            elif dep_config.condition == "on_failure":
                if dep_state.status != "failed" and dep_name not in context.failed:
                    logger.info(f"Skipping {fault_name} due to dependency success")
                    return True

            # Apply delay
            if dep_config.delay_seconds:
                await asyncio.sleep(dep_config.delay_seconds)

        # Execute the fault
        try:
            logger.info(f"Executing fault: {fault_name}")
            state.status = "running"
            state.start_time = asyncio.get_event_loop().time()

            # Inject the fault
            result = await self.fault_orchestrator.inject_fault(state.config)

            state.status = "completed"
            state.end_time = asyncio.get_event_loop().time()
            state.result = result
            context.completed.add(fault_name)

            logger.info(f"Fault {fault_name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Fault {fault_name} failed: {e}")
            state.status = "failed"
            state.end_time = asyncio.get_event_loop().time()
            state.result = e
            context.failed.add(fault_name)

            return False

    def get_scenario_status(self, scenario_name: str) -> Optional[dict[str, Any]]:
        """Get status of a running scenario."""

        if scenario_name not in self.active_scenarios:
            return None

        context = self.active_scenarios[scenario_name]

        return {
            "scenario_name": scenario_name,
            "completed_faults": len(context.completed),
            "failed_faults": len(context.failed),
            "total_faults": len(context.execution_graph),
            "ready_queue_size": len(context.ready_queue),
            "execution_states": {
                name: {
                    "status": state.status,
                    "start_time": state.start_time,
                    "end_time": state.end_time,
                    "duration": (state.end_time - state.start_time) if state.start_time and state.end_time else None,
                }
                for name, state in context.execution_graph.items()
            },
        }
