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


"""verl fault injection - CLI wrapper for injecting faults into distributed training."""

# Import injectors to register them
from . import injectors  # noqa: F401
from .base import BaseFaultInjector, FaultContext, FaultResult
from .config import (
    BaseFaultConfig,
    FaultConfig,
    FaultDependencyConfig,
    FaultInjectionConfig,
    FaultLayer,
    FaultScenarioConfig,
    FaultScenarioTemplate,
    FaultTarget,
    FaultTrigger,
    FaultType,
    NetworkFaultConfig,
    ProcessFaultConfig,
)
from .integration import (
    FaultInjectionHooks,
    RayFaultInjectionManager,
    create_fault_injection_hooks,
    initialize_ray_fault_injection,
)
from .orchestrator import FaultOrchestrator
from .scenario import FaultScenarioOrchestrator
from .scenario_loader import ScenarioConfigLoader
from .triggers import (
    BaseTrigger,
    ConditionalTrigger,
    CountTrigger,
    ImmediateTrigger,
    ProbabilisticTrigger,
    TimedTrigger,
    TriggerManager,
)

__all__ = [
    "FaultOrchestrator",
    "FaultInjectionConfig",
    "FaultConfig",
    "FaultLayer",
    "FaultType",
    "FaultTrigger",
    "FaultTarget",
    "BaseFaultConfig",
    "ProcessFaultConfig",
    "NetworkFaultConfig",
    "FaultDependencyConfig",
    "FaultScenarioConfig",
    "FaultScenarioTemplate",
    "FaultContext",
    "FaultResult",
    "BaseFaultInjector",
    "RayFaultInjectionManager",
    "initialize_ray_fault_injection",
    "FaultInjectionHooks",
    "create_fault_injection_hooks",
    "FaultScenarioOrchestrator",
    "TriggerManager",
    "BaseTrigger",
    "ImmediateTrigger",
    "TimedTrigger",
    "CountTrigger",
    "ProbabilisticTrigger",
    "ConditionalTrigger",
    "ScenarioConfigLoader",
]
