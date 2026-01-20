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

"""Fault injectors for different types of faults."""

from .engine import (
    CheckpointCorruptionInjector,
    DeviceMapErrorInjector,
    DeviceMeshErrorInjector,
    EngineHangInjector,
    EngineInitFailureInjector,
    NCCLFailureInjector,
    PrecisionErrorInjector,
)
from .inference import (
    CompilationFailureInjector,
    InferenceOOMInjector,
    SchedulerDeadlockInjector,
)
from .network import NetworkDelayInjector, NetworkPartitionInjector
from .process import ProcessExitInjector, ProcessHangInjector, ProcessKillInjector
from .ui import (
    CLIArgErrorInjector,
    EnvVarErrorInjector,
    HydraConfigErrorInjector,
    RayInitFailureInjector,
    UICrashInjector,
    UIFreezeInjector,
)
from .worker import (
    CudaMemoryFragmentationInjector,
    CudaOOMWorkerInjector,
    FSDPShardingErrorInjector,
    FSDPSyncFailureInjector,
    GradientSyncTimeoutInjector,
    MegatronPipelineErrorInjector,
    MegatronSyncFailureInjector,
    WorkerCrashInjector,
    WorkerHangInjector,
)

__all__ = [
    "ProcessKillInjector",
    "ProcessExitInjector",
    "ProcessHangInjector",
    "NetworkDelayInjector",
    "NetworkPartitionInjector",
    "HydraConfigErrorInjector",
    "RayInitFailureInjector",
    "CLIArgErrorInjector",
    "EnvVarErrorInjector",
    "UIFreezeInjector",
    "UICrashInjector",
    "FSDPSyncFailureInjector",
    "FSDPShardingErrorInjector",
    "MegatronSyncFailureInjector",
    "MegatronPipelineErrorInjector",
    "GradientSyncTimeoutInjector",
    "CudaOOMWorkerInjector",
    "CudaMemoryFragmentationInjector",
    "WorkerCrashInjector",
    "WorkerHangInjector",
    "EngineInitFailureInjector",
    "EngineHangInjector",
    "CheckpointCorruptionInjector",
    "NCCLFailureInjector",
    "DeviceMeshErrorInjector",
    "PrecisionErrorInjector",
    "DeviceMapErrorInjector",
    "InferenceOOMInjector",
    "SchedulerDeadlockInjector",
    "CompilationFailureInjector",
]
