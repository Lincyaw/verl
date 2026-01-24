"""
Proxy classes that intercept and inject faults into target functions.

Organized by layer: L0 (Ray), L1 (Distributed), L2 (verl), L3 (Resource).
"""

from ralph.proxies.base import BaseProxy

# L0 Ray proxies - imported conditionally to handle missing torch/ray
try:
    from ralph.proxies.l0_ray import (
        DummyObjectRef,
        ExecuteAllProxy,
        ObjectLostError,
        ObjectStoreFullError,
        RayGetProxy,
        RayPutProxy,
        WorkerDeathError,
    )
except ImportError:
    RayGetProxy = None
    RayPutProxy = None
    ExecuteAllProxy = None
    ObjectLostError = None
    ObjectStoreFullError = None
    DummyObjectRef = None
    WorkerDeathError = None

# L1 Distributed proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.l1_distributed import AllGatherProxy, AllReduceProxy, BarrierProxy
except ImportError:
    AllReduceProxy = None
    AllGatherProxy = None
    BarrierProxy = None

# L2 verl proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.l2_verl import (
        CheckpointLoadProxy,
        CheckpointSaveProxy,
        RewardManagerProxy,
        UpdateActorProxy,
    )
except ImportError:
    CheckpointLoadProxy = None
    CheckpointSaveProxy = None
    RewardManagerProxy = None
    UpdateActorProxy = None

# Algorithm proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.algorithm import GAEProxy, GRPOProxy, KLPenaltyProxy
except ImportError:
    GAEProxy = None
    GRPOProxy = None
    KLPenaltyProxy = None

# Inference proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.inference import GenerateProxy, UpdateWeightsProxy
except ImportError:
    GenerateProxy = None
    UpdateWeightsProxy = None

# Worker proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.worker import ComputeValuesProxy
except ImportError:
    ComputeValuesProxy = None

# L3 Resource proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.l3_resource import GPUMemoryInjector
except ImportError:
    GPUMemoryInjector = None

# Data pipeline proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.data_pipeline import DataProtoChunkProxy, DataProtoConcatProxy
except ImportError:
    DataProtoConcatProxy = None
    DataProtoChunkProxy = None

# Agent proxies - no torch dependency
try:
    from ralph.proxies.agent import CallToolProxy, ToolParserProxy
except ImportError:
    CallToolProxy = None
    ToolParserProxy = None

__all__ = [
    "BaseProxy",
    "RayGetProxy",
    "RayPutProxy",
    "ExecuteAllProxy",
    "ObjectLostError",
    "ObjectStoreFullError",
    "DummyObjectRef",
    "WorkerDeathError",
    "AllReduceProxy",
    "AllGatherProxy",
    "BarrierProxy",
    "RewardManagerProxy",
    "CheckpointSaveProxy",
    "CheckpointLoadProxy",
    "UpdateActorProxy",
    "GAEProxy",
    "GRPOProxy",
    "KLPenaltyProxy",
    "GenerateProxy",
    "UpdateWeightsProxy",
    "ComputeValuesProxy",
    "GPUMemoryInjector",
    "DataProtoConcatProxy",
    "DataProtoChunkProxy",
    "CallToolProxy",
    "ToolParserProxy",
]
