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
    from ralph.proxies.l2_verl import CheckpointSaveProxy, RewardManagerProxy
except ImportError:
    CheckpointSaveProxy = None
    RewardManagerProxy = None

# Algorithm proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.algorithm import GAEProxy
except ImportError:
    GAEProxy = None

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
    "GAEProxy",
]
