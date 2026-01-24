"""
Proxy classes that intercept and inject faults into target functions.

Organized by layer: L0 (Ray), L1 (Distributed), L2 (verl), L3 (Resource).
"""

from ralph.proxies.base import BaseProxy

# L0 Ray proxies - imported conditionally to handle missing torch/ray
try:
    from ralph.proxies.l0_ray import ObjectLostError, RayGetProxy
except ImportError:
    RayGetProxy = None
    ObjectLostError = None

# L1 Distributed proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.l1_distributed import AllReduceProxy
except ImportError:
    AllReduceProxy = None

# L2 verl proxies - imported conditionally to handle missing torch
try:
    from ralph.proxies.l2_verl import RewardManagerProxy
except ImportError:
    RewardManagerProxy = None

__all__ = [
    "BaseProxy",
    "RayGetProxy",
    "ObjectLostError",
    "AllReduceProxy",
    "RewardManagerProxy",
]
