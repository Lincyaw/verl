"""
Proxy classes that intercept and inject faults into target functions.

Organized by layer: L0 (Ray), L1 (Distributed), L2 (verl), L3 (Resource).
"""

from ralph.proxies.base import BaseProxy

__all__ = ["BaseProxy"]
