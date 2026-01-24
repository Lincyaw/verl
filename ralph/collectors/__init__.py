"""
Data collection components for capturing fault injection telemetry.

Includes dual stream collector for JSONL output and log parsers.
"""

from ralph.collectors.dual_stream import DualStreamCollector

__all__ = [
    "DualStreamCollector",
]
