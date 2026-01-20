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

"""Performance optimization and overhead control for fault injection system."""

import asyncio
import functools
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Performance configuration flags
ENABLE_ASYNC_INJECTION = os.getenv("VERL_FAULT_INJECTION_ASYNC", "true").lower() == "true"
ENABLE_CACHING = os.getenv("VERL_FAULT_INJECTION_CACHE", "true").lower() == "true"
ENABLE_BATCH_PROCESSING = os.getenv("VERL_FAULT_INJECTION_BATCH", "true").lower() == "true"
ENABLE_MONITORING = os.getenv("VERL_FAULT_INJECTION_MONITORING", "false").lower() == "true"
ENABLE_PROFILING = os.getenv("VERL_FAULT_INJECTION_PROFILING", "false").lower() == "true"

# Performance thresholds
MAX_CACHE_SIZE = int(os.getenv("VERL_FAULT_INJECTION_MAX_CACHE_SIZE", "1000"))
BATCH_SIZE = int(os.getenv("VERL_FAULT_INJECTION_BATCH_SIZE", "10"))
CACHE_TTL_SECONDS = int(os.getenv("VERL_FAULT_INJECTION_CACHE_TTL", "300"))

# Type variables
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class PerformanceMode(Enum):
    """Performance optimization modes."""

    DISABLED = "disabled"
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetrics:
    """Performance metrics for fault injection operations."""

    injection_count: int = 0
    injection_latency_ms: list[float] = field(default_factory=list)
    cache_hit_rate: float = 0.0
    batch_processed: int = 0
    async_operations: int = 0
    lock_contention_time_ms: float = 0.0
    memory_usage_mb: float = 0.0


class PerformanceConfig:
    """Configuration for performance optimizations."""

    def __init__(self):
        self.mode = PerformanceMode(os.getenv("VERL_FAULT_INJECTION_MODE", "balanced"))
        self.enable_async = ENABLE_ASYNC_INJECTION
        self.enable_caching = ENABLE_CACHING
        self.enable_batching = ENABLE_BATCH_PROCESSING
        self.enable_monitoring = ENABLE_MONITORING
        self.enable_profiling = ENABLE_PROFILING
        self.max_cache_size = MAX_CACHE_SIZE
        self.batch_size = BATCH_SIZE
        self.cache_ttl = CACHE_TTL_SECONDS

        # Mode-specific overrides
        if self.mode == PerformanceMode.DISABLED:
            self.enable_async = False
            self.enable_caching = False
            self.enable_batching = False
            self.enable_monitoring = False
        elif self.mode == PerformanceMode.MINIMAL:
            self.enable_caching = True
            self.enable_async = False
            self.enable_batching = False
            self.enable_monitoring = False
        elif self.mode == PerformanceMode.AGGRESSIVE:
            self.enable_async = True
            self.enable_caching = True
            self.enable_batching = True
            self.enable_monitoring = False  # Monitoring adds overhead


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock() if ENABLE_ASYNC_INJECTION else None
        self._hits = 0
        self._misses = 0

    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from cache asynchronously."""
        if not ENABLE_CACHING:
            return None

        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]

            self._misses += 1
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache synchronously."""
        if not ENABLE_CACHING:
            return None

        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            else:
                # Expired
                del self._cache[key]

        self._misses += 1
        return None

    async def set_async(self, key: str, value: Any) -> None:
        """Set value in cache asynchronously."""
        if not ENABLE_CACHING:
            return

        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())

            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache synchronously."""
        if not ENABLE_CACHING:
            return

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())

        # Evict oldest if over capacity
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class BatchProcessor:
    """Batch processing for fault injection operations."""

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self._batch_queue: list[Any] = []
        self._processing = False
        self._lock = asyncio.Lock() if ENABLE_ASYNC_INJECTION else None

    async def add_async(self, item: Any) -> Optional[list[Any]]:
        """Add item to batch queue asynchronously."""
        if not ENABLE_BATCH_PROCESSING:
            return [item]

        async with self._lock:
            self._batch_queue.append(item)

            if len(self._batch_queue) >= self.batch_size:
                batch = self._batch_queue[: self.batch_size]
                self._batch_queue = self._batch_queue[self.batch_size :]
                return batch

            return None

    def add(self, item: Any) -> Optional[list[Any]]:
        """Add item to batch queue synchronously."""
        if not ENABLE_BATCH_PROCESSING:
            return [item]

        self._batch_queue.append(item)

        if len(self._batch_queue) >= self.batch_size:
            batch = self._batch_queue[: self.batch_size]
            self._batch_queue = self._batch_queue[self.batch_size :]
            return batch

        return None

    async def flush_async(self) -> list[Any]:
        """Flush remaining items in batch queue asynchronously."""
        async with self._lock:
            batch = self._batch_queue.copy()
            self._batch_queue.clear()
            return batch

    def flush(self) -> list[Any]:
        """Flush remaining items in batch queue synchronously."""
        batch = self._batch_queue.copy()
        self._batch_queue.clear()
        return batch


class PerformanceProfiler:
    """Performance profiling for fault injection operations."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._operation_timers: dict[str, float] = {}

    @contextmanager
    def profile(self, operation: str):
        """Context manager for profiling operations."""
        if not ENABLE_PROFILING:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if operation == "injection":
                self.metrics.injection_count += 1
                self.metrics.injection_latency_ms.append(elapsed_ms)

            logger.debug(f"Operation '{operation}' took {elapsed_ms:.2f}ms")

    @asynccontextmanager
    async def profile_async(self, operation: str):
        """Async context manager for profiling operations."""
        if not ENABLE_PROFILING:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if operation == "injection":
                self.metrics.injection_count += 1
                self.metrics.injection_latency_ms.append(elapsed_ms)

            logger.debug(f"Async operation '{operation}' took {elapsed_ms:.2f}ms")

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics


def conditional_injection(enabled: bool = True):
    """Decorator for conditional fault injection based on performance mode."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled or not ENABLE_CACHING:
                return None
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not enabled or not ENABLE_CACHING:
                return None
            return await func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


def optimize_performance(mode: PerformanceMode = PerformanceMode.BALANCED):
    """Class decorator for performance optimization."""

    def decorator(cls):
        # Add performance monitoring to key methods
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                if asyncio.iscoroutinefunction(attr):
                    setattr(cls, attr_name, profile_async_method(attr_name)(attr))
                else:
                    setattr(cls, attr_name, profile_method(attr_name)(attr))

        # Add performance configuration
        cls._performance_config = PerformanceConfig()
        cls._performance_config.mode = mode

        return cls

    return decorator


def profile_method(operation_name: str):
    """Decorator for profiling method execution time."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not ENABLE_PROFILING:
                return func(self, *args, **kwargs)

            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                if hasattr(self, "_profiler"):
                    self._profiler.metrics.injection_latency_ms.append(elapsed_ms)
                logger.debug(f"Method '{operation_name}' took {elapsed_ms:.2f}ms")

        return wrapper

    return decorator


def profile_async_method(operation_name: str):
    """Decorator for profiling async method execution time."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not ENABLE_PROFILING:
                return await func(self, *args, **kwargs)

            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                if hasattr(self, "_profiler"):
                    self._profiler.metrics.injection_latency_ms.append(elapsed_ms)
                logger.debug(f"Async method '{operation_name}' took {elapsed_ms:.2f}ms")

        return wrapper

    return decorator


# Global instances
_performance_config = PerformanceConfig()
_target_cache = LRUCache()
_batch_processor = BatchProcessor()
_profiler = PerformanceProfiler()


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration."""
    return _performance_config


def get_target_cache() -> LRUCache:
    """Get the global target cache."""
    return _target_cache


def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor."""
    return _batch_processor


def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler."""
    return _profiler
