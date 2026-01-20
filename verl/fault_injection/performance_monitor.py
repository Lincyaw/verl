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

"""Performance monitoring and benchmarking for fault injection system."""

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""

    timestamp: float
    injection_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cache_hit_rate: float
    batch_efficiency: float
    memory_usage_mb: float
    active_faults: int


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""

    name: str
    duration_seconds: float
    total_operations: int
    operations_per_second: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    latency_percentiles: dict[str, float]
    errors: int


class PerformanceMonitor:
    """Monitor performance of fault injection system."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._injection_latencies: deque[float] = deque(maxlen=window_size)
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_sizes: deque[int] = deque(maxlen=window_size)
        self._memory_snapshots: deque[float] = deque(maxlen=100)
        self._start_time = time.time()
        self._injection_count = 0
        self._error_count = 0

    def record_injection(self, latency_ms: float, success: bool = True) -> None:
        """Record a fault injection operation."""
        self._injection_latencies.append(latency_ms)
        self._injection_count += 1
        if not success:
            self._error_count += 1

    def record_cache_operation(self, hit: bool) -> None:
        """Record a cache operation."""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_batch(self, size: int) -> None:
        """Record a batch operation."""
        self._batch_sizes.append(size)

    def record_memory_usage(self, usage_mb: float) -> None:
        """Record memory usage snapshot."""
        self._memory_snapshots.append(usage_mb)

    def get_snapshot(self, active_faults: int = 0) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        if not self._injection_latencies:
            return PerformanceSnapshot(
                timestamp=time.time(),
                injection_count=self._injection_count,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                cache_hit_rate=0.0,
                batch_efficiency=0.0,
                memory_usage_mb=statistics.mean(self._memory_snapshots) if self._memory_snapshots else 0.0,
                active_faults=active_faults,
            )

        # Calculate latency percentiles
        sorted_latencies = sorted(self._injection_latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_latencies[min(idx, n - 1)]

        # Calculate cache hit rate
        total_cache_ops = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

        # Calculate batch efficiency
        avg_batch_size = statistics.mean(self._batch_sizes) if self._batch_sizes else 0.0
        batch_efficiency = min(avg_batch_size / 10, 1.0)  # Normalize to 10 as optimal

        return PerformanceSnapshot(
            timestamp=time.time(),
            injection_count=self._injection_count,
            avg_latency_ms=statistics.mean(self._injection_latencies),
            p50_latency_ms=percentile(50),
            p95_latency_ms=percentile(95),
            p99_latency_ms=percentile(99),
            cache_hit_rate=cache_hit_rate,
            batch_efficiency=batch_efficiency,
            memory_usage_mb=statistics.mean(self._memory_snapshots) if self._memory_snapshots else 0.0,
            active_faults=active_faults,
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._injection_latencies.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_sizes.clear()
        self._memory_snapshots.clear()
        self._injection_count = 0
        self._error_count = 0
        self._start_time = time.time()


class PerformanceBenchmark:
    """Benchmark fault injection performance."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    async def benchmark_async_injection(
        self,
        injector_factory: Callable[[], Any],
        num_operations: int = 1000,
        max_concurrent: int = 10,
    ) -> BenchmarkResult:
        """Benchmark async fault injection performance."""
        from ..async_injectors import run_batch_async_fault_injection

        latencies: list[float] = []
        errors = 0
        start_time = time.time()

        # Create injectors
        injectors = [injector_factory() for _ in range(num_operations)]
        contexts = [None] * num_operations

        # Run benchmark
        for i in range(0, num_operations, max_concurrent):
            batch_injectors = injectors[i : i + max_concurrent]
            batch_contexts = contexts[i : i + max_concurrent]

            batch_start = time.time()
            results = await run_batch_async_fault_injection(
                batch_injectors,
                batch_contexts,
                max_concurrent,
            )
            batch_end = time.time()

            # Record latencies
            batch_latency_ms = (batch_end - batch_start) * 1000
            for _ in results:
                latencies.append(batch_latency_ms / len(results))

            # Count errors
            for result in results:
                if result.status == result.Status.FAILED:
                    errors += 1

        duration = time.time() - start_time

        result = BenchmarkResult(
            name="async_injection",
            duration_seconds=duration,
            total_operations=num_operations,
            operations_per_second=num_operations / duration,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            latency_percentiles=self._calculate_percentiles(latencies),
            errors=errors,
        )

        self.results.append(result)
        return result

    def benchmark_sync_injection(
        self,
        injector_factory: Callable[[], Any],
        num_operations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark sync fault injection performance."""
        latencies: list[float] = []
        errors = 0
        start_time = time.time()

        for _ in range(num_operations):
            injector = injector_factory()

            op_start = time.time()
            try:
                result = injector.execute(None)
                op_end = time.time()

                latencies.append((op_end - op_start) * 1000)

                if result.status == result.Status.FAILED:
                    errors += 1
            except Exception as e:
                errors += 1
                logger.error(f"Injection failed: {e}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            name="sync_injection",
            duration_seconds=duration,
            total_operations=num_operations,
            operations_per_second=num_operations / duration,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            latency_percentiles=self._calculate_percentiles(latencies) if latencies else {},
            errors=errors,
        )

        self.results.append(result)
        return result

    def benchmark_cache_performance(
        self,
        cache_factory: Callable[[], Any],
        num_operations: int = 10000,
        hit_ratio: float = 0.8,
    ) -> BenchmarkResult:
        """Benchmark cache performance."""
        cache = cache_factory()
        latencies: list[float] = []
        errors = 0
        start_time = time.time()

        # Pre-populate cache
        num_pre_populate = int(num_operations * hit_ratio)
        for i in range(num_pre_populate):
            cache.set(f"key_{i}", f"value_{i}")

        # Benchmark operations
        for i in range(num_operations):
            key = f"key_{i % num_pre_populate}" if i < num_pre_populate * hit_ratio else f"key_missing_{i}"

            op_start = time.time()
            try:
                result = cache.get(key)
                op_end = time.time()

                latencies.append((op_end - op_start) * 1000)
            except Exception as e:
                errors += 1
                logger.error(f"Cache operation failed: {e}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            name="cache_performance",
            duration_seconds=duration,
            total_operations=num_operations,
            operations_per_second=num_operations / duration,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            latency_percentiles=self._calculate_percentiles(latencies) if latencies else {},
            errors=errors,
        )

        self.results.append(result)
        return result

    def _calculate_percentiles(self, values: list[float]) -> dict[str, float]:
        """Calculate latency percentiles."""
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        percentiles = {
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
        }

        return percentiles

    def compare_results(self) -> dict[str, Any]:
        """Compare all benchmark results."""
        if not self.results:
            return {}

        comparison = {
            "benchmarks": {},
            "summary": {},
        }

        # Collect results
        for result in self.results:
            comparison["benchmarks"][result.name] = {
                "ops_per_second": result.operations_per_second,
                "avg_latency_ms": result.avg_latency_ms,
                "p95_latency_ms": result.latency_percentiles.get("p95", 0),
                "error_rate": result.errors / result.total_operations if result.total_operations > 0 else 0,
            }

        # Calculate speedup
        if "sync_injection" in comparison["benchmarks"] and "async_injection" in comparison["benchmarks"]:
            sync_ops = comparison["benchmarks"]["sync_injection"]["ops_per_second"]
            async_ops = comparison["benchmarks"]["async_injection"]["ops_per_second"]
            comparison["summary"]["async_speedup"] = async_ops / sync_ops if sync_ops > 0 else 0

        return comparison

    def print_report(self) -> None:
        """Print benchmark report."""
        print("\n" + "=" * 60)
        print("Fault Injection Performance Benchmark Report")
        print("=" * 60)

        for result in self.results:
            print(f"\nBenchmark: {result.name}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Total Operations: {result.total_operations}")
            print(f"  Operations/Second: {result.operations_per_second:.2f}")
            print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
            print(f"  Min/Max Latency: {result.min_latency_ms:.2f}ms / {result.max_latency_ms:.2f}ms")
            print(f"  Errors: {result.errors}")

            if result.latency_percentiles:
                print("  Latency Percentiles:")
                for p, v in result.latency_percentiles.items():
                    print(f"    {p}: {v:.2f}ms")

        # Comparison
        comparison = self.compare_results()
        if "summary" in comparison:
            print("\nSummary:")
            for key, value in comparison["summary"].items():
                print(f"  {key}: {value:.2f}x")

        print("\n" + "=" * 60)
