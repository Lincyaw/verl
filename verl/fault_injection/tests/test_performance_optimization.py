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


"""Tests for fault injection performance optimizations."""

import asyncio
import time
import unittest
from unittest.mock import Mock, patch

import pytest

from verl.fault_injection.async_injectors import (
    AsyncBaseFaultInjector,
    AsyncBatchFaultInjector,
    AsyncCachedFaultInjector,
)
from verl.fault_injection.config import FaultConfig, FaultInjectionConfig, FaultLayer, FaultType
from verl.fault_injection.orchestrator_optimized import OptimizedFaultOrchestrator
from verl.fault_injection.performance import (
    BatchProcessor,
    LRUCache,
    PerformanceConfig,
    PerformanceMode,
    PerformanceProfiler,
    conditional_injection,
    profile_method,
)
from verl.fault_injection.performance_monitor import PerformanceBenchmark, PerformanceMonitor


class TestPerformanceConfig(unittest.TestCase):
    """Test performance configuration."""

    def test_default_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        self.assertEqual(config.mode, PerformanceMode.BALANCED)
        self.assertTrue(config.enable_async)
        self.assertTrue(config.enable_caching)
        self.assertTrue(config.enable_batching)
        self.assertFalse(config.enable_monitoring)
        self.assertFalse(config.enable_profiling)

    def test_disabled_mode(self):
        """Test disabled performance mode."""
        config = PerformanceConfig()
        config.mode = PerformanceMode.DISABLED
        config._apply_mode_overrides()

        self.assertFalse(config.enable_async)
        self.assertFalse(config.enable_caching)
        self.assertFalse(config.enable_batching)
        self.assertFalse(config.enable_monitoring)

    def test_aggressive_mode(self):
        """Test aggressive performance mode."""
        config = PerformanceConfig()
        config.mode = PerformanceMode.AGGRESSIVE
        config._apply_mode_overrides()

        self.assertTrue(config.enable_async)
        self.assertTrue(config.enable_caching)
        self.assertTrue(config.enable_batching)
        self.assertFalse(config.enable_monitoring)  # Monitoring adds overhead


class TestLRUCache(unittest.TestCase):
    """Test LRU cache implementation."""

    def setUp(self):
        self.cache = LRUCache(max_size=3, ttl=1)

    def test_basic_operations(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

        # Test non-existent key
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        # Access key1 to make it recently used
        self.cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        self.cache.set("key4", "value4")

        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        self.cache.set("key1", "value1")
        time.sleep(1.1)  # Wait for TTL to expire

        self.assertIsNone(self.cache.get("key1"))

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # hit
        self.cache.get("key2")  # miss
        self.cache.get("key1")  # hit

        self.assertEqual(self.cache.hit_rate(), 2.0 / 3.0)

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async cache operations."""
        await self.cache.set_async("key1", "value1")
        value = await self.cache.get_async("key1")
        self.assertEqual(value, "value1")


class TestBatchProcessor(unittest.TestCase):
    """Test batch processor implementation."""

    def setUp(self):
        self.processor = BatchProcessor(batch_size=3)

    def test_batch_formation(self):
        """Test batch formation."""
        # Add items one by one
        result1 = self.processor.add("item1")
        self.assertIsNone(result1)

        result2 = self.processor.add("item2")
        self.assertIsNone(result2)

        # Third item should trigger batch
        result3 = self.processor.add("item3")
        self.assertEqual(result3, ["item1", "item2", "item3"])

    def test_flush(self):
        """Test flushing remaining items."""
        self.processor.add("item1")
        self.processor.add("item2")

        # Flush should return remaining items
        remaining = self.processor.flush()
        self.assertEqual(remaining, ["item1", "item2"])

    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing."""
        result1 = await self.processor.add_async("item1")
        self.assertIsNone(result1)

        result2 = await self.processor.add_async("item2")
        self.assertIsNone(result2)

        result3 = await self.processor.add_async("item3")
        self.assertEqual(result3, ["item1", "item2", "item3"])


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiler."""

    def setUp(self):
        self.profiler = PerformanceProfiler()

    def test_profile_context_manager(self):
        """Test profiling with context manager."""
        with self.profiler.profile("test_operation"):
            time.sleep(0.1)

        metrics = self.profiler.get_metrics()
        self.assertEqual(metrics.injection_count, 1)
        self.assertGreater(len(metrics.injection_latency_ms), 0)

    @pytest.mark.asyncio
    async def test_async_profile_context_manager(self):
        """Test async profiling with context manager."""
        async with self.profiler.profile_async("test_async_operation"):
            await asyncio.sleep(0.1)

        metrics = self.profiler.get_metrics()
        self.assertEqual(metrics.injection_count, 1)


class TestDecorators(unittest.TestCase):
    """Test performance decorators."""

    def test_conditional_injection(self):
        """Test conditional injection decorator."""

        @conditional_injection(enabled=True)
        def test_func():
            return "executed"

        result = test_func()
        self.assertEqual(result, "executed")

        @conditional_injection(enabled=False)
        def test_func_disabled():
            return "executed"

        result = test_func_disabled()
        self.assertIsNone(result)

    def test_profile_method_decorator(self):
        """Test method profiling decorator."""

        class TestClass:
            def __init__(self):
                self._profiler = PerformanceProfiler()

            @profile_method("test_method")
            def test_method(self):
                time.sleep(0.01)
                return "result"

        obj = TestClass()
        result = obj.test_method()
        self.assertEqual(result, "result")


class TestOptimizedOrchestrator(unittest.TestCase):
    """Test optimized fault orchestrator."""

    def setUp(self):
        self.config = FaultInjectionConfig(
            enabled=True,
            max_concurrent_faults=5,
            faults=[
                FaultConfig(
                    fault_id="test_fault_1",
                    fault_type=FaultType.PROCESS_CRASH,
                    layer=FaultLayer.WORKER,
                    enabled=True,
                ),
                FaultConfig(
                    fault_id="test_fault_2",
                    fault_type=FaultType.NETWORK_DELAY,
                    layer=FaultLayer.ORCHESTRATION,
                    enabled=True,
                ),
            ],
        )

    @patch("verl.fault_injection.orchestrator_optimized.FaultInjectorRegistry")
    def test_orchestrator_initialization(self, mock_registry):
        """Test optimized orchestrator initialization."""
        # Mock injector creation
        mock_injector = Mock()
        mock_injector.fault_id = "test_fault_1"
        mock_injector.config = self.config.faults[0]
        mock_registry.create.return_value = mock_injector

        orchestrator = OptimizedFaultOrchestrator(self.config)

        self.assertEqual(len(orchestrator._injectors), 2)
        self.assertIsNotNone(orchestrator._target_cache)
        self.assertIsNotNone(orchestrator._batch_processor)
        self.assertIsNotNone(orchestrator._profiler)

    @patch("verl.fault_injection.orchestrator_optimized.FaultInjectorRegistry")
    def test_cached_fault_injection(self, mock_registry):
        """Test cached fault injection."""
        # Mock injector
        mock_injector = Mock()
        mock_injector.fault_id = "test_fault_1"
        mock_injector.config = self.config.faults[0]
        mock_injector.should_inject.return_value = True
        mock_result = Mock()
        mock_result.status.value = "COMPLETED"
        mock_result.timestamp = time.time()
        mock_injector.execute.return_value = mock_result
        mock_registry.create.return_value = mock_injector

        orchestrator = OptimizedFaultOrchestrator(self.config)

        # First injection
        result1 = orchestrator.inject_fault("test_fault_1")
        self.assertIsNotNone(result1)

        # Second injection should use cache
        result2 = orchestrator.inject_fault("test_fault_1")
        self.assertIsNotNone(result2)


class TestAsyncInjectors(unittest.TestCase):
    """Test async fault injectors."""

    @pytest.mark.asyncio
    async def test_async_base_injector(self):
        """Test async base fault injector."""

        class TestAsyncInjector(AsyncBaseFaultInjector):
            async def _inject_async(self, target_context=None):
                await asyncio.sleep(0.01)
                from verl.fault_injection.base import FaultResult

                return FaultResult(
                    fault_id=self.fault_id,
                    status=FaultResult.Status.COMPLETED,
                    message="Async injection completed",
                )

            async def _cleanup_async(self):
                pass

        config = Mock()
        config.fault_id = "test_async"
        injector = TestAsyncInjector(config)

        result = await injector.execute_async()
        self.assertEqual(result.status.value, "COMPLETED")

    @pytest.mark.asyncio
    async def test_batch_async_injection(self):
        """Test batch async fault injection."""

        class TestBatchInjector(AsyncBatchFaultInjector):
            async def _inject_batch_async(self, contexts):
                results = []
                for ctx in contexts:
                    from verl.fault_injection.base import FaultResult

                    results.append(
                        FaultResult(
                            fault_id=self.fault_id,
                            status=FaultResult.Status.COMPLETED,
                            target_context=ctx,
                        )
                    )
                return results

        config = Mock()
        config.fault_id = "test_batch"
        injector = TestBatchInjector(config, batch_size=3)

        contexts = [Mock(), Mock(), Mock()]
        results = await injector.execute_batch_async(contexts)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.status.value, "COMPLETED")

    @pytest.mark.asyncio
    async def test_cached_async_injector(self):
        """Test cached async fault injector."""

        class TestCachedInjector(AsyncCachedFaultInjector):
            def __init__(self, config):
                super().__init__(config, cache_ttl=1)
                self.call_count = 0

            async def _inject_async(self, target_context=None):
                self.call_count += 1
                from verl.fault_injection.base import FaultResult

                return FaultResult(
                    fault_id=self.fault_id,
                    status=FaultResult.Status.COMPLETED,
                    message=f"Call {self.call_count}",
                )

        config = Mock()
        config.fault_id = "test_cached"
        injector = TestCachedInjector(config)

        # First call
        result1 = await injector.execute_cached_async()
        self.assertIn("Call 1", result1.message)

        # Second call should use cache
        result2 = await injector.execute_cached_async()
        self.assertEqual(result1.message, result2.message)
        self.assertEqual(injector.call_count, 1)  # Only called once


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring."""

    def setUp(self):
        self.monitor = PerformanceMonitor(window_size=100)

    def test_record_operations(self):
        """Test recording operations."""
        # Record injections
        for i in range(10):
            self.monitor.record_injection(i + 1, success=i % 2 == 0)

        # Record cache operations
        self.monitor.record_cache_operation(True)  # hit
        self.monitor.record_cache_operation(False)  # miss

        # Record batch
        self.monitor.record_batch(5)

        # Get snapshot
        snapshot = self.monitor.get_snapshot(active_faults=3)

        self.assertEqual(snapshot.injection_count, 10)
        self.assertEqual(snapshot.active_faults, 3)
        self.assertAlmostEqual(snapshot.cache_hit_rate, 0.5, places=2)

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        # Record known latencies
        latencies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for lat in latencies:
            self.monitor.record_injection(lat)

        snapshot = self.monitor.get_snapshot()

        self.assertEqual(snapshot.p50_latency_ms, 5.5)  # Median
        self.assertEqual(snapshot.p95_latency_ms, 9.5)  # 95th percentile


class TestPerformanceBenchmark(unittest.TestCase):
    """Test performance benchmarking."""

    def setUp(self):
        self.benchmark = PerformanceBenchmark()

    def test_sync_injection_benchmark(self):
        """Test sync injection benchmark."""

        def mock_injector_factory():
            injector = Mock()
            result = Mock()
            result.status = result.Status.COMPLETED
            injector.execute.return_value = result
            return injector

        result = self.benchmark.benchmark_sync_injection(
            mock_injector_factory,
            num_operations=100,
        )

        self.assertEqual(result.name, "sync_injection")
        self.assertEqual(result.total_operations, 100)
        self.assertGreater(result.operations_per_second, 0)
        self.assertGreater(result.avg_latency_ms, 0)

    @pytest.mark.asyncio
    async def test_async_injection_benchmark(self):
        """Test async injection benchmark."""

        def mock_injector_factory():
            from verl.fault_injection.async_injectors import AsyncBaseFaultInjector

            class MockAsyncInjector(AsyncBaseFaultInjector):
                async def _inject_async(self, target_context=None):
                    from verl.fault_injection.base import FaultResult

                    return FaultResult(
                        fault_id="mock",
                        status=FaultResult.Status.COMPLETED,
                    )

                async def _cleanup_async(self):
                    pass

            config = Mock()
            config.fault_id = "mock"
            return MockAsyncInjector(config)

        result = await self.benchmark.benchmark_async_injection(
            mock_injector_factory,
            num_operations=50,
            max_concurrent=5,
        )

        self.assertEqual(result.name, "async_injection")
        self.assertEqual(result.total_operations, 50)
        self.assertGreater(result.operations_per_second, 0)

    def test_benchmark_comparison(self):
        """Test benchmark result comparison."""
        # Add mock results
        self.benchmark.results = [
            Mock(
                name="sync_injection",
                operations_per_second=100,
                avg_latency_ms=10,
                latency_percentiles={"p95": 20},
                total_operations=100,
                errors=0,
            ),
            Mock(
                name="async_injection",
                operations_per_second=200,
                avg_latency_ms=5,
                latency_percentiles={"p95": 10},
                total_operations=100,
                errors=0,
            ),
        ]

        comparison = self.benchmark.compare_results()

        self.assertIn("benchmarks", comparison)
        self.assertIn("summary", comparison)
        self.assertAlmostEqual(comparison["summary"]["async_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
