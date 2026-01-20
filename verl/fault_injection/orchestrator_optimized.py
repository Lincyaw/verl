# Copyright 2026 Aoyang Fang Ltd. and/or its affiliates
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

"""Optimized fault injection orchestrator with performance improvements."""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import (
    BaseFaultInjector,
    FaultContext,
    FaultInjectorRegistry,
    FaultResult,
    FaultStatus,
    create_target_selector,
)
from .config import FaultConfig, FaultInjectionConfig, FaultLayer
from .monitoring.integration import MonitoringIntegration
from .performance import (
    get_batch_processor,
    get_performance_config,
    get_profiler,
    get_target_cache,
    profile_async_method,
    profile_method,
)
from .recovery.integration import RecoveryOrchestratorIntegration, create_recovery_integration

logger = logging.getLogger(__name__)


@dataclass
class OptimizedFaultOrchestratorMetrics:
    """Optimized metrics for fault injection orchestrator."""

    total_faults: int = 0
    active_faults: int = 0
    completed_faults: int = 0
    failed_faults: int = 0
    recovered_faults: int = 0
    injection_duration: dict[str, float] = field(default_factory=dict)
    layer_stats: dict[FaultLayer, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0
    batch_operations: int = 0
    async_operations: int = 0


class OptimizedFaultOrchestrator:
    """Optimized orchestrator with performance improvements for fault injection."""

    def __init__(self, config: FaultInjectionConfig):
        self.config = config
        self._injectors: dict[str, BaseFaultInjector] = {}
        self._active_faults: dict[str, BaseFaultInjector] = {}
        self._fault_history: list[FaultResult] = []
        self._metrics = OptimizedFaultOrchestratorMetrics()
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock() if get_performance_config().enable_async else None
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_faults)
        self._shutdown = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Performance components
        self._perf_config = get_performance_config()
        self._profiler = get_profiler()
        self._target_cache = get_target_cache()
        self._batch_processor = get_batch_processor()

        # Initialize recovery integration if enabled
        self._recovery_integration: Optional[RecoveryOrchestratorIntegration] = None
        if config.recovery.enabled:
            self._recovery_integration = create_recovery_integration(self, config)

        # Initialize monitoring integration (optional for performance)
        self._monitoring_integration: Optional[MonitoringIntegration] = None
        if self._perf_config.enable_monitoring:
            self._monitoring_integration = MonitoringIntegration(config.monitoring, self)

        # Target management
        self._available_targets: list[FaultContext] = []
        self._target_update_callbacks: list[callable] = []
        self._target_cache_key = "available_targets"

        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))

        # Initialize injectors from config
        self._initialize_injectors()

    def _initialize_injectors(self) -> None:
        """Initialize fault injectors from configuration."""
        for fault_config in self.config.faults:
            if fault_config.enabled:
                try:
                    injector = FaultInjectorRegistry.create(fault_config)
                    self._injectors[injector.fault_id] = injector
                    logger.info(f"Initialized fault injector: {injector.fault_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize fault injector: {e}")

    @profile_method("register_target_update_callback")
    def register_target_update_callback(self, callback: callable) -> None:
        """Register a callback for target updates."""
        self._target_update_callbacks.append(callback)

    @profile_method("update_available_targets")
    def update_available_targets(self, targets: list[FaultContext]) -> None:
        """Update the list of available fault targets with caching."""
        with self._lock:
            self._available_targets = targets
            # Update cache
            if self._perf_config.enable_caching:
                self._target_cache.set(self._target_cache_key, targets)
            logger.debug(f"Updated available targets: {len(targets)} targets")

            # Notify callbacks
            for callback in self._target_update_callbacks:
                try:
                    callback(targets)
                except Exception as e:
                    logger.error(f"Target update callback failed: {e}")

    def _get_cached_targets(self) -> Optional[list[FaultContext]]:
        """Get targets from cache if available."""
        if not self._perf_config.enable_caching:
            return None

        cached = self._target_cache.get(self._target_cache_key)
        if cached:
            self._metrics.cache_hits += 1
            logger.debug("Using cached targets")
        else:
            self._metrics.cache_misses += 1

        return cached

    @profile_method("add_fault")
    def add_fault(self, fault_config: FaultConfig) -> str:
        """Add a new fault to the orchestrator."""
        with self._lock:
            injector = FaultInjectorRegistry.create(fault_config)
            self._injectors[injector.fault_id] = injector
            logger.info(f"Added fault injector: {injector.fault_id}")
            return injector.fault_id

    @profile_method("remove_fault")
    def remove_fault(self, fault_id: str) -> bool:
        """Remove a fault from the orchestrator."""
        with self._lock:
            if fault_id in self._injectors:
                del self._injectors[fault_id]
                if fault_id in self._active_faults:
                    del self._active_faults[fault_id]
                logger.info(f"Removed fault injector: {fault_id}")
                return True
            return False

    @profile_method("inject_fault")
    def inject_fault(self, fault_id: str, target_context: Optional[FaultContext] = None) -> Optional[FaultResult]:
        """Inject a specific fault with caching and batching support."""
        # Check cache first
        cache_key = f"fault_result:{fault_id}"
        if self._perf_config.enable_caching:
            cached_result = self._target_cache.get(cache_key)
            if cached_result and time.time() - cached_result.timestamp < 60:  # 1 minute TTL for results
                logger.debug(f"Using cached fault result for {fault_id}")
                return cached_result

        with self._lock:
            if fault_id not in self._injectors:
                logger.error(f"Fault injector not found: {fault_id}")
                return None

            injector = self._injectors[fault_id]

            # Check if we can inject more faults
            if len(self._active_faults) >= self.config.max_concurrent_faults:
                logger.warning(f"Maximum concurrent faults reached: {self.config.max_concurrent_faults}")
                return None

            # Select target if not provided
            if target_context is None:
                # Try cache first
                cached_targets = self._get_cached_targets()
                if cached_targets:
                    # Use cached targets for selection
                    temp_available = self._available_targets
                    self._available_targets = cached_targets
                    target_context = self._select_target_for_fault(injector.config)
                    self._available_targets = temp_available
                else:
                    target_context = self._select_target_for_fault(injector.config)

                if target_context is None:
                    logger.warning(f"No suitable target found for fault: {fault_id}")
                    return None

            # Check if fault should be injected based on trigger
            if not injector.should_inject(target_context):
                return None

            # Execute fault injection
            self._active_faults[fault_id] = injector
            self._metrics.active_faults += 1

        # Execute outside of lock with profiling
        with self._profiler.profile("fault_injection"):
            try:
                result = injector.execute(target_context)

                # Cache result if enabled
                if self._perf_config.enable_caching:
                    self._target_cache.set(cache_key, result)

                with self._lock:
                    self._fault_history.append(result)
                    self._metrics.total_faults += 1
                    self._metrics.active_faults -= 1
                    self._metrics.completed_faults += 1
                    self._metrics.injection_duration[fault_id] = result.duration or 0
                    self._metrics.layer_stats[injector.config.layer] += 1

                    if result.status == FaultStatus.FAILED:
                        self._metrics.failed_faults += 1

                    if fault_id in self._active_faults:
                        del self._active_faults[fault_id]

                logger.info(f"Fault injection completed: {fault_id} - Status: {result.status.value}")

                # Notify monitoring integration (async if enabled)
                if self._monitoring_integration:
                    if self._perf_config.enable_async:
                        asyncio.create_task(self._monitoring_integration._on_fault_completed(result))
                    else:
                        self._monitoring_integration._on_fault_completed(result)

                # Trigger recovery if enabled
                if self._recovery_integration and result.status in [FaultStatus.COMPLETED, FaultStatus.FAILED]:
                    recovered_result = self._recovery_integration.on_fault_completed(result, target_context)
                    if recovered_result:
                        with self._lock:
                            self._metrics.recovered_faults += 1
                        return recovered_result

                return result

            except Exception as e:
                logger.error(f"Fault injection failed: {fault_id} - Error: {e}")
                with self._lock:
                    self._metrics.active_faults -= 1
                    if fault_id in self._active_faults:
                        del self._active_faults[fault_id]
                return None

    @profile_async_method("inject_fault_async")
    async def inject_fault_async(
        self, fault_id: str, target_context: Optional[FaultContext] = None
    ) -> Optional[FaultResult]:
        """Asynchronously inject a specific fault."""
        if not self._perf_config.enable_async:
            # Fall back to sync version
            return self.inject_fault(fault_id, target_context)

        self._metrics.async_operations += 1

        # Check cache first
        cache_key = f"fault_result:{fault_id}"
        if self._perf_config.enable_caching:
            cached_result = await self._target_cache.get_async(cache_key)
            if cached_result and time.time() - cached_result.timestamp < 60:
                logger.debug(f"Using cached fault result for {fault_id}")
                return cached_result

        async with self._async_lock:
            if fault_id not in self._injectors:
                logger.error(f"Fault injector not found: {fault_id}")
                return None

            injector = self._injectors[fault_id]

            # Check concurrent fault limit
            if len(self._active_faults) >= self.config.max_concurrent_faults:
                logger.warning(f"Maximum concurrent faults reached: {self.config.max_concurrent_faults}")
                return None

            # Select target if not provided
            if target_context is None:
                target_context = await self._select_target_for_fault_async(injector.config)
                if target_context is None:
                    logger.warning(f"No suitable target found for fault: {fault_id}")
                    return None

            # Check trigger
            if not await self._check_trigger_async(injector, target_context):
                return None

            self._active_faults[fault_id] = injector
            self._metrics.active_faults += 1

        # Execute asynchronously
        async with self._profiler.profile_async("fault_injection_async"):
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._executor, injector.execute, target_context)

                # Cache result
                if self._perf_config.enable_caching:
                    await self._target_cache.set_async(cache_key, result)

                async with self._async_lock:
                    self._fault_history.append(result)
                    self._metrics.total_faults += 1
                    self._metrics.active_faults -= 1
                    self._metrics.completed_faults += 1
                    self._metrics.injection_duration[fault_id] = result.duration or 0
                    self._metrics.layer_stats[injector.config.layer] += 1

                    if result.status == FaultStatus.FAILED:
                        self._metrics.failed_faults += 1

                    if fault_id in self._active_faults:
                        del self._active_faults[fault_id]

                logger.info(f"Async fault injection completed: {fault_id} - Status: {result.status.value}")

                # Handle recovery asynchronously
                if self._recovery_integration and result.status in [FaultStatus.COMPLETED, FaultStatus.FAILED]:
                    recovered_result = await self._recovery_integration.on_fault_completed_async(result, target_context)
                    if recovered_result:
                        async with self._async_lock:
                            self._metrics.recovered_faults += 1
                        return recovered_result

                return result

            except Exception as e:
                logger.error(f"Async fault injection failed: {fault_id} - Error: {e}")
                async with self._async_lock:
                    self._metrics.active_faults -= 1
                    if fault_id in self._active_faults:
                        del self._active_faults[fault_id]
                return None

    async def _check_trigger_async(self, injector: BaseFaultInjector, target_context: FaultContext) -> bool:
        """Check trigger asynchronously."""
        # Run trigger check in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, injector.should_inject, target_context)

    async def _select_target_for_fault_async(self, fault_config: FaultConfig) -> Optional[FaultContext]:
        """Select target for fault asynchronously."""
        # Try cache first
        cached_targets = await self._get_cached_targets_async()
        if cached_targets:
            # Use cached targets
            async with self._async_lock:
                temp_available = self._available_targets
                self._available_targets = cached_targets
                result = self._select_target_for_fault(fault_config)
                self._available_targets = temp_available
                return result

        # No cached targets, use current
        return self._select_target_for_fault(fault_config)

    async def _get_cached_targets_async(self) -> Optional[list[FaultContext]]:
        """Get targets from cache asynchronously."""
        if not self._perf_config.enable_caching:
            return None

        cached = await self._target_cache.get_async(self._target_cache_key)
        if cached:
            self._metrics.cache_hits += 1
            logger.debug("Using cached targets (async)")
        else:
            self._metrics.cache_misses += 1

        return cached

    @profile_method("inject_all_enabled")
    def inject_all_enabled(self) -> list[FaultResult]:
        """Inject all enabled faults with batching support."""
        results = []

        with self._lock:
            fault_ids = list(self._injectors.keys())

        # Process in batches if enabled
        if self._perf_config.enable_batching:
            for i in range(0, len(fault_ids), self._perf_config.batch_size):
                batch = fault_ids[i : i + self._perf_config.batch_size]
                batch_results = self._process_fault_batch(batch)
                results.extend(batch_results)
                self._metrics.batch_operations += 1
        else:
            # Process sequentially
            for fault_id in fault_ids:
                if self._shutdown:
                    break
                result = self.inject_fault(fault_id)
                if result:
                    results.append(result)

        return results

    def _process_fault_batch(self, fault_ids: list[str]) -> list[FaultResult]:
        """Process a batch of faults."""
        results = []

        # Use thread pool for parallel execution
        futures = []
        for fault_id in fault_ids:
            if self._shutdown:
                break
            future = self._executor.submit(self.inject_fault, fault_id)
            futures.append((fault_id, future))

        # Collect results
        for fault_id, future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Batch fault injection failed for {fault_id}: {e}")

        return results

    def _select_target_for_fault(self, fault_config: FaultConfig) -> Optional[FaultContext]:
        """Select a target for the fault."""
        if not self._available_targets:
            return None

        # Use target selector from config
        selector = create_target_selector(fault_config.target_selector)
        return selector.select_target(self._available_targets, fault_config)

    def get_metrics(self) -> OptimizedFaultOrchestratorMetrics:
        """Get orchestrator metrics."""
        with self._lock:
            return OptimizedFaultOrchestratorMetrics(
                total_faults=self._metrics.total_faults,
                active_faults=self._metrics.active_faults,
                completed_faults=self._metrics.completed_faults,
                failed_faults=self._metrics.failed_faults,
                recovered_faults=self._metrics.recovered_faults,
                injection_duration=self._metrics.injection_duration.copy(),
                layer_stats=self._metrics.layer_stats.copy(),
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                batch_operations=self._metrics.batch_operations,
                async_operations=self._metrics.async_operations,
            )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = self.get_metrics()
        perf_metrics = self._profiler.get_metrics()

        return {
            "orchestrator": {
                "total_faults": metrics.total_faults,
                "active_faults": metrics.active_faults,
                "cache_hit_rate": metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
                if (metrics.cache_hits + metrics.cache_misses) > 0
                else 0,
                "batch_operations": metrics.batch_operations,
                "async_operations": metrics.async_operations,
            },
            "profiler": {
                "injection_count": perf_metrics.injection_count,
                "avg_injection_latency_ms": sum(perf_metrics.injection_latency_ms)
                / len(perf_metrics.injection_latency_ms)
                if perf_metrics.injection_latency_ms
                else 0,
                "cache_hit_rate": self._target_cache.hit_rate(),
            },
        }

    def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down optimized fault orchestrator")
        self._shutdown = True

        # Shutdown executor
        self._executor.shutdown(wait=True)

        # Clear caches
        if self._perf_config.enable_caching:
            self._target_cache.clear()

        logger.info("Optimized fault orchestrator shutdown complete")
