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

"""Optimized integration hooks for fault injection with minimal overhead."""

import asyncio
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Optional

from ..base import FaultContext, FaultLayer
from ..performance import (
    get_performance_config,
    get_profiler,
    profile_method,
)

logger = logging.getLogger(__name__)

# Performance flags
HOOK_ENABLED = os.getenv("VERL_FAULT_INJECTION_HOOKS", "true").lower() == "true"
HOOK_ASYNC = os.getenv("VERL_FAULT_INJECTION_HOOKS_ASYNC", "true").lower() == "true"
HOOK_CACHE_RESULTS = os.getenv("VERL_FAULT_INJECTION_HOOKS_CACHE", "true").lower() == "true"
HOOK_BATCH_INJECTIONS = os.getenv("VERL_FAULT_INJECTION_HOOKS_BATCH", "true").lower() == "true"

# Performance thresholds
HOOK_MIN_INTERVAL_MS = float(os.getenv("VERL_FAULT_INJECTION_HOOKS_MIN_INTERVAL", "10.0"))
HOOK_MAX_BATCH_SIZE = int(os.getenv("VERL_FAULT_INJECTION_HOOKS_BATCH_SIZE", "5"))


class OptimizedFaultInjectionHooks:
    """Optimized hooks for fault injection with minimal performance overhead."""

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._hooks_enabled = HOOK_ENABLED
        self._hook_counters: dict[str, int] = {}
        self._hook_last_call: dict[str, float] = {}
        self._pending_injections: dict[str, list] = {}
        self._perf_config = get_performance_config()
        self._profiler = get_profiler()

    def disable_hooks(self) -> None:
        """Disable fault injection hooks."""
        self._hooks_enabled = False
        logger.info("Fault injection hooks disabled")

    def enable_hooks(self) -> None:
        """Enable fault injection hooks."""
        self._hooks_enabled = True
        logger.info("Fault injection hooks enabled")

    def _should_skip_hook(self, hook_name: str) -> bool:
        """Check if hook should be skipped based on performance settings."""
        if not self._hooks_enabled or not HOOK_ENABLED:
            return True

        # Check minimum interval
        if HOOK_MIN_INTERVAL_MS > 0:
            current_time = time.time() * 1000
            last_call = self._hook_last_call.get(hook_name, 0)
            if current_time - last_call < HOOK_MIN_INTERVAL_MS:
                return True
            self._hook_last_call[hook_name] = current_time

        return False

    def _increment_counter(self, hook_name: str) -> int:
        """Increment and return hook counter."""
        self._hook_counters[hook_name] = self._hook_counters.get(hook_name, 0) + 1
        return self._hook_counters[hook_name]

    @profile_method("_try_inject_faults_optimized")
    def _try_inject_faults(self, layer: FaultLayer, hook_name: str, context: Optional[dict[str, Any]] = None) -> None:
        """Try to inject faults for a specific layer and hook with optimization."""
        if self._should_skip_hook(hook_name):
            return

        # Create fault context
        fault_context = FaultContext(
            layer=layer, metadata={"hook": hook_name, "counter": self._increment_counter(hook_name), **(context or {})}
        )

        # Batch injections if enabled
        if HOOK_BATCH_INJECTIONS:
            self._batch_injection_request(layer, hook_name, fault_context)
        else:
            self._immediate_injection(layer, hook_name, fault_context)

    def _batch_injection_request(self, layer: FaultLayer, hook_name: str, context: FaultContext) -> None:
        """Queue injection for batch processing."""
        key = f"{layer.value}:{hook_name}"

        if key not in self._pending_injections:
            self._pending_injections[key] = []

        self._pending_injections[key].append(context)

        # Process batch if size reached
        if len(self._pending_injections[key]) >= HOOK_MAX_BATCH_SIZE:
            self._process_batch(key, layer, hook_name)

    def _process_batch(self, key: str, layer: FaultLayer, hook_name: str) -> None:
        """Process a batch of pending injections."""
        contexts = self._pending_injections.pop(key, [])

        # Run batch injection in background thread
        if HOOK_ASYNC and hasattr(self.orchestrator, "inject_fault_async"):
            asyncio.create_task(self._async_batch_injection(layer, contexts))
        else:
            # Use thread pool for sync batch processing
            import threading

            threading.Thread(target=self._sync_batch_injection, args=(layer, contexts), daemon=True).start()

    async def _async_batch_injection(self, layer: FaultLayer, contexts: list) -> None:
        """Async batch fault injection."""
        try:
            # Get enabled faults for layer
            fault_ids = self._get_enabled_faults_for_layer(layer)

            # Inject each fault for each context
            tasks = []
            for fault_id in fault_ids:
                for context in contexts:
                    if hasattr(self.orchestrator, "inject_fault_async"):
                        task = self.orchestrator.inject_fault_async(fault_id, context)
                    else:
                        # Fallback to sync in thread pool
                        loop = asyncio.get_event_loop()
                        task = loop.run_in_executor(None, self.orchestrator.inject_fault, fault_id, context)
                    tasks.append(task)

            # Wait for all injections
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Async batch injection failed: {e}")

    def _sync_batch_injection(self, layer: FaultLayer, contexts: list) -> None:
        """Sync batch fault injection."""
        try:
            fault_ids = self._get_enabled_faults_for_layer(layer)

            for fault_id in fault_ids:
                for context in contexts:
                    try:
                        self.orchestrator.inject_fault(fault_id, context)
                    except Exception as e:
                        logger.error(f"Batch injection failed for {fault_id}: {e}")

        except Exception as e:
            logger.error(f"Sync batch injection failed: {e}")

    def _immediate_injection(self, layer: FaultLayer, hook_name: str, context: FaultContext) -> None:
        """Immediate fault injection."""
        try:
            fault_ids = self._get_enabled_faults_for_layer(layer)

            # Run injections
            if HOOK_ASYNC and hasattr(self.orchestrator, "inject_fault_async"):
                # Schedule async injections
                for fault_id in fault_ids:
                    asyncio.create_task(self.orchestrator.inject_fault_async(fault_id, context))
            else:
                # Sync injection
                for fault_id in fault_ids:
                    self.orchestrator.inject_fault(fault_id, context)

        except Exception as e:
            logger.error(f"Fault injection failed for {layer.value}/{hook_name}: {e}")

    def _get_enabled_faults_for_layer(self, layer: FaultLayer) -> list:
        """Get enabled fault IDs for a specific layer."""
        # Cache results if enabled
        if HOOK_CACHE_RESULTS:
            cache_key = f"enabled_faults:{layer.value}"
            if hasattr(self, "_enabled_faults_cache"):
                cached = self._enabled_faults_cache.get(cache_key)
                if cached and time.time() - cached[1] < 60:  # 1 minute TTL
                    return cached[0]

            # Build cache
            if not hasattr(self, "_enabled_faults_cache"):
                self._enabled_faults_cache = {}

            fault_ids = [
                fid
                for fid, injector in self.orchestrator._injectors.items()
                if injector.config.layer == layer and injector.config.enabled
            ]

            self._enabled_faults_cache[cache_key] = (fault_ids, time.time())
            return fault_ids

        # No cache, query directly
        return [
            fid
            for fid, injector in self.orchestrator._injectors.items()
            if injector.config.layer == layer and injector.config.enabled
        ]

    def process_pending_batches(self) -> None:
        """Process any remaining pending batches."""
        for key, contexts in list(self._pending_injections.items()):
            if contexts:
                layer_str, hook_name = key.split(":", 1)
                layer = FaultLayer(layer_str)
                self._process_batch(key, layer, hook_name)


# Optimized hook decorators
def optimized_fault_hook(layer: FaultLayer, hook_name: str, context_extractor: Optional[Callable] = None):
    """Optimized decorator for fault injection hooks."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Quick check to avoid overhead
            if not HOOK_ENABLED:
                return func(*args, **kwargs)

            # Extract context if provided
            context = None
            if context_extractor:
                try:
                    context = context_extractor(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Context extraction failed: {e}")

            # Get hooks instance from args
            hooks = None
            for arg in args:
                if hasattr(arg, "_hooks_enabled"):
                    hooks = arg
                    break

            if hooks and hasattr(hooks, "_try_inject_faults"):
                # Use optimized injection
                hooks._try_inject_faults(layer, hook_name, context)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-compiled hook functions for common operations
class PrecompiledHooks:
    """Pre-compiled hook functions for minimal overhead."""

    @staticmethod
    def rollout_generation_hook(hooks: OptimizedFaultInjectionHooks, context: Optional[dict] = None) -> None:
        """Pre-compiled rollout generation hook."""
        if hooks._should_skip_hook("rollout_generation"):
            return
        hooks._immediate_injection(
            FaultLayer.WORKER,
            "rollout_generation",
            FaultContext(layer=FaultLayer.WORKER, metadata={"hook": "rollout_generation", **(context or {})}),
        )

    @staticmethod
    def actor_update_hook(hooks: OptimizedFaultInjectionHooks, context: Optional[dict] = None) -> None:
        """Pre-compiled actor update hook."""
        if hooks._should_skip_hook("actor_update"):
            return
        hooks._immediate_injection(
            FaultLayer.WORKER,
            "actor_update",
            FaultContext(layer=FaultLayer.WORKER, metadata={"hook": "actor_update", **(context or {})}),
        )

    @staticmethod
    def critic_update_hook(hooks: OptimizedFaultInjectionHooks, context: Optional[dict] = None) -> None:
        """Pre-compiled critic update hook."""
        if hooks._should_skip_hook("critic_update"):
            return
        hooks._immediate_injection(
            FaultLayer.WORKER,
            "critic_update",
            FaultContext(layer=FaultLayer.WORKER, metadata={"hook": "critic_update", **(context or {})}),
        )
