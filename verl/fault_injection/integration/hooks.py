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

"""Integration hooks for fault injection in verl training."""

import logging
from functools import wraps
from typing import Any, Callable, Optional

from ..base import FaultContext, FaultLayer
from ..orchestrator import FaultOrchestrator

logger = logging.getLogger(__name__)


class FaultInjectionHooks:
    """Hooks for injecting faults at various points in verl training."""

    def __init__(self, orchestrator: FaultOrchestrator):
        self.orchestrator = orchestrator
        self._hooks_enabled = True
        self._hook_counters: dict[str, int] = {}

    def disable_hooks(self) -> None:
        """Disable fault injection hooks."""
        self._hooks_enabled = False
        logger.info("Fault injection hooks disabled")

    def enable_hooks(self) -> None:
        """Enable fault injection hooks."""
        self._hooks_enabled = True
        logger.info("Fault injection hooks enabled")

    def _increment_counter(self, hook_name: str) -> int:
        """Increment and return hook counter."""
        self._hook_counters[hook_name] = self._hook_counters.get(hook_name, 0) + 1
        return self._hook_counters[hook_name]

    def _try_inject_faults(self, layer: FaultLayer, hook_name: str, context: Optional[dict[str, Any]] = None) -> None:
        """Try to inject faults for a specific layer and hook."""
        if not self._hooks_enabled:
            return

        # Create fault context
        fault_context = FaultContext(
            layer=layer,
            metadata={"hook": hook_name, "counter": self._hook_counters.get(hook_name, 0), **(context or {})},
        )

        # Update orchestrator targets if needed
        # This would be done by Ray integration in real implementation

        # Try to inject enabled faults for this layer
        active_faults = self.orchestrator.get_active_faults()
        for fault_id, injector in active_faults.items():
            if injector.config.layer == layer:
                if injector.should_inject(fault_context):
                    logger.info(f"Injecting fault at {hook_name}: {fault_id}")
                    self.orchestrator.inject_fault(fault_id, fault_context)

    def hook_ui_initialization(self, func: Callable) -> Callable:
        """Hook for UI layer initialization (config parsing, etc)."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "ui_initialization"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.UI, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.UI, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.UI, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def hook_ray_init(self, func: Callable) -> Callable:
        """Hook for Ray initialization."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "ray_init"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.ORCHESTRATION, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.ORCHESTRATION, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(
                    FaultLayer.ORCHESTRATION, f"{hook_name}_post", {"success": False, "error": str(e)}
                )
                raise

        return wrapper

    def hook_worker_creation(self, func: Callable) -> Callable:
        """Hook for worker creation."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "worker_creation"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def hook_engine_init(self, func: Callable) -> Callable:
        """Hook for engine initialization."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "engine_init"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def hook_training_step(self, func: Callable) -> Callable:
        """Hook for training step execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "training_step"
            step = self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_pre", {"step": step})

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_post", {"step": step, "success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(
                    FaultLayer.WORKER, f"{hook_name}_post", {"step": step, "success": False, "error": str(e)}
                )
                raise

        return wrapper

    def hook_inference_step(self, func: Callable) -> Callable:
        """Hook for inference step execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "inference_step"
            step = self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.INFERENCE, f"{hook_name}_pre", {"step": step})

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.INFERENCE, f"{hook_name}_post", {"step": step, "success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(
                    FaultLayer.INFERENCE, f"{hook_name}_post", {"step": step, "success": False, "error": str(e)}
                )
                raise

        return wrapper

    def hook_checkpoint_save(self, func: Callable) -> Callable:
        """Hook for checkpoint saving."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "checkpoint_save"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def hook_checkpoint_load(self, func: Callable) -> Callable:
        """Hook for checkpoint loading."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "checkpoint_load"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.ENGINE, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def hook_gradient_sync(self, func: Callable) -> Callable:
        """Hook for gradient synchronization."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            hook_name = "gradient_sync"
            self._increment_counter(hook_name)

            # Pre-injection
            self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_pre")

            try:
                result = func(*args, **kwargs)
                # Post-injection
                self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_post", {"success": True})
                return result
            except Exception as e:
                # Post-injection on error
                self._try_inject_faults(FaultLayer.WORKER, f"{hook_name}_post", {"success": False, "error": str(e)})
                raise

        return wrapper

    def get_hook_stats(self) -> dict[str, int]:
        """Get statistics about hook invocations."""
        return self._hook_counters.copy()


# Convenience function to create hooks
def create_fault_injection_hooks(orchestrator: FaultOrchestrator) -> FaultInjectionHooks:
    """Create fault injection hooks."""
    return FaultInjectionHooks(orchestrator)
