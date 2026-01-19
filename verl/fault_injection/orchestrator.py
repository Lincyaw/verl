"""Fault injection orchestrator for managing faults across the system."""

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .base import (
    BaseFaultInjector,
    FaultContext,
    FaultInjectorRegistry,
    FaultResult,
    FaultStatus,
    FaultTargetConfig,
    create_target_selector,
)
from .config import FaultConfig, FaultInjectionConfig, FaultLayer

logger = logging.getLogger(__name__)


@dataclass
class FaultOrchestratorMetrics:
    """Metrics for fault injection orchestrator."""

    total_faults: int = 0
    active_faults: int = 0
    completed_faults: int = 0
    failed_faults: int = 0
    recovered_faults: int = 0
    injection_duration: Dict[str, float] = field(default_factory=dict)
    layer_stats: Dict[FaultLayer, int] = field(default_factory=lambda: defaultdict(int))


class FaultOrchestrator:
    """Orchestrator for managing fault injection across the verl system."""

    def __init__(self, config: FaultInjectionConfig):
        self.config = config
        self._injectors: Dict[str, BaseFaultInjector] = {}
        self._active_faults: Dict[str, BaseFaultInjector] = {}
        self._fault_history: List[FaultResult] = []
        self._metrics = FaultOrchestratorMetrics()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_faults)
        self._shutdown = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._available_targets: List[FaultContext] = []
        self._target_update_callbacks: List[callable] = []

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

    def register_target_update_callback(self, callback: callable) -> None:
        """Register a callback for target updates."""
        self._target_update_callbacks.append(callback)

    def update_available_targets(self, targets: List[FaultContext]) -> None:
        """Update the list of available fault targets."""
        with self._lock:
            self._available_targets = targets
            logger.debug(f"Updated available targets: {len(targets)} targets")

            # Notify callbacks
            for callback in self._target_update_callbacks:
                try:
                    callback(targets)
                except Exception as e:
                    logger.error(f"Target update callback failed: {e}")

    def add_fault(self, fault_config: FaultConfig) -> str:
        """Add a new fault to the orchestrator."""
        with self._lock:
            injector = FaultInjectorRegistry.create(fault_config)
            self._injectors[injector.fault_id] = injector
            logger.info(f"Added fault injector: {injector.fault_id}")
            return injector.fault_id

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

    def inject_fault(self, fault_id: str, target_context: Optional[FaultContext] = None) -> Optional[FaultResult]:
        """Inject a specific fault."""
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

        # Execute outside of lock
        try:
            result = injector.execute(target_context)

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
            return result

        except Exception as e:
            logger.error(f"Fault injection failed: {fault_id} - Error: {e}")
            with self._lock:
                self._metrics.active_faults -= 1
                if fault_id in self._active_faults:
                    del self._active_faults[fault_id]
            return None

    def inject_all_enabled(self) -> List[FaultResult]:
        """Inject all enabled faults."""
        results = []

        with self._lock:
            fault_ids = list(self._injectors.keys())

        for fault_id in fault_ids:
            if self._shutdown:
                break

            result = self.inject_fault(fault_id)
            if result:
                results.append(result)

        return results

    def recover_fault(self, fault_id: str) -> bool:
        """Recover from a specific fault."""
        with self._lock:
            if fault_id not in self._injectors:
                logger.error(f"Fault injector not found: {fault_id}")
                return False

            injector = self._injectors[fault_id]

            # Find a suitable target for recovery
            target_context = self._select_target_for_fault(injector.config)
            if target_context is None:
                logger.warning(f"No suitable target found for recovery: {fault_id}")
                return False

        # Execute recovery outside of lock
        try:
            injector.execute_recovery(target_context)

            with self._lock:
                self._metrics.recovered_faults += 1

            logger.info(f"Fault recovery completed: {fault_id}")
            return True

        except Exception as e:
            logger.error(f"Fault recovery failed: {fault_id} - Error: {e}")
            return False

    def recover_all(self) -> int:
        """Recover from all faults."""
        recovered = 0

        with self._lock:
            fault_ids = list(self._injectors.keys())

        for fault_id in fault_ids:
            if self.recover_fault(fault_id):
                recovered += 1

        return recovered

    def _select_target_for_fault(self, fault_config: FaultConfig) -> Optional[FaultContext]:
        """Select a target for fault injection."""
        if not self._available_targets:
            return None

        # Create target selector
        selector = create_target_selector(fault_config.target)

        # Filter targets by layer if specified
        candidates = self._available_targets
        if fault_config.layer:
            candidates = [t for t in candidates if t.layer == fault_config.layer]

        # Select targets
        selected = selector.select_targets(candidates)

        # Return first selected target
        return selected[0] if selected else None

    def get_metrics(self) -> FaultOrchestratorMetrics:
        """Get current metrics."""
        with self._lock:
            return self._metrics

    def get_fault_history(self, limit: Optional[int] = None) -> List[FaultResult]:
        """Get fault injection history."""
        with self._lock:
            if limit:
                return self._fault_history[-limit:]
            return self._fault_history.copy()

    def get_active_faults(self) -> Dict[str, BaseFaultInjector]:
        """Get currently active faults."""
        with self._lock:
            return self._active_faults.copy()

    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._monitor_thread is not None:
            return

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Fault injection monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._shutdown = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("Fault injection monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Fault injection monitor started")

        while not self._shutdown:
            try:
                # Check for faults that need recovery
                with self._lock:
                    current_time = time.time()

                    for fault_id, injector in list(self._active_faults.items()):
                        result = injector.get_result()
                        if result and result.status == FaultStatus.COMPLETED:
                            # Check if recovery is needed
                            if injector.config.duration_seconds:
                                if current_time - result.end_time >= injector.config.duration_seconds:
                                    logger.info(f"Auto-recovering fault: {fault_id}")
                                    self.recover_fault(fault_id)

                # Sleep for monitoring interval
                time.sleep(1)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down fault injection orchestrator")

        # Stop monitoring
        self.stop_monitoring()

        # Recover all faults
        self.recover_all()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("Fault injection orchestrator shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
