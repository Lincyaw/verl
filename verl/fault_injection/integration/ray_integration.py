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

"""Ray integration for fault injection in verl."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import ray

from ..base import FaultContext, FaultLayer
from ..orchestrator import FaultOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class RayWorkerInfo:
    """Information about a Ray worker."""

    worker_id: str
    node_id: str
    node_ip: str
    pid: int
    actor_id: Optional[str] = None
    process_type: Optional[str] = None


class RayFaultInjectionManager:
    """Manages fault injection for Ray workers in verl."""

    def __init__(self, orchestrator: FaultOrchestrator):
        self.orchestrator = orchestrator
        self._worker_info: dict[str, RayWorkerInfo] = {}
        self._registered = False

    def register_with_ray(self) -> None:
        """Register fault injection hooks with Ray."""
        if self._registered:
            return

        try:
            # Register worker info collector
            ray.worker.global_worker.run_function_on_all_workers(self._collect_worker_info)
            self._registered = True
            logger.info("Ray fault injection hooks registered")
        except Exception as e:
            logger.error(f"Failed to register Ray hooks: {e}")

    def _collect_worker_info(self, worker_info: Optional[dict] = None) -> None:
        """Collect information about Ray workers."""
        try:
            worker = ray.worker.global_worker
            node_info = ray._private.services.get_node_ip_address()

            info = RayWorkerInfo(
                worker_id=worker.worker_id,
                node_id=worker.current_node_id,
                node_ip=node_info,
                pid=os.getpid(),
                actor_id=getattr(worker, "actor_id", None),
            )

            # Determine process type from environment or actor class
            info.process_type = self._determine_process_type()

            self._worker_info[info.worker_id] = info

            # Update orchestrator with new target
            self._create_fault_context(info)
            self._update_orchestrator_targets()

        except Exception as e:
            logger.error(f"Failed to collect worker info: {e}")

    def _determine_process_type(self) -> Optional[str]:
        """Determine the type of process (actor, critic, rollout, etc)."""
        # Check environment variables
        process_type = os.environ.get("VERL_PROCESS_TYPE")
        if process_type:
            return process_type

        # Check actor class name
        try:
            worker = ray.worker.global_worker
            if hasattr(worker, "actor_class"):
                class_name = worker.actor_class.__name__
                if "Actor" in class_name:
                    return "actor"
                elif "Critic" in class_name:
                    return "critic"
                elif "Reward" in class_name:
                    return "reward"
                elif "Rollout" in class_name:
                    return "rollout"
        except Exception:
            pass

        return "unknown"

    def _create_fault_context(self, worker_info: RayWorkerInfo) -> FaultContext:
        """Create fault context from worker info."""
        # Determine layer based on process type
        layer = self._determine_layer(worker_info.process_type)

        return FaultContext(
            worker_id=worker_info.worker_id,
            rank=self._get_rank_from_worker(worker_info),
            host=worker_info.node_ip,
            process_type=worker_info.process_type,
            layer=layer,
            metadata={"pid": worker_info.pid, "node_id": worker_info.node_id, "actor_id": worker_info.actor_id},
        )

    def _determine_layer(self, process_type: Optional[str]) -> FaultLayer:
        """Determine fault layer based on process type."""
        if not process_type:
            return FaultLayer.WORKER

        if process_type in ["actor", "critic"]:
            return FaultLayer.WORKER
        elif process_type == "rollout":
            return FaultLayer.INFERENCE
        else:
            return FaultLayer.WORKER

    def _get_rank_from_worker(self, worker_info: RayWorkerInfo) -> Optional[int]:
        """Get rank from worker info."""
        # Try to get rank from environment
        rank = os.environ.get("RANK")
        if rank:
            try:
                return int(rank)
            except ValueError:
                pass

        # Try to infer from actor ID or worker ID
        # This is a simplified implementation
        try:
            if worker_info.actor_id:
                # Extract rank from actor ID if possible
                return hash(worker_info.actor_id) % 1000
        except Exception:
            pass

        return None

    def _update_orchestrator_targets(self) -> None:
        """Update orchestrator with current worker targets."""
        contexts = []
        for info in self._worker_info.values():
            context = self._create_fault_context(info)
            contexts.append(context)

        self.orchestrator.update_available_targets(contexts)
        logger.debug(f"Updated orchestrator with {len(contexts)} targets")

    def inject_fault_on_worker(self, worker_id: str, fault_id: str) -> bool:
        """Inject a fault on a specific worker."""
        try:
            worker_info = self._worker_info.get(worker_id)
            if not worker_info:
                logger.error(f"Worker not found: {worker_id}")
                return False

            context = self._create_fault_context(worker_info)
            result = self.orchestrator.inject_fault(fault_id, context)

            return result is not None and result.status.value == "completed"

        except Exception as e:
            logger.error(f"Failed to inject fault on worker {worker_id}: {e}")
            return False

    def inject_fault_on_rank(self, rank: int, fault_id: str) -> bool:
        """Inject a fault on a specific rank."""
        # Find worker with matching rank
        for worker_info in self._worker_info.values():
            worker_rank = self._get_rank_from_worker(worker_info)
            if worker_rank == rank:
                return self.inject_fault_on_worker(worker_info.worker_id, fault_id)

        logger.error(f"No worker found for rank {rank}")
        return False

    def inject_fault_on_process_type(self, process_type: str, fault_id: str) -> list[bool]:
        """Inject a fault on all workers of a specific process type."""
        results = []
        for worker_info in self._worker_info.values():
            if worker_info.process_type == process_type:
                result = self.inject_fault_on_worker(worker_info.worker_id, fault_id)
                results.append(result)

        if not results:
            logger.warning(f"No workers found for process type: {process_type}")

        return results

    def get_worker_summary(self) -> dict[str, Any]:
        """Get summary of all workers."""
        summary = {"total_workers": len(self._worker_info), "workers_by_type": {}, "workers_by_host": {}, "workers": []}

        for info in self._worker_info.values():
            # Count by type
            ptype = info.process_type or "unknown"
            summary["workers_by_type"][ptype] = summary["workers_by_type"].get(ptype, 0) + 1

            # Count by host
            host = info.node_ip
            summary["workers_by_host"][host] = summary["workers_by_host"].get(host, 0) + 1

            # Add worker details
            summary["workers"].append(
                {
                    "worker_id": info.worker_id,
                    "process_type": ptype,
                    "host": host,
                    "pid": info.pid,
                    "rank": self._get_rank_from_worker(info),
                }
            )

        return summary


# Global instance
_ray_fault_manager: Optional[RayFaultInjectionManager] = None


def initialize_ray_fault_injection(orchestrator: FaultOrchestrator) -> RayFaultInjectionManager:
    """Initialize Ray fault injection."""
    global _ray_fault_manager

    if _ray_fault_manager is None:
        _ray_fault_manager = RayFaultInjectionManager(orchestrator)
        _ray_fault_manager.register_with_ray()

    return _ray_fault_manager


def get_ray_fault_manager() -> Optional[RayFaultInjectionManager]:
    """Get the global Ray fault injection manager."""
    return _ray_fault_manager
