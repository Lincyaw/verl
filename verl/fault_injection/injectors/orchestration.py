"""Orchestration layer fault injectors for verl fault injection system."""

import os
import time
import signal
import socket
import threading
from typing import Any, Dict, List, Optional

import ray
from ray.util import placement_group
from ray.util.placement_group import PlacementGroup

from verl.fault_injection.base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from verl.fault_injection.config import FaultType, OrchestrationFaultConfig


@FaultInjectorRegistry.register(FaultType.RAY_CLUSTER_FAILURE)
class RayClusterFailureInjector(BaseFaultInjector):
    """Injector for Ray cluster failures."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config
        self._original_nodes = []
        self._failure_thread = None
        self._recovery_event = threading.Event()

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject Ray cluster failure."""
        failure_type = self.config.cluster_failure_type or "gcs"

        if failure_type == "gcs":
            # Simulate GCS failure by disconnecting from GCS
            try:
                # Get the GCS address
                gcs_address = ray._private.worker.global_worker.gcs_client.address

                # Create a thread that will simulate GCS disconnection
                def gcs_failure_simulator():
                    # Block GCS communication by modifying routing
                    original_timeout = os.environ.get("RAY_GCS_RPC_TIMEOUT_S", "60")
                    os.environ["RAY_GCS_RPC_TIMEOUT_S"] = "1"

                    # Wait for the specified duration
                    time.sleep(self.config.failure_duration)

                    # Restore original timeout
                    os.environ["RAY_GCS_RPC_TIMEOUT_S"] = original_timeout
                    self._recovery_event.set()

                self._failure_thread = threading.Thread(target=gcs_failure_simulator)
                self._failure_thread.start()

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.INJECTING,
                    start_time=time.time(),
                    metadata={"failure_type": failure_type, "gcs_address": gcs_address}
                )

            except Exception as e:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error=e,
                    metadata={"failure_type": failure_type}
                )

        elif failure_type == "node":
            # Simulate node failure
            try:
                # Get current node
                current_node = ray.util.get_node_ip_address()
                affected_nodes = self.config.affected_nodes or [current_node]

                # Simulate node death by creating a failing actor
                @ray.remote
                class NodeKiller:
                    def __init__(self, node_ip):
                        self.node_ip = node_ip

                    def kill(self):
                        # This will cause the actor to die
                        os._exit(1)

                # Place actors on specified nodes and kill them
                for node in affected_nodes:
                    killer = NodeKiller.options(
                        resources={f"node:{node}": 0.1}
                    ).remote()

                    # Schedule the kill after a delay
                    def delayed_kill():
                        time.sleep(self.config.failure_duration)
                        try:
                            ray.get(killer.kill.remote())
                        except:
                            pass

                    threading.Thread(target=delayed_kill).start()

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.INJECTING,
                    start_time=time.time(),
                    metadata={"failure_type": failure_type, "affected_nodes": affected_nodes}
                )

            except Exception as e:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error=e,
                    metadata={"failure_type": failure_type}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error=ValueError(f"Unknown cluster failure type: {failure_type}"),
            metadata={"failure_type": failure_type}
        )


@FaultInjectorRegistry.register(FaultType.ACTOR_CRASH)
@FaultInjectorRegistry.register(FaultType.ACTOR_DEATH)
class ActorCrashInjector(BaseFaultInjector):
    """Injector for Ray actor crashes and deaths."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config
        self._crashed_actors = []

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject actor crash/death."""
        actor_name = self.config.actor_name or "*"
        death_type = self.config.actor_death_type or "exception"

        try:
            # Get all actors
            actors = ray.util.list_named_actors()

            # Filter by name pattern
            target_actors = [a for a in actors if actor_name == "*" or actor_name in a["name"]]

            if not target_actors:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error=ValueError(f"No actors found matching pattern: {actor_name}"),
                    metadata={"actor_name": actor_name}
                )

            # Schedule crashes after delay
            def crash_actors():
                time.sleep(self.config.crash_delay)

                for actor_info in target_actors:
                    try:
                        # Get actor handle
                        actor = ray.get_actor(actor_info["name"])

                        if death_type == "exception":
                            # Cause exception in actor
                            actor.__ray_terminate__.remote()
                        elif death_type == "exit":
                            # Force exit
                            actor.__ray_kill__.remote()
                        elif death_type == "kill":
                            # Send kill signal
                            import signal
                            # This would need to be done from within the actor
                            pass

                        self._crashed_actors.append(actor_info["name"])

                    except Exception as e:
                        # Actor might already be dead
                        pass

            # Start crash thread
            crash_thread = threading.Thread(target=crash_actors)
            crash_thread.start()

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "actor_name": actor_name,
                    "death_type": death_type,
                    "target_count": len(target_actors),
                    "crash_delay": self.config.crash_delay
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"actor_name": actor_name}
            )


@FaultInjectorRegistry.register(FaultType.RESOURCE_EXHAUSTION)
class ResourceExhaustionInjector(BaseFaultInjector):
    """Injector for resource exhaustion faults."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config
        self._exhaustion_threads = []
        self._stop_exhaustion = threading.Event()

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject resource exhaustion."""
        resource_type = self.config.resource_type or "memory"

        try:
            if resource_type == "memory":
                # Exhaust memory
                amount = self.config.exhaustion_amount or 1024  # MB

                def exhaust_memory():
                    allocated = []
                    try:
                        # Allocate memory in chunks
                        chunk_size = 100  # MB
                        chunks = amount // chunk_size

                        for _ in range(chunks):
                            if self._stop_exhaustion.is_set():
                                break
                            # Allocate memory
                            chunk = bytearray(chunk_size * 1024 * 1024)
                            allocated.append(chunk)
                            time.sleep(0.1)

                        # Hold memory for the duration
                        time.sleep(self.config.exhaustion_duration)

                    finally:
                        # Release memory
                        allocated.clear()

                # Start memory exhaustion in thread
                thread = threading.Thread(target=exhaust_memory)
                thread.start()
                self._exhaustion_threads.append(thread)

            elif resource_type == "cpu":
                # Exhaust CPU
                def exhaust_cpu():
                    start_time = time.time()
                    while time.time() - start_time < self.config.exhaustion_duration:
                        if self._stop_exhaustion.is_set():
                            break
                        # Busy loop
                        _ = sum(i*i for i in range(1000000))

                # Start multiple CPU exhaustion threads
                for _ in range(4):  # Use 4 threads to load multiple cores
                    thread = threading.Thread(target=exhaust_cpu)
                    thread.start()
                    self._exhaustion_threads.append(thread)

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "resource_type": resource_type,
                    "exhaustion_amount": self.config.exhaustion_amount,
                    "duration": self.config.exhaustion_duration
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"resource_type": resource_type}
            )


@FaultInjectorRegistry.register(FaultType.TASK_SCHEDULING_FAILURE)
class TaskSchedulingFailureInjector(BaseFaultInjector):
    """Injector for task scheduling failures."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config
        self._original_task_options = {}

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject task scheduling failures."""
        task_pattern = self.config.task_name_pattern or "*"
        failure_rate = self.config.failure_rate or 1.0
        failure_type = self.config.failure_type or "exception"

        try:
            # Monkey patch ray.remote to inject failures
            original_remote = ray.remote

            def patched_remote(*args, **kwargs):
                def decorator(func_or_class):
                    # Get the original remote decorator
                    remote_func = original_remote(*args, **kwargs)(func_or_class)

                    # Check if this task matches our pattern
                    task_name = getattr(func_or_class, "__name__", str(func_or_class))
                    if task_pattern != "*" and task_pattern not in task_name:
                        return remote_func

                    # Wrap the remote function to inject failures
                    if hasattr(remote_func, "remote"):
                        original_remote_method = remote_func.remote

                        def failing_remote(*args, **kwargs):
                            import random
                            if random.random() < failure_rate:
                                if failure_type == "exception":
                                    # Return a failing object
                                    class FailingObject:
                                        def __init__(self):
                                            self._exception = Exception(f"Simulated task failure: {task_name}")

                                        def __getattr__(self, name):
                                            raise self._exception

                                    return FailingObject()
                                elif failure_type == "hang":
                                    # Return an object that hangs
                                    class HangingObject:
                                        def __getattr__(self, name):
                                            import time
                                            time.sleep(3600)  # Hang for an hour

                                    return HangingObject()
                                elif failure_type == "lost":
                                    # Return None to simulate lost task
                                    return None

                            # Normal execution
                            return original_remote_method(*args, **kwargs)

                        remote_func.remote = failing_remote

                    return remote_func

                return decorator

            # Apply the patch
            ray.remote = patched_remote

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "task_pattern": task_pattern,
                    "failure_rate": failure_rate,
                    "failure_type": failure_type
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"task_pattern": task_pattern}
            )


@FaultInjectorRegistry.register(FaultType.RESOURCE_POOL_FAULT)
class ResourcePoolFaultInjector(BaseFaultInjector):
    """Injector for resource pool faults."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject resource pool faults."""
        pool_name = self.config.pool_name or "default"
        operation = self.config.pool_operation or "acquire"

        try:
            # Create a blocking resource request
            if operation == "acquire":
                # Request all resources in the pool
                @ray.remote
                class ResourceHog:
                    def __init__(self):
                        self.held_resources = True

                    def hold(self):
                        # Just hold the resource
                        time.sleep(self.config.pool_block_duration)

                # Request resources with the pool name
                hog = ResourceHog.options(
                    resources={pool_name: 1000000}  # Request huge amount
                ).remote()

                # Start holding
                ray.get(hog.hold.remote())

            elif operation == "release":
                # Make resources immediately available but fail releases
                # This would require deeper integration with Ray's resource manager
                pass

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "pool_name": pool_name,
                    "operation": operation,
                    "block_duration": self.config.pool_block_duration
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"pool_name": pool_name}
            )


@FaultInjectorRegistry.register(FaultType.GCS_FAILURE)
class GCSFailureInjector(BaseFaultInjector):
    """Injector for GCS (Global Control Store) failures."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject GCS failure."""
        failure_type = self.config.gcs_failure_type or "disconnect"

        try:
            if failure_type == "disconnect":
                # Simulate GCS disconnection
                # This is done by creating a network partition to the GCS
                gcs_address = ray._private.worker.global_worker.gcs_client.address

                # Block GCS port
                gcs_host, gcs_port = gcs_address.split(":")
                gcs_port = int(gcs_port)

                # Create a socket that blocks the GCS port
                blocker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                blocker_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                try:
                    # Try to bind to the GCS port to block it
                    blocker_socket.bind((gcs_host, gcs_port))
                    blocker_socket.listen(1)

                    # Hold the connection for the specified duration
                    time.sleep(self.config.gcs_isolation_duration)

                except:
                    # Port might be in use, just wait
                    time.sleep(self.config.gcs_isolation_duration)
                finally:
                    blocker_socket.close()

            elif failure_type == "slow":
                # Make GCS operations slow
                original_timeout = os.environ.get("RAY_GCS_RPC_TIMEOUT_S", "60")
                os.environ["RAY_GCS_RPC_TIMEOUT_S"] = "300"  # 5 minutes

                # Wait for duration
                time.sleep(self.config.gcs_isolation_duration)

                # Restore
                os.environ["RAY_GCS_RPC_TIMEOUT_S"] = original_timeout

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "failure_type": failure_type,
                    "duration": self.config.gcs_isolation_duration
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"failure_type": failure_type}
            )


@FaultInjectorRegistry.register(FaultType.NETWORK_PARTITION)
class NetworkPartitionInjector(BaseFaultInjector):
    """Injector for network partition faults."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject network partition."""
        partition_type = self.config.partition_type or "partial"
        isolated_ranks = self.config.isolated_ranks or []

        try:
            # Get current worker info
            current_rank = context.rank or 0
            current_ip = context.host or ray.util.get_node_ip_address()

            # Check if this worker should be isolated
            should_isolate = (
                partition_type == "total" or
                current_rank in isolated_ranks or
                (partition_type == "partial" and current_rank % 2 == 0)
            )

            if should_isolate:
                # Create network isolation by blocking ports
                # Block common Ray ports
                ray_ports = [6379, 6380, 10001, 10002, 12345, 12346]

                blocking_sockets = []

                for port in ray_ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind(("0.0.0.0", port))
                        sock.listen(1)
                        blocking_sockets.append(sock)
                    except:
                        # Port might be in use
                        pass

                # Hold the block for the specified duration
                time.sleep(self.config.partition_duration)

                # Release the sockets
                for sock in blocking_sockets:
                    sock.close()

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "partition_type": partition_type,
                    "isolated_ranks": isolated_ranks,
                    "current_rank": current_rank,
                    "should_isolate": should_isolate,
                    "duration": self.config.partition_duration
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"partition_type": partition_type}
            )


@FaultInjectorRegistry.register(FaultType.PLACEMENT_GROUP_FAULT)
class PlacementGroupFaultInjector(BaseFaultInjector):
    """Injector for placement group faults."""

    def __init__(self, config: OrchestrationFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject placement group fault."""
        pg_name = self.config.placement_group_name or "fault_pg"
        fault_type = self.config.pg_fault_type or "creation_failure"

        try:
            if fault_type == "creation_failure":
                # Try to create a placement group with impossible requirements
                try:
                    pg = placement_group(
                        [{"GPU": 1000000, "CPU": 1000000}] * 100,  # Impossible requirements
                        name=pg_name,
                        strategy="STRICT_PACK"
                    )

                    # This should timeout or fail
                    ray.get(pg.ready(), timeout=10)

                except Exception as e:
                    # Expected to fail
                    return FaultResult(
                        fault_id=self.config.name,
                        status=FaultStatus.COMPLETED,
                        start_time=time.time(),
                        metadata={
                            "fault_type": fault_type,
                            "pg_name": pg_name,
                            "error": str(e)
                        }
                    )

            elif fault_type == "removal_failure":
                # Create a placement group and make it impossible to remove
                pg = placement_group(
                    [{"CPU": 1}] * 2,
                    name=pg_name,
                    strategy="PACK"
                )

                ray.get(pg.ready())

                # Create actors that hold the placement group
                @ray.remote
                class PGOccupier:
                    def __init__(self):
                        self.occupied = True

                    def occupy(self):
                        while True:
                            time.sleep(1)

                # Place actors in the placement group
                occupiers = []
                for i in range(2):
                    occupier = PGOccupier.options(
                        placement_group=pg,
                        placement_group_bundle_index=i
                    ).remote()
                    occupiers.append(occupier)

                # Now try to remove - this should fail
                try:
                    placement_group.remove_placement_group(pg)
                except:
                    pass  # Expected to fail

                # Clean up
                for occupier in occupiers:
                    ray.kill(occupier)

            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.INJECTING,
                start_time=time.time(),
                metadata={
                    "fault_type": fault_type,
                    "pg_name": pg_name
                }
            )

        except Exception as e:
            return FaultResult(
                fault_id=self.config.name,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=e,
                metadata={"fault_type": fault_type}
            )