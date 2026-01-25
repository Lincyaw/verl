"""
L0 Ray proxies for Ralph fault injection framework.

Contains proxy classes for Ray-level operations like ray.get, ray.put, etc.
"""

import random
from typing import Any, Optional

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy

# Attempt to import ray exceptions, falling back to custom exceptions if not available
try:
    import ray
    import ray.exceptions

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class ObjectLostError(Exception):
    """
    Fallback exception for when Ray is not installed.

    Mimics ray.exceptions.ObjectLostError for testing purposes.
    """

    def __init__(
        self,
        object_ref: Any = None,
        owner_address: str = "",
        call_site: str = "",
    ):
        self.object_ref = object_ref
        self.owner_address = owner_address
        self.call_site = call_site
        super().__init__(f"Object lost: ref={object_ref}, owner={owner_address}, call_site={call_site}")


@ProxyRegistry.register("ray.get")
class RayGetProxy(BaseProxy, DelayMixin, ExceptionMixin, TensorCorruptionMixin):
    """
    Proxy for ray.get() operations.

    Supports fault injection at the Ray object store level (L0).

    Supported strategies:
    - DELAY: Adds delay before calling original ray.get
    - RAISE_EXCEPTION: Raises a configurable exception instead of calling ray.get
    - CORRUPT_TENSOR: Corrupts tensor data in ray.get results with Gaussian noise
    - OBJECT_LOST: Raises ObjectLostError simulating lost objects
    - PARTIAL_FAILURE: Sets a ratio of results to None simulating partial failures

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - RAISE_EXCEPTION: exc_type (str, required), message (str, optional)
    - CORRUPT_TENSOR: noise_scale (float, default 0.1)
    - OBJECT_LOST: No specific parameters
    - PARTIAL_FAILURE: fail_ratio (float, default 0.5) - ratio of results to set to None
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.OBJECT_LOST,
        StrategyType.PARTIAL_FAILURE,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L0"

    def _strategy_object_lost(
        self,
        object_refs: Any | list[Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate ray ObjectLostError.

        Raises ObjectLostError as if the object was lost from the object store.
        The original function is never called.

        Args:
            object_refs: Single object reference or list of references
            *args: Additional positional arguments (ignored)
            timeout: Timeout parameter (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            ray.exceptions.ObjectLostError: If Ray is installed
            ObjectLostError: If Ray is not installed (fallback)
        """
        # Get the first reference to report in the error
        ref = object_refs[0] if isinstance(object_refs, list) else object_refs

        if HAS_RAY:
            raise ray.exceptions.ObjectLostError(
                object_ref=ref,
                owner_address="ralph_simulated",
                call_site="ralph_fault_injection",
            )
        else:
            raise ObjectLostError(
                object_ref=ref,
                owner_address="ralph_simulated",
                call_site="ralph_fault_injection",
            )

    def _strategy_partial_failure(
        self,
        object_refs: Any | list[Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate partial failure where some results are None.

        Calls the original ray.get, then sets a ratio of the results to None
        to simulate partial failures in the object store.

        Args:
            object_refs: Single object reference or list of references
            *args: Additional positional arguments to pass to original
            timeout: Timeout parameter to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Results with fail_ratio of them set to None.
            For single reference, may return None or the actual result.
            For list of references, returns list with some elements as None.

        Config parameters:
            fail_ratio (float): Ratio of results to set to None (0.0 to 1.0, default 0.5)
        """
        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout

        # Call original to get results
        results = self._original(object_refs, *args, **call_kwargs)

        # Get fail ratio from config
        fail_ratio = self._config.parameters.get("fail_ratio", 0.5)

        # Handle single reference case
        if not isinstance(object_refs, list):
            if random.random() < fail_ratio:
                return None
            return results

        # Handle list of references - results should be a list
        if not isinstance(results, list):
            # If results is not a list, wrap and unwrap
            results = [results]
            for i in range(len(results)):
                if random.random() < fail_ratio:
                    results[i] = None
            return results[0]

        # Process list results
        for i in range(len(results)):
            if random.random() < fail_ratio:
                results[i] = None

        return results


class ObjectStoreFullError(Exception):
    """
    Fallback exception for when Ray is not installed.

    Mimics ray.exceptions.ObjectStoreFullError for testing purposes.
    """

    def __init__(self, message: str = "Object store is full"):
        self.message = message
        super().__init__(message)


class DummyObjectRef:
    """
    A dummy ObjectRef for simulating silent drops.

    This class mimics a Ray ObjectRef but contains no actual data.
    When ray.get() is called on this, it will fail or return unexpected results.
    """

    def __init__(self, object_id: str = "dummy"):
        self._object_id = object_id

    def __repr__(self) -> str:
        return f"DummyObjectRef({self._object_id})"

    def __hash__(self) -> int:
        return hash(self._object_id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DummyObjectRef):
            return self._object_id == other._object_id
        return False


class WorkerDeathError(Exception):
    """
    Exception raised when a worker death is simulated.

    This exception simulates a Ray actor death scenario where a worker
    is killed during execution.
    """

    def __init__(
        self,
        worker_index: int,
        message: str = "Worker killed by fault injection",
    ):
        self.worker_index = worker_index
        self.message = message
        super().__init__(f"Worker {worker_index} death: {message}")


@ProxyRegistry.register("ray.put")
class RayPutProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ExceptionMixin):
    """
    Proxy for ray.put() operations.

    Supports fault injection at the Ray object store level (L0).

    Supported strategies:
    - DELAY: Adds delay before calling original ray.put
    - CORRUPT_TENSOR: Corrupts tensor data before storing
    - STORE_FULL: Raises ObjectStoreFullError simulating storage full
    - SILENT_DROP: Returns dummy ObjectRef without storing data

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - CORRUPT_TENSOR: noise_scale (float, default 0.1)
    - STORE_FULL: message (str, optional) - custom error message
    - SILENT_DROP: object_id (str, optional) - custom object ID for dummy ref
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.STORE_FULL,
        StrategyType.SILENT_DROP,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L0"

    def _strategy_corrupt_tensor(
        self,
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt tensor data before storing.

        Modifies tensor values before putting them in the object store.
        This differs from RayGetProxy's corruption which corrupts on retrieval.

        Args:
            value: The value to store (tensor or container with tensors)
            *args: Additional positional arguments to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            ObjectRef pointing to corrupted data
        """
        noise_scale = self._config.parameters.get("noise_scale", 0.1)
        corrupted_value = self._corrupt_result_tensors(value, noise_scale)
        return self._original(corrupted_value, *args, **kwargs)

    def _strategy_store_full(
        self,
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate ray ObjectStoreFullError.

        Raises ObjectStoreFullError as if the object store ran out of space.
        The original function is never called.

        Args:
            value: The value to store (ignored)
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            ray.exceptions.ObjectStoreFullError: If Ray is installed
            ObjectStoreFullError: If Ray is not installed (fallback)
        """
        # Get custom message from config if provided
        message = self._config.parameters.get("message", "Object store full: cannot store object")

        if HAS_RAY:
            # Ray's ObjectStoreFullError may have different signature
            try:
                raise ray.exceptions.ObjectStoreFullError(message)
            except (AttributeError, TypeError) as e:
                # Fallback if the exception class has different signature
                raise ObjectStoreFullError(message) from e
        else:
            raise ObjectStoreFullError(message)

    def _strategy_silent_drop(
        self,
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Silently drop the object and return a dummy ObjectRef.

        Instead of storing the actual object, returns a dummy reference
        that will fail when ray.get() is called on it. This simulates
        a silent data loss scenario.

        Args:
            value: The value to store (ignored, not actually stored)
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            DummyObjectRef: A fake ObjectRef that contains no data
        """
        # Get custom object_id from config if provided
        object_id = self._config.parameters.get("object_id", f"dropped_{id(value)}")
        return DummyObjectRef(object_id)


@ProxyRegistry.register("execute_all_sync")
class ExecuteAllProxy(BaseProxy, DelayMixin):
    """
    Proxy for execute_all_sync() operations.

    Supports fault injection at the Ray worker execution level (L0).
    This proxy intercepts batch execution calls that run a method across
    all workers in a worker group.

    Supported strategies:
    - WORKER_DEATH: Simulates killing a specific worker using ray.kill
    - STRAGGLER: Adds delay to a specific worker's execution
    - SKIP_WORKER: Skips execution on a specific worker (removes from result)
    - DUPLICATE_CALL: Duplicates the call to a specific worker

    Config parameters:
    - WORKER_DEATH: kill_worker_idx (int, required) - index of worker to kill
    - STRAGGLER: straggler_idx (int, required) - index of worker to delay
                 delay_seconds (float, default 60.0) - delay amount
    - SKIP_WORKER: skip_idx (int, required) - index of worker to skip
    - DUPLICATE_CALL: duplicate_idx (int, required) - index of worker to duplicate
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.WORKER_DEATH,
        StrategyType.STRAGGLER,
        StrategyType.SKIP_WORKER,
        StrategyType.DUPLICATE_CALL,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L0"

    def _strategy_worker_death(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate worker death by killing a specific worker.

        Uses ray.kill() to terminate the worker at the specified index.
        If Ray is not available, raises WorkerDeathError instead.

        Args:
            method_name: The method name to execute on workers
            *args: Additional positional arguments to pass to execute_all_sync
            **kwargs: Additional keyword arguments to pass to execute_all_sync

        Raises:
            ValueError: If kill_worker_idx is not specified in config
            WorkerDeathError: If Ray is not available (fallback)
            ray.exceptions.RayActorError: After killing the worker (if Ray available)

        Config parameters:
            kill_worker_idx (int): Index of the worker to kill (required)
        """
        kill_worker_idx = self._config.parameters.get("kill_worker_idx")
        if kill_worker_idx is None:
            raise ValueError("worker_death strategy requires 'kill_worker_idx' parameter")

        # Get the workers attribute from the object this method is bound to
        # In verl, execute_all_sync is typically a method on a WorkerGroup
        # that has a 'workers' attribute containing the Ray actors
        workers = getattr(self, "_workers", None)

        if workers is not None and HAS_RAY:
            if 0 <= kill_worker_idx < len(workers):
                worker = workers[kill_worker_idx]
                try:
                    ray.kill(worker)
                except Exception:
                    # If kill fails, continue anyway - the fault was attempted
                    pass

        # For testing without actual workers or Ray, raise a descriptive error
        if not HAS_RAY:
            raise WorkerDeathError(
                worker_index=kill_worker_idx,
                message="Worker killed by fault injection (Ray not available)",
            )

        # Call original - this should now fail because the worker is dead
        return self._original(method_name, *args, **kwargs)

    def _strategy_straggler(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate a straggler worker by adding delay.

        Adds artificial delay before executing the method, simulating
        a slow worker that holds up the entire batch operation.

        Note: In a real implementation, this would need to intercept
        the individual worker calls. For now, this delays the entire
        execute_all_sync call to simulate the observable effect.

        Args:
            method_name: The method name to execute on workers
            *args: Additional positional arguments to pass to execute_all_sync
            **kwargs: Additional keyword arguments to pass to execute_all_sync

        Returns:
            Results from the original execute_all_sync call

        Config parameters:
            straggler_idx (int): Index of the worker to slow down (required for validation)
            delay_seconds (float): Delay in seconds (default: 60.0)
        """
        import time

        straggler_idx = self._config.parameters.get("straggler_idx")
        if straggler_idx is None:
            raise ValueError("straggler strategy requires 'straggler_idx' parameter")

        delay_seconds = self._config.parameters.get("delay_seconds", 60.0)

        # Apply delay to simulate straggler effect
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        return self._original(method_name, *args, **kwargs)

    def _strategy_skip_worker(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Skip execution on a specific worker.

        Calls the original execute_all_sync, then removes the result
        at the specified index from the results list. This simulates
        a worker that silently failed to execute or return its result.

        Args:
            method_name: The method name to execute on workers
            *args: Additional positional arguments to pass to execute_all_sync
            **kwargs: Additional keyword arguments to pass to execute_all_sync

        Returns:
            Results list with the skipped worker's result removed

        Config parameters:
            skip_idx (int): Index of the worker to skip (required)
        """
        skip_idx = self._config.parameters.get("skip_idx")
        if skip_idx is None:
            raise ValueError("skip_worker strategy requires 'skip_idx' parameter")

        # Call original to get all results
        results = self._original(method_name, *args, **kwargs)

        # Remove the result at the specified index
        if isinstance(results, list):
            if 0 <= skip_idx < len(results):
                results = results[:skip_idx] + results[skip_idx + 1 :]

        return results

    def _strategy_duplicate_call(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Duplicate the call to a specific worker.

        Calls the original execute_all_sync, then duplicates the result
        at the specified index. This simulates a scenario where a worker's
        method is called twice, potentially causing state inconsistencies.

        Args:
            method_name: The method name to execute on workers
            *args: Additional positional arguments to pass to execute_all_sync
            **kwargs: Additional keyword arguments to pass to execute_all_sync

        Returns:
            Results list with the specified worker's result duplicated

        Config parameters:
            duplicate_idx (int): Index of the worker to duplicate (required)
        """
        duplicate_idx = self._config.parameters.get("duplicate_idx")
        if duplicate_idx is None:
            raise ValueError("duplicate_call strategy requires 'duplicate_idx' parameter")

        # Call original to get all results
        results = self._original(method_name, *args, **kwargs)

        # Duplicate the result at the specified index
        if isinstance(results, list):
            if 0 <= duplicate_idx < len(results):
                duplicated_result = results[duplicate_idx]
                results = results[: duplicate_idx + 1] + [duplicated_result] + results[duplicate_idx + 1 :]

        return results
