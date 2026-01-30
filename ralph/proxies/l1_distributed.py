"""
L1 Distributed proxies for Ralph fault injection framework.

Contains proxy classes for torch.distributed operations like all_reduce, all_gather, barrier, etc.
"""

import os
import time
from typing import Any

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy

# Constants for deadlock simulation
_DEADLOCK_CHECK_INTERVAL = 10  # seconds
_DEADLOCK_STOP_FILE_PREFIX = "/tmp/ralph_stop_deadlock_"

# Attempt to import torch.distributed for rank detection
try:
    import torch.distributed as dist

    HAS_DIST = True
except ImportError:
    HAS_DIST = False


@ProxyRegistry.register("torch.distributed.all_reduce")
class AllReduceProxy(BaseProxy, DelayMixin, TensorCorruptionMixin):
    """
    Proxy for torch.distributed.all_reduce() operations.

    Supports fault injection at the distributed communication level (L1).

    Supported strategies:
    - DELAY: Adds delay before calling original all_reduce
    - CORRUPT_TENSOR: Corrupts tensor data before all_reduce with Gaussian noise
    - INJECT_NAN: Injects NaN values into the tensor before all_reduce
    - INJECT_INF: Injects Inf values into the tensor before all_reduce
    - DEADLOCK: Causes specified rank to hang indefinitely, simulating a deadlock

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - CORRUPT_TENSOR: noise_scale (float, default 0.1)
    - INJECT_NAN: nan_ratio (float, default 0.001) - ratio of elements to set to NaN
    - INJECT_INF: inf_ratio (float, default 0.001), positive (bool, default True)
    - DEADLOCK: deadlock_rank (int, required) - the rank to hang
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.INJECT_NAN,
        StrategyType.INJECT_INF,
        StrategyType.DEADLOCK,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L1"

    def _get_current_rank(self) -> int:
        """
        Get the current process rank.

        Returns:
            The current rank if distributed is initialized, otherwise 0.
        """
        if HAS_DIST and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _strategy_corrupt_tensor(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt tensor before all_reduce.

        Adds Gaussian noise to the tensor before performing the all_reduce operation.
        This simulates data corruption in the communication layer.

        NOTE: This strategy modifies the input tensor IN-PLACE, which is consistent
        with all_reduce's expected behavior where the input tensor is both the
        source and destination. The tensor is corrupted before the collective
        operation, so all ranks will see the corruption propagate through the
        reduction.

        Args:
            tensor: The tensor to be reduced (modified in-place).
            op: The reduction operation (e.g., ReduceOp.SUM).
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_reduce with corrupted input tensor.

        Config parameters:
            noise_scale (float): Standard deviation of Gaussian noise (default 0.1).
        """
        noise_scale = self._config.parameters.get("noise_scale", 0.1)

        # Corrupt the tensor in-place before all_reduce
        # Note: In-place modification is intentional and matches all_reduce semantics
        corrupted = self._corrupt_tensor(tensor, noise_scale)
        tensor.copy_(corrupted)

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if op is not None:
            call_kwargs["op"] = op
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        return self._original(tensor, **call_kwargs)

    def _strategy_inject_nan(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Inject NaN values into tensor before all_reduce.

        Sets a ratio of tensor elements to NaN before performing all_reduce.
        This simulates numerical instability in the communication layer.

        NOTE: This strategy modifies the input tensor IN-PLACE, which is consistent
        with all_reduce's expected behavior where the input tensor is both the
        source and destination.

        Args:
            tensor: The tensor to be reduced (modified in-place).
            op: The reduction operation (e.g., ReduceOp.SUM).
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_reduce with NaN-injected input tensor.

        Config parameters:
            nan_ratio (float): Fraction of elements to set to NaN (default 0.001).
        """
        nan_ratio = self._config.parameters.get("nan_ratio", 0.001)

        # Inject NaN into the tensor in-place before all_reduce
        # Note: In-place modification is intentional and matches all_reduce semantics
        nan_tensor = self._inject_nan(tensor, nan_ratio)
        tensor.copy_(nan_tensor)

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if op is not None:
            call_kwargs["op"] = op
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        return self._original(tensor, **call_kwargs)

    def _strategy_inject_inf(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Inject Inf values into tensor before all_reduce.

        Sets a ratio of tensor elements to +/-Inf before performing all_reduce.
        This simulates numerical overflow in the communication layer.

        NOTE: This strategy modifies the input tensor IN-PLACE, which is consistent
        with all_reduce's expected behavior where the input tensor is both the
        source and destination.

        Args:
            tensor: The tensor to be reduced (modified in-place).
            op: The reduction operation (e.g., ReduceOp.SUM).
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_reduce with Inf-injected input tensor.

        Config parameters:
            inf_ratio (float): Fraction of elements to set to Inf (default 0.001).
            positive (bool): If True, inject +Inf; if False, inject -Inf (default True).
        """
        inf_ratio = self._config.parameters.get("inf_ratio", 0.001)
        positive = self._config.parameters.get("positive", True)

        # Inject Inf into the tensor in-place before all_reduce
        # Note: In-place modification is intentional and matches all_reduce semantics
        inf_tensor = self._inject_inf(tensor, inf_ratio, positive)
        tensor.copy_(inf_tensor)

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if op is not None:
            call_kwargs["op"] = op
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        return self._original(tensor, **call_kwargs)

    def _strategy_deadlock(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Cause specified rank to hang, simulating a deadlock.

        The specified rank will enter a sleep loop, while other ranks
        will proceed normally. This creates a deadlock situation where other ranks
        wait for the hanging rank during collective operations.

        The deadlock can be terminated by:
        1. Exceeding max_deadlock_time (default: 3600s)
        2. Creating a stop signal file: /tmp/ralph_stop_deadlock_{pid}

        Args:
            tensor: The tensor to be reduced.
            op: The reduction operation (e.g., ReduceOp.SUM).
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Never returns for the deadlock rank (raises TimeoutError/InterruptedError).
            Result of original all_reduce for other ranks.

        Config parameters:
            deadlock_rank (int): The rank that should hang (required).
            max_deadlock_time (float): Maximum time to simulate deadlock in seconds
                                      (default: 3600). Set to 0 for infinite.

        Raises:
            ValueError: If deadlock_rank is not specified in config parameters.
            TimeoutError: When max_deadlock_time is exceeded.
            InterruptedError: When stop signal file is detected.
        """
        deadlock_rank = self._config.parameters.get("deadlock_rank")
        max_deadlock_time = self._config.parameters.get("max_deadlock_time", 3600)

        if deadlock_rank is None:
            raise ValueError("deadlock_rank must be specified in config parameters for DEADLOCK strategy")

        current_rank = self._get_current_rank()

        if current_rank == deadlock_rank:
            start_time = time.time()
            stop_file = f"{_DEADLOCK_STOP_FILE_PREFIX}{os.getpid()}"

            while True:
                # Check timeout (if max_deadlock_time > 0)
                if max_deadlock_time > 0 and (time.time() - start_time) > max_deadlock_time:
                    raise TimeoutError(f"Deadlock simulation exceeded {max_deadlock_time}s timeout")

                # Check for stop signal file
                if os.path.exists(stop_file):
                    try:
                        os.remove(stop_file)
                    except OSError:
                        pass  # Ignore removal errors
                    raise InterruptedError("Deadlock simulation stopped by signal file")

                time.sleep(_DEADLOCK_CHECK_INTERVAL)

        # Other ranks proceed normally
        call_kwargs = dict(kwargs)
        if op is not None:
            call_kwargs["op"] = op
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        return self._original(tensor, **call_kwargs)


@ProxyRegistry.register("torch.distributed.all_gather")
class AllGatherProxy(BaseProxy, DelayMixin, TensorCorruptionMixin):
    """
    Proxy for torch.distributed.all_gather() operations.

    Supports fault injection at the distributed communication level (L1).

    Supported strategies:
    - DELAY: Adds delay before calling original all_gather
    - CORRUPT_GATHERED: Corrupts gathered tensor data for specified ranks
    - MISSING_RANK: Zeros out data for specified rank, simulating missing data
    - SHAPE_MISMATCH: Alters tensor shape to cause shape mismatch errors

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - CORRUPT_GATHERED: noise_scale (float, default 0.1), corrupt_ranks (list[int], optional)
    - MISSING_RANK: missing_rank (int, required) - the rank whose data should be zeroed
    - SHAPE_MISMATCH: size_delta (int, default 1) - how much to change size
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.CORRUPT_GATHERED,
        StrategyType.MISSING_RANK,
        StrategyType.SHAPE_MISMATCH,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L1"

    def _get_current_rank(self) -> int:
        """
        Get the current process rank.

        Returns:
            The current rank if distributed is initialized, otherwise 0.
        """
        if HAS_DIST and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _get_world_size(self) -> int:
        """
        Get the world size (number of processes).

        Returns:
            The world size if distributed is initialized, otherwise 1.
        """
        if HAS_DIST and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _strategy_corrupt_gathered(
        self,
        tensor_list: list,
        tensor: torch.Tensor,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt gathered tensor data for specified ranks.

        Calls the original all_gather, then corrupts the gathered data
        for specified ranks with Gaussian noise.

        Args:
            tensor_list: List of tensors to gather into (output).
            tensor: The tensor to send from this rank.
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_gather with corrupted output tensors.

        Config parameters:
            noise_scale (float): Standard deviation of Gaussian noise (default 0.1).
            corrupt_ranks (list[int]): Ranks whose gathered data to corrupt.
                If not specified, corrupts all ranks.
        """
        noise_scale = self._config.parameters.get("noise_scale", 0.1)
        corrupt_ranks = self._config.parameters.get("corrupt_ranks", None)

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        # Call original all_gather first
        result = self._original(tensor_list, tensor, **call_kwargs)

        # If async_op, we can't modify the result synchronously
        if async_op:
            return result

        # Corrupt the gathered tensors for specified ranks
        for rank_idx, gathered_tensor in enumerate(tensor_list):
            if corrupt_ranks is None or rank_idx in corrupt_ranks:
                corrupted = self._corrupt_tensor(gathered_tensor, noise_scale)
                gathered_tensor.copy_(corrupted)

        return result

    def _strategy_missing_rank(
        self,
        tensor_list: list,
        tensor: torch.Tensor,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Zero out data for specified rank, simulating missing data.

        Calls the original all_gather, then zeros out the gathered data
        for the specified rank, simulating a scenario where that rank's
        data was lost or corrupted.

        Args:
            tensor_list: List of tensors to gather into (output).
            tensor: The tensor to send from this rank.
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_gather with zeroed data for missing rank.

        Config parameters:
            missing_rank (int): The rank whose data should be zeroed (required).

        Raises:
            ValueError: If missing_rank is not specified in config parameters.
        """
        missing_rank = self._config.parameters.get("missing_rank")

        if missing_rank is None:
            raise ValueError("missing_rank must be specified in config parameters for MISSING_RANK strategy")

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        # Call original all_gather first
        result = self._original(tensor_list, tensor, **call_kwargs)

        # If async_op, we can't modify the result synchronously
        if async_op:
            return result

        # Zero out the data for the missing rank
        if 0 <= missing_rank < len(tensor_list):
            tensor_list[missing_rank].zero_()

        return result

    def _strategy_shape_mismatch(
        self,
        tensor_list: list,
        tensor: torch.Tensor,
        group: Any = None,
        async_op: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Alter tensor shape to cause shape mismatch errors.

        Modifies the input tensor's shape before calling all_gather,
        which will cause shape mismatch errors in distributed operations.
        This simulates bugs where tensors have inconsistent shapes across ranks.

        Args:
            tensor_list: List of tensors to gather into (output).
            tensor: The tensor to send from this rank.
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original all_gather (may raise an error due to mismatch).

        Config parameters:
            size_delta (int): How much to change the first dimension size (default 1).
            mismatch_rank (int): Only apply mismatch on this rank (optional).
                If not specified, applies to current rank 0 only.
        """
        size_delta = self._config.parameters.get("size_delta", 1)
        mismatch_rank = self._config.parameters.get("mismatch_rank", 0)

        current_rank = self._get_current_rank()

        # Only modify on the specified rank
        if current_rank == mismatch_rank:
            # Create a tensor with different shape
            if tensor.dim() > 0:
                # Add or remove elements from first dimension
                new_size = max(1, tensor.size(0) + size_delta)
                if size_delta > 0:
                    # Expand tensor
                    padding = torch.zeros(size_delta, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=0)
                elif size_delta < 0 and tensor.size(0) > abs(size_delta):
                    # Shrink tensor
                    tensor = tensor[:new_size]

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op

        # Call original - this may raise an error due to shape mismatch
        return self._original(tensor_list, tensor, **call_kwargs)


@ProxyRegistry.register("torch.distributed.barrier")
class BarrierProxy(BaseProxy, DelayMixin):
    """
    Proxy for torch.distributed.barrier() operations.

    Supports fault injection at the distributed communication level (L1).

    Supported strategies:
    - DELAY: Adds delay before calling original barrier
    - BARRIER_TIMEOUT: Causes specified rank to hang indefinitely, simulating timeout
    - BARRIER_SKIP: Skips the barrier call entirely, returning immediately
    - ASYNC_DESYNC: Adds random delay per rank to desynchronize processes

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - BARRIER_TIMEOUT: timeout_rank (int, required) - the rank to hang
    - BARRIER_SKIP: (no parameters required)
    - ASYNC_DESYNC: max_delay (float, default 5.0) - max random delay per rank
                    seed (int, optional) - random seed for reproducibility
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.BARRIER_TIMEOUT,
        StrategyType.BARRIER_SKIP,
        StrategyType.ASYNC_DESYNC,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L1"

    def _get_current_rank(self) -> int:
        """
        Get the current process rank.

        Returns:
            The current rank if distributed is initialized, otherwise 0.
        """
        if HAS_DIST and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _strategy_barrier_timeout(
        self,
        group: Any = None,
        async_op: bool = False,
        device_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Cause specified rank to hang, simulating a barrier timeout.

        The specified rank will enter a sleep loop, while other ranks
        will proceed normally. This creates a timeout situation where other ranks
        wait for the hanging rank at the barrier.

        The timeout simulation can be terminated by:
        1. Exceeding max_timeout_time (default: 3600s)
        2. Creating a stop signal file: /tmp/ralph_stop_deadlock_{pid}

        Args:
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            device_ids: Device IDs for NCCL backend.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Never returns for the timeout rank (raises TimeoutError/InterruptedError).
            Result of original barrier for other ranks.

        Config parameters:
            timeout_rank (int): The rank that should hang (required).
            max_timeout_time (float): Maximum time to simulate timeout in seconds
                                     (default: 3600). Set to 0 for infinite.

        Raises:
            ValueError: If timeout_rank is not specified in config parameters.
            TimeoutError: When max_timeout_time is exceeded.
            InterruptedError: When stop signal file is detected.
        """
        timeout_rank = self._config.parameters.get("timeout_rank")
        max_timeout_time = self._config.parameters.get("max_timeout_time", 3600)

        if timeout_rank is None:
            raise ValueError("timeout_rank must be specified in config parameters for BARRIER_TIMEOUT strategy")

        current_rank = self._get_current_rank()

        if current_rank == timeout_rank:
            start_time = time.time()
            stop_file = f"{_DEADLOCK_STOP_FILE_PREFIX}{os.getpid()}"

            while True:
                # Check timeout (if max_timeout_time > 0)
                if max_timeout_time > 0 and (time.time() - start_time) > max_timeout_time:
                    raise TimeoutError(f"Barrier timeout simulation exceeded {max_timeout_time}s")

                # Check for stop signal file
                if os.path.exists(stop_file):
                    try:
                        os.remove(stop_file)
                    except OSError:
                        pass
                    raise InterruptedError("Barrier timeout simulation stopped by signal file")

                time.sleep(_DEADLOCK_CHECK_INTERVAL)

        # Other ranks proceed normally
        call_kwargs = dict(kwargs)
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op
        if device_ids is not None:
            call_kwargs["device_ids"] = device_ids

        return self._original(**call_kwargs)

    def _strategy_barrier_skip(
        self,
        group: Any = None,
        async_op: bool = False,
        device_ids: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Skip the barrier call entirely, returning immediately.

        This simulates a scenario where a process skips synchronization,
        which can lead to race conditions and inconsistent state across processes.

        Args:
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            device_ids: Device IDs for NCCL backend.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            None without calling the original barrier.
        """
        # Simply return without calling original
        return None

    def _strategy_async_desync(
        self,
        group: Any = None,
        async_op: bool = False,
        device_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Add random delay per rank to desynchronize processes.

        Each rank will experience a different random delay before the barrier,
        simulating network jitter or varying process execution times.
        This can help test how the system handles desynchronization.

        Args:
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            device_ids: Device IDs for NCCL backend.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Result of the original barrier after the random delay.

        Config parameters:
            max_delay (float): Maximum delay in seconds (default 5.0).
            seed (int, optional): Random seed for reproducibility.
        """
        import random

        max_delay = self._config.parameters.get("max_delay", 5.0)
        seed = self._config.parameters.get("seed")

        # Use current rank to create deterministic but different delays per rank
        current_rank = self._get_current_rank()

        if seed is not None:
            # Use seed + rank for deterministic but rank-specific delays
            rng = random.Random(seed + current_rank)
        else:
            rng = random.Random()

        delay = rng.uniform(0, max_delay)
        time.sleep(delay)

        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if group is not None:
            call_kwargs["group"] = group
        call_kwargs["async_op"] = async_op
        if device_ids is not None:
            call_kwargs["device_ids"] = device_ids

        return self._original(**call_kwargs)
