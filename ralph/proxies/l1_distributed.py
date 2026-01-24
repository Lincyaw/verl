"""
L1 Distributed proxies for Ralph fault injection framework.

Contains proxy classes for torch.distributed operations like all_reduce, all_gather, barrier, etc.
"""

import time
from typing import Any, Optional, Set

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


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

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
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

        Args:
            tensor: The tensor to be reduced.
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

        Args:
            tensor: The tensor to be reduced.
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

        Args:
            tensor: The tensor to be reduced.
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
        Cause specified rank to hang indefinitely, simulating a deadlock.

        The specified rank will enter an infinite sleep loop, while other ranks
        will proceed normally. This creates a deadlock situation where other ranks
        wait for the hanging rank during collective operations.

        Args:
            tensor: The tensor to be reduced.
            op: The reduction operation (e.g., ReduceOp.SUM).
            group: The process group to work on.
            async_op: If True, returns a distributed request object.
            **kwargs: Additional keyword arguments for the original function.

        Returns:
            Never returns for the deadlock rank.
            Result of original all_reduce for other ranks.

        Config parameters:
            deadlock_rank (int): The rank that should hang (required).

        Raises:
            ValueError: If deadlock_rank is not specified in config parameters.
        """
        deadlock_rank = self._config.parameters.get("deadlock_rank")

        if deadlock_rank is None:
            raise ValueError(
                "deadlock_rank must be specified in config parameters for DEADLOCK strategy"
            )

        current_rank = self._get_current_rank()

        if current_rank == deadlock_rank:
            # This rank hangs forever
            while True:
                time.sleep(3600)  # Sleep for 1 hour in a loop

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

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
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
            raise ValueError(
                "missing_rank must be specified in config parameters for MISSING_RANK strategy"
            )

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
                    padding = torch.zeros(
                        size_delta, *tensor.shape[1:],
                        dtype=tensor.dtype,
                        device=tensor.device
                    )
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
