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
