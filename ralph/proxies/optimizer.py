"""
Optimizer proxies for Ralph fault injection framework.

Contains proxy classes for PyTorch optimizer operations like optimizer.step() and lr_scheduler.step().
"""

from typing import Any, Dict, Optional, Set

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.skip import SkipMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("optimizer.step")
class OptimizerStepProxy(BaseProxy, DelayMixin, SkipMixin, TensorCorruptionMixin):
    """
    Proxy for optimizer.step() operations.

    Supports fault injection at the optimizer step level (Optimizer P3).

    Supported strategies:
    - SKIP: Skip the optimizer step entirely (no parameter update)
    - REPEAT: Repeat the optimizer step multiple times
    - CORRUPTED_MOMENTUM: Corrupt momentum buffers with Gaussian noise
    - RESET_STATE: Reset optimizer state (momentum, variance, etc.)

    Config parameters:
    - SKIP: default_return (Any, default None) - value to return when skipping
    - REPEAT: times (int, default 2) - number of times to repeat step
    - CORRUPTED_MOMENTUM: noise_scale (float, default 0.1) - noise magnitude
    - RESET_STATE: (no parameters) - resets all optimizer state buffers

    Expected input/output:
    - Input: optimizer.step(closure=None)
    - Output: None or loss value if closure provided

    Note:
        This proxy expects the optimizer to be passed as the first argument
        or accessed via self._optimizer attribute if available.
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.SKIP,
        StrategyType.REPEAT,
        StrategyType.CORRUPTED_MOMENTUM,
        StrategyType.RESET_STATE,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Optimizer"

    def _strategy_corrupted_momentum(
        self,
        *args: Any,
        closure: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt momentum buffers before optimizer step.

        Adds Gaussian noise to all momentum-related state tensors in the optimizer.
        This simulates corrupted gradient history that can destabilize training.

        Args:
            *args: Additional positional arguments (typically none for optimizer.step)
            closure: Optional closure that recomputes loss
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from optimizer.step() after corrupting momentum.

        Config parameters:
            noise_scale (float): Standard deviation of Gaussian noise (default 0.1).
        """
        noise_scale = self._config.parameters.get("noise_scale", 0.1)

        # Try to access the optimizer state
        optimizer = self._get_optimizer_from_context()

        if optimizer is not None:
            self._corrupt_optimizer_state(optimizer, noise_scale)

        # Now call the original step
        if closure is not None:
            return self._original(closure=closure, **kwargs)
        return self._original(*args, **kwargs)

    def _strategy_reset_state(
        self,
        *args: Any,
        closure: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Reset optimizer state before step.

        Clears all accumulated state (momentum buffers, adaptive learning rate history, etc.)
        as if the optimizer was just initialized. This can cause training instability.

        Args:
            *args: Additional positional arguments (typically none for optimizer.step)
            closure: Optional closure that recomputes loss
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from optimizer.step() after resetting state.
        """
        # Try to access the optimizer state
        optimizer = self._get_optimizer_from_context()

        if optimizer is not None:
            self._reset_optimizer_state(optimizer)

        # Now call the original step
        if closure is not None:
            return self._original(closure=closure, **kwargs)
        return self._original(*args, **kwargs)

    def _get_optimizer_from_context(self) -> Optional[Any]:
        """
        Try to get the optimizer instance from the calling context.

        Attempts to find the optimizer through various means:
        1. Check if original function has __self__ (bound method)
        2. Check _optimizer attribute set by injection engine
        3. Return None if not found (strategies will still run but won't modify state)

        Returns:
            The optimizer instance or None if not accessible.
        """
        # Check if this is a bound method (optimizer.step)
        if hasattr(self._original, "__self__"):
            return self._original.__self__

        # Check if _optimizer was set externally
        if hasattr(self, "_optimizer"):
            return self._optimizer

        return None

    def _corrupt_optimizer_state(self, optimizer: Any, noise_scale: float) -> None:
        """
        Add Gaussian noise to all momentum-related tensors in optimizer state.

        Handles common optimizer state keys:
        - 'momentum_buffer': SGD momentum
        - 'exp_avg': Adam first moment estimate
        - 'exp_avg_sq': Adam second moment estimate
        - 'exp_inf': AdaMax exponential infinity norm
        - 'step': Step counter (left unchanged)

        Args:
            optimizer: PyTorch optimizer instance
            noise_scale: Standard deviation of Gaussian noise to add
        """
        if not hasattr(optimizer, "state"):
            return

        momentum_keys = {
            "momentum_buffer",
            "exp_avg",
            "exp_avg_sq",
            "exp_inf",
            "max_exp_avg_sq",  # Amsgrad
            "square_avg",  # RMSprop
            "acc_delta",  # Adadelta
            "grad_avg",  # ASGD
        }

        for param_state in optimizer.state.values():
            if not isinstance(param_state, dict):
                continue

            for key in momentum_keys:
                if key in param_state:
                    tensor = param_state[key]
                    if isinstance(tensor, torch.Tensor):
                        # Add noise in-place
                        noise = torch.randn_like(tensor) * noise_scale
                        param_state[key] = tensor + noise

    def _reset_optimizer_state(self, optimizer: Any) -> None:
        """
        Reset all optimizer state buffers.

        Clears the state dictionary for all parameters, effectively resetting
        momentum, adaptive learning rate estimates, etc.

        Args:
            optimizer: PyTorch optimizer instance
        """
        if hasattr(optimizer, "state"):
            optimizer.state.clear()


@ProxyRegistry.register("lr_scheduler.step")
class LRSchedulerProxy(BaseProxy, DelayMixin, SkipMixin):
    """
    Proxy for lr_scheduler.step() operations.

    Supports fault injection at the learning rate scheduler level (Optimizer P3).

    Supported strategies:
    - SKIP: Skip the scheduler step (learning rate won't update)
    - WRONG_LR: Set learning rate to an incorrect value
    - LR_SPIKE: Temporarily spike learning rate to a very high value
    - LR_ZERO: Set learning rate to zero (freezes learning)

    Config parameters:
    - SKIP: default_return (Any, default None) - value to return when skipping
    - WRONG_LR: lr_factor (float, default 10.0) - multiply current LR by this factor
    - LR_SPIKE: spike_value (float, default 1.0) - absolute LR value to set
    - LR_ZERO: (no parameters) - sets LR to 0

    Expected input/output:
    - Input: scheduler.step(epoch=None, metrics=None)
    - Output: None (typically)
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.SKIP,
        StrategyType.WRONG_LR,
        StrategyType.LR_SPIKE,
        StrategyType.LR_ZERO,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Optimizer"

    def _strategy_wrong_lr(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply incorrect learning rate scaling.

        Calls the original scheduler step, then multiplies all learning rates
        by a configurable factor, simulating a misconfigured scheduler.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from scheduler.step() after modifying LR.

        Config parameters:
            lr_factor (float): Factor to multiply learning rate by (default 10.0).
        """
        # Call original step first
        result = self._original(*args, **kwargs)

        # Get the scheduler and modify LR
        lr_factor = self._config.parameters.get("lr_factor", 10.0)
        scheduler = self._get_scheduler_from_context()

        if scheduler is not None:
            self._scale_learning_rate(scheduler, lr_factor)

        return result

    def _strategy_lr_spike(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Spike learning rate to a high value.

        Calls the original scheduler step, then sets learning rate to
        an absolute high value, simulating a sudden LR spike.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from scheduler.step() after spiking LR.

        Config parameters:
            spike_value (float): Absolute LR value to set (default 1.0).
        """
        # Call original step first
        result = self._original(*args, **kwargs)

        # Get the scheduler and set absolute LR
        spike_value = self._config.parameters.get("spike_value", 1.0)
        scheduler = self._get_scheduler_from_context()

        if scheduler is not None:
            self._set_learning_rate(scheduler, spike_value)

        return result

    def _strategy_lr_zero(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Set learning rate to zero.

        Calls the original scheduler step, then sets learning rate to zero,
        effectively freezing all parameter updates.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from scheduler.step() after zeroing LR.
        """
        # Call original step first
        result = self._original(*args, **kwargs)

        # Get the scheduler and zero LR
        scheduler = self._get_scheduler_from_context()

        if scheduler is not None:
            self._set_learning_rate(scheduler, 0.0)

        return result

    def _get_scheduler_from_context(self) -> Optional[Any]:
        """
        Try to get the scheduler instance from the calling context.

        Returns:
            The scheduler instance or None if not accessible.
        """
        # Check if this is a bound method
        if hasattr(self._original, "__self__"):
            return self._original.__self__

        # Check if _scheduler was set externally
        if hasattr(self, "_scheduler"):
            return self._scheduler

        return None

    def _scale_learning_rate(self, scheduler: Any, factor: float) -> None:
        """
        Scale learning rate in all parameter groups by a factor.

        Args:
            scheduler: PyTorch LR scheduler instance
            factor: Factor to multiply learning rate by
        """
        optimizer = getattr(scheduler, "optimizer", None)
        if optimizer is None:
            return

        for param_group in optimizer.param_groups:
            if "lr" in param_group:
                param_group["lr"] *= factor

    def _set_learning_rate(self, scheduler: Any, lr_value: float) -> None:
        """
        Set learning rate in all parameter groups to an absolute value.

        Args:
            scheduler: PyTorch LR scheduler instance
            lr_value: Absolute learning rate value to set
        """
        optimizer = getattr(scheduler, "optimizer", None)
        if optimizer is None:
            return

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_value
