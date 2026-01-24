"""
Megatron proxies for Ralph fault injection framework.

Contains proxy classes for Megatron-LM operations including MegatronEngine.optimizer_step().
These proxies target Megatron-specific behavior for distributed training fault injection.
"""

from typing import Any, Dict, Optional, Set

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.skip import SkipMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("MegatronEngine.optimizer_step")
class MegatronOptimizerProxy(BaseProxy, DelayMixin, SkipMixin, TensorCorruptionMixin):
    """
    Proxy for MegatronEngine.optimizer_step() operations.

    Supports fault injection at the Megatron optimizer step level (Megatron P3).

    Supported strategies:
    - SKIP: Skip the optimizer step entirely (no parameter update)
    - GRADIENT_OVERFLOW: Simulate gradient overflow condition
    - NAN_PARAMS: Inject NaN values into model parameters
    - WRONG_LR: Apply incorrect learning rate

    Config parameters:
    - SKIP: default_return (Any, default None) - value to return when skipping
    - GRADIENT_OVERFLOW: (no parameters) - triggers overflow flag
    - NAN_PARAMS: nan_ratio (float, default 0.001) - ratio of params to corrupt
    - WRONG_LR: lr_factor (float, default 10.0) - multiply LR by this factor

    Expected input/output:
    - Input: MegatronEngine.optimizer_step(args, timers)
    - Output: Typically None or success indicator

    Note:
        This proxy targets Megatron-LM's specific optimizer step handling
        which includes gradient clipping, overflow detection, and distributed
        optimizer synchronization.
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.SKIP,
        StrategyType.GRADIENT_OVERFLOW,
        StrategyType.NAN_PARAMS,
        StrategyType.WRONG_LR,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Megatron"

    def _strategy_gradient_overflow(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate gradient overflow condition.

        Sets the overflow flag in the Megatron optimizer to simulate
        a gradient overflow scenario. This causes the optimizer to skip
        the current step and potentially trigger loss scaling adjustments.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from optimizer_step() with overflow flag set.

        Config parameters:
            set_flag_only (bool): If True, only set flag without calling original (default False).
        """
        set_flag_only = self._config.parameters.get("set_flag_only", False)

        # Try to access the engine and set overflow flag
        engine = self._get_megatron_engine()
        if engine is not None:
            self._set_overflow_flag(engine)

        if set_flag_only:
            # Return without calling original - simulate overflow skip behavior
            return None

        # Call original - Megatron will handle the overflow flag
        return self._original(*args, **kwargs)

    def _strategy_nan_params(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Inject NaN values into model parameters.

        Corrupts a fraction of model parameters by setting them to NaN,
        simulating numerical instability or corrupted gradient updates.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from optimizer_step() after corrupting parameters.

        Config parameters:
            nan_ratio (float): Ratio of elements to set to NaN (default 0.001).
            param_names (list): Specific parameter names to corrupt (optional).
        """
        nan_ratio = self._config.parameters.get("nan_ratio", 0.001)
        param_names = self._config.parameters.get("param_names", None)

        # Try to access the engine and corrupt parameters
        engine = self._get_megatron_engine()
        if engine is not None:
            self._inject_nan_into_params(engine, nan_ratio, param_names)

        # Call original step
        return self._original(*args, **kwargs)

    def _strategy_wrong_lr(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply incorrect learning rate.

        Scales the learning rate by a configurable factor before the optimizer
        step, simulating a misconfigured learning rate schedule.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from optimizer_step() with modified learning rate.

        Config parameters:
            lr_factor (float): Factor to multiply learning rate by (default 10.0).
            restore_after (bool): Restore original LR after step (default False).
        """
        lr_factor = self._config.parameters.get("lr_factor", 10.0)
        restore_after = self._config.parameters.get("restore_after", False)

        # Try to access the optimizer and scale LR
        engine = self._get_megatron_engine()
        original_lrs: Optional[Dict[int, float]] = None

        if engine is not None:
            original_lrs = self._scale_megatron_lr(engine, lr_factor)

        # Call original step
        result = self._original(*args, **kwargs)

        # Restore original LR if requested
        if restore_after and original_lrs is not None and engine is not None:
            self._restore_megatron_lr(engine, original_lrs)

        return result

    def _get_megatron_engine(self) -> Optional[Any]:
        """
        Try to get the MegatronEngine instance from the calling context.

        Attempts to find the engine through various means:
        1. Check if original function has __self__ (bound method)
        2. Check _engine attribute set by injection engine
        3. Return None if not found

        Returns:
            The MegatronEngine instance or None if not accessible.
        """
        # Check if this is a bound method
        if hasattr(self._original, "__self__"):
            return self._original.__self__

        # Check if _engine was set externally
        if hasattr(self, "_engine"):
            return self._engine

        return None

    def _set_overflow_flag(self, engine: Any) -> None:
        """
        Set the gradient overflow flag in the Megatron engine.

        The overflow flag is typically stored in the optimizer or a dedicated
        overflow detector. Setting it triggers Megatron's overflow handling.

        Args:
            engine: MegatronEngine instance
        """
        # Try various common locations for overflow flag
        if hasattr(engine, "optimizer"):
            optimizer = engine.optimizer
            # Megatron-LM style
            if hasattr(optimizer, "overflow"):
                optimizer.overflow = True
            # FP16 optimizer style
            if hasattr(optimizer, "found_inf"):
                optimizer.found_inf = torch.tensor([1.0])
            # Apex style
            if hasattr(optimizer, "_overflow_buf"):
                optimizer._overflow_buf.fill_(1)

        # Direct engine flag
        if hasattr(engine, "overflow"):
            engine.overflow = True

        # Grad scaler style
        if hasattr(engine, "grad_scaler"):
            scaler = engine.grad_scaler
            if hasattr(scaler, "_found_inf_per_device"):
                # Mark all devices as having inf
                for device_tensor in scaler._found_inf_per_device.values():
                    device_tensor.fill_(1)

    def _inject_nan_into_params(
        self,
        engine: Any,
        nan_ratio: float,
        param_names: Optional[list] = None,
    ) -> None:
        """
        Inject NaN values into model parameters.

        Args:
            engine: MegatronEngine instance
            nan_ratio: Ratio of elements to set to NaN
            param_names: Specific parameter names to target (None = all)
        """
        model = self._get_model_from_engine(engine)
        if model is None:
            return

        for name, param in model.named_parameters():
            # Filter by name if specified
            if param_names is not None and name not in param_names:
                continue

            if param.requires_grad and param.data is not None:
                # Use TensorCorruptionMixin's method
                corrupted = self._inject_nan(param.data, nan_ratio)
                param.data.copy_(corrupted)

    def _get_model_from_engine(self, engine: Any) -> Optional[Any]:
        """
        Get the model from a MegatronEngine instance.

        Args:
            engine: MegatronEngine instance

        Returns:
            The model or None if not found.
        """
        # Try common model access patterns
        if hasattr(engine, "model"):
            model = engine.model
            # Handle list of model chunks (pipeline parallel)
            if isinstance(model, (list, tuple)):
                return model[0] if model else None
            return model

        if hasattr(engine, "module"):
            return engine.module

        return None

    def _scale_megatron_lr(
        self,
        engine: Any,
        factor: float,
    ) -> Dict[int, float]:
        """
        Scale learning rate in Megatron optimizer.

        Args:
            engine: MegatronEngine instance
            factor: Factor to multiply learning rate by

        Returns:
            Dictionary mapping param group index to original LR values.
        """
        original_lrs: Dict[int, float] = {}

        optimizer = getattr(engine, "optimizer", None)
        if optimizer is None:
            return original_lrs

        # Handle wrapped optimizers (e.g., Float16OptimizerWithFloat16Params)
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer

        if hasattr(actual_optimizer, "param_groups"):
            for i, param_group in enumerate(actual_optimizer.param_groups):
                if "lr" in param_group:
                    original_lrs[i] = param_group["lr"]
                    param_group["lr"] *= factor

        return original_lrs

    def _restore_megatron_lr(
        self,
        engine: Any,
        original_lrs: Dict[int, float],
    ) -> None:
        """
        Restore original learning rates in Megatron optimizer.

        Args:
            engine: MegatronEngine instance
            original_lrs: Dictionary mapping param group index to original LR values
        """
        optimizer = getattr(engine, "optimizer", None)
        if optimizer is None:
            return

        # Handle wrapped optimizers
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer

        if hasattr(actual_optimizer, "param_groups"):
            for i, original_lr in original_lrs.items():
                if i < len(actual_optimizer.param_groups):
                    actual_optimizer.param_groups[i]["lr"] = original_lr
