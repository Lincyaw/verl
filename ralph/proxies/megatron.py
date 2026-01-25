"""
Megatron proxies for Ralph fault injection framework.

Contains proxy classes for Megatron-LM operations including MegatronEngine.optimizer_step()
and parallel execution operations.
These proxies target Megatron-specific behavior for distributed training fault injection.
"""

import random
import time
from typing import Any, Optional

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

    SUPPORTED_STRATEGIES: set[StrategyType] = {
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
        original_lrs: Optional[dict[int, float]] = None

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
    ) -> dict[int, float]:
        """
        Scale learning rate in Megatron optimizer.

        Args:
            engine: MegatronEngine instance
            factor: Factor to multiply learning rate by

        Returns:
            Dictionary mapping param group index to original LR values.
        """
        original_lrs: dict[int, float] = {}

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
        original_lrs: dict[int, float],
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


@ProxyRegistry.register("megatron.parallelism")
class ParallelismProxy(BaseProxy, DelayMixin, TensorCorruptionMixin):
    """
    Proxy for Megatron parallel execution operations.

    Supports fault injection at the parallelism layer (Megatron P3).

    Supported strategies:
    - PP_STAGE_FAILURE: Simulate pipeline parallel stage failure
    - TP_DESYNC: Cause tensor parallel desynchronization
    - WRONG_MICRO_BATCH_ROUTING: Route micro-batches incorrectly
    - ACTIVATION_CORRUPTION: Corrupt activations during forward/backward pass

    Config parameters:
    - PP_STAGE_FAILURE:
        - fail_stage (int, required): Which pipeline stage to fail
        - failure_type (str): "hang", "crash", "silent" (default "hang")
        - hang_duration (float): Seconds to hang (default 300.0)
    - TP_DESYNC:
        - desync_rank (int, optional): Specific rank to desync (default: random)
        - delay_seconds (float): Delay to introduce (default 5.0)
        - desync_type (str): "delay", "skip", "duplicate" (default "delay")
    - WRONG_MICRO_BATCH_ROUTING:
        - swap_stages (list): Pairs of stages to swap, e.g., [[0, 1], [2, 3]]
        - drop_ratio (float): Ratio of micro-batches to drop (default 0.0)
        - duplicate_ratio (float): Ratio of micro-batches to duplicate (default 0.0)
    - ACTIVATION_CORRUPTION:
        - noise_scale (float): Scale of Gaussian noise to add (default 0.01)
        - corrupt_ratio (float): Ratio of activations to corrupt (default 0.1)
        - target_layers (list): Specific layer names to corrupt (optional)
        - corrupt_type (str): "noise", "nan", "zero", "scale" (default "noise")

    Note:
        This proxy targets Megatron-LM's parallel execution including pipeline
        parallelism (PP) and tensor parallelism (TP) operations.
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.PP_STAGE_FAILURE,
        StrategyType.TP_DESYNC,
        StrategyType.WRONG_MICRO_BATCH_ROUTING,
        StrategyType.ACTIVATION_CORRUPTION,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Megatron"

    def _get_current_rank(self) -> int:
        """
        Get the current process rank in the distributed setting.

        Returns:
            Current rank (0 if not in distributed mode).
        """
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_rank()
        except (ImportError, RuntimeError):
            pass
        return 0

    def _get_pipeline_stage(self) -> int:
        """
        Get the current pipeline parallel stage.

        Returns:
            Pipeline stage (0 if not in pipeline parallel mode).
        """
        # Try to get from Megatron's parallel state
        try:
            from megatron.core import parallel_state

            if hasattr(parallel_state, "get_pipeline_model_parallel_rank"):
                return parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

        # Fallback: try from engine
        engine = self._get_parallel_context()
        if engine is not None:
            if hasattr(engine, "pipeline_rank"):
                return engine.pipeline_rank
            if hasattr(engine, "pp_rank"):
                return engine.pp_rank

        return 0

    def _get_tensor_parallel_rank(self) -> int:
        """
        Get the current tensor parallel rank.

        Returns:
            Tensor parallel rank (0 if not in tensor parallel mode).
        """
        # Try to get from Megatron's parallel state
        try:
            from megatron.core import parallel_state

            if hasattr(parallel_state, "get_tensor_model_parallel_rank"):
                return parallel_state.get_tensor_model_parallel_rank()
        except ImportError:
            pass

        # Fallback: try from engine
        engine = self._get_parallel_context()
        if engine is not None:
            if hasattr(engine, "tensor_rank"):
                return engine.tensor_rank
            if hasattr(engine, "tp_rank"):
                return engine.tp_rank

        return 0

    def _get_parallel_context(self) -> Optional[Any]:
        """
        Get the parallel execution context from the calling context.

        Returns:
            Parallel context or None if not available.
        """
        # Check if original function has __self__ (bound method)
        if hasattr(self._original, "__self__"):
            return self._original.__self__

        # Check if _context was set externally
        if hasattr(self, "_context"):
            return self._context

        return None

    def _strategy_pp_stage_failure(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate pipeline parallel stage failure.

        Causes a specific pipeline stage to fail, simulating hardware
        failure, OOM, or other issues in a specific PP stage.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from original function or failure behavior.

        Config parameters:
            fail_stage (int): Which pipeline stage to fail (required).
            failure_type (str): Type of failure - "hang", "crash", "silent".
            hang_duration (float): How long to hang in seconds (for "hang" type).
        """
        fail_stage = self._config.parameters.get("fail_stage")
        if fail_stage is None:
            raise ValueError("PP_STAGE_FAILURE requires 'fail_stage' parameter")

        failure_type = self._config.parameters.get("failure_type", "hang")
        hang_duration = self._config.parameters.get("hang_duration", 300.0)

        current_stage = self._get_pipeline_stage()

        # Only affect the specified stage
        if current_stage != fail_stage:
            return self._original(*args, **kwargs)

        if failure_type == "hang":
            # Simulate a hung process
            time.sleep(hang_duration)
            return self._original(*args, **kwargs)
        elif failure_type == "crash":
            # Simulate a crash by raising an exception
            raise RuntimeError(f"Simulated crash at pipeline stage {fail_stage}: PP stage failure injection")
        elif failure_type == "silent":
            # Silently skip execution and return None
            return None
        else:
            # Unknown failure type, default to hang
            time.sleep(hang_duration)
            return self._original(*args, **kwargs)

    def _strategy_tp_desync(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Cause tensor parallel desynchronization.

        Introduces desynchronization between tensor parallel ranks,
        simulating network delays or inconsistent execution.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from original function after desync.

        Config parameters:
            desync_rank (int): Specific rank to desync (optional, random if not set).
            delay_seconds (float): Delay to introduce (default 5.0).
            desync_type (str): Type of desync - "delay", "skip", "duplicate".
        """
        desync_rank = self._config.parameters.get("desync_rank")
        delay_seconds = self._config.parameters.get("delay_seconds", 5.0)
        desync_type = self._config.parameters.get("desync_type", "delay")

        current_tp_rank = self._get_tensor_parallel_rank()

        # If no specific rank specified, pick randomly
        if desync_rank is None:
            # Use deterministic random based on step for reproducibility
            seed = self._current_step if hasattr(self, "_current_step") else 0
            rng = random.Random(seed)
            desync_rank = rng.randint(0, 7)  # Assume max 8 TP ranks

        # Only affect the specified rank
        if current_tp_rank != desync_rank:
            return self._original(*args, **kwargs)

        if desync_type == "delay":
            # Add delay before execution
            time.sleep(delay_seconds)
            return self._original(*args, **kwargs)
        elif desync_type == "skip":
            # Skip execution entirely
            return None
        elif desync_type == "duplicate":
            # Execute twice and return last result
            self._original(*args, **kwargs)
            return self._original(*args, **kwargs)
        else:
            # Default to delay
            time.sleep(delay_seconds)
            return self._original(*args, **kwargs)

    def _strategy_wrong_micro_batch_routing(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Route micro-batches incorrectly in pipeline parallel execution.

        Manipulates micro-batch routing to cause inconsistent execution
        across pipeline stages.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from original function with modified micro-batch handling.

        Config parameters:
            swap_stages (list): Pairs of stages to swap, e.g., [[0, 1], [2, 3]].
            drop_ratio (float): Ratio of micro-batches to drop (default 0.0).
            duplicate_ratio (float): Ratio of micro-batches to duplicate (default 0.0).
        """
        swap_stages: list[list[int]] = self._config.parameters.get("swap_stages", [])
        drop_ratio = self._config.parameters.get("drop_ratio", 0.0)
        duplicate_ratio = self._config.parameters.get("duplicate_ratio", 0.0)

        current_stage = self._get_pipeline_stage()

        # Check if current stage should be swapped
        target_stage = current_stage
        for swap_pair in swap_stages:
            if len(swap_pair) >= 2:
                if current_stage == swap_pair[0]:
                    target_stage = swap_pair[1]
                    break
                elif current_stage == swap_pair[1]:
                    target_stage = swap_pair[0]
                    break

        # Handle dropping micro-batches
        if drop_ratio > 0:
            seed = self._current_step if hasattr(self, "_current_step") else 0
            rng = random.Random(seed + current_stage)
            if rng.random() < drop_ratio:
                # Drop this micro-batch
                return None

        # Handle duplicating micro-batches
        if duplicate_ratio > 0:
            seed = self._current_step if hasattr(self, "_current_step") else 0
            rng = random.Random(seed + current_stage + 1000)
            if rng.random() < duplicate_ratio:
                # Execute twice
                self._original(*args, **kwargs)

        # Modify the stage info if we're swapping
        if target_stage != current_stage:
            # Try to modify stage in the context
            context = self._get_parallel_context()
            if context is not None:
                original_stage = None
                if hasattr(context, "pipeline_rank"):
                    original_stage = context.pipeline_rank
                    context.pipeline_rank = target_stage
                elif hasattr(context, "pp_rank"):
                    original_stage = context.pp_rank
                    context.pp_rank = target_stage

                try:
                    result = self._original(*args, **kwargs)
                finally:
                    # Restore original stage
                    if original_stage is not None:
                        if hasattr(context, "pipeline_rank"):
                            context.pipeline_rank = original_stage
                        elif hasattr(context, "pp_rank"):
                            context.pp_rank = original_stage

                return result

        return self._original(*args, **kwargs)

    def _strategy_activation_corruption(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt activations during forward/backward pass.

        Adds noise or corruption to activations passed between
        pipeline stages or tensor parallel ranks.

        Args:
            *args: Positional arguments to pass to original
            **kwargs: Keyword arguments to pass to original

        Returns:
            Result from original function with corrupted activations.

        Config parameters:
            noise_scale (float): Scale of Gaussian noise (default 0.01).
            corrupt_ratio (float): Ratio of activations to corrupt (default 0.1).
            target_layers (list): Specific layer names to target (optional).
            corrupt_type (str): "noise", "nan", "zero", "scale" (default "noise").
        """
        noise_scale = self._config.parameters.get("noise_scale", 0.01)
        corrupt_ratio = self._config.parameters.get("corrupt_ratio", 0.1)
        target_layers: Optional[list[str]] = self._config.parameters.get("target_layers", None)
        corrupt_type = self._config.parameters.get("corrupt_type", "noise")

        # Call original function first
        result = self._original(*args, **kwargs)

        # Corrupt the result
        return self._apply_activation_corruption(result, noise_scale, corrupt_ratio, target_layers, corrupt_type)

    def _apply_activation_corruption(
        self,
        result: Any,
        noise_scale: float,
        corrupt_ratio: float,
        target_layers: Optional[list[str]],
        corrupt_type: str,
    ) -> Any:
        """
        Apply corruption to activations recursively.

        Args:
            result: Result to corrupt (tensor, dict, list, tuple)
            noise_scale: Scale of noise to add
            corrupt_ratio: Ratio of elements to corrupt
            target_layers: Specific layer names to target (for dicts)
            corrupt_type: Type of corruption to apply

        Returns:
            Corrupted result.
        """
        if isinstance(result, torch.Tensor):
            return self._corrupt_single_activation(result, noise_scale, corrupt_ratio, corrupt_type)
        elif isinstance(result, dict):
            return {
                key: self._apply_activation_corruption(value, noise_scale, corrupt_ratio, target_layers, corrupt_type)
                if target_layers is None or key in target_layers
                else value
                for key, value in result.items()
            }
        elif isinstance(result, (list, tuple)):
            corrupted = [
                self._apply_activation_corruption(item, noise_scale, corrupt_ratio, target_layers, corrupt_type)
                for item in result
            ]
            return type(result)(corrupted)
        else:
            return result

    def _corrupt_single_activation(
        self,
        tensor: torch.Tensor,
        noise_scale: float,
        corrupt_ratio: float,
        corrupt_type: str,
    ) -> torch.Tensor:
        """
        Corrupt a single activation tensor.

        Args:
            tensor: Tensor to corrupt
            noise_scale: Scale of noise
            corrupt_ratio: Ratio of elements to corrupt
            corrupt_type: Type of corruption

        Returns:
            Corrupted tensor.
        """
        # Determine which elements to corrupt
        seed = self._current_step if hasattr(self, "_current_step") else 0
        random.Random(seed)

        # Create corruption mask
        numel = tensor.numel()
        num_corrupt = int(numel * corrupt_ratio)

        if num_corrupt == 0:
            return tensor

        # Clone to avoid modifying original
        result = tensor.clone()

        if corrupt_type == "noise":
            # Add Gaussian noise
            noise = torch.randn_like(result) * noise_scale
            result = result + noise
        elif corrupt_type == "nan":
            # Inject NaN values
            result = self._inject_nan(result, corrupt_ratio)
        elif corrupt_type == "zero":
            # Zero out activations
            flat = result.view(-1)
            indices = torch.randperm(numel)[:num_corrupt]
            flat[indices] = 0.0
            result = flat.view(result.shape)
        elif corrupt_type == "scale":
            # Scale by large factor
            result = result * noise_scale
        else:
            # Default to noise
            noise = torch.randn_like(result) * noise_scale
            result = result + noise

        return result
