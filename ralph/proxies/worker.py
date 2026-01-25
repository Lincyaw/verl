"""
Worker proxies for Ralph fault injection framework.

Contains proxy classes for worker-level operations like value computation
in the critic worker.
"""

from typing import Any

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("CriticWorker.compute_values")
class ComputeValuesProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    Proxy for CriticWorker.compute_values() operations.

    Supports fault injection at the worker level for value function computation
    in the critic network. This allows testing how the training pipeline handles
    corrupted or incorrect value estimates from the critic.

    Supported strategies:
    - DELAY: Adds delay before computing values
    - WRONG_VALUES: Adds noise to computed values
    - CONSTANT_VALUES: Returns constant value for all predictions
    - NAN_VALUES: Injects NaN into value predictions
    - INVERTED_VALUES: Negates the computed values

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - WRONG_VALUES: noise_scale (float, default 0.1) - scale of noise to add
    - CONSTANT_VALUES: constant_value (float, default 0.0) - constant to return
    - NAN_VALUES: nan_ratio (float, default 0.1) - ratio of values to set to NaN
    - INVERTED_VALUES: No specific parameters

    Expected input/output:
    - Input: Batch of states/observations for value prediction
    - Output: Value tensor or dict containing 'values', 'value', 'v' keys
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.WRONG_VALUES,
        StrategyType.CONSTANT_VALUES,
        StrategyType.NAN_VALUES,
        StrategyType.INVERTED_VALUES,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Worker"

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add delay before computing values.

        Applies a delay before calling the original value computation.
        This simulates slow computation or resource contention.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result from original function after delay.

        Config parameters:
            delay_seconds (float): Seconds to delay (default 10.0).
        """
        delay_seconds = self._config.parameters.get("delay_seconds", 10.0)
        self._apply_delay(delay_seconds)
        return self._original(*args, **kwargs)

    def _strategy_wrong_values(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add noise to computed values.

        Calls the original value computation and then adds Gaussian noise
        to all value predictions. This simulates noisy or inaccurate value
        function estimates.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with noise added to value predictions.

        Config parameters:
            noise_scale (float): Standard deviation of noise (default 0.1).
        """
        result = self._original(*args, **kwargs)
        noise_scale = self._config.parameters.get("noise_scale", 0.1)
        return self._apply_noise_to_values(result, noise_scale)

    def _apply_noise_to_values(self, result: Any, noise_scale: float) -> Any:
        """
        Recursively add noise to value tensors in the result.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            noise_scale: Standard deviation of Gaussian noise to add

        Returns:
            Result with noise added to value predictions
        """
        if isinstance(result, torch.Tensor):
            # Add Gaussian noise to the tensor
            return result + torch.randn_like(result) * noise_scale
        elif isinstance(result, dict):
            # Only modify value-related keys
            value_keys = {"values", "value", "v", "critic_values", "state_values", "v_values"}
            modified = {}
            for k, v in result.items():
                if k.lower() in value_keys:
                    modified[k] = self._apply_noise_to_values(v, noise_scale)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            import random
            return type(result)(result + random.gauss(0, noise_scale))
        elif isinstance(result, list):
            return [self._apply_noise_to_values(item, noise_scale) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_noise_to_values(item, noise_scale) for item in result)
        else:
            return result

    def _strategy_constant_values(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return constant value for all predictions.

        Calls the original value computation to get the result structure, then
        replaces all value predictions with a constant. This simulates a
        completely uninformative value function.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with all value predictions set to constant_value.

        Config parameters:
            constant_value (float): Value to return for all predictions (default 0.0).
        """
        result = self._original(*args, **kwargs)
        constant_value = self._config.parameters.get("constant_value", 0.0)
        return self._apply_constant_to_values(result, constant_value)

    def _apply_constant_to_values(self, result: Any, constant_value: float) -> Any:
        """
        Recursively replace value tensors with a constant.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            constant_value: Constant value to set for all predictions

        Returns:
            Result with value predictions replaced by constant
        """
        if isinstance(result, torch.Tensor):
            return torch.full_like(result.float(), constant_value).to(result.dtype)
        elif isinstance(result, dict):
            # Only modify value-related keys
            value_keys = {"values", "value", "v", "critic_values", "state_values", "v_values"}
            modified = {}
            for k, v in result.items():
                if k.lower() in value_keys:
                    modified[k] = self._apply_constant_to_values(v, constant_value)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(constant_value)
        elif isinstance(result, list):
            return [self._apply_constant_to_values(item, constant_value) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_constant_to_values(item, constant_value) for item in result)
        else:
            return result

    def _strategy_nan_values(self, *args: Any, **kwargs: Any) -> Any:
        """
        Inject NaN into value predictions.

        Calls the original value computation and then injects NaN values
        into a ratio of the predictions. This simulates numerical instability
        or corrupted computation.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with NaN injected into value predictions.

        Config parameters:
            nan_ratio (float): Ratio of values to set to NaN (default 0.1).
        """
        result = self._original(*args, **kwargs)
        nan_ratio = self._config.parameters.get("nan_ratio", 0.1)
        return self._apply_nan_to_values(result, nan_ratio)

    def _apply_nan_to_values(self, result: Any, nan_ratio: float) -> Any:
        """
        Recursively inject NaN into value tensors in the result.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            nan_ratio: Ratio of elements to set to NaN (0.0-1.0)

        Returns:
            Result with NaN injected into value predictions
        """
        if isinstance(result, torch.Tensor):
            # Clone to avoid modifying original
            output = result.clone().float()
            # Create mask for NaN injection
            mask = torch.rand_like(output) < nan_ratio
            output[mask] = float('nan')
            return output.to(result.dtype)
        elif isinstance(result, dict):
            # Only modify value-related keys
            value_keys = {"values", "value", "v", "critic_values", "state_values", "v_values"}
            modified = {}
            for k, v in result.items():
                if k.lower() in value_keys:
                    modified[k] = self._apply_nan_to_values(v, nan_ratio)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            import random
            if random.random() < nan_ratio:
                return float('nan')
            return result
        elif isinstance(result, list):
            return [self._apply_nan_to_values(item, nan_ratio) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_nan_to_values(item, nan_ratio) for item in result)
        else:
            return result

    def _strategy_inverted_values(self, *args: Any, **kwargs: Any) -> Any:
        """
        Negate the computed values.

        Calls the original value computation and then negates all value
        predictions. This simulates a sign bug that would cause the critic
        to estimate the opposite of the true value.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with negated value predictions.
        """
        result = self._original(*args, **kwargs)
        return self._apply_inversion_to_values(result)

    def _apply_inversion_to_values(self, result: Any) -> Any:
        """
        Recursively negate value tensors in the result.

        Args:
            result: Result to process (tensor, dict, list, tuple)

        Returns:
            Result with value predictions negated
        """
        if isinstance(result, torch.Tensor):
            return -result
        elif isinstance(result, dict):
            # Only modify value-related keys
            value_keys = {"values", "value", "v", "critic_values", "state_values", "v_values"}
            modified = {}
            for k, v in result.items():
                if k.lower() in value_keys:
                    modified[k] = self._apply_inversion_to_values(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(-result)
        elif isinstance(result, list):
            return [self._apply_inversion_to_values(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_inversion_to_values(item) for item in result)
        else:
            return result
