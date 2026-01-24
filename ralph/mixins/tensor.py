"""
TensorCorruptionMixin provides tensor corruption capabilities for fault injection proxies.

This mixin enables proxies to corrupt tensor data in various ways:
- Adding Gaussian noise
- Injecting NaN values
- Injecting Inf values
- Scaling tensor values
- Recursively corrupting tensors in nested structures (dict/list/tuple)
"""
import torch
from typing import Any, Union


class TensorCorruptionMixin:
    """
    Mixin class providing tensor corruption capabilities.

    This mixin should be inherited by proxy classes that need to support
    tensor corruption strategies (CORRUPT_TENSOR, INJECT_NAN, INJECT_INF).

    Expected attributes on the inheriting class:
    - _config: FaultConfig instance with parameters dict
    - _original: The original function being proxied
    """

    def _corrupt_tensor(self, tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """
        Add Gaussian noise to tensor.

        Args:
            tensor: Input tensor to corrupt.
            noise_scale: Standard deviation of the Gaussian noise.

        Returns:
            New tensor with added noise (tensor + randn_like(tensor) * noise_scale).
        """
        noise = torch.randn_like(tensor.float()) * noise_scale
        return tensor + noise.to(tensor.dtype)

    def _inject_nan(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Inject NaN values into a tensor.

        Args:
            tensor: Input tensor to corrupt.
            ratio: Fraction of elements to set to NaN (0.0 to 1.0).

        Returns:
            New tensor with specified ratio of elements set to NaN.
        """
        mask = torch.rand_like(tensor.float()) < ratio
        result = tensor.clone().float()
        result[mask] = float('nan')
        return result.to(tensor.dtype)

    def _inject_inf(
        self,
        tensor: torch.Tensor,
        ratio: float,
        positive: bool = True
    ) -> torch.Tensor:
        """
        Inject Inf values into a tensor.

        Args:
            tensor: Input tensor to corrupt.
            ratio: Fraction of elements to set to Inf (0.0 to 1.0).
            positive: If True, inject +Inf; if False, inject -Inf.

        Returns:
            New tensor with specified ratio of elements set to +/-Inf.
        """
        mask = torch.rand_like(tensor.float()) < ratio
        result = tensor.clone().float()
        inf_value = float('inf') if positive else float('-inf')
        result[mask] = inf_value
        return result.to(tensor.dtype)

    def _scale_tensor(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale tensor by a constant factor.

        Args:
            tensor: Input tensor to scale.
            scale: Scaling factor to multiply tensor by.

        Returns:
            Scaled tensor (tensor * scale).
        """
        return tensor * scale

    def _corrupt_result_tensors(
        self,
        result: Any,
        noise_scale: float
    ) -> Any:
        """
        Recursively corrupt all tensors in a result structure.

        Handles nested dicts, lists, tuples, and bare tensors.

        Args:
            result: The result to process (tensor, dict, list, tuple, or other).
            noise_scale: Standard deviation of Gaussian noise to add.

        Returns:
            Result with all tensors corrupted by Gaussian noise.
            Non-tensor values are returned unchanged.
        """
        if isinstance(result, torch.Tensor):
            return self._corrupt_tensor(result, noise_scale)
        elif isinstance(result, dict):
            return {
                k: self._corrupt_result_tensors(v, noise_scale)
                for k, v in result.items()
            }
        elif isinstance(result, list):
            return [
                self._corrupt_result_tensors(item, noise_scale)
                for item in result
            ]
        elif isinstance(result, tuple):
            return tuple(
                self._corrupt_result_tensors(item, noise_scale)
                for item in result
            )
        else:
            return result

    def _strategy_corrupt_tensor(self, *args: Any, **kwargs: Any) -> Any:
        """
        Corrupt tensor strategy implementation.

        Executes the original function, then corrupts all tensors in the result
        by adding Gaussian noise.

        Reads 'noise_scale' from self._config.parameters (defaults to 0.1).

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with all tensors corrupted.
        """
        result = self._original(*args, **kwargs)
        noise_scale = self._config.parameters.get("noise_scale", 0.1)
        return self._corrupt_result_tensors(result, noise_scale)

    def _strategy_inject_nan(self, *args: Any, **kwargs: Any) -> Any:
        """
        Inject NaN strategy implementation.

        Executes the original function, then injects NaN values into all tensors
        in the result.

        Reads 'nan_ratio' from self._config.parameters (defaults to 0.001).

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with NaN values injected into tensors.
        """
        result = self._original(*args, **kwargs)
        nan_ratio = self._config.parameters.get("nan_ratio", 0.001)
        return self._inject_nan_result(result, nan_ratio)

    def _inject_nan_result(self, result: Any, nan_ratio: float) -> Any:
        """
        Recursively inject NaN into all tensors in a result structure.

        Args:
            result: The result to process (tensor, dict, list, tuple, or other).
            nan_ratio: Fraction of elements to set to NaN.

        Returns:
            Result with NaN injected into all tensors.
        """
        if isinstance(result, torch.Tensor):
            return self._inject_nan(result, nan_ratio)
        elif isinstance(result, dict):
            return {
                k: self._inject_nan_result(v, nan_ratio)
                for k, v in result.items()
            }
        elif isinstance(result, list):
            return [self._inject_nan_result(item, nan_ratio) for item in result]
        elif isinstance(result, tuple):
            return tuple(
                self._inject_nan_result(item, nan_ratio) for item in result
            )
        else:
            return result

    def _strategy_inject_inf(self, *args: Any, **kwargs: Any) -> Any:
        """
        Inject Inf strategy implementation.

        Executes the original function, then injects Inf values into all tensors
        in the result.

        Reads 'inf_ratio' (defaults to 0.001) and 'positive' (defaults to True)
        from self._config.parameters.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with Inf values injected into tensors.
        """
        result = self._original(*args, **kwargs)
        inf_ratio = self._config.parameters.get("inf_ratio", 0.001)
        positive = self._config.parameters.get("positive", True)
        return self._inject_inf_result(result, inf_ratio, positive)

    def _inject_inf_result(
        self,
        result: Any,
        inf_ratio: float,
        positive: bool
    ) -> Any:
        """
        Recursively inject Inf into all tensors in a result structure.

        Args:
            result: The result to process (tensor, dict, list, tuple, or other).
            inf_ratio: Fraction of elements to set to Inf.
            positive: If True, inject +Inf; if False, inject -Inf.

        Returns:
            Result with Inf injected into all tensors.
        """
        if isinstance(result, torch.Tensor):
            return self._inject_inf(result, inf_ratio, positive)
        elif isinstance(result, dict):
            return {
                k: self._inject_inf_result(v, inf_ratio, positive)
                for k, v in result.items()
            }
        elif isinstance(result, list):
            return [
                self._inject_inf_result(item, inf_ratio, positive)
                for item in result
            ]
        elif isinstance(result, tuple):
            return tuple(
                self._inject_inf_result(item, inf_ratio, positive)
                for item in result
            )
        else:
            return result
