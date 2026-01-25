"""
ResultModificationMixin provides result modification capabilities for fault injection proxies.

This mixin enables proxies to modify function results in various ways:
- Apply arbitrary modifier functions
- Negate numeric results (flip sign)
- Set results to constant values
- Handle nested structures (dict/list/tuple) recursively
"""

from typing import Any, Callable

import torch

Number = int | float


class ResultModificationMixin:
    """
    Mixin class providing result modification capabilities.

    This mixin should be inherited by proxy classes that need to support
    result modification strategies (MODIFY_RESULT, REWARD_FLIP, CONSTANT_REWARD).

    Expected attributes on the inheriting class:
    - _config: FaultConfig instance with parameters dict
    - _original: The original function being proxied
    """

    def _modify_result(self, result: Any, modifier_fn: Callable[[Any], Any]) -> Any:
        """
        Apply an arbitrary modifier function to the result.

        Args:
            result: The result to modify.
            modifier_fn: A callable that takes a value and returns the modified value.

        Returns:
            The result of applying modifier_fn to result.
        """
        return modifier_fn(result)

    def _negate_result(self, result: Any) -> Any:
        """
        Negate a result value (flip the sign).

        Handles:
        - Numbers (int, float): Returns -result
        - Tensors: Returns -result
        - Dicts: Recursively negates all values
        - Lists: Recursively negates all items
        - Tuples: Recursively negates all items (returns tuple)

        Args:
            result: The result to negate.

        Returns:
            Negated result. For unsupported types, returns the original value unchanged.
        """
        if isinstance(result, torch.Tensor):
            return -result
        elif isinstance(result, (int, float)):
            return -result
        elif isinstance(result, dict):
            return {k: self._negate_result(v) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._negate_result(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._negate_result(item) for item in result)
        else:
            # Unsupported types are returned unchanged
            return result

    def _set_constant(self, result: Any, value: Number) -> Any:
        """
        Set the result to a constant value.

        Handles:
        - Tensors: Returns torch.full_like(result, value)
        - Numbers (int, float): Returns the constant value
        - Dicts: Recursively sets all values to constant
        - Lists: Recursively sets all items to constant
        - Tuples: Recursively sets all items to constant (returns tuple)

        Args:
            result: The result to replace (used for shape inference for tensors).
            value: The constant value to use.

        Returns:
            Result with all numeric values replaced by the constant.
            For unsupported types, returns the original value unchanged.
        """
        if isinstance(result, torch.Tensor):
            return torch.full_like(result.float(), value).to(result.dtype)
        elif isinstance(result, (int, float)):
            # Return same type as input
            return type(result)(value)
        elif isinstance(result, dict):
            return {k: self._set_constant(v, value) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._set_constant(item, value) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._set_constant(item, value) for item in result)
        else:
            # Unsupported types are returned unchanged
            return result

    def _strategy_modify_result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Modify result strategy implementation.

        Executes the original function, then applies the modifier function
        specified in config to the result.

        Reads 'modifier_fn' from self._config.parameters.
        If no modifier_fn is provided, returns the original result unchanged.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with modifier applied.

        Config Parameters:
            modifier_fn (Callable): Function to apply to the result (optional).
        """
        result = self._original(*args, **kwargs)
        modifier_fn = self._config.parameters.get("modifier_fn", None)
        if modifier_fn is not None:
            return self._modify_result(result, modifier_fn)
        return result

    def _strategy_reward_flip(self, *args: Any, **kwargs: Any) -> Any:
        """
        Reward flip strategy implementation.

        Executes the original function, then negates all numeric values
        in the result, effectively flipping rewards from positive to negative
        and vice versa.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with all numeric values negated.
        """
        result = self._original(*args, **kwargs)
        return self._negate_result(result)

    def _strategy_constant_reward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Constant reward strategy implementation.

        Executes the original function, then replaces all numeric values
        in the result with a constant value.

        Reads 'constant_value' from self._config.parameters (defaults to 0.0).

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            Result of original function with all numeric values set to constant.

        Config Parameters:
            constant_value (Number): The constant value to use (optional, defaults to 0.0).
        """
        result = self._original(*args, **kwargs)
        constant_value = self._config.parameters.get("constant_value", 0.0)
        return self._set_constant(result, constant_value)
