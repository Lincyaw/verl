"""
L2 verl proxies for Ralph fault injection framework.

Contains proxy classes for verl-specific operations like RewardManager, FSDPCheckpointManager, etc.
"""

from typing import Any, Dict, Optional, Set

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("RewardManager.__call__")
class RewardManagerProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    Proxy for RewardManager.__call__() operations.

    Supports fault injection at the verl reward computation level (L2).

    Supported strategies:
    - DELAY: Adds delay before calling original RewardManager
    - CORRUPT_TENSOR: Corrupts tensor data in reward results with Gaussian noise
    - REWARD_FLIP: Negates all reward values (positive becomes negative and vice versa)
    - CONSTANT_REWARD: Returns a constant value for all rewards

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - CORRUPT_TENSOR: noise_scale (float, default 0.1)
    - REWARD_FLIP: No specific parameters
    - CONSTANT_REWARD: constant_value (float, default 0.0) - the constant reward value

    Expected input/output:
    - Input: data (DataProto or dict), return_dict (bool)
    - Output: Typically returns dict with 'rewards' tensor or similar structure
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.REWARD_FLIP,
        StrategyType.CONSTANT_REWARD,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L2"

    def _strategy_corrupt_tensor(
        self,
        data: Any,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Corrupt tensor data in reward results with Gaussian noise.

        Calls the original RewardManager, then corrupts all tensors in the result
        using Gaussian noise. This simulates noisy reward signals.

        Args:
            data: Input data (DataProto or dict containing prompts, responses, etc.)
            *args: Additional positional arguments to pass to original
            return_dict: Whether to return results as dict (passed to original)
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from RewardManager with all tensor values corrupted by noise.

        Config parameters:
            noise_scale (float): Standard deviation of Gaussian noise (default 0.1).
        """
        # Call original to get results
        call_kwargs = dict(kwargs)
        call_kwargs["return_dict"] = return_dict
        result = self._original(data, *args, **call_kwargs)

        # Get noise scale from config
        noise_scale = self._config.parameters.get("noise_scale", 0.1)

        # Corrupt all tensors in the result
        return self._corrupt_result_tensors(result, noise_scale)

    def _strategy_reward_flip(
        self,
        data: Any,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Negate all reward values in the result.

        Calls the original RewardManager, then negates all numeric values
        (tensors, floats, ints) in the result. This effectively flips
        positive rewards to negative and vice versa.

        Args:
            data: Input data (DataProto or dict containing prompts, responses, etc.)
            *args: Additional positional arguments to pass to original
            return_dict: Whether to return results as dict (passed to original)
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from RewardManager with all numeric values negated.
        """
        # Call original to get results
        call_kwargs = dict(kwargs)
        call_kwargs["return_dict"] = return_dict
        result = self._original(data, *args, **call_kwargs)

        # Negate all numeric values in the result
        return self._negate_result(result)

    def _strategy_constant_reward(
        self,
        data: Any,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Return a constant value for all rewards.

        Calls the original RewardManager, then replaces all numeric values
        (tensors, floats, ints) in the result with a constant value.
        This simulates a broken or uninformative reward signal.

        Args:
            data: Input data (DataProto or dict containing prompts, responses, etc.)
            *args: Additional positional arguments to pass to original
            return_dict: Whether to return results as dict (passed to original)
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from RewardManager with all numeric values set to constant.

        Config parameters:
            constant_value (float): The constant reward value (default 0.0).
        """
        # Call original to get results
        call_kwargs = dict(kwargs)
        call_kwargs["return_dict"] = return_dict
        result = self._original(data, *args, **call_kwargs)

        # Get constant value from config
        constant_value = self._config.parameters.get("constant_value", 0.0)

        # Set all numeric values to constant
        return self._set_constant(result, constant_value)
