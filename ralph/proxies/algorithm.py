"""
Algorithm proxies for Ralph fault injection framework.

Contains proxy classes for algorithm-level operations like GAE computation,
KL penalty calculation, and GRPO advantage estimation.
"""

from typing import Any, Set

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("compute_gae_advantage_return")
class GAEProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    Proxy for compute_gae_advantage_return() operations.

    Supports fault injection at the algorithm level for Generalized Advantage
    Estimation (GAE) computation. This allows testing how the training pipeline
    handles corrupted or incorrect advantage estimates.

    Supported strategies:
    - WRONG_ADVANTAGE: Returns random noise instead of computed advantages
    - ZERO_ADVANTAGE: Returns zeros for all advantages
    - INVERTED_ADVANTAGE: Negates the computed advantages
    - SCALED_ADVANTAGE: Multiplies advantages by a scale factor
    - DELAYED_ADVANTAGE: Adds delay before computing advantages

    Config parameters:
    - WRONG_ADVANTAGE: noise_scale (float, default 1.0) - scale of random noise
    - ZERO_ADVANTAGE: No specific parameters
    - INVERTED_ADVANTAGE: No specific parameters
    - SCALED_ADVANTAGE: scale_factor (float, default 0.0) - multiplier for advantages
    - DELAYED_ADVANTAGE: delay_seconds (float, default 10.0) - seconds to delay

    Expected input/output:
    - Input: values, rewards, response_mask, gamma, lam (typical GAE args)
    - Output: Dict with 'advantages' and 'returns' tensors or similar structure
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.WRONG_ADVANTAGE,
        StrategyType.ZERO_ADVANTAGE,
        StrategyType.INVERTED_ADVANTAGE,
        StrategyType.SCALED_ADVANTAGE,
        StrategyType.DELAYED_ADVANTAGE,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Algorithm"

    def _strategy_wrong_advantage(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return random noise instead of computed advantages.

        Calls the original GAE computation to get the result structure, then
        replaces all advantage values with random Gaussian noise. This simulates
        a completely broken advantage computation.

        Args:
            *args: Positional arguments to pass to original GAE computation
            **kwargs: Keyword arguments to pass to original GAE computation

        Returns:
            Result structure with advantages replaced by random noise.

        Config parameters:
            noise_scale (float): Standard deviation of random noise (default 1.0).
        """
        # Call original to get result structure
        result = self._original(*args, **kwargs)

        # Get noise scale from config
        noise_scale = self._config.parameters.get("noise_scale", 1.0)

        # Replace advantage tensors with random noise
        return self._apply_wrong_advantage(result, noise_scale)

    def _apply_wrong_advantage(self, result: Any, noise_scale: float) -> Any:
        """
        Recursively replace advantage tensors with random noise.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            noise_scale: Standard deviation of random noise

        Returns:
            Result with advantage values replaced by noise
        """
        if isinstance(result, torch.Tensor):
            # Replace tensor values with random Gaussian noise
            return torch.randn_like(result) * noise_scale
        elif isinstance(result, dict):
            # Only modify advantage-related keys, leave others unchanged
            advantage_keys = {"advantages", "advantage", "adv", "gae", "gae_advantages"}
            return_keys = {"returns", "return", "value_targets"}
            modified = {}
            for k, v in result.items():
                if k.lower() in advantage_keys:
                    modified[k] = self._apply_wrong_advantage(v, noise_scale)
                elif k.lower() in return_keys:
                    # Also corrupt returns if present
                    modified[k] = self._apply_wrong_advantage(v, noise_scale)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            return [self._apply_wrong_advantage(item, noise_scale) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_wrong_advantage(item, noise_scale) for item in result)
        else:
            return result

    def _strategy_zero_advantage(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return zeros for all advantages.

        Calls the original GAE computation to get the result structure, then
        replaces all advantage values with zeros. This simulates a condition
        where the advantage estimate provides no learning signal.

        Args:
            *args: Positional arguments to pass to original GAE computation
            **kwargs: Keyword arguments to pass to original GAE computation

        Returns:
            Result structure with advantages replaced by zeros.
        """
        # Call original to get result structure
        result = self._original(*args, **kwargs)

        # Set all advantage values to zero
        return self._apply_zero_advantage(result)

    def _apply_zero_advantage(self, result: Any) -> Any:
        """
        Recursively replace advantage tensors with zeros.

        Args:
            result: Result to process (tensor, dict, list, tuple)

        Returns:
            Result with advantage values replaced by zeros
        """
        if isinstance(result, torch.Tensor):
            return torch.zeros_like(result)
        elif isinstance(result, dict):
            # Only modify advantage-related keys, leave others unchanged
            advantage_keys = {"advantages", "advantage", "adv", "gae", "gae_advantages"}
            return_keys = {"returns", "return", "value_targets"}
            modified = {}
            for k, v in result.items():
                if k.lower() in advantage_keys:
                    modified[k] = self._apply_zero_advantage(v)
                elif k.lower() in return_keys:
                    # Also zero out returns if present
                    modified[k] = self._apply_zero_advantage(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            return [self._apply_zero_advantage(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_zero_advantage(item) for item in result)
        else:
            return result

    def _strategy_inverted_advantage(self, *args: Any, **kwargs: Any) -> Any:
        """
        Negate the computed advantages.

        Calls the original GAE computation to get the result, then negates all
        advantage values. This simulates a bug where the sign of advantages is
        flipped, which would cause the policy to learn the opposite behavior.

        Args:
            *args: Positional arguments to pass to original GAE computation
            **kwargs: Keyword arguments to pass to original GAE computation

        Returns:
            Result structure with negated advantages.
        """
        # Call original to get result
        result = self._original(*args, **kwargs)

        # Negate advantage values using the inherited method
        return self._apply_inverted_advantage(result)

    def _apply_inverted_advantage(self, result: Any) -> Any:
        """
        Recursively negate advantage tensors.

        Args:
            result: Result to process (tensor, dict, list, tuple)

        Returns:
            Result with advantage values negated
        """
        if isinstance(result, torch.Tensor):
            return -result
        elif isinstance(result, dict):
            # Only modify advantage-related keys, leave others unchanged
            advantage_keys = {"advantages", "advantage", "adv", "gae", "gae_advantages"}
            return_keys = {"returns", "return", "value_targets"}
            modified = {}
            for k, v in result.items():
                if k.lower() in advantage_keys:
                    modified[k] = self._apply_inverted_advantage(v)
                elif k.lower() in return_keys:
                    # Also invert returns if present
                    modified[k] = self._apply_inverted_advantage(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            return [self._apply_inverted_advantage(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_inverted_advantage(item) for item in result)
        else:
            return result

    def _strategy_scaled_advantage(self, *args: Any, **kwargs: Any) -> Any:
        """
        Scale the computed advantages by a factor.

        Calls the original GAE computation to get the result, then multiplies
        all advantage values by a scale factor. This can simulate issues like:
        - scale_factor=0: No learning signal (same as zero_advantage)
        - scale_factor<1: Reduced learning signal
        - scale_factor>1: Amplified learning signal
        - scale_factor<0: Inverted learning signal

        Args:
            *args: Positional arguments to pass to original GAE computation
            **kwargs: Keyword arguments to pass to original GAE computation

        Returns:
            Result structure with scaled advantages.

        Config parameters:
            scale_factor (float): Multiplier for advantages (default 0.0).
        """
        # Call original to get result
        result = self._original(*args, **kwargs)

        # Get scale factor from config
        scale_factor = self._config.parameters.get("scale_factor", 0.0)

        # Scale advantage values
        return self._apply_scaled_advantage(result, scale_factor)

    def _apply_scaled_advantage(self, result: Any, scale_factor: float) -> Any:
        """
        Recursively scale advantage tensors.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            scale_factor: Multiplier for advantage values

        Returns:
            Result with advantage values scaled
        """
        if isinstance(result, torch.Tensor):
            return result * scale_factor
        elif isinstance(result, dict):
            # Only modify advantage-related keys, leave others unchanged
            advantage_keys = {"advantages", "advantage", "adv", "gae", "gae_advantages"}
            return_keys = {"returns", "return", "value_targets"}
            modified = {}
            for k, v in result.items():
                if k.lower() in advantage_keys:
                    modified[k] = self._apply_scaled_advantage(v, scale_factor)
                elif k.lower() in return_keys:
                    # Also scale returns if present
                    modified[k] = self._apply_scaled_advantage(v, scale_factor)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            return [self._apply_scaled_advantage(item, scale_factor) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_scaled_advantage(item, scale_factor) for item in result)
        else:
            return result

    def _strategy_delayed_advantage(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add delay before computing advantages.

        Applies a delay before calling the original GAE computation. This
        simulates slow computation or resource contention that could affect
        training throughput.

        Args:
            *args: Positional arguments to pass to original GAE computation
            **kwargs: Keyword arguments to pass to original GAE computation

        Returns:
            Result from original GAE computation after delay.

        Config parameters:
            delay_seconds (float): Seconds to delay (default 10.0).
        """
        # Use the DelayMixin's _strategy_delay implementation
        delay_seconds = self._config.parameters.get("delay_seconds", 10.0)
        self._apply_delay(delay_seconds)
        return self._original(*args, **kwargs)
