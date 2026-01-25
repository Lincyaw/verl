"""
Algorithm proxies for Ralph fault injection framework.

Contains proxy classes for algorithm-level operations like GAE computation,
KL penalty calculation, and GRPO advantage estimation.
"""

from typing import Any

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

    SUPPORTED_STRATEGIES: set[StrategyType] = {
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


@ProxyRegistry.register("apply_kl_penalty")
class KLPenaltyProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    Proxy for apply_kl_penalty() operations.

    Supports fault injection at the algorithm level for KL divergence penalty
    calculations used in PPO and similar algorithms. This allows testing how
    the training pipeline handles corrupted or incorrect KL penalty values.

    Supported strategies:
    - DELAY: Adds delay before computing KL penalty
    - WRONG_KL: Scales KL penalty by a configurable factor
    - ZERO_KL: Returns zero KL penalty
    - EXTREME_KL: Returns very large KL value
    - NEGATIVE_KL: Returns negative KL penalty

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - WRONG_KL: scale_factor (float, default 10.0) - multiplier for KL penalty
    - ZERO_KL: No specific parameters
    - EXTREME_KL: extreme_value (float, default 1e6) - extreme KL value
    - NEGATIVE_KL: No specific parameters (negates the computed KL)

    Expected input/output:
    - Input: Varies by implementation (typically policy logprobs, ref logprobs, etc.)
    - Output: KL penalty tensor or dict containing 'kl', 'kl_penalty' keys
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.WRONG_KL,
        StrategyType.ZERO_KL,
        StrategyType.EXTREME_KL,
        StrategyType.NEGATIVE_KL,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Algorithm"

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add delay before computing KL penalty.

        Applies a delay before calling the original KL penalty computation.
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

    def _strategy_wrong_kl(self, *args: Any, **kwargs: Any) -> Any:
        """
        Scale KL penalty by a configurable factor.

        Calls the original KL computation and then multiplies the result
        by a scale factor. This simulates miscalculated or corrupted KL
        penalty values.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with KL values multiplied by scale_factor.

        Config parameters:
            scale_factor (float): Multiplier for KL values (default 10.0).
        """
        result = self._original(*args, **kwargs)
        scale_factor = self._config.parameters.get("scale_factor", 10.0)
        return self._apply_scale_kl(result, scale_factor)

    def _apply_scale_kl(self, result: Any, scale_factor: float) -> Any:
        """
        Recursively scale KL values in the result.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            scale_factor: Multiplier for KL values

        Returns:
            Result with KL values scaled
        """
        if isinstance(result, torch.Tensor):
            return result * scale_factor
        elif isinstance(result, dict):
            # Only modify KL-related keys
            kl_keys = {"kl", "kl_penalty", "kl_divergence", "kl_loss", "kl_coef"}
            modified = {}
            for k, v in result.items():
                if k.lower() in kl_keys:
                    modified[k] = self._apply_scale_kl(v, scale_factor)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(result * scale_factor)
        elif isinstance(result, list):
            return [self._apply_scale_kl(item, scale_factor) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_scale_kl(item, scale_factor) for item in result)
        else:
            return result

    def _strategy_zero_kl(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return zero KL penalty.

        Calls the original KL computation to get the result structure, then
        replaces all KL values with zeros. This simulates a condition where
        the KL penalty provides no regularization signal.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with all KL values replaced by zeros.
        """
        result = self._original(*args, **kwargs)
        return self._apply_zero_kl(result)

    def _apply_zero_kl(self, result: Any) -> Any:
        """
        Recursively replace KL values with zeros.

        Args:
            result: Result to process (tensor, dict, list, tuple)

        Returns:
            Result with KL values replaced by zeros
        """
        if isinstance(result, torch.Tensor):
            return torch.zeros_like(result)
        elif isinstance(result, dict):
            # Only modify KL-related keys
            kl_keys = {"kl", "kl_penalty", "kl_divergence", "kl_loss", "kl_coef"}
            modified = {}
            for k, v in result.items():
                if k.lower() in kl_keys:
                    modified[k] = self._apply_zero_kl(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(0)
        elif isinstance(result, list):
            return [self._apply_zero_kl(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_zero_kl(item) for item in result)
        else:
            return result

    def _strategy_extreme_kl(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return very large KL value.

        Calls the original KL computation to get the result structure, then
        replaces all KL values with an extreme value. This simulates a
        catastrophic divergence between policy and reference distributions.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with all KL values set to extreme_value.

        Config parameters:
            extreme_value (float): Extreme KL value to use (default 1e6).
        """
        result = self._original(*args, **kwargs)
        extreme_value = self._config.parameters.get("extreme_value", 1e6)
        return self._apply_extreme_kl(result, extreme_value)

    def _apply_extreme_kl(self, result: Any, extreme_value: float) -> Any:
        """
        Recursively replace KL values with an extreme value.

        Args:
            result: Result to process (tensor, dict, list, tuple)
            extreme_value: Value to set for all KL values

        Returns:
            Result with KL values replaced by extreme_value
        """
        if isinstance(result, torch.Tensor):
            return torch.full_like(result, extreme_value)
        elif isinstance(result, dict):
            # Only modify KL-related keys
            kl_keys = {"kl", "kl_penalty", "kl_divergence", "kl_loss", "kl_coef"}
            modified = {}
            for k, v in result.items():
                if k.lower() in kl_keys:
                    modified[k] = self._apply_extreme_kl(v, extreme_value)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(extreme_value)
        elif isinstance(result, list):
            return [self._apply_extreme_kl(item, extreme_value) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_extreme_kl(item, extreme_value) for item in result)
        else:
            return result

    def _strategy_negative_kl(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return negative KL penalty.

        Calls the original KL computation and then negates all KL values.
        This simulates a bug where the sign of the KL penalty is flipped,
        which would encourage divergence rather than penalizing it.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with all KL values negated.
        """
        result = self._original(*args, **kwargs)
        return self._apply_negative_kl(result)

    def _apply_negative_kl(self, result: Any) -> Any:
        """
        Recursively negate KL values.

        Args:
            result: Result to process (tensor, dict, list, tuple)

        Returns:
            Result with KL values negated
        """
        if isinstance(result, torch.Tensor):
            return -result
        elif isinstance(result, dict):
            # Only modify KL-related keys
            kl_keys = {"kl", "kl_penalty", "kl_divergence", "kl_loss", "kl_coef"}
            modified = {}
            for k, v in result.items():
                if k.lower() in kl_keys:
                    modified[k] = self._apply_negative_kl(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, (int, float)):
            return type(result)(-result)
        elif isinstance(result, list):
            return [self._apply_negative_kl(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_negative_kl(item) for item in result)
        else:
            return result


@ProxyRegistry.register("compute_grpo_outcome_advantage")
class GRPOProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    Proxy for compute_grpo_outcome_advantage() operations.

    Supports fault injection at the algorithm level for Group Relative Policy
    Optimization (GRPO) advantage computation. This allows testing how the
    training pipeline handles corrupted grouping, normalization, or advantage
    estimates in GRPO-based training.

    GRPO groups samples by index and normalizes rewards within each group to
    compute advantages. The strategies here target different aspects of this
    computation.

    Supported strategies:
    - WRONG_GROUPING: Uses shuffled or random group indices
    - WRONG_NORMALIZATION: Applies incorrect mean/std computation
    - SKIP_NORMALIZATION: Skips normalization, returns raw scores
    - SINGLE_SAMPLE_GROUPS: Treats each sample as its own group

    Config parameters:
    - WRONG_GROUPING: shuffle_seed (int, optional) - seed for reproducible shuffling
    - WRONG_NORMALIZATION: mean_scale (float, default 2.0), std_scale (float, default 0.5)
    - SKIP_NORMALIZATION: No specific parameters
    - SINGLE_SAMPLE_GROUPS: No specific parameters

    Expected input/output:
    - Input: token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo
    - Output: Tuple of (advantages, returns) tensors, both shape (bs, response_length)
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.WRONG_GROUPING,
        StrategyType.WRONG_NORMALIZATION,
        StrategyType.SKIP_NORMALIZATION,
        StrategyType.SINGLE_SAMPLE_GROUPS,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Algorithm"

    def _strategy_wrong_grouping(self, *args: Any, **kwargs: Any) -> Any:
        """
        Use shuffled or random group indices for GRPO computation.

        Extracts the index array from arguments and shuffles it before passing
        to the original function. This causes samples to be grouped incorrectly,
        potentially mixing samples from different prompts/groups.

        Args:
            *args: Positional arguments (token_level_rewards, response_mask, index, ...)
            **kwargs: Keyword arguments

        Returns:
            Result with incorrect grouping applied.

        Config parameters:
            shuffle_seed (int, optional): Seed for reproducible shuffling.
            randomize (bool, default False): If True, use completely random indices.
        """
        import random

        import numpy as np

        # Extract index from args or kwargs
        # Signature: (token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo)
        args_list = list(args)

        if "index" in kwargs:
            index = kwargs["index"]
            index_source = "kwargs"
        elif len(args) >= 3:
            index = args[2]
            index_source = "args"
        else:
            # No index found, just call original
            return self._original(*args, **kwargs)

        # Get config parameters
        shuffle_seed = self._config.parameters.get("shuffle_seed", None)
        randomize = self._config.parameters.get("randomize", False)

        # Create corrupted index
        if isinstance(index, np.ndarray):
            corrupted_index = index.copy()
        else:
            corrupted_index = np.array(index)

        if randomize:
            # Completely random indices
            if shuffle_seed is not None:
                np.random.seed(shuffle_seed)
            unique_indices = np.unique(corrupted_index)
            corrupted_index = np.random.choice(unique_indices, size=len(corrupted_index))
        else:
            # Shuffle the existing indices
            if shuffle_seed is not None:
                random.seed(shuffle_seed)
            random.shuffle(corrupted_index)

        # Apply corrupted index
        if index_source == "kwargs":
            kwargs["index"] = corrupted_index
        else:
            args_list[2] = corrupted_index
            args = tuple(args_list)

        return self._original(*args, **kwargs)

    def _strategy_wrong_normalization(self, *args: Any, **kwargs: Any) -> Any:
        """
        Apply incorrect mean/std computation for GRPO normalization.

        Calls the original function to get the result, then applies additional
        incorrect scaling to simulate bugs in the normalization computation.
        This can cause advantages to be miscalibrated.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with additional scaling applied to simulate wrong normalization.

        Config parameters:
            mean_scale (float, default 2.0): Factor to scale mean subtraction effect.
            std_scale (float, default 0.5): Factor to scale std division effect.
        """
        # Get the original result
        result = self._original(*args, **kwargs)

        # Get scaling factors
        mean_scale = self._config.parameters.get("mean_scale", 2.0)
        std_scale = self._config.parameters.get("std_scale", 0.5)

        # Apply wrong scaling to simulate incorrect normalization
        # The idea is: if original did (x - mean) / std,
        # we modify it to look like it was computed with wrong mean/std
        return self._apply_wrong_normalization(result, mean_scale, std_scale)

    def _apply_wrong_normalization(self, result: Any, mean_scale: float, std_scale: float) -> Any:
        """
        Recursively apply wrong normalization scaling to advantage tensors.

        Args:
            result: Result to process (tensor, tuple, dict, list)
            mean_scale: Factor to scale the mean-subtracted component
            std_scale: Factor to scale the std-divided component

        Returns:
            Result with modified normalization
        """
        if isinstance(result, torch.Tensor):
            # Apply combined scaling: essentially scale * x + bias
            # This simulates effect of wrong mean/std computation
            return result * mean_scale * std_scale
        elif isinstance(result, tuple):
            return tuple(self._apply_wrong_normalization(item, mean_scale, std_scale) for item in result)
        elif isinstance(result, dict):
            return {k: self._apply_wrong_normalization(v, mean_scale, std_scale) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._apply_wrong_normalization(item, mean_scale, std_scale) for item in result]
        else:
            return result

    def _strategy_skip_normalization(self, *args: Any, **kwargs: Any) -> Any:
        """
        Skip normalization and return raw scores as advantages.

        Instead of computing normalized advantages within groups, this strategy
        returns the raw summed scores without any normalization. This simulates
        a bug where the normalization step is bypassed.

        Args:
            *args: Positional arguments (token_level_rewards, response_mask, ...)
            **kwargs: Keyword arguments

        Returns:
            Raw scores without group normalization applied.
        """
        # Extract token_level_rewards and response_mask
        if len(args) >= 2:
            token_level_rewards = args[0]
            response_mask = args[1]
        else:
            token_level_rewards = kwargs.get("token_level_rewards")
            response_mask = kwargs.get("response_mask")

        if token_level_rewards is None or response_mask is None:
            # Fallback to original if we can't extract tensors
            return self._original(*args, **kwargs)

        # Compute raw scores without normalization
        # This is what GRPO does before normalizing within groups
        with torch.no_grad():
            scores = token_level_rewards.sum(dim=-1)
            # Broadcast to response length without normalization
            advantages = scores.unsqueeze(-1) * response_mask

        # GRPO returns (advantages, returns) where both are the same
        return advantages, advantages.clone()

    def _strategy_single_sample_groups(self, *args: Any, **kwargs: Any) -> Any:
        """
        Treat each sample as its own group.

        Modifies the index array so each sample has a unique group index,
        effectively eliminating the grouping benefit of GRPO. With single-sample
        groups, the mean equals the sample value and std is 0 (or 1 if handled),
        resulting in zero or undefined advantages.

        Args:
            *args: Positional arguments (token_level_rewards, response_mask, index, ...)
            **kwargs: Keyword arguments

        Returns:
            Result computed with each sample in its own group.

        Config parameters:
            zero_std_handling (str, default "one"): How to handle zero std -
                "one" sets std to 1, "epsilon" uses small value.
        """
        import numpy as np

        # Extract tensor shape to determine batch size
        if len(args) >= 1:
            token_level_rewards = args[0]
        else:
            token_level_rewards = kwargs.get("token_level_rewards")

        if token_level_rewards is None:
            return self._original(*args, **kwargs)

        batch_size = token_level_rewards.shape[0]

        # Create unique index for each sample
        single_sample_index = np.arange(batch_size)

        # Replace index in args/kwargs
        args_list = list(args)
        if "index" in kwargs:
            kwargs["index"] = single_sample_index
        elif len(args) >= 3:
            args_list[2] = single_sample_index
            args = tuple(args_list)
        else:
            # If no index parameter position found, add to kwargs
            kwargs["index"] = single_sample_index

        return self._original(*args, **kwargs)
