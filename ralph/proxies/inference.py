"""
Inference proxies for Ralph fault injection framework.

Contains proxy classes for inference-level operations like text generation
from vLLM/SGLang engines.
"""

from typing import Any, Dict, List, Optional, Set, Union

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("vllm.generate")
class GenerateProxy(BaseProxy, DelayMixin):
    """
    Proxy for vLLM/SGLang generate() operations.

    Supports fault injection at the inference level for text generation
    operations. This allows testing how the training pipeline handles
    generation failures, timeouts, and corrupted outputs.

    Supported strategies:
    - DELAY: Standard delay before generation (inherited from DelayMixin)
    - GENERATION_TIMEOUT: Very long delay simulating a hung generation
    - EMPTY_RESPONSE: Returns empty sequences
    - TRUNCATED_OUTPUT: Limits output to shorter than expected length
    - GARBAGE_OUTPUT: Returns random tokens instead of generated text

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - GENERATION_TIMEOUT: timeout_seconds (float, default 300.0) - very long delay
    - EMPTY_RESPONSE: No specific parameters
    - TRUNCATED_OUTPUT: max_tokens (int, default 1) - maximum output tokens
    - GARBAGE_OUTPUT: vocab_size (int, default 32000) - vocabulary size for random tokens

    Expected input/output:
    - Input: prompts or input_ids, sampling parameters, etc.
    - Output: Generated sequences (list of token IDs or strings)
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.GENERATION_TIMEOUT,
        StrategyType.EMPTY_RESPONSE,
        StrategyType.TRUNCATED_OUTPUT,
        StrategyType.GARBAGE_OUTPUT,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Inference"

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add standard delay before generation.

        Applies a delay before calling the original generation function.
        This simulates slow generation or resource contention.

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

    def _strategy_generation_timeout(self, *args: Any, **kwargs: Any) -> Any:
        """
        Simulate a generation timeout with very long delay.

        Applies a very long delay to simulate a hung or extremely slow
        generation process. This tests timeout handling in the training
        pipeline.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result from original function after very long delay.

        Config parameters:
            timeout_seconds (float): Very long delay in seconds (default 300.0).
        """
        timeout_seconds = self._config.parameters.get("timeout_seconds", 300.0)
        self._apply_delay(timeout_seconds)
        return self._original(*args, **kwargs)

    def _strategy_empty_response(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return empty sequences instead of generated output.

        Calls the original generation to get the result structure, then
        replaces all generated content with empty sequences. This simulates
        a generation failure or empty response from the model.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result structure with empty sequences.
        """
        result = self._original(*args, **kwargs)
        return self._apply_empty_response(result)

    def _apply_empty_response(self, result: Any) -> Any:
        """
        Recursively replace generated content with empty sequences.

        Args:
            result: Result to process (tensor, dict, list, or RequestOutput-like)

        Returns:
            Result with generated content replaced by empty sequences
        """
        if isinstance(result, torch.Tensor):
            # Return tensor with zero tokens (empty along sequence dimension)
            # Assuming shape is [batch, seq_len] or [seq_len]
            if result.dim() >= 2:
                # Keep batch dimension, zero sequence
                shape = list(result.shape)
                shape[-1] = 0
                return torch.empty(shape, dtype=result.dtype, device=result.device)
            else:
                # Single sequence - return empty
                return torch.empty(0, dtype=result.dtype, device=result.device)
        elif isinstance(result, dict):
            # Handle dict-like outputs (common in many frameworks)
            output_keys = {
                "sequences", "generated_ids", "output_ids", "token_ids",
                "outputs", "generated_tokens", "generated_text", "text"
            }
            modified = {}
            for k, v in result.items():
                if k.lower() in output_keys:
                    modified[k] = self._apply_empty_response(v)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            # Handle list of outputs (batch generation)
            return [self._apply_empty_response(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_empty_response(item) for item in result)
        elif isinstance(result, str):
            # Return empty string for text outputs
            return ""
        elif hasattr(result, "outputs"):
            # Handle vLLM RequestOutput-like objects
            # Create a modified version with empty outputs
            try:
                # Try to create a similar object with empty outputs
                for output in result.outputs:
                    if hasattr(output, "token_ids"):
                        output.token_ids = []
                    if hasattr(output, "text"):
                        output.text = ""
            except (AttributeError, TypeError):
                pass
            return result
        else:
            return result

    def _strategy_truncated_output(self, *args: Any, **kwargs: Any) -> Any:
        """
        Truncate generated output to a shorter length.

        Calls the original generation, then truncates the output to be
        shorter than expected. This simulates early stopping or buffer
        overflow scenarios.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with truncated sequences.

        Config parameters:
            max_tokens (int): Maximum number of output tokens (default 1).
        """
        result = self._original(*args, **kwargs)
        max_tokens = self._config.parameters.get("max_tokens", 1)
        return self._apply_truncated_output(result, max_tokens)

    def _apply_truncated_output(self, result: Any, max_tokens: int) -> Any:
        """
        Recursively truncate sequences to max_tokens.

        Args:
            result: Result to process (tensor, dict, list, etc.)
            max_tokens: Maximum number of tokens to keep

        Returns:
            Result with truncated sequences
        """
        if isinstance(result, torch.Tensor):
            # Truncate along the sequence dimension (typically last dim)
            if result.dim() >= 2:
                # Truncate sequence dimension
                return result[..., :max_tokens]
            elif result.dim() == 1:
                return result[:max_tokens]
            else:
                return result
        elif isinstance(result, dict):
            # Handle dict-like outputs
            output_keys = {
                "sequences", "generated_ids", "output_ids", "token_ids",
                "outputs", "generated_tokens", "generated_text", "text"
            }
            modified = {}
            for k, v in result.items():
                if k.lower() in output_keys:
                    modified[k] = self._apply_truncated_output(v, max_tokens)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            # Truncate list of tokens or outputs
            if all(isinstance(item, (int, float)) for item in result):
                # List of token IDs
                return result[:max_tokens]
            else:
                # List of outputs (batch)
                return [self._apply_truncated_output(item, max_tokens) for item in result]
        elif isinstance(result, tuple):
            if all(isinstance(item, (int, float)) for item in result):
                # Tuple of token IDs
                return tuple(result[:max_tokens])
            else:
                return tuple(self._apply_truncated_output(item, max_tokens) for item in result)
        elif isinstance(result, str):
            # Truncate string output (approximate by characters, not tokens)
            # Use a rough estimate of 4 chars per token
            return result[:max_tokens * 4]
        elif hasattr(result, "outputs"):
            # Handle vLLM RequestOutput-like objects
            try:
                for output in result.outputs:
                    if hasattr(output, "token_ids"):
                        output.token_ids = output.token_ids[:max_tokens]
                    if hasattr(output, "text"):
                        output.text = output.text[:max_tokens * 4]
            except (AttributeError, TypeError):
                pass
            return result
        else:
            return result

    def _strategy_garbage_output(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return random tokens instead of generated output.

        Calls the original generation to get the result structure, then
        replaces all generated tokens with random values. This simulates
        model corruption, memory errors, or similar issues.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with random tokens instead of generated content.

        Config parameters:
            vocab_size (int): Vocabulary size for random tokens (default 32000).
            seed (int, optional): Random seed for reproducibility.
        """
        result = self._original(*args, **kwargs)
        vocab_size = self._config.parameters.get("vocab_size", 32000)
        seed = self._config.parameters.get("seed", None)
        return self._apply_garbage_output(result, vocab_size, seed)

    def _apply_garbage_output(
        self,
        result: Any,
        vocab_size: int,
        seed: Optional[int] = None
    ) -> Any:
        """
        Recursively replace sequences with random tokens.

        Args:
            result: Result to process (tensor, dict, list, etc.)
            vocab_size: Vocabulary size for random token generation
            seed: Optional random seed

        Returns:
            Result with random tokens
        """
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(result, torch.Tensor):
            # Generate random tokens with same shape
            return torch.randint(
                0, vocab_size, result.shape,
                dtype=result.dtype, device=result.device
            )
        elif isinstance(result, dict):
            # Handle dict-like outputs
            output_keys = {
                "sequences", "generated_ids", "output_ids", "token_ids",
                "outputs", "generated_tokens"
            }
            # Exclude text keys since we handle them differently
            text_keys = {"generated_text", "text"}
            modified = {}
            for k, v in result.items():
                if k.lower() in output_keys:
                    modified[k] = self._apply_garbage_output(v, vocab_size, seed)
                elif k.lower() in text_keys:
                    # Generate random text for text outputs
                    modified[k] = self._generate_garbage_text(v, vocab_size)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, list):
            if all(isinstance(item, int) for item in result):
                # List of token IDs - replace with random
                import random
                if seed is not None:
                    random.seed(seed)
                return [random.randint(0, vocab_size - 1) for _ in result]
            else:
                # List of outputs (batch)
                return [self._apply_garbage_output(item, vocab_size, seed) for item in result]
        elif isinstance(result, tuple):
            if all(isinstance(item, int) for item in result):
                import random
                if seed is not None:
                    random.seed(seed)
                return tuple(random.randint(0, vocab_size - 1) for _ in result)
            else:
                return tuple(self._apply_garbage_output(item, vocab_size, seed) for item in result)
        elif isinstance(result, str):
            # Generate random text (garbage characters)
            return self._generate_garbage_text(result, vocab_size)
        elif hasattr(result, "outputs"):
            # Handle vLLM RequestOutput-like objects
            try:
                import random
                if seed is not None:
                    random.seed(seed)
                for output in result.outputs:
                    if hasattr(output, "token_ids") and output.token_ids:
                        output.token_ids = [
                            random.randint(0, vocab_size - 1)
                            for _ in output.token_ids
                        ]
                    if hasattr(output, "text"):
                        output.text = self._generate_garbage_text(output.text, vocab_size)
            except (AttributeError, TypeError):
                pass
            return result
        else:
            return result

    def _generate_garbage_text(self, original: str, vocab_size: int) -> str:
        """
        Generate random garbage text with approximately same length.

        Args:
            original: Original text to replace
            vocab_size: Used to seed randomness (not directly used for text)

        Returns:
            Random string of similar length
        """
        import random
        import string

        # Generate random printable characters
        length = len(original) if isinstance(original, str) else 100
        chars = string.ascii_letters + string.digits + " " * 10  # Add spaces for readability
        return "".join(random.choice(chars) for _ in range(length))


@ProxyRegistry.register("rollout.update_weights")
class UpdateWeightsProxy(BaseProxy, DelayMixin):
    """
    Proxy for rollout weight update operations.

    Supports fault injection at the weight synchronization level for inference
    worker weight updates. This allows testing how the training pipeline handles
    weight synchronization failures, mismatches, and corruption.

    Supported strategies:
    - DELAY: Standard delay before weight update (inherited from DelayMixin)
    - WEIGHT_MISMATCH: Returns weights with wrong shape or structure
    - PARTIAL_UPDATE: Only updates some parameters, leaves others unchanged
    - CORRUPT_WEIGHTS: Corrupts weight tensors with noise
    - OLD_WEIGHTS: Returns stale/old weights instead of new ones

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - WEIGHT_MISMATCH: mismatch_ratio (float, default 0.1) - how much to alter dimensions
    - PARTIAL_UPDATE: update_ratio (float, default 0.5) - fraction of params to update
    - CORRUPT_WEIGHTS: noise_scale (float, default 0.01) - Gaussian noise scale
    - OLD_WEIGHTS: staleness_steps (int, default 1) - how many steps old the weights are

    Expected input/output:
    - Input: New weights (state_dict or tensor dict)
    - Output: Updated weights (potentially modified)
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.WEIGHT_MISMATCH,
        StrategyType.PARTIAL_UPDATE,
        StrategyType.CORRUPT_WEIGHTS,
        StrategyType.OLD_WEIGHTS,
    }

    def __init__(self, original_fn: Any, collector: Any = None):
        """
        Initialize UpdateWeightsProxy.

        Args:
            original_fn: The original function to wrap
            collector: Optional collector for recording fault injections
        """
        super().__init__(original_fn, collector)
        # Store historical weights for OLD_WEIGHTS strategy
        self._weight_history: List[Any] = []
        self._max_history_size = 10

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Inference"

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add standard delay before weight update.

        Applies a delay before calling the original weight update function.
        This simulates slow weight transfer or synchronization delays.

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
        result = self._original(*args, **kwargs)
        self._store_weights(args, kwargs)
        return result

    def _strategy_weight_mismatch(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return weights with wrong shape or structure.

        Calls the original function, then alters the weight tensor shapes
        to simulate shape mismatches during weight synchronization.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with altered weight shapes.

        Config parameters:
            mismatch_ratio (float): How much to alter dimensions (default 0.1).
        """
        result = self._original(*args, **kwargs)
        self._store_weights(args, kwargs)
        mismatch_ratio = self._config.parameters.get("mismatch_ratio", 0.1)
        return self._apply_weight_mismatch(result, mismatch_ratio)

    def _apply_weight_mismatch(self, result: Any, mismatch_ratio: float) -> Any:
        """
        Recursively apply shape mismatch to weight tensors.

        Args:
            result: Result to process (tensor, dict, list, etc.)
            mismatch_ratio: Fraction to alter dimensions by

        Returns:
            Result with altered tensor shapes
        """
        if isinstance(result, torch.Tensor):
            # Alter the tensor shape by truncating or padding
            new_shape = list(result.shape)
            if len(new_shape) > 0:
                # Alter the last dimension (typically hidden size)
                dim_idx = -1
                original_size = new_shape[dim_idx]
                delta = int(original_size * mismatch_ratio)
                if delta > 0:
                    # Randomly choose to increase or decrease
                    import random
                    if random.random() > 0.5:
                        new_shape[dim_idx] = original_size + delta
                        # Pad with zeros
                        padded = torch.zeros(new_shape, dtype=result.dtype, device=result.device)
                        slices = [slice(None)] * len(new_shape)
                        slices[dim_idx] = slice(0, original_size)
                        padded[tuple(slices)] = result
                        return padded
                    else:
                        new_shape[dim_idx] = max(1, original_size - delta)
                        # Truncate
                        slices = [slice(None)] * len(new_shape)
                        slices[dim_idx] = slice(0, new_shape[dim_idx])
                        return result[tuple(slices)].clone()
            return result
        elif isinstance(result, dict):
            return {k: self._apply_weight_mismatch(v, mismatch_ratio) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._apply_weight_mismatch(item, mismatch_ratio) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_weight_mismatch(item, mismatch_ratio) for item in result)
        else:
            return result

    def _strategy_partial_update(self, *args: Any, **kwargs: Any) -> Any:
        """
        Only update some parameters, leave others unchanged.

        Calls the original function, then selectively keeps only some
        parameters updated while reverting others to their previous values.
        This simulates partial weight synchronization failures.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with only partial weight updates.

        Config parameters:
            update_ratio (float): Fraction of params to update (default 0.5).
            keys_to_skip (list, optional): Specific keys to skip updating.
        """
        result = self._original(*args, **kwargs)
        self._store_weights(args, kwargs)
        update_ratio = self._config.parameters.get("update_ratio", 0.5)
        keys_to_skip = self._config.parameters.get("keys_to_skip", None)
        return self._apply_partial_update(result, update_ratio, keys_to_skip)

    def _apply_partial_update(
        self, result: Any, update_ratio: float, keys_to_skip: Optional[List[str]] = None
    ) -> Any:
        """
        Recursively apply partial update to weights.

        Args:
            result: Result to process (dict of weights)
            update_ratio: Fraction of parameters to keep updated
            keys_to_skip: Specific keys to skip (revert to zeros)

        Returns:
            Result with only some weights updated
        """
        import random

        if isinstance(result, dict):
            modified = {}
            keys = list(result.keys())

            if keys_to_skip:
                # Skip specific keys
                skip_set = set(keys_to_skip)
            else:
                # Randomly skip some keys based on update_ratio
                num_to_skip = int(len(keys) * (1 - update_ratio))
                skip_set = set(random.sample(keys, min(num_to_skip, len(keys))))

            for k, v in result.items():
                if k in skip_set:
                    # Revert to zeros (simulating failed update)
                    if isinstance(v, torch.Tensor):
                        modified[k] = torch.zeros_like(v)
                    else:
                        modified[k] = self._apply_partial_update(v, 0.0, None)
                else:
                    modified[k] = v
            return modified
        elif isinstance(result, torch.Tensor):
            # For tensors at root level, return zeros based on ratio
            import random
            if random.random() > update_ratio:
                return torch.zeros_like(result)
            return result
        elif isinstance(result, list):
            return [self._apply_partial_update(item, update_ratio, keys_to_skip) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_partial_update(item, update_ratio, keys_to_skip) for item in result)
        else:
            return result

    def _strategy_corrupt_weights(self, *args: Any, **kwargs: Any) -> Any:
        """
        Corrupt weight tensors with Gaussian noise.

        Calls the original function, then adds noise to the weight tensors
        to simulate memory corruption or transmission errors.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result with corrupted weight tensors.

        Config parameters:
            noise_scale (float): Scale of Gaussian noise (default 0.01).
        """
        result = self._original(*args, **kwargs)
        self._store_weights(args, kwargs)
        noise_scale = self._config.parameters.get("noise_scale", 0.01)
        return self._apply_weight_corruption(result, noise_scale)

    def _apply_weight_corruption(self, result: Any, noise_scale: float) -> Any:
        """
        Recursively add Gaussian noise to weight tensors.

        Args:
            result: Result to process (tensor, dict, list, etc.)
            noise_scale: Scale of Gaussian noise to add

        Returns:
            Result with noise added to tensors
        """
        if isinstance(result, torch.Tensor):
            if result.is_floating_point():
                noise = torch.randn_like(result) * noise_scale
                return result + noise
            else:
                # For non-float tensors, convert, add noise, convert back
                float_tensor = result.float()
                noise = torch.randn_like(float_tensor) * noise_scale
                return (float_tensor + noise).to(result.dtype)
        elif isinstance(result, dict):
            return {k: self._apply_weight_corruption(v, noise_scale) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._apply_weight_corruption(item, noise_scale) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_weight_corruption(item, noise_scale) for item in result)
        else:
            return result

    def _strategy_old_weights(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return stale/old weights instead of new ones.

        Returns previously stored weights instead of the newly synchronized
        weights. This simulates synchronization failures or version mismatches.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Old weights from history instead of new weights.

        Config parameters:
            staleness_steps (int): How many steps old the weights are (default 1).
        """
        # Get the current weights first
        current_result = self._original(*args, **kwargs)
        staleness_steps = self._config.parameters.get("staleness_steps", 1)

        # Try to return old weights
        if len(self._weight_history) >= staleness_steps:
            old_weights = self._weight_history[-staleness_steps]
            # Store current weights for future use
            self._store_weights(args, kwargs, current_result)
            return old_weights
        else:
            # Not enough history - return current but store for future
            self._store_weights(args, kwargs, current_result)
            return current_result

    def _store_weights(
        self,
        args: tuple,
        kwargs: dict,
        result: Optional[Any] = None
    ) -> None:
        """
        Store weights in history for OLD_WEIGHTS strategy.

        Args:
            args: Positional arguments (may contain weights)
            kwargs: Keyword arguments (may contain weights)
            result: Optional result to store directly
        """
        if result is not None:
            weights_to_store = result
        elif len(args) > 0:
            # Assume first arg is weights
            weights_to_store = args[0]
        elif "weights" in kwargs:
            weights_to_store = kwargs["weights"]
        elif "state_dict" in kwargs:
            weights_to_store = kwargs["state_dict"]
        else:
            # Try to store the result of the call
            return

        # Deep copy tensors to avoid storing references
        weights_copy = self._deep_copy_weights(weights_to_store)
        self._weight_history.append(weights_copy)

        # Trim history if too large
        if len(self._weight_history) > self._max_history_size:
            self._weight_history = self._weight_history[-self._max_history_size:]

    def _deep_copy_weights(self, weights: Any) -> Any:
        """
        Deep copy weight tensors to avoid storing references.

        Args:
            weights: Weights to copy (tensor, dict, list, etc.)

        Returns:
            Deep copy of weights
        """
        if isinstance(weights, torch.Tensor):
            return weights.clone().detach()
        elif isinstance(weights, dict):
            return {k: self._deep_copy_weights(v) for k, v in weights.items()}
        elif isinstance(weights, list):
            return [self._deep_copy_weights(item) for item in weights]
        elif isinstance(weights, tuple):
            return tuple(self._deep_copy_weights(item) for item in weights)
        else:
            return weights
