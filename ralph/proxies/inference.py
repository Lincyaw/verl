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
