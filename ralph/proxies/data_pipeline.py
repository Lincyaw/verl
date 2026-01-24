"""
Data pipeline proxies for Ralph fault injection framework.

Contains proxy classes for data pipeline operations like DataProto concatenation
and chunking. These proxies enable testing of data handling resilience in verl's
DataProto-based data flow.
"""

import random
from typing import Any, List, Optional, Set

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("DataProto.concat")
class DataProtoConcatProxy(BaseProxy, DelayMixin):
    """
    Proxy for DataProto.concat() operations.

    Supports fault injection at the data pipeline level for DataProto concatenation.
    This allows testing how the training pipeline handles corrupted or misaligned
    data batches.

    Supported strategies:
    - DATA_MISMATCH: Returns result with inconsistent batch sizes
    - LOST_ITEMS: Drops some input DataProtos before concatenation
    - DUPLICATE_ITEMS: Repeats some input DataProtos
    - WRONG_ORDER: Shuffles the order of input DataProtos

    Config parameters:
    - DATA_MISMATCH: mismatch_ratio (float, default 0.1) - how much to alter batch size
    - LOST_ITEMS: drop_ratio (float, default 0.2) - fraction of items to drop
    - DUPLICATE_ITEMS: duplicate_ratio (float, default 0.2) - fraction of items to duplicate
    - WRONG_ORDER: seed (int, optional) - random seed for shuffle reproducibility

    Expected input/output:
    - Input: List of DataProto objects or similar iterable
    - Output: Concatenated DataProto or similar structure
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DATA_MISMATCH,
        StrategyType.LOST_ITEMS,
        StrategyType.DUPLICATE_ITEMS,
        StrategyType.WRONG_ORDER,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "DataPipeline"

    def _strategy_data_mismatch(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return result with inconsistent batch sizes.

        Calls the original concat to get the result, then modifies the batch
        dimension to create inconsistency. This simulates data corruption that
        leads to shape mismatches downstream.

        Args:
            *args: Positional arguments, first arg expected to be list of DataProtos
            **kwargs: Keyword arguments to pass to original concat

        Returns:
            Result with modified batch dimensions.

        Config parameters:
            mismatch_ratio (float): How much to alter batch size (default 0.1).
                Positive values increase, negative values decrease.
        """
        # Call original to get result
        result = self._original(*args, **kwargs)

        # Get mismatch ratio from config
        mismatch_ratio = self._config.parameters.get("mismatch_ratio", 0.1)

        # Apply mismatch to result
        return self._apply_data_mismatch(result, mismatch_ratio)

    def _apply_data_mismatch(self, result: Any, mismatch_ratio: float) -> Any:
        """
        Apply batch size mismatch to the result.

        Recursively processes the result structure and modifies tensor shapes
        to create inconsistency.

        Args:
            result: Result to modify (dict, tensor, or other structure)
            mismatch_ratio: How much to alter batch size

        Returns:
            Modified result with mismatched batch sizes
        """
        try:
            import torch
        except ImportError:
            return result

        if isinstance(result, torch.Tensor) and result.dim() > 0:
            # Modify the batch dimension (dim 0)
            batch_size = result.shape[0]
            delta = int(batch_size * mismatch_ratio)

            if delta > 0:
                # Increase batch size by duplicating some entries
                extra_indices = torch.randint(0, batch_size, (delta,))
                extra_data = result[extra_indices]
                return torch.cat([result, extra_data], dim=0)
            elif delta < 0 and batch_size > abs(delta):
                # Decrease batch size by removing entries
                new_size = batch_size + delta  # delta is negative
                return result[:new_size]
            return result
        elif isinstance(result, dict):
            return {k: self._apply_data_mismatch(v, mismatch_ratio) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._apply_data_mismatch(item, mismatch_ratio) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._apply_data_mismatch(item, mismatch_ratio) for item in result)
        else:
            # Non-tensor types or objects with batch attribute
            if hasattr(result, "batch") and hasattr(result.batch, "__len__"):
                # Try to modify batch attribute if it's a dict or similar
                try:
                    result.batch = self._apply_data_mismatch(result.batch, mismatch_ratio)
                except (AttributeError, TypeError):
                    pass
            return result

    def _strategy_lost_items(self, *args: Any, **kwargs: Any) -> Any:
        """
        Drop some input DataProtos before concatenation.

        Removes a fraction of the input items before calling the original concat.
        This simulates data loss scenarios where some batches are dropped.

        Args:
            *args: Positional arguments, first arg expected to be list of DataProtos
            **kwargs: Keyword arguments to pass to original concat

        Returns:
            Result of concatenating remaining items.

        Config parameters:
            drop_ratio (float): Fraction of items to drop (default 0.2).
                Value between 0.0 and 1.0.
        """
        # Get drop ratio from config
        drop_ratio = self._config.parameters.get("drop_ratio", 0.2)
        drop_ratio = max(0.0, min(1.0, drop_ratio))  # Clamp to [0, 1]

        # Extract the list of items to concatenate
        items = self._extract_items(args, kwargs)

        if items is None or len(items) == 0:
            # No items to process, call original
            return self._original(*args, **kwargs)

        # Calculate how many items to drop
        num_to_drop = max(0, int(len(items) * drop_ratio))

        if num_to_drop >= len(items):
            # Don't drop everything - keep at least one
            num_to_drop = len(items) - 1 if len(items) > 1 else 0

        if num_to_drop > 0:
            # Randomly select indices to keep
            indices_to_keep = sorted(
                random.sample(range(len(items)), len(items) - num_to_drop)
            )
            filtered_items = [items[i] for i in indices_to_keep]
        else:
            filtered_items = list(items)

        # Call original with filtered items
        return self._call_with_modified_items(args, kwargs, filtered_items)

    def _strategy_duplicate_items(self, *args: Any, **kwargs: Any) -> Any:
        """
        Repeat some input DataProtos.

        Duplicates a fraction of the input items before calling the original concat.
        This simulates scenarios where data is accidentally processed multiple times.

        Args:
            *args: Positional arguments, first arg expected to be list of DataProtos
            **kwargs: Keyword arguments to pass to original concat

        Returns:
            Result of concatenating items including duplicates.

        Config parameters:
            duplicate_ratio (float): Fraction of items to duplicate (default 0.2).
                Value between 0.0 and 1.0.
        """
        # Get duplicate ratio from config
        duplicate_ratio = self._config.parameters.get("duplicate_ratio", 0.2)
        duplicate_ratio = max(0.0, min(1.0, duplicate_ratio))  # Clamp to [0, 1]

        # Extract the list of items to concatenate
        items = self._extract_items(args, kwargs)

        if items is None or len(items) == 0:
            # No items to process, call original
            return self._original(*args, **kwargs)

        # Calculate how many items to duplicate
        num_to_duplicate = max(0, int(len(items) * duplicate_ratio))

        if num_to_duplicate > 0:
            # Randomly select indices to duplicate
            indices_to_duplicate = random.sample(
                range(len(items)), min(num_to_duplicate, len(items))
            )
            duplicated_items = list(items)
            for idx in indices_to_duplicate:
                duplicated_items.append(items[idx])
        else:
            duplicated_items = list(items)

        # Call original with duplicated items
        return self._call_with_modified_items(args, kwargs, duplicated_items)

    def _strategy_wrong_order(self, *args: Any, **kwargs: Any) -> Any:
        """
        Shuffle the order of input DataProtos.

        Randomizes the order of input items before calling the original concat.
        This simulates scenarios where data ordering is corrupted or lost.

        Args:
            *args: Positional arguments, first arg expected to be list of DataProtos
            **kwargs: Keyword arguments to pass to original concat

        Returns:
            Result of concatenating items in shuffled order.

        Config parameters:
            seed (int, optional): Random seed for shuffle reproducibility.
        """
        # Get optional seed from config
        seed = self._config.parameters.get("seed", None)

        # Extract the list of items to concatenate
        items = self._extract_items(args, kwargs)

        if items is None or len(items) <= 1:
            # Nothing to shuffle, call original
            return self._original(*args, **kwargs)

        # Shuffle items
        shuffled_items = list(items)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(shuffled_items)
        else:
            random.shuffle(shuffled_items)

        # Call original with shuffled items
        return self._call_with_modified_items(args, kwargs, shuffled_items)

    def _extract_items(self, args: tuple, kwargs: dict) -> Optional[List[Any]]:
        """
        Extract the list of items to concatenate from arguments.

        Handles various calling conventions:
        - concat([item1, item2, ...])
        - concat(item1, item2, ...)
        - concat(data_protos=[item1, item2, ...])

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            List of items to concatenate, or None if not found.
        """
        # Check for keyword argument 'data_protos' or similar
        for key in ["data_protos", "protos", "items", "data"]:
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, (list, tuple)):
                    return list(val)

        # Check first positional argument
        if args:
            first_arg = args[0]
            if isinstance(first_arg, (list, tuple)):
                return list(first_arg)
            # Could be multiple positional arguments
            if len(args) > 1:
                return list(args)

        return None

    def _call_with_modified_items(
        self, original_args: tuple, original_kwargs: dict, modified_items: List[Any]
    ) -> Any:
        """
        Call the original function with modified items.

        Reconstructs the arguments with the modified item list.

        Args:
            original_args: Original positional arguments
            original_kwargs: Original keyword arguments
            modified_items: Modified list of items

        Returns:
            Result from calling original with modified items.
        """
        # Check if items were passed via keyword
        for key in ["data_protos", "protos", "items", "data"]:
            if key in original_kwargs:
                new_kwargs = dict(original_kwargs)
                new_kwargs[key] = modified_items
                return self._original(*original_args, **new_kwargs)

        # Items were in positional args
        if original_args:
            first_arg = original_args[0]
            if isinstance(first_arg, (list, tuple)):
                # First arg was the list
                return self._original(modified_items, *original_args[1:], **original_kwargs)
            else:
                # Multiple positional args
                return self._original(*modified_items, **original_kwargs)

        # Fallback: pass as first positional arg
        return self._original(modified_items, **original_kwargs)
