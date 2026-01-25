"""
L2 verl proxies for Ralph fault injection framework.

Contains proxy classes for verl-specific operations like RewardManager, FSDPCheckpointManager, etc.
"""

from typing import Any, Optional

import torch

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.mixins.skip import SkipMixin
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

    SUPPORTED_STRATEGIES: set[StrategyType] = {
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


@ProxyRegistry.register("FSDPCheckpointManager.save_checkpoint")
class CheckpointSaveProxy(BaseProxy, DelayMixin, ExceptionMixin):
    """
    Proxy for FSDPCheckpointManager.save_checkpoint() operations.

    Supports fault injection at the verl checkpoint saving level (L2).

    Supported strategies:
    - DELAY: Adds delay before calling original save_checkpoint
    - RAISE_EXCEPTION: Raises configurable exception instead of saving

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - RAISE_EXCEPTION: exc_type (str, required) - exception type to raise,
                       message (str, optional) - error message

    Expected input/output:
    - Input: local_path (str), hdfs_path (str/optional), global_step (int), max_ckpt (int)
    - Output: None or dict with checkpoint metadata
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L2"

    def _strategy_raise_exception(
        self,
        local_path: Optional[str] = None,
        hdfs_path: Optional[str] = None,
        global_step: Optional[int] = None,
        max_ckpt: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Raise an exception instead of saving checkpoint.

        This strategy is a terminal operation - the original save_checkpoint
        is never called when this strategy triggers.

        Args:
            local_path: Local filesystem path for checkpoint
            hdfs_path: HDFS path for checkpoint (optional)
            global_step: Current training step
            max_ckpt: Maximum number of checkpoints to keep
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            The exception type specified in config with checkpoint-specific context.

        Config parameters:
            exc_type (str): The exception type name (required). Valid types include:
                IOError, RuntimeError, PermissionError, OSError, FileNotFoundError, etc.
            message (str): Custom error message (optional). If not provided, a
                checkpoint-specific message is generated.
        """
        exc_type = self._config.parameters.get("exc_type")
        if exc_type is None:
            raise ValueError("exc_type must be specified in config parameters")

        # Build checkpoint-specific default message with context
        default_message = (
            f"Fault injection: {exc_type} during checkpoint save (local_path={local_path}, global_step={global_step})"
        )
        message = self._config.parameters.get("message", default_message)

        self._raise_exception(exc_type, message)


@ProxyRegistry.register("FSDPCheckpointManager.load_checkpoint")
class CheckpointLoadProxy(BaseProxy, TensorCorruptionMixin, ExceptionMixin):
    """
    Proxy for FSDPCheckpointManager.load_checkpoint() operations.

    Supports fault injection at the verl checkpoint loading level (L2).

    Supported strategies:
    - RAISE_EXCEPTION: Raises configurable exception instead of loading
    - FILE_NOT_FOUND: Raises FileNotFoundError simulating missing checkpoint
    - CORRUPT_STATE_DICT: Adds noise to loaded parameters
    - PARTIAL_LOAD: Removes some keys from loaded state dict

    Config parameters:
    - RAISE_EXCEPTION: exc_type (str, required), message (str, optional)
    - FILE_NOT_FOUND: message (str, optional)
    - CORRUPT_STATE_DICT: noise_scale (float, default 0.01)
    - PARTIAL_LOAD: drop_ratio (float, default 0.1), drop_keys (list, optional)

    Expected input/output:
    - Input: checkpoint_path (str), model, optimizer (optional), etc.
    - Output: Typically returns dict with state_dict or loaded model state
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.RAISE_EXCEPTION,
        StrategyType.FILE_NOT_FOUND,
        StrategyType.CORRUPT_STATE_DICT,
        StrategyType.PARTIAL_LOAD,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L2"

    def _strategy_raise_exception(
        self,
        checkpoint_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Raise an exception instead of loading checkpoint.

        This strategy is a terminal operation - the original load_checkpoint
        is never called when this strategy triggers.

        Args:
            checkpoint_path: Path to the checkpoint file/directory
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            The exception type specified in config with checkpoint-specific context.

        Config parameters:
            exc_type (str): The exception type name (required).
            message (str): Custom error message (optional).
        """
        exc_type = self._config.parameters.get("exc_type")
        if exc_type is None:
            raise ValueError("exc_type must be specified in config parameters")

        default_message = f"Fault injection: {exc_type} during checkpoint load (checkpoint_path={checkpoint_path})"
        message = self._config.parameters.get("message", default_message)

        self._raise_exception(exc_type, message)

    def _strategy_file_not_found(
        self,
        checkpoint_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Raise FileNotFoundError simulating a missing checkpoint.

        This strategy simulates the scenario where a checkpoint file
        is expected but does not exist, which can happen due to storage
        failures, incorrect paths, or incomplete writes.

        Args:
            checkpoint_path: Path to the checkpoint file/directory
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            FileNotFoundError: Always raised with checkpoint path context.

        Config parameters:
            message (str): Custom error message (optional).
        """
        default_message = f"Checkpoint not found: {checkpoint_path}"
        message = self._config.parameters.get("message", default_message)

        raise FileNotFoundError(message)

    def _strategy_corrupt_state_dict(
        self,
        checkpoint_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Load checkpoint but add noise to all tensor parameters.

        This strategy simulates corrupted checkpoint files or storage
        bit flips by adding Gaussian noise to all tensor values in
        the loaded state dict.

        Args:
            checkpoint_path: Path to the checkpoint file/directory
            *args: Additional positional arguments for original
            **kwargs: Additional keyword arguments for original

        Returns:
            State dict with all tensor values corrupted by noise.

        Config parameters:
            noise_scale (float): Standard deviation of Gaussian noise (default 0.01).
        """
        # Call original to get results
        result = self._original(checkpoint_path, *args, **kwargs)

        # Get noise scale from config (default lower than typical to be subtle)
        noise_scale = self._config.parameters.get("noise_scale", 0.01)

        # Corrupt all tensors in the result
        return self._corrupt_state_dict_tensors(result, noise_scale)

    def _corrupt_state_dict_tensors(self, obj: Any, noise_scale: float) -> Any:
        """
        Recursively corrupt tensor values in a state dict.

        Args:
            obj: Object to corrupt (dict, list, tuple, or tensor)
            noise_scale: Standard deviation of Gaussian noise

        Returns:
            Object with all tensors corrupted
        """
        if isinstance(obj, torch.Tensor):
            return self._corrupt_tensor(obj, noise_scale)
        elif isinstance(obj, dict):
            return {k: self._corrupt_state_dict_tensors(v, noise_scale) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._corrupt_state_dict_tensors(item, noise_scale) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._corrupt_state_dict_tensors(item, noise_scale) for item in obj)
        else:
            return obj

    def _strategy_partial_load(
        self,
        checkpoint_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Load checkpoint but remove some keys from the state dict.

        This strategy simulates incomplete checkpoint files or version
        mismatches where some expected keys are missing from the state dict.

        Args:
            checkpoint_path: Path to the checkpoint file/directory
            *args: Additional positional arguments for original
            **kwargs: Additional keyword arguments for original

        Returns:
            State dict with some keys removed.

        Config parameters:
            drop_ratio (float): Fraction of keys to drop randomly (default 0.1).
            drop_keys (list): Specific key names to drop (optional).
                If provided, drop_ratio is ignored.
        """

        # Call original to get results
        result = self._original(checkpoint_path, *args, **kwargs)

        # Get config parameters
        drop_keys = self._config.parameters.get("drop_keys")
        drop_ratio = self._config.parameters.get("drop_ratio", 0.1)

        # Apply partial load to the result
        return self._apply_partial_load(result, drop_keys, drop_ratio)

    def _apply_partial_load(
        self,
        obj: Any,
        drop_keys: Optional[list],
        drop_ratio: float,
    ) -> Any:
        """
        Remove keys from a state dict structure.

        Args:
            obj: Object to process (dict, list, tuple, or other)
            drop_keys: Specific keys to drop (if None, use drop_ratio)
            drop_ratio: Fraction of keys to drop randomly

        Returns:
            Object with specified keys removed
        """
        import random

        if isinstance(obj, dict):
            if drop_keys is not None:
                # Drop specific keys
                return {
                    k: self._apply_partial_load(v, drop_keys, drop_ratio) for k, v in obj.items() if k not in drop_keys
                }
            else:
                # Drop random fraction of keys
                keys = list(obj.keys())
                num_to_drop = max(1, int(len(keys) * drop_ratio))
                keys_to_drop = set(random.sample(keys, min(num_to_drop, len(keys))))
                return {
                    k: self._apply_partial_load(v, drop_keys, drop_ratio)
                    for k, v in obj.items()
                    if k not in keys_to_drop
                }
        elif isinstance(obj, list):
            return [self._apply_partial_load(item, drop_keys, drop_ratio) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._apply_partial_load(item, drop_keys, drop_ratio) for item in obj)
        else:
            return obj


@ProxyRegistry.register("RayPPOTrainer._update_actor")
class UpdateActorProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, SkipMixin):
    """
    Proxy for RayPPOTrainer._update_actor() operations.

    Supports fault injection at the verl actor update level (L2).

    Supported strategies:
    - DELAY: Adds delay before calling original _update_actor
    - NAN_INPUT: Injects NaN values into input batch before update
    - EXPLODING_GRADIENTS: Scales input by 1e6 to simulate gradient explosion
    - VANISHING_GRADIENTS: Scales input by 1e-8 to simulate gradient vanishing
    - SKIP_UPDATE: Skips the update entirely, returning without calling original
    - DOUBLE_UPDATE: Calls the update twice with the same batch

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - NAN_INPUT: nan_ratio (float, default 0.1) - fraction of values to set to NaN
    - EXPLODING_GRADIENTS: scale (float, default 1e6) - multiplier for input
    - VANISHING_GRADIENTS: scale (float, default 1e-8) - multiplier for input
    - SKIP_UPDATE: default_return (Any, optional) - value to return (default None)
    - DOUBLE_UPDATE: (no specific parameters)

    Expected input/output:
    - Input: batch (DataProto or dict), other parameters for actor update
    - Output: Typically returns dict with update metrics (loss, grad_norm, etc.)
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.NAN_INPUT,
        StrategyType.EXPLODING_GRADIENTS,
        StrategyType.VANISHING_GRADIENTS,
        StrategyType.SKIP_UPDATE,
        StrategyType.DOUBLE_UPDATE,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L2"

    def _strategy_nan_input(
        self,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Inject NaN values into the input batch before update.

        This strategy corrupts the input data by injecting NaN values,
        which can cause numerical instability during the actor update.
        The original function is called with the corrupted batch.

        Args:
            batch: Input batch (DataProto or dict containing tensors)
            *args: Additional positional arguments to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from original _update_actor with corrupted input.

        Config parameters:
            nan_ratio (float): Fraction of tensor values to set to NaN (default 0.1).
        """
        nan_ratio = self._config.parameters.get("nan_ratio", 0.1)
        corrupted_batch = self._inject_nan_into_batch(batch, nan_ratio)
        return self._original(corrupted_batch, *args, **kwargs)

    def _inject_nan_into_batch(self, obj: Any, nan_ratio: float) -> Any:
        """
        Recursively inject NaN values into tensors in a batch.

        Args:
            obj: Object to corrupt (dict, list, tuple, or tensor)
            nan_ratio: Fraction of values to set to NaN

        Returns:
            Object with NaN values injected into tensors
        """
        if isinstance(obj, torch.Tensor):
            return self._inject_nan(obj, nan_ratio)
        elif isinstance(obj, dict):
            return {k: self._inject_nan_into_batch(v, nan_ratio) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._inject_nan_into_batch(item, nan_ratio) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._inject_nan_into_batch(item, nan_ratio) for item in obj)
        else:
            return obj

    def _strategy_exploding_gradients(
        self,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Scale input by a large factor to simulate gradient explosion.

        This strategy multiplies all tensor values in the input batch
        by a very large factor (default 1e6), which can cause gradient
        explosion during the backward pass.

        Args:
            batch: Input batch (DataProto or dict containing tensors)
            *args: Additional positional arguments to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from original _update_actor with scaled input.

        Config parameters:
            scale (float): Multiplier for tensor values (default 1e6).
        """
        scale = self._config.parameters.get("scale", 1e6)
        scaled_batch = self._scale_batch_tensors(batch, scale)
        return self._original(scaled_batch, *args, **kwargs)

    def _strategy_vanishing_gradients(
        self,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Scale input by a tiny factor to simulate gradient vanishing.

        This strategy multiplies all tensor values in the input batch
        by a very small factor (default 1e-8), which can cause gradient
        vanishing during the backward pass.

        Args:
            batch: Input batch (DataProto or dict containing tensors)
            *args: Additional positional arguments to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from original _update_actor with scaled input.

        Config parameters:
            scale (float): Multiplier for tensor values (default 1e-8).
        """
        scale = self._config.parameters.get("scale", 1e-8)
        scaled_batch = self._scale_batch_tensors(batch, scale)
        return self._original(scaled_batch, *args, **kwargs)

    def _scale_batch_tensors(self, obj: Any, scale: float) -> Any:
        """
        Recursively scale all tensors in a batch.

        Args:
            obj: Object to scale (dict, list, tuple, or tensor)
            scale: Multiplier for tensor values

        Returns:
            Object with all tensors scaled
        """
        if isinstance(obj, torch.Tensor):
            return self._scale_tensor(obj, scale)
        elif isinstance(obj, dict):
            return {k: self._scale_batch_tensors(v, scale) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._scale_batch_tensors(item, scale) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._scale_batch_tensors(item, scale) for item in obj)
        else:
            return obj

    def _strategy_skip_update(
        self,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Skip the actor update entirely.

        This strategy returns without calling the original _update_actor,
        effectively skipping a training step. This can simulate dropped
        updates due to failures or intentional skipping.

        Args:
            batch: Input batch (ignored)
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            The default_return value from config, or None if not specified.

        Config parameters:
            default_return (Any): Value to return (optional, defaults to None).
        """
        default_return = self._config.parameters.get("default_return", None)
        return self._skip_operation(default_return)

    def _strategy_double_update(
        self,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Call the actor update twice with the same batch.

        This strategy calls the original _update_actor twice with the
        same input batch, simulating duplicate updates that could occur
        due to retry logic bugs or network issues.

        Args:
            batch: Input batch (DataProto or dict)
            *args: Additional positional arguments to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Result from the second call to _update_actor.
        """
        # Call the first time (result discarded)
        self._original(batch, *args, **kwargs)
        # Call the second time (result returned)
        return self._original(batch, *args, **kwargs)
