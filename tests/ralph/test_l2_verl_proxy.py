"""
Unit tests for L2 verl proxies.

Tests cover RewardManagerProxy with all 4 supported strategies:
- DELAY: Adds delay before calling original RewardManager
- CORRUPT_TENSOR: Corrupts tensor data with Gaussian noise
- REWARD_FLIP: Negates all reward values
- CONSTANT_REWARD: Returns constant value for all rewards

Tests cover CheckpointSaveProxy with 2 supported strategies:
- DELAY: Adds delay before calling original save_checkpoint
- RAISE_EXCEPTION: Raises configurable exception instead of saving

Tests cover CheckpointLoadProxy with 4 supported strategies:
- RAISE_EXCEPTION: Raises configurable exception instead of loading
- FILE_NOT_FOUND: Raises FileNotFoundError simulating missing checkpoint
- CORRUPT_STATE_DICT: Adds noise to loaded parameters
- PARTIAL_LOAD: Removes some keys from loaded state dict

Tests cover UpdateActorProxy with 6 supported strategies:
- DELAY: Adds delay before calling original _update_actor
- NAN_INPUT: Injects NaN values into input batch
- EXPLODING_GRADIENTS: Scales input by 1e6 to simulate gradient explosion
- VANISHING_GRADIENTS: Scales input by 1e-8 to simulate gradient vanishing
- SKIP_UPDATE: Skips the update entirely
- DOUBLE_UPDATE: Calls the update twice with the same batch
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.l2_verl import (
    CheckpointLoadProxy,
    CheckpointSaveProxy,
    RewardManagerProxy,
    UpdateActorProxy,
)


class TestRewardManagerProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_reward_manager_target(self):
        """RewardManagerProxy is registered for 'RewardManager.__call__' target."""
        assert ProxyRegistry.is_registered("RewardManager.__call__")
        assert ProxyRegistry.get_proxy("RewardManager.__call__") is RewardManagerProxy

    def test_supported_strategies(self):
        """RewardManagerProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("RewardManager.__call__")
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.REWARD_FLIP,
            StrategyType.CONSTANT_REWARD,
        }
        assert strategies == expected


class TestRewardManagerProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_l2(self):
        """_get_layer returns 'L2'."""
        proxy = RewardManagerProxy(lambda x: x)
        assert proxy._get_layer() == "L2"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        data = {"prompts": "test"}
        result = proxy(data, return_dict=True)
        original.assert_called_once_with(data, return_dict=True)
        assert "rewards" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        data = {"prompts": "test"}
        result = proxy(data)
        original.assert_called_once()
        assert "rewards" in result

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = RewardManagerProxy(lambda x: x)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by RewardManagerProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},  # Very short delay for tests
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy({"prompts": "test"})
            mock_sleep.assert_called_once_with(0.01)
            assert "rewards" in result

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},  # No delay_seconds
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy({"prompts": "test"})
            mock_sleep.assert_called_once_with(10.0)  # Default is 10 seconds

    def test_delay_passes_return_dict_kwarg(self):
        """DELAY strategy passes return_dict to original."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy({"prompts": "test"}, return_dict=False)
            original.assert_called_once()
            _, kwargs = original.call_args
            # Note: DelayMixin passes kwargs directly, so return_dict should be in kwargs
            assert kwargs.get("return_dict") is False or "return_dict" not in kwargs


class TestCorruptTensorStrategy:
    """Tests for CORRUPT_TENSOR strategy."""

    def test_corrupt_tensor_single_tensor_in_dict(self):
        """CORRUPT_TENSOR strategy corrupts single tensor in dict."""
        rewards_tensor = torch.ones(10)
        original = MagicMock(return_value={"rewards": rewards_tensor.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        # Result should have corrupted rewards
        assert "rewards" in result
        assert result["rewards"].shape == rewards_tensor.shape
        # With noise_scale=1.0, result should be different from original
        assert not torch.allclose(result["rewards"], rewards_tensor)

    def test_corrupt_tensor_multiple_tensors(self):
        """CORRUPT_TENSOR strategy corrupts multiple tensors in result."""
        original_result = {
            "rewards": torch.ones(5),
            "values": torch.zeros(5),
            "log_probs": torch.full((5,), -1.0),
        }
        original = MagicMock(return_value={k: v.clone() for k, v in original_result.items()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-multi",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        # All tensors should be corrupted
        assert len(result) == 3
        for key in ["rewards", "values", "log_probs"]:
            assert key in result
            assert result[key].shape == original_result[key].shape

    def test_corrupt_tensor_default_noise_scale(self):
        """CORRUPT_TENSOR strategy uses default noise scale."""
        rewards_tensor = torch.zeros(100)
        original = MagicMock(return_value={"rewards": rewards_tensor.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-default",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={},  # No noise_scale, default is 0.1
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        # With default noise_scale=0.1, std should be approximately 0.1
        assert result["rewards"].std().item() < 0.5  # Should be around 0.1 but allow variance

    def test_corrupt_tensor_preserves_non_tensors(self):
        """CORRUPT_TENSOR strategy preserves non-tensor values."""
        original_result = {
            "rewards": torch.ones(5),
            "batch_size": 32,
            "model_name": "test_model",
            "metadata": {"step": 100},
        }
        original = MagicMock(return_value=original_result.copy())
        # Need to clone the tensor
        original.return_value["rewards"] = original_result["rewards"].clone()
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-preserve",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result["batch_size"] == 32
        assert result["model_name"] == "test_model"
        assert result["metadata"] == {"step": 100}


class TestRewardFlipStrategy:
    """Tests for REWARD_FLIP strategy."""

    def test_reward_flip_single_tensor(self):
        """REWARD_FLIP strategy negates single tensor."""
        rewards = torch.tensor([1.0, 2.0, -3.0, 0.5])
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-flip",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        expected = torch.tensor([-1.0, -2.0, 3.0, -0.5])
        assert torch.allclose(result["rewards"], expected)

    def test_reward_flip_multiple_tensors(self):
        """REWARD_FLIP strategy negates multiple tensors."""
        original_result = {
            "rewards": torch.tensor([1.0, -2.0]),
            "advantages": torch.tensor([0.5, -0.5]),
        }
        original = MagicMock(return_value={k: v.clone() for k, v in original_result.items()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-flip-multi",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert torch.allclose(result["rewards"], torch.tensor([-1.0, 2.0]))
        assert torch.allclose(result["advantages"], torch.tensor([-0.5, 0.5]))

    def test_reward_flip_nested_structure(self):
        """REWARD_FLIP strategy handles nested structures."""
        original_result = {
            "outer": {
                "inner_rewards": torch.tensor([1.0, 2.0]),
                "scalar": 5.0,
            },
            "list_rewards": [torch.tensor([3.0]), torch.tensor([-4.0])],
        }
        original = MagicMock(return_value={
            "outer": {
                "inner_rewards": original_result["outer"]["inner_rewards"].clone(),
                "scalar": 5.0,
            },
            "list_rewards": [t.clone() for t in original_result["list_rewards"]],
        })
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-flip-nested",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert torch.allclose(result["outer"]["inner_rewards"], torch.tensor([-1.0, -2.0]))
        assert result["outer"]["scalar"] == -5.0
        assert torch.allclose(result["list_rewards"][0], torch.tensor([-3.0]))
        assert torch.allclose(result["list_rewards"][1], torch.tensor([4.0]))

    def test_reward_flip_preserves_non_numeric(self):
        """REWARD_FLIP strategy preserves non-numeric values."""
        original_result = {
            "rewards": torch.tensor([1.0]),
            "name": "test",
            "data": {"nested": "value"},
            "metadata": None,
        }
        original = MagicMock(return_value={
            "rewards": original_result["rewards"].clone(),
            "name": "test",
            "data": {"nested": "value"},
            "metadata": None,
        })
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-flip-preserve",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result["name"] == "test"
        assert result["data"]["nested"] == "value"
        assert result["metadata"] is None

    def test_reward_flip_scalar_result(self):
        """REWARD_FLIP strategy handles scalar numeric result."""
        original = MagicMock(return_value=5.0)
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-flip-scalar",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result == -5.0


class TestConstantRewardStrategy:
    """Tests for CONSTANT_REWARD strategy."""

    def test_constant_reward_single_tensor(self):
        """CONSTANT_REWARD strategy sets single tensor to constant."""
        rewards = torch.tensor([1.0, 2.0, -3.0, 0.5])
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        expected = torch.full((4,), 0.5)
        assert torch.allclose(result["rewards"], expected)

    def test_constant_reward_default_zero(self):
        """CONSTANT_REWARD strategy uses default value of 0.0."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-default",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={},  # No constant_value, default is 0.0
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        expected = torch.zeros(3)
        assert torch.allclose(result["rewards"], expected)

    def test_constant_reward_negative_value(self):
        """CONSTANT_REWARD strategy supports negative constant."""
        rewards = torch.tensor([1.0, 2.0])
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-negative",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": -1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        expected = torch.full((2,), -1.0)
        assert torch.allclose(result["rewards"], expected)

    def test_constant_reward_multiple_tensors(self):
        """CONSTANT_REWARD strategy sets all tensors to constant."""
        original_result = {
            "rewards": torch.tensor([1.0, -2.0]),
            "values": torch.tensor([0.5, -0.5]),
        }
        original = MagicMock(return_value={k: v.clone() for k, v in original_result.items()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-multi",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert torch.allclose(result["rewards"], torch.ones(2))
        assert torch.allclose(result["values"], torch.ones(2))

    def test_constant_reward_preserves_shape(self):
        """CONSTANT_REWARD strategy preserves tensor shapes."""
        rewards = torch.randn(8, 16)
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-shape",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result["rewards"].shape == (8, 16)
        assert torch.all(result["rewards"] == 0.0)

    def test_constant_reward_scalar(self):
        """CONSTANT_REWARD strategy handles scalar results."""
        original = MagicMock(return_value=5.0)
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-scalar",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": 42.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result == 42.0

    def test_constant_reward_preserves_dtype(self):
        """CONSTANT_REWARD strategy preserves tensor dtype."""
        rewards = torch.tensor([1, 2, 3], dtype=torch.int64)
        original = MagicMock(return_value={"rewards": rewards.clone()})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-constant-dtype",
            strategy=StrategyType.CONSTANT_REWARD,
            trigger=trigger,
            parameters={"constant_value": 5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy({"prompts": "test"})

        assert result["rewards"].dtype == torch.int64


class TestRewardManagerProxyIntegration:
    """Integration tests for RewardManagerProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0-4 should return positive rewards
        for step in range(5):
            proxy.set_step(step)
            original.return_value = {"rewards": torch.ones(5)}
            result = proxy({"prompts": "test"})
            assert torch.all(result["rewards"] > 0)

        # Step 5 should flip rewards
        proxy.set_step(5)
        original.return_value = {"rewards": torch.ones(5)}
        result = proxy({"prompts": "test"})
        assert torch.all(result["rewards"] < 0)

        # Step 6 should return positive again (one-shot)
        proxy.set_step(6)
        original.return_value = {"rewards": torch.ones(5)}
        result = proxy({"prompts": "test"})
        assert torch.all(result["rewards"] > 0)

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Track which steps trigger flipping
        flip_steps = []
        for step in range(10):
            proxy.set_step(step)
            original.return_value = {"rewards": torch.ones(5)}
            result = proxy({"prompts": "test"})
            if torch.all(result["rewards"] < 0):
                flip_steps.append(step)

        # Steps 0, 3, 6, 9 should trigger (step % 3 == 0)
        assert flip_steps == [0, 3, 6, 9]

    def test_step_based_trigger(self):
        """Step-based trigger triggers within range."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=3, end_step=6)
        config = FaultConfig(
            id="test-step-based",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)

        flip_steps = []
        for step in range(10):
            proxy.set_step(step)
            original.return_value = {"rewards": torch.ones(5)}
            result = proxy({"prompts": "test"})
            if torch.all(result["rewards"] < 0):
                flip_steps.append(step)

        # Steps 3-6 inclusive should trigger
        assert flip_steps == [3, 4, 5, 6]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-456"
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
            severity="high",
            expected_behavior="Should flip all rewards",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy({"prompts": "test"})

        # Check injection was recorded
        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "reward_flip"
        assert call_kwargs["target_layer"] == "L2"
        assert call_kwargs["severity"] == "high"
        assert call_kwargs["expected_behavior"] == "Should flip all rewards"

        # Check outcome was recorded
        collector.record_fault_outcome.assert_called_once()
        args = collector.record_fault_outcome.call_args[0]
        assert args[0] == "fault-456"
        assert args[1] == "success"

    def test_exception_in_strategy_recorded(self):
        """Exceptions during strategy execution are recorded."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-789"
        # Make original raise an exception
        original = MagicMock(side_effect=RuntimeError("Test error"))
        proxy = RewardManagerProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-exception",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError):
            proxy({"prompts": "test"})

        # Check outcome recorded exception
        collector.record_fault_outcome.assert_called_once()
        args = collector.record_fault_outcome.call_args[0]
        assert args[0] == "fault-789"
        assert "exception" in args[1]
        assert "RuntimeError" in args[1]

    def test_passes_additional_kwargs(self):
        """Proxy passes additional kwargs to original."""
        original = MagicMock(return_value={"rewards": torch.ones(5)})
        proxy = RewardManagerProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-kwargs",
            strategy=StrategyType.REWARD_FLIP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy({"prompts": "test"}, return_dict=True, custom_param=42)

        # Original should be called with all kwargs
        call_args = original.call_args
        assert call_args[1]["return_dict"] is True
        assert call_args[1]["custom_param"] == 42


# =============================================================================
# CheckpointSaveProxy Tests
# =============================================================================


class TestCheckpointSaveProxyRegistration:
    """Tests for CheckpointSaveProxy registration."""

    def test_registered_with_checkpoint_manager_target(self):
        """CheckpointSaveProxy is registered for 'FSDPCheckpointManager.save_checkpoint' target."""
        assert ProxyRegistry.is_registered("FSDPCheckpointManager.save_checkpoint")
        assert ProxyRegistry.get_proxy("FSDPCheckpointManager.save_checkpoint") is CheckpointSaveProxy

    def test_supported_strategies(self):
        """CheckpointSaveProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("FSDPCheckpointManager.save_checkpoint")
        expected = {
            StrategyType.DELAY,
            StrategyType.RAISE_EXCEPTION,
        }
        assert strategies == expected


class TestCheckpointSaveProxyBasics:
    """Tests for basic CheckpointSaveProxy functionality."""

    def test_get_layer_returns_l2(self):
        """_get_layer returns 'L2'."""
        proxy = CheckpointSaveProxy(lambda *args, **kwargs: None)
        assert proxy._get_layer() == "L2"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        proxy("/path/to/local", "/path/to/hdfs", global_step=100, max_ckpt=5)
        original.assert_called_once_with("/path/to/local", "/path/to/hdfs", global_step=100, max_ckpt=5)

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        proxy("/path/to/local", global_step=100)
        original.assert_called_once()

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = CheckpointSaveProxy(lambda *args, **kwargs: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.REWARD_FLIP,  # Not supported by CheckpointSaveProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestCheckpointSaveDelayStrategy:
    """Tests for CheckpointSaveProxy DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy("/path/to/checkpoint", global_step=50)
            mock_sleep.assert_called_once_with(0.01)
            original.assert_called_once()

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-default",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy("/path/to/checkpoint")
            mock_sleep.assert_called_once_with(10.0)  # Default is 10 seconds

    def test_delay_passes_all_kwargs(self):
        """DELAY strategy passes all arguments to original."""
        original = MagicMock(return_value={"status": "saved"})
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-kwargs",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            result = proxy(
                "/local/path",
                "/hdfs/path",
                global_step=1000,
                max_ckpt=10,
                extra_option=True,
            )
            original.assert_called_once()
            args, kwargs = original.call_args
            assert args[0] == "/local/path"
            assert args[1] == "/hdfs/path"
            assert kwargs["global_step"] == 1000
            assert kwargs["max_ckpt"] == 10
            assert kwargs["extra_option"] is True


class TestCheckpointSaveRaiseExceptionStrategy:
    """Tests for CheckpointSaveProxy RAISE_EXCEPTION strategy."""

    def test_raise_exception_ioerror(self):
        """RAISE_EXCEPTION strategy raises IOError."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-ioerror",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "IOError", "message": "Disk full"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(IOError) as exc_info:
            proxy("/path/to/checkpoint", global_step=100)
        assert "Disk full" in str(exc_info.value)
        original.assert_not_called()

    def test_raise_exception_permission_error(self):
        """RAISE_EXCEPTION strategy raises PermissionError."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-permission",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "PermissionError", "message": "Access denied"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(PermissionError) as exc_info:
            proxy("/protected/path")
        assert "Access denied" in str(exc_info.value)

    def test_raise_exception_oserror(self):
        """RAISE_EXCEPTION strategy raises OSError."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-oserror",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "OSError", "message": "System error"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(OSError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "System error" in str(exc_info.value)

    def test_raise_exception_runtime_error(self):
        """RAISE_EXCEPTION strategy raises RuntimeError."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-runtime",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError", "message": "Serialization failed"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "Serialization failed" in str(exc_info.value)

    def test_raise_exception_default_message(self):
        """RAISE_EXCEPTION strategy uses checkpoint-specific default message."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-default-msg",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "IOError"},  # No message provided
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(IOError) as exc_info:
            proxy("/local/checkpoint", global_step=500)
        error_msg = str(exc_info.value)
        assert "IOError" in error_msg
        assert "checkpoint" in error_msg.lower()
        assert "/local/checkpoint" in error_msg
        assert "500" in error_msg

    def test_raise_exception_missing_exc_type(self):
        """RAISE_EXCEPTION strategy raises ValueError if exc_type not provided."""
        original = MagicMock()
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-type",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={},  # No exc_type
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "exc_type" in str(exc_info.value)


class TestCheckpointSaveProxyIntegration:
    """Integration tests for CheckpointSaveProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "IOError", "message": "Injection"},
        )
        proxy.set_config(config)

        # Steps 0-4 should call original normally
        for step in range(5):
            proxy.set_step(step)
            original.reset_mock()
            proxy("/path", global_step=step)
            original.assert_called_once()

        # Step 5 should raise exception
        proxy.set_step(5)
        with pytest.raises(IOError):
            proxy("/path", global_step=5)

        # Step 6 should call original again (one-shot already fired)
        proxy.set_step(6)
        original.reset_mock()
        proxy("/path", global_step=6)
        original.assert_called_once()

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "IOError"},
        )
        proxy.set_config(config)

        # Track which steps trigger exception
        exception_steps = []
        for step in range(10):
            proxy.set_step(step)
            try:
                proxy("/path", global_step=step)
            except IOError:
                exception_steps.append(step)

        # Steps 0, 3, 6, 9 should trigger (step % 3 == 0)
        assert exception_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-ckpt-001"
        original = MagicMock(return_value=None)
        proxy = CheckpointSaveProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            severity="critical",
            expected_behavior="Should fail checkpoint save",
            parameters={"exc_type": "IOError", "message": "Disk failure"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(IOError):
            proxy("/path/to/checkpoint", global_step=100)

        # Check injection was recorded
        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "raise_exception"
        assert call_kwargs["target_layer"] == "L2"
        assert call_kwargs["severity"] == "critical"
        assert call_kwargs["expected_behavior"] == "Should fail checkpoint save"

        # Check outcome was recorded
        collector.record_fault_outcome.assert_called_once()
        args = collector.record_fault_outcome.call_args[0]
        assert args[0] == "fault-ckpt-001"
        # The outcome should indicate the exception was raised (success for injection)
        assert "exception" in args[1] or args[1] == "success"

    def test_delay_then_original_succeeds(self):
        """DELAY strategy delays then calls original successfully."""
        result_data = {"checkpoint_path": "/saved/path", "step": 100}
        original = MagicMock(return_value=result_data)
        proxy = CheckpointSaveProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-success",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy("/local/path", global_step=100)
            mock_sleep.assert_called_once()
            assert result == result_data
            original.assert_called_once()


# =============================================================================
# CheckpointLoadProxy Tests
# =============================================================================


class TestCheckpointLoadProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_checkpoint_manager_target(self):
        """CheckpointLoadProxy is registered for 'FSDPCheckpointManager.load_checkpoint' target."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        assert ProxyRegistry.is_registered("FSDPCheckpointManager.load_checkpoint")
        assert ProxyRegistry.get_proxy("FSDPCheckpointManager.load_checkpoint") is CheckpointLoadProxy

    def test_supported_strategies(self):
        """CheckpointLoadProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("FSDPCheckpointManager.load_checkpoint")
        expected = {
            StrategyType.RAISE_EXCEPTION,
            StrategyType.FILE_NOT_FOUND,
            StrategyType.CORRUPT_STATE_DICT,
            StrategyType.PARTIAL_LOAD,
        }
        assert strategies == expected


class TestCheckpointLoadProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_l2(self):
        """_get_layer returns 'L2'."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        proxy = CheckpointLoadProxy(lambda x: x)
        assert proxy._get_layer() == "L2"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"model.weight": torch.ones(5)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        result = proxy("/path/to/checkpoint")
        original.assert_called_once_with("/path/to/checkpoint")
        assert result == state_dict

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"model.weight": torch.ones(5)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("/path/to/checkpoint")
        original.assert_called_once()
        assert result == state_dict

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        proxy = CheckpointLoadProxy(lambda x: x)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by CheckpointLoadProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestCheckpointLoadRaiseExceptionStrategy:
    """Tests for RAISE_EXCEPTION strategy."""

    def test_raise_exception_ioerror(self):
        """RAISE_EXCEPTION strategy raises IOError."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "IOError", "message": "Disk read failure"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(IOError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "Disk read failure" in str(exc_info.value)
        original.assert_not_called()

    def test_raise_exception_runtime_error(self):
        """RAISE_EXCEPTION strategy raises RuntimeError."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError", "message": "Checkpoint corrupted"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "Checkpoint corrupted" in str(exc_info.value)

    def test_raise_exception_default_message(self):
        """RAISE_EXCEPTION uses default message with checkpoint context."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "OSError"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(OSError) as exc_info:
            proxy("/checkpoints/step_100")
        assert "checkpoint load" in str(exc_info.value).lower()
        assert "/checkpoints/step_100" in str(exc_info.value)

    def test_raise_exception_missing_exc_type(self):
        """RAISE_EXCEPTION raises ValueError if exc_type not specified."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={},  # Missing exc_type
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "exc_type" in str(exc_info.value)


class TestCheckpointLoadFileNotFoundStrategy:
    """Tests for FILE_NOT_FOUND strategy."""

    def test_file_not_found_raises_error(self):
        """FILE_NOT_FOUND strategy raises FileNotFoundError."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(FileNotFoundError) as exc_info:
            proxy("/checkpoints/step_100")
        assert "/checkpoints/step_100" in str(exc_info.value)
        original.assert_not_called()

    def test_file_not_found_custom_message(self):
        """FILE_NOT_FOUND uses custom message when provided."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
            parameters={"message": "Checkpoint was deleted"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(FileNotFoundError) as exc_info:
            proxy("/path/to/checkpoint")
        assert "Checkpoint was deleted" in str(exc_info.value)

    def test_file_not_found_default_message(self):
        """FILE_NOT_FOUND includes path in default message."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(FileNotFoundError) as exc_info:
            proxy("/data/model/checkpoint_500.pt")
        assert "not found" in str(exc_info.value).lower()


class TestCheckpointLoadCorruptStateDictStrategy:
    """Tests for CORRUPT_STATE_DICT strategy."""

    def test_corrupt_state_dict_adds_noise(self):
        """CORRUPT_STATE_DICT adds noise to tensor values."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original_tensor = torch.ones(10, 10)
        state_dict = {"model.weight": original_tensor.clone()}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")
        original.assert_called_once()

        # Result should be different from original
        assert not torch.allclose(result["model.weight"], original_tensor)
        # But shape should be preserved
        assert result["model.weight"].shape == original_tensor.shape

    def test_corrupt_state_dict_default_noise_scale(self):
        """CORRUPT_STATE_DICT uses default noise scale of 0.01."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        original_tensor = torch.zeros(5, 5)
        state_dict = {"layer.bias": original_tensor.clone()}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            # No noise_scale specified, should use default 0.01
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # With small noise scale on zeros, result should be close to zero but not exactly
        assert not torch.allclose(result["layer.bias"], original_tensor, atol=0)
        # But should be small values (around 0.01 std)
        assert torch.abs(result["layer.bias"]).max() < 0.1

    def test_corrupt_state_dict_multiple_tensors(self):
        """CORRUPT_STATE_DICT corrupts all tensors in state dict."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "encoder.weight": torch.ones(5, 5),
            "encoder.bias": torch.zeros(5),
            "decoder.weight": torch.ones(10, 5),
        }
        original_values = {k: v.clone() for k, v in state_dict.items()}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # All tensors should be corrupted
        for key in original_values:
            assert not torch.allclose(result[key], original_values[key])
            assert result[key].shape == original_values[key].shape

    def test_corrupt_state_dict_nested_structure(self):
        """CORRUPT_STATE_DICT handles nested state dict structures."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "model": {
                "layer1": {"weight": torch.ones(3, 3), "bias": torch.zeros(3)},
                "layer2": {"weight": torch.ones(5, 3)},
            },
            "optimizer": {"step": 100},  # Non-tensor value
        }
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            parameters={"noise_scale": 0.2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # Nested tensors should be corrupted
        assert not torch.allclose(result["model"]["layer1"]["weight"], torch.ones(3, 3))
        assert not torch.allclose(result["model"]["layer1"]["bias"], torch.zeros(3))
        assert not torch.allclose(result["model"]["layer2"]["weight"], torch.ones(5, 3))
        # Non-tensor values should be preserved
        assert result["optimizer"]["step"] == 100

    def test_corrupt_state_dict_preserves_dtype(self):
        """CORRUPT_STATE_DICT preserves tensor dtypes."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "float32": torch.ones(5, dtype=torch.float32),
            "float64": torch.ones(5, dtype=torch.float64),
        }
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        assert result["float32"].dtype == torch.float32
        assert result["float64"].dtype == torch.float64


class TestCheckpointLoadPartialLoadStrategy:
    """Tests for PARTIAL_LOAD strategy."""

    def test_partial_load_drops_keys(self):
        """PARTIAL_LOAD drops specific keys from state dict."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "layer1.weight": torch.ones(5),
            "layer1.bias": torch.ones(5),
            "layer2.weight": torch.ones(5),
            "layer2.bias": torch.ones(5),
        }
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_LOAD,
            trigger=trigger,
            parameters={"drop_keys": ["layer1.bias", "layer2.bias"]},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        assert "layer1.weight" in result
        assert "layer2.weight" in result
        assert "layer1.bias" not in result
        assert "layer2.bias" not in result

    def test_partial_load_drop_ratio(self):
        """PARTIAL_LOAD drops random ratio of keys."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {f"layer{i}.weight": torch.ones(5) for i in range(10)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_LOAD,
            trigger=trigger,
            parameters={"drop_ratio": 0.3},  # Drop 30%
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # Should have fewer keys than original
        assert len(result) < len(state_dict)
        # Should have dropped at least 1 key (30% of 10 = 3)
        assert len(result) <= 9

    def test_partial_load_default_drop_ratio(self):
        """PARTIAL_LOAD uses default drop_ratio of 0.1."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {f"param{i}": torch.ones(5) for i in range(20)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_LOAD,
            trigger=trigger,
            # No drop_ratio specified, should use default 0.1
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # With 20 keys and 10% drop ratio, should drop at least 1-2 keys
        assert len(result) < len(state_dict)

    def test_partial_load_preserves_values(self):
        """PARTIAL_LOAD preserves values for non-dropped keys."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "keep_this": torch.tensor([1.0, 2.0, 3.0]),
            "drop_this": torch.tensor([4.0, 5.0, 6.0]),
        }
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_LOAD,
            trigger=trigger,
            parameters={"drop_keys": ["drop_this"]},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        assert "keep_this" in result
        assert torch.allclose(result["keep_this"], torch.tensor([1.0, 2.0, 3.0]))

    def test_partial_load_nested_structure(self):
        """PARTIAL_LOAD handles nested state dict structures."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {
            "model": {
                "encoder": {"weight": torch.ones(5), "bias": torch.ones(5)},
                "decoder": {"weight": torch.ones(5)},
            }
        }
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_LOAD,
            trigger=trigger,
            parameters={"drop_keys": ["bias"]},  # Drop all bias keys at any level
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # Nested bias should be dropped
        assert "weight" in result["model"]["encoder"]
        assert "bias" not in result["model"]["encoder"]
        assert "weight" in result["model"]["decoder"]


class TestCheckpointLoadProxyIntegration:
    """Integration tests for CheckpointLoadProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategies only trigger at configured step."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"weight": torch.ones(5)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0-4 should call original
        for step in range(5):
            proxy.set_step(step)
            result = proxy("/path/to/checkpoint")
            assert result == state_dict

        # Step 5 should trigger
        proxy.set_step(5)
        with pytest.raises(FileNotFoundError):
            proxy("/path/to/checkpoint")

    def test_periodic_trigger(self):
        """Periodic trigger fires at correct intervals."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"weight": torch.ones(5)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
        )
        proxy.set_config(config)

        exception_steps = []
        for step in range(10):
            proxy.set_step(step)
            try:
                proxy("/path/to/checkpoint")
            except FileNotFoundError:
                exception_steps.append(step)

        # Steps 0, 3, 6, 9 should trigger (step % 3 == 0)
        assert exception_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-load-001"
        original = MagicMock(return_value={})
        proxy = CheckpointLoadProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.FILE_NOT_FOUND,
            trigger=trigger,
            severity="high",
            expected_behavior="Should fail checkpoint load",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(FileNotFoundError):
            proxy("/path/to/checkpoint")

        # Check injection was recorded
        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "file_not_found"
        assert call_kwargs["target_layer"] == "L2"
        assert call_kwargs["severity"] == "high"
        assert call_kwargs["expected_behavior"] == "Should fail checkpoint load"

        # Check outcome was recorded
        collector.record_fault_outcome.assert_called_once()

    def test_corrupt_then_return_success(self):
        """CORRUPT_STATE_DICT corrupts then returns successfully."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"model.weight": torch.ones(10)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-success",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("/path/to/checkpoint")

        # Result should be returned (not an exception)
        assert result is not None
        assert "model.weight" in result
        original.assert_called_once()

    def test_passes_additional_kwargs(self):
        """Proxy passes additional kwargs to original function."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        state_dict = {"weight": torch.ones(5)}
        original = MagicMock(return_value=state_dict)
        proxy = CheckpointLoadProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_STATE_DICT,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(
            "/path/to/checkpoint",
            map_location="cpu",
            strict=False,
            extra_param="value",
        )

        original.assert_called_once_with(
            "/path/to/checkpoint",
            map_location="cpu",
            strict=False,
            extra_param="value",
        )


# =============================================================================
# UpdateActorProxy Tests
# =============================================================================


class TestUpdateActorProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_update_actor_target(self):
        """UpdateActorProxy is registered for 'RayPPOTrainer._update_actor' target."""
        assert ProxyRegistry.is_registered("RayPPOTrainer._update_actor")
        assert ProxyRegistry.get_proxy("RayPPOTrainer._update_actor") is UpdateActorProxy

    def test_supported_strategies(self):
        """UpdateActorProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("RayPPOTrainer._update_actor")
        expected = {
            StrategyType.DELAY,
            StrategyType.NAN_INPUT,
            StrategyType.EXPLODING_GRADIENTS,
            StrategyType.VANISHING_GRADIENTS,
            StrategyType.SKIP_UPDATE,
            StrategyType.DOUBLE_UPDATE,
        }
        assert strategies == expected


class TestUpdateActorProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_l2(self):
        """_get_layer returns 'L2'."""
        proxy = UpdateActorProxy(lambda x: x)
        assert proxy._get_layer() == "L2"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"loss": 0.5, "grad_norm": 1.0})
        proxy = UpdateActorProxy(original)
        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)
        original.assert_called_once_with(batch)
        assert result == {"loss": 0.5, "grad_norm": 1.0}

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)
        original.assert_called_once()
        assert result == {"loss": 0.5}

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = UpdateActorProxy(lambda x: x)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.REWARD_FLIP,  # Not supported by UpdateActorProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestUpdateActorDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy({"input_ids": torch.ones(2, 10)})
            mock_sleep.assert_called_once_with(0.01)
            assert result == {"loss": 0.5}

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy({"input_ids": torch.ones(2, 10)})
            mock_sleep.assert_called_once_with(10.0)


class TestNanInputStrategy:
    """Tests for NAN_INPUT strategy."""

    def test_nan_input_injects_nan(self):
        """NAN_INPUT strategy injects NaN into input batch."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan",
            strategy=StrategyType.NAN_INPUT,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(10, 10)}
        result = proxy(batch)

        # Verify original was called
        original.assert_called_once()
        # Verify the batch passed to original has NaN values
        called_batch = original.call_args[0][0]
        assert torch.isnan(called_batch["input_ids"]).any()
        assert result == {"loss": 0.5}

    def test_nan_input_default_ratio(self):
        """NAN_INPUT strategy uses default ratio (0.1) when not specified."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan",
            strategy=StrategyType.NAN_INPUT,
            trigger=trigger,
            parameters={},  # No nan_ratio specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(100, 100)}
        proxy(batch)

        # With 10% ratio on 10000 elements, we expect some NaN values
        called_batch = original.call_args[0][0]
        nan_count = torch.isnan(called_batch["input_ids"]).sum().item()
        # Should have approximately 10% NaN values (allow for randomness)
        assert 500 < nan_count < 1500  # Roughly 10% of 10000

    def test_nan_input_multiple_tensors(self):
        """NAN_INPUT strategy injects NaN into all tensors in batch."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan",
            strategy=StrategyType.NAN_INPUT,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {
            "input_ids": torch.ones(10, 10),
            "attention_mask": torch.ones(10, 10),
        }
        proxy(batch)

        called_batch = original.call_args[0][0]
        assert torch.isnan(called_batch["input_ids"]).any()
        assert torch.isnan(called_batch["attention_mask"]).any()

    def test_nan_input_preserves_non_tensors(self):
        """NAN_INPUT strategy preserves non-tensor values."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan",
            strategy=StrategyType.NAN_INPUT,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {
            "input_ids": torch.ones(10, 10),
            "batch_size": 10,
            "name": "test",
        }
        proxy(batch)

        called_batch = original.call_args[0][0]
        assert called_batch["batch_size"] == 10
        assert called_batch["name"] == "test"


class TestExplodingGradientsStrategy:
    """Tests for EXPLODING_GRADIENTS strategy."""

    def test_exploding_gradients_scales_by_1e6(self):
        """EXPLODING_GRADIENTS strategy scales input by 1e6."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-explode",
            strategy=StrategyType.EXPLODING_GRADIENTS,
            trigger=trigger,
            parameters={},  # Use default scale
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 2)}
        proxy(batch)

        called_batch = original.call_args[0][0]
        expected = torch.ones(2, 2) * 1e6
        assert torch.allclose(called_batch["input_ids"], expected)

    def test_exploding_gradients_custom_scale(self):
        """EXPLODING_GRADIENTS strategy uses configured scale."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-explode",
            strategy=StrategyType.EXPLODING_GRADIENTS,
            trigger=trigger,
            parameters={"scale": 100.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 2) * 2.0}
        proxy(batch)

        called_batch = original.call_args[0][0]
        expected = torch.ones(2, 2) * 200.0  # 2.0 * 100.0
        assert torch.allclose(called_batch["input_ids"], expected)

    def test_exploding_gradients_multiple_tensors(self):
        """EXPLODING_GRADIENTS strategy scales all tensors in batch."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-explode",
            strategy=StrategyType.EXPLODING_GRADIENTS,
            trigger=trigger,
            parameters={"scale": 10.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {
            "input_ids": torch.ones(2, 2),
            "attention_mask": torch.ones(2, 2) * 0.5,
        }
        proxy(batch)

        called_batch = original.call_args[0][0]
        assert torch.allclose(called_batch["input_ids"], torch.ones(2, 2) * 10.0)
        assert torch.allclose(called_batch["attention_mask"], torch.ones(2, 2) * 5.0)


class TestVanishingGradientsStrategy:
    """Tests for VANISHING_GRADIENTS strategy."""

    def test_vanishing_gradients_scales_by_1e_minus_8(self):
        """VANISHING_GRADIENTS strategy scales input by 1e-8."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-vanish",
            strategy=StrategyType.VANISHING_GRADIENTS,
            trigger=trigger,
            parameters={},  # Use default scale
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 2)}
        proxy(batch)

        called_batch = original.call_args[0][0]
        expected = torch.ones(2, 2) * 1e-8
        assert torch.allclose(called_batch["input_ids"], expected)

    def test_vanishing_gradients_custom_scale(self):
        """VANISHING_GRADIENTS strategy uses configured scale."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-vanish",
            strategy=StrategyType.VANISHING_GRADIENTS,
            trigger=trigger,
            parameters={"scale": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 2) * 100.0}
        proxy(batch)

        called_batch = original.call_args[0][0]
        expected = torch.ones(2, 2) * 0.1  # 100.0 * 0.001
        assert torch.allclose(called_batch["input_ids"], expected)

    def test_vanishing_gradients_preserves_shape(self):
        """VANISHING_GRADIENTS strategy preserves tensor shape."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-vanish",
            strategy=StrategyType.VANISHING_GRADIENTS,
            trigger=trigger,
            parameters={"scale": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(3, 5, 7)}
        proxy(batch)

        called_batch = original.call_args[0][0]
        assert called_batch["input_ids"].shape == torch.Size([3, 5, 7])


class TestSkipUpdateStrategy:
    """Tests for SKIP_UPDATE strategy."""

    def test_skip_update_does_not_call_original(self):
        """SKIP_UPDATE strategy does not call original function."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)

        original.assert_not_called()
        assert result is None

    def test_skip_update_returns_default(self):
        """SKIP_UPDATE strategy returns configured default value."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
            parameters={"default_return": {"skipped": True}},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)

        original.assert_not_called()
        assert result == {"skipped": True}

    def test_skip_update_default_returns_none(self):
        """SKIP_UPDATE strategy returns None when no default specified."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
            parameters={},  # No default_return
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)

        assert result is None


class TestDoubleUpdateStrategy:
    """Tests for DOUBLE_UPDATE strategy."""

    def test_double_update_calls_original_twice(self):
        """DOUBLE_UPDATE strategy calls original function twice."""
        call_count = 0
        results = [{"loss": 0.5, "call": 1}, {"loss": 0.3, "call": 2}]

        def track_calls(*args, **kwargs):
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        original = MagicMock(side_effect=track_calls)
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-double",
            strategy=StrategyType.DOUBLE_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)

        assert call_count == 2
        assert original.call_count == 2
        # Should return result from second call
        assert result == {"loss": 0.3, "call": 2}

    def test_double_update_same_args(self):
        """DOUBLE_UPDATE strategy passes same arguments to both calls."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-double",
            strategy=StrategyType.DOUBLE_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        proxy(batch, learning_rate=0.001)

        assert original.call_count == 2
        # Both calls should have the same arguments
        for call in original.call_args_list:
            assert call[0][0] is batch
            assert call[1]["learning_rate"] == 0.001

    def test_double_update_returns_second_result(self):
        """DOUBLE_UPDATE strategy returns result from second call."""
        original = MagicMock(side_effect=[{"loss": 0.5}, {"loss": 0.3}])
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-double",
            strategy=StrategyType.DOUBLE_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        result = proxy(batch)

        assert result == {"loss": 0.3}


class TestUpdateActorProxyIntegration:
    """Integration tests for UpdateActorProxy."""

    def test_step_based_trigger(self):
        """Proxy triggers only within specified step range."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)

        batch = {"input_ids": torch.ones(2, 10)}

        # Before range - should call original
        proxy.set_step(3)
        proxy(batch)
        assert original.call_count == 1

        # In range - should skip
        proxy.set_step(7)
        result = proxy(batch)
        assert original.call_count == 1  # Still 1 (skipped)
        assert result is None

        # After range - should call original
        proxy.set_step(15)
        proxy(batch)
        assert original.call_count == 2

    def test_periodic_trigger(self):
        """Proxy triggers every N steps."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=5,
        )
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
        )
        proxy.set_config(config)

        batch = {"input_ids": torch.ones(2, 10)}
        skipped_steps = []

        for step in range(15):
            proxy.set_step(step)
            result = proxy(batch)
            if result is None:
                skipped_steps.append(step)

        # Should skip at steps 0, 5, 10
        assert skipped_steps == [0, 5, 10]

    def test_collector_recording(self):
        """Proxy records injection to collector."""
        original = MagicMock(return_value={"loss": 0.5})
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault-123")
        collector.record_fault_outcome = MagicMock()

        proxy = UpdateActorProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-record",
            strategy=StrategyType.SKIP_UPDATE,
            trigger=trigger,
            severity="high",
            expected_behavior="Update skipped",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}
        proxy(batch)

        # Should record injection start
        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "skip_update"
        assert call_kwargs["target_layer"] == "L2"
        assert call_kwargs["severity"] == "high"

        # Should record injection end
        collector.record_fault_outcome.assert_called_once()
        outcome_kwargs = collector.record_fault_outcome.call_args[1]
        assert outcome_kwargs["fault_id"] == "fault-123"
        assert outcome_kwargs["outcome"] == "success"

    def test_collector_records_failure(self):
        """Proxy records failure to collector when original raises."""
        original = MagicMock(side_effect=RuntimeError("Training error"))
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault-456")
        collector.record_fault_outcome = MagicMock()

        proxy = UpdateActorProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-failure",
            strategy=StrategyType.NAN_INPUT,
            trigger=trigger,
            parameters={"nan_ratio": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}

        with pytest.raises(RuntimeError, match="Training error"):
            proxy(batch)

        # Should record failure
        collector.record_fault_outcome.assert_called_once()
        outcome_kwargs = collector.record_fault_outcome.call_args[1]
        assert outcome_kwargs["outcome"] == "error"

    def test_nested_batch_structures(self):
        """Proxy handles nested batch structures correctly."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nested",
            strategy=StrategyType.EXPLODING_GRADIENTS,
            trigger=trigger,
            parameters={"scale": 10.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {
            "inputs": {
                "input_ids": torch.ones(2, 2),
                "attention_mask": torch.ones(2, 2),
            },
            "labels": torch.ones(2, 2),
            "meta": {"name": "test"},  # Non-tensor nested
        }
        proxy(batch)

        called_batch = original.call_args[0][0]
        assert torch.allclose(called_batch["inputs"]["input_ids"], torch.ones(2, 2) * 10.0)
        assert torch.allclose(called_batch["inputs"]["attention_mask"], torch.ones(2, 2) * 10.0)
        assert torch.allclose(called_batch["labels"], torch.ones(2, 2) * 10.0)
        assert called_batch["meta"]["name"] == "test"  # Non-tensor preserved

    def test_kwargs_passing(self):
        """Proxy correctly passes all kwargs to original."""
        original = MagicMock(return_value={"loss": 0.5})
        proxy = UpdateActorProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-kwargs",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        batch = {"input_ids": torch.ones(2, 10)}

        with patch("time.sleep"):
            proxy(
                batch,
                learning_rate=0.001,
                clip_grad=1.0,
                extra_param="value",
            )

        original.assert_called_once_with(
            batch,
            learning_rate=0.001,
            clip_grad=1.0,
            extra_param="value",
        )
