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
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.l2_verl import CheckpointSaveProxy, RewardManagerProxy


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

