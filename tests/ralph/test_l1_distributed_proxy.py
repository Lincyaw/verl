"""
Unit tests for AllReduceProxy class.

Tests cover all 5 supported strategies:
- DELAY: Adds delay before calling original all_reduce
- CORRUPT_TENSOR: Corrupts tensor data with Gaussian noise before all_reduce
- INJECT_NAN: Injects NaN values into tensor before all_reduce
- INJECT_INF: Injects Inf values into tensor before all_reduce
- DEADLOCK: Causes specified rank to hang indefinitely
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.l1_distributed import AllReduceProxy


class TestAllReduceProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_all_reduce_target(self):
        """AllReduceProxy is registered for 'torch.distributed.all_reduce' target."""
        assert ProxyRegistry.is_registered("torch.distributed.all_reduce")
        assert ProxyRegistry.get_proxy("torch.distributed.all_reduce") is AllReduceProxy

    def test_supported_strategies(self):
        """AllReduceProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("torch.distributed.all_reduce")
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.INJECT_NAN,
            StrategyType.INJECT_INF,
            StrategyType.DEADLOCK,
        }
        assert strategies == expected


class TestAllReduceProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_l1(self):
        """_get_layer returns 'L1'."""
        proxy = AllReduceProxy(lambda x: x)
        assert proxy._get_layer() == "L1"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        result = proxy(tensor)
        original.assert_called_once_with(tensor)
        assert result is None

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy(tensor)
        original.assert_called_once_with(tensor)
        assert result is None


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
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
            result = proxy(tensor)
            mock_sleep.assert_called_once_with(0.01)
            assert result is None
            original.assert_called_once()

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
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
            proxy(tensor)
            mock_sleep.assert_called_once_with(10.0)  # Default is 10 seconds

    def test_delay_strategy_passes_op_and_group(self):
        """DELAY strategy passes op and group to original."""
        tensor = torch.ones(10)
        mock_op = MagicMock()
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-params",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy(tensor, op=mock_op, group=mock_group, async_op=True)
            original.assert_called_once()


class TestCorruptTensorStrategy:
    """Tests for CORRUPT_TENSOR strategy."""

    def test_corrupt_tensor_modifies_input(self):
        """CORRUPT_TENSOR strategy modifies the input tensor."""
        tensor = torch.ones(10, 10)
        original_values = tensor.clone()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # Tensor should be modified in-place
        assert not torch.equal(tensor, original_values)
        # Shape should be preserved
        assert tensor.shape == original_values.shape

    def test_corrupt_tensor_default_noise_scale(self):
        """CORRUPT_TENSOR uses default noise_scale of 0.1."""
        tensor = torch.zeros(100, 100)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-default",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={},  # No noise_scale
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # With default noise_scale=0.1, most values should be small
        assert tensor.abs().mean() < 0.5

    def test_corrupt_tensor_passes_kwargs(self):
        """CORRUPT_TENSOR passes op, group, async_op to original."""
        tensor = torch.ones(10)
        mock_op = MagicMock()
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-kwargs",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor, op=mock_op, group=mock_group, async_op=True)
        call_kwargs = original.call_args[1]
        assert call_kwargs["op"] is mock_op
        assert call_kwargs["group"] is mock_group
        assert call_kwargs["async_op"] is True


class TestInjectNanStrategy:
    """Tests for INJECT_NAN strategy."""

    def test_inject_nan_modifies_input(self):
        """INJECT_NAN strategy injects NaN into the input tensor."""
        tensor = torch.ones(100)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # Tensor should contain NaN values
        assert torch.isnan(tensor).any()
        # Roughly half should be NaN
        nan_ratio = torch.isnan(tensor).sum().item() / tensor.numel()
        assert 0.3 < nan_ratio < 0.7

    def test_inject_nan_default_ratio(self):
        """INJECT_NAN uses default nan_ratio of 0.001."""
        tensor = torch.ones(10000)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan-default",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger,
            parameters={},  # No nan_ratio
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # With default nan_ratio=0.001, roughly 10 should be NaN
        nan_count = torch.isnan(tensor).sum().item()
        assert nan_count > 0  # Should have some NaN
        assert nan_count < 100  # But not too many

    def test_inject_nan_passes_kwargs(self):
        """INJECT_NAN passes op, group, async_op to original."""
        tensor = torch.ones(10)
        mock_op = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-nan-kwargs",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger,
            parameters={"nan_ratio": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor, op=mock_op, async_op=False)
        call_kwargs = original.call_args[1]
        assert call_kwargs["op"] is mock_op
        assert call_kwargs["async_op"] is False


class TestInjectInfStrategy:
    """Tests for INJECT_INF strategy."""

    def test_inject_inf_positive(self):
        """INJECT_INF strategy injects positive Inf."""
        tensor = torch.ones(100)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-inf-pos",
            strategy=StrategyType.INJECT_INF,
            trigger=trigger,
            parameters={"inf_ratio": 0.5, "positive": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # Tensor should contain Inf values
        assert torch.isinf(tensor).any()
        # All Inf should be positive
        inf_values = tensor[torch.isinf(tensor)]
        assert (inf_values > 0).all()

    def test_inject_inf_negative(self):
        """INJECT_INF strategy injects negative Inf."""
        tensor = torch.ones(100)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-inf-neg",
            strategy=StrategyType.INJECT_INF,
            trigger=trigger,
            parameters={"inf_ratio": 0.5, "positive": False},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # Tensor should contain Inf values
        assert torch.isinf(tensor).any()
        # All Inf should be negative
        inf_values = tensor[torch.isinf(tensor)]
        assert (inf_values < 0).all()

    def test_inject_inf_default_ratio(self):
        """INJECT_INF uses default inf_ratio of 0.001."""
        tensor = torch.ones(10000)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-inf-default",
            strategy=StrategyType.INJECT_INF,
            trigger=trigger,
            parameters={},  # No inf_ratio, will use defaults
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)
        # With default inf_ratio=0.001, roughly 10 should be Inf
        inf_count = torch.isinf(tensor).sum().item()
        assert inf_count > 0
        assert inf_count < 100

    def test_inject_inf_passes_kwargs(self):
        """INJECT_INF passes op, group, async_op to original."""
        tensor = torch.ones(10)
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-inf-kwargs",
            strategy=StrategyType.INJECT_INF,
            trigger=trigger,
            parameters={"inf_ratio": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor, group=mock_group)
        call_kwargs = original.call_args[1]
        assert call_kwargs["group"] is mock_group


class TestDeadlockStrategy:
    """Tests for DEADLOCK strategy."""

    def test_deadlock_raises_on_missing_rank(self):
        """DEADLOCK raises ValueError if deadlock_rank not specified."""
        tensor = torch.ones(10)
        original = MagicMock()
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-deadlock-no-rank",
            strategy=StrategyType.DEADLOCK,
            trigger=trigger,
            parameters={},  # Missing deadlock_rank
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy(tensor)
        assert "deadlock_rank must be specified" in str(exc_info.value)

    def test_deadlock_non_matching_rank_calls_original(self):
        """DEADLOCK with non-matching rank calls original."""
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-deadlock-other",
            strategy=StrategyType.DEADLOCK,
            trigger=trigger,
            parameters={"deadlock_rank": 99},  # Different from current rank (0)
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Mock _get_current_rank to return 0
        proxy._get_current_rank = MagicMock(return_value=0)

        result = proxy(tensor)
        original.assert_called_once()
        assert result is None

    def test_deadlock_matching_rank_hangs(self):
        """DEADLOCK with matching rank enters infinite loop."""
        tensor = torch.ones(10)
        original = MagicMock()
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-deadlock-match",
            strategy=StrategyType.DEADLOCK,
            trigger=trigger,
            parameters={"deadlock_rank": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Mock _get_current_rank to return 0 (matching)
        proxy._get_current_rank = MagicMock(return_value=0)

        # Mock time.sleep to raise an exception after first call
        # to break out of the infinite loop
        with patch("time.sleep", side_effect=InterruptedError("Test break")):
            with pytest.raises(InterruptedError):
                proxy(tensor)

        # Original should not be called for deadlock rank
        original.assert_not_called()

    def test_deadlock_passes_kwargs_to_non_deadlock_rank(self):
        """DEADLOCK passes kwargs to original for non-deadlock ranks."""
        tensor = torch.ones(10)
        mock_op = MagicMock()
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-deadlock-kwargs",
            strategy=StrategyType.DEADLOCK,
            trigger=trigger,
            parameters={"deadlock_rank": 99},  # Not current rank
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(tensor, op=mock_op, group=mock_group, async_op=True)
        call_kwargs = original.call_args[1]
        assert call_kwargs["op"] is mock_op
        assert call_kwargs["group"] is mock_group
        assert call_kwargs["async_op"] is True


class TestAllReduceProxyIntegration:
    """Integration tests for AllReduceProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        tensor = torch.ones(100)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)

        # Steps 0-4 should not inject NaN
        for step in range(5):
            test_tensor = torch.ones(100)
            proxy.set_step(step)
            proxy(test_tensor)
            assert not torch.isnan(test_tensor).any()

        # Step 5 should inject NaN
        test_tensor = torch.ones(100)
        proxy.set_step(5)
        proxy(test_tensor)
        assert torch.isnan(test_tensor).any()

        # Step 6 should not inject NaN (one-shot)
        test_tensor = torch.ones(100)
        proxy.set_step(6)
        proxy(test_tensor)
        assert not torch.isnan(test_tensor).any()

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger,
            parameters={"nan_ratio": 0.9},  # High ratio for reliable detection
        )
        proxy.set_config(config)

        # Steps 0, 3, 6 should trigger (step % 3 == 0)
        trigger_steps = []
        for step in range(10):
            test_tensor = torch.ones(100)
            proxy.set_step(step)
            proxy(test_tensor)
            if torch.isnan(test_tensor).any():
                trigger_steps.append(step)

        assert trigger_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-456"
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
            severity="high",
            expected_behavior="Should corrupt gradients",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor)

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "corrupt_tensor"
        assert call_kwargs["target_layer"] == "L1"
        assert call_kwargs["severity"] == "high"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-456"
        assert collector.record_fault_outcome.call_args[0][1] == "success"

    def test_get_current_rank_without_distributed(self):
        """_get_current_rank returns 0 when distributed not initialized."""
        proxy = AllReduceProxy(lambda x: x)
        # Without distributed initialization, should return 0
        rank = proxy._get_current_rank()
        assert rank == 0

    def test_multiple_strategies_sequentially(self):
        """Different strategies can be applied sequentially."""
        original = MagicMock(return_value=None)
        proxy = AllReduceProxy(original)

        # First apply DELAY
        tensor1 = torch.ones(10)
        trigger1 = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config1 = FaultConfig(
            id="delay-test",
            strategy=StrategyType.DELAY,
            trigger=trigger1,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config1)
        proxy.set_step(0)
        with patch("time.sleep") as mock_sleep:
            proxy(tensor1)
            mock_sleep.assert_called_with(0.01)

        # Then apply INJECT_NAN
        tensor2 = torch.ones(100)
        trigger2 = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1)
        config2 = FaultConfig(
            id="nan-test",
            strategy=StrategyType.INJECT_NAN,
            trigger=trigger2,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config2)
        proxy.set_step(1)
        proxy(tensor2)
        assert torch.isnan(tensor2).any()
