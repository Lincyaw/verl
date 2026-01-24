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


# =============================================================================
# AllGatherProxy Tests
# =============================================================================

from ralph.proxies.l1_distributed import AllGatherProxy


class TestAllGatherProxyRegistration:
    """Tests for AllGatherProxy registration."""

    def test_registered_with_all_gather_target(self):
        """AllGatherProxy is registered for 'torch.distributed.all_gather' target."""
        assert ProxyRegistry.is_registered("torch.distributed.all_gather")
        assert ProxyRegistry.get_proxy("torch.distributed.all_gather") is AllGatherProxy

    def test_supported_strategies(self):
        """AllGatherProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("torch.distributed.all_gather")
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_GATHERED,
            StrategyType.MISSING_RANK,
            StrategyType.SHAPE_MISMATCH,
        }
        assert strategies == expected


class TestAllGatherProxyBasics:
    """Tests for basic AllGatherProxy functionality."""

    def test_get_layer_returns_l1(self):
        """_get_layer returns 'L1'."""
        proxy = AllGatherProxy(lambda tensor_list, tensor: None)
        assert proxy._get_layer() == "L1"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        tensor_list = [torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
        result = proxy(tensor_list, tensor)
        original.assert_called_once_with(tensor_list, tensor)
        assert result is None

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        tensor_list = [torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy(tensor_list, tensor)
        original.assert_called_once_with(tensor_list, tensor)
        assert result is None

    def test_unsupported_strategy_raises_error(self):
        """Unsupported strategy raises ValueError."""
        proxy = AllGatherProxy(lambda tensor_list, tensor: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-unsupported",
            strategy=StrategyType.DEADLOCK,  # Not supported by AllGatherProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value).lower()


class TestAllGatherDelayStrategy:
    """Tests for AllGatherProxy DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        tensor_list = [torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
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
            result = proxy(tensor_list, tensor)
            mock_sleep.assert_called_once_with(0.01)
            assert result is None
            original.assert_called_once()

    def test_delay_strategy_passes_group(self):
        """DELAY strategy passes group to original."""
        tensor_list = [torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-group",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy(tensor_list, tensor, group=mock_group)
            original.assert_called_once()


class TestCorruptGatheredStrategy:
    """Tests for CORRUPT_GATHERED strategy."""

    def test_corrupt_gathered_modifies_all_ranks(self):
        """CORRUPT_GATHERED strategy corrupts all gathered tensors by default."""
        tensor_list = [torch.ones(10), torch.ones(10), torch.ones(10)]
        tensor = torch.ones(10)
        original_values = [t.clone() for t in tensor_list]

        def mock_original(tl, t, **kwargs):
            # Simulate all_gather filling tensor_list
            for i, tensor_out in enumerate(tl):
                tensor_out.fill_(float(i + 1))
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-all",
            strategy=StrategyType.CORRUPT_GATHERED,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor_list, tensor)

        # All tensors should be corrupted (different from expected values)
        for i, gathered_tensor in enumerate(tensor_list):
            expected_val = float(i + 1)
            # At least some values should differ from expected due to noise
            assert not torch.allclose(gathered_tensor, torch.full_like(gathered_tensor, expected_val))

    def test_corrupt_gathered_specific_ranks(self):
        """CORRUPT_GATHERED strategy corrupts only specified ranks."""
        tensor_list = [torch.zeros(10), torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)

        def mock_original(tl, t, **kwargs):
            for i, tensor_out in enumerate(tl):
                tensor_out.fill_(1.0)  # All ones
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-specific",
            strategy=StrategyType.CORRUPT_GATHERED,
            trigger=trigger,
            parameters={"noise_scale": 1.0, "corrupt_ranks": [1]},  # Only corrupt rank 1
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor_list, tensor)

        # Rank 0 and 2 should be unchanged (all ones)
        assert torch.allclose(tensor_list[0], torch.ones(10))
        assert torch.allclose(tensor_list[2], torch.ones(10))
        # Rank 1 should be corrupted
        assert not torch.allclose(tensor_list[1], torch.ones(10))

    def test_corrupt_gathered_default_noise_scale(self):
        """CORRUPT_GATHERED uses default noise_scale of 0.1."""
        tensor_list = [torch.zeros(100)]
        tensor = torch.ones(100)

        def mock_original(tl, t, **kwargs):
            tl[0].fill_(0.0)
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-default",
            strategy=StrategyType.CORRUPT_GATHERED,
            trigger=trigger,
            parameters={},  # No noise_scale
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor_list, tensor)
        # With default noise_scale=0.1, mean absolute value should be small
        assert tensor_list[0].abs().mean() < 0.5

    def test_corrupt_gathered_async_returns_immediately(self):
        """CORRUPT_GATHERED with async_op=True returns without corrupting."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)
        mock_result = MagicMock()

        def mock_original(tl, t, **kwargs):
            return mock_result

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-async",
            strategy=StrategyType.CORRUPT_GATHERED,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(tensor_list, tensor, async_op=True)
        # Should return the async result without modifications
        assert result is mock_result


class TestMissingRankStrategy:
    """Tests for MISSING_RANK strategy."""

    def test_missing_rank_zeros_out_data(self):
        """MISSING_RANK strategy zeros out specified rank's data."""
        tensor_list = [torch.zeros(10), torch.zeros(10), torch.zeros(10)]
        tensor = torch.ones(10)

        def mock_original(tl, t, **kwargs):
            for i, tensor_out in enumerate(tl):
                tensor_out.fill_(float(i + 1))  # Fill with 1, 2, 3
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={"missing_rank": 1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor_list, tensor)

        # Rank 0 should have value 1
        assert torch.allclose(tensor_list[0], torch.ones(10) * 1.0)
        # Rank 1 should be zeroed out
        assert torch.allclose(tensor_list[1], torch.zeros(10))
        # Rank 2 should have value 3
        assert torch.allclose(tensor_list[2], torch.ones(10) * 3.0)

    def test_missing_rank_requires_parameter(self):
        """MISSING_RANK raises ValueError if missing_rank not specified."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)
        original = MagicMock()
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-no-rank",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={},  # Missing 'missing_rank'
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy(tensor_list, tensor)
        assert "missing_rank must be specified" in str(exc_info.value)

    def test_missing_rank_out_of_bounds_is_safe(self):
        """MISSING_RANK with out-of-bounds rank is handled safely."""
        tensor_list = [torch.ones(10), torch.ones(10)]
        tensor = torch.ones(10)

        def mock_original(tl, t, **kwargs):
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-oob",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={"missing_rank": 99},  # Out of bounds
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Should not raise, just skip zeroing
        proxy(tensor_list, tensor)
        # Tensors should be unchanged (still ones from initialization)
        assert torch.allclose(tensor_list[0], torch.ones(10))
        assert torch.allclose(tensor_list[1], torch.ones(10))

    def test_missing_rank_async_returns_immediately(self):
        """MISSING_RANK with async_op=True returns without modifying."""
        tensor_list = [torch.ones(10)]
        tensor = torch.ones(10)
        mock_result = MagicMock()

        original = MagicMock(return_value=mock_result)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-async",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={"missing_rank": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(tensor_list, tensor, async_op=True)
        assert result is mock_result


class TestShapeMismatchStrategy:
    """Tests for SHAPE_MISMATCH strategy."""

    def test_shape_mismatch_expands_tensor(self):
        """SHAPE_MISMATCH strategy can expand tensor size."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)

        call_args = []

        def mock_original(tl, t, **kwargs):
            call_args.append((tl, t.clone()))
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-shape-expand",
            strategy=StrategyType.SHAPE_MISMATCH,
            trigger=trigger,
            parameters={"size_delta": 2, "mismatch_rank": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Mock rank to be 0 (the mismatch rank)
        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(tensor_list, tensor)

        # The tensor passed to original should be expanded
        _, called_tensor = call_args[0]
        assert called_tensor.shape[0] == 12  # 10 + 2

    def test_shape_mismatch_shrinks_tensor(self):
        """SHAPE_MISMATCH strategy can shrink tensor size."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)

        call_args = []

        def mock_original(tl, t, **kwargs):
            call_args.append((tl, t.clone()))
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-shape-shrink",
            strategy=StrategyType.SHAPE_MISMATCH,
            trigger=trigger,
            parameters={"size_delta": -3, "mismatch_rank": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(tensor_list, tensor)

        _, called_tensor = call_args[0]
        assert called_tensor.shape[0] == 7  # 10 - 3

    def test_shape_mismatch_only_on_configured_rank(self):
        """SHAPE_MISMATCH only modifies tensor on configured rank."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)

        call_args = []

        def mock_original(tl, t, **kwargs):
            call_args.append((tl, t.clone()))
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-shape-other-rank",
            strategy=StrategyType.SHAPE_MISMATCH,
            trigger=trigger,
            parameters={"size_delta": 5, "mismatch_rank": 1},  # Only rank 1
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Current rank is 0, mismatch_rank is 1
        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(tensor_list, tensor)

        # Tensor should not be modified
        _, called_tensor = call_args[0]
        assert called_tensor.shape[0] == 10  # Unchanged

    def test_shape_mismatch_default_rank_is_zero(self):
        """SHAPE_MISMATCH defaults to mismatch_rank=0."""
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)

        call_args = []

        def mock_original(tl, t, **kwargs):
            call_args.append((tl, t.clone()))
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-shape-default-rank",
            strategy=StrategyType.SHAPE_MISMATCH,
            trigger=trigger,
            parameters={"size_delta": 2},  # No mismatch_rank specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(tensor_list, tensor)

        # Should apply to rank 0 by default
        _, called_tensor = call_args[0]
        assert called_tensor.shape[0] == 12  # 10 + 2


class TestAllGatherProxyIntegration:
    """Integration tests for AllGatherProxy."""

    def test_step_based_triggering(self):
        """Strategy only triggers within configured step range."""
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=7)
        config = FaultConfig(
            id="test-step-based",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={"missing_rank": 0},
        )
        proxy.set_config(config)

        triggered_steps = []
        for step in range(10):
            proxy.set_step(step)
            tensor_list = [torch.ones(10)]
            tensor = torch.ones(10)

            # Reset tensor_list after original is called
            def reset_original(tl, t, **kwargs):
                tl[0].fill_(1.0)
                return None

            original.side_effect = reset_original

            proxy(tensor_list, tensor)

            # If rank 0 is zeroed, the strategy triggered
            if torch.allclose(tensor_list[0], torch.zeros(10)):
                triggered_steps.append(step)

        assert triggered_steps == [5, 6, 7]

    def test_periodic_triggering(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value=None)
        proxy = AllGatherProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={"missing_rank": 0},
        )
        proxy.set_config(config)

        triggered_steps = []
        for step in range(10):
            proxy.set_step(step)
            tensor_list = [torch.ones(10)]
            tensor = torch.ones(10)

            def reset_original(tl, t, **kwargs):
                tl[0].fill_(1.0)
                return None

            original.side_effect = reset_original

            proxy(tensor_list, tensor)

            if torch.allclose(tensor_list[0], torch.zeros(10)):
                triggered_steps.append(step)

        assert triggered_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-789"
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)

        def mock_original(tl, t, **kwargs):
            tl[0].fill_(1.0)
            return None

        original = MagicMock(side_effect=mock_original)
        proxy = AllGatherProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.CORRUPT_GATHERED,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
            severity="medium",
            expected_behavior="Should corrupt gathered data",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor_list, tensor)

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "corrupt_gathered"
        assert call_kwargs["target_layer"] == "L1"
        assert call_kwargs["severity"] == "medium"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-789"
        assert collector.record_fault_outcome.call_args[0][1] == "success"

    def test_collector_records_failure(self):
        """Collector records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-err"
        tensor_list = [torch.zeros(10)]
        tensor = torch.ones(10)
        original = MagicMock()
        proxy = AllGatherProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-failure-recording",
            strategy=StrategyType.MISSING_RANK,
            trigger=trigger,
            parameters={},  # Missing required 'missing_rank'
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError):
            proxy(tensor_list, tensor)

        # Should record failure
        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-err"
        assert collector.record_fault_outcome.call_args[0][1] == "failure"

    def test_get_world_size_without_distributed(self):
        """_get_world_size returns 1 when distributed not initialized."""
        proxy = AllGatherProxy(lambda tl, t: None)
        world_size = proxy._get_world_size()
        assert world_size == 1


# =============================================================================
# BarrierProxy Tests
# =============================================================================

from ralph.proxies.l1_distributed import BarrierProxy


class TestBarrierProxyRegistration:
    """Tests for BarrierProxy registration."""

    def test_registered_with_barrier_target(self):
        """BarrierProxy is registered for 'torch.distributed.barrier' target."""
        assert ProxyRegistry.is_registered("torch.distributed.barrier")
        assert ProxyRegistry.get_proxy("torch.distributed.barrier") is BarrierProxy

    def test_supported_strategies(self):
        """BarrierProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("torch.distributed.barrier")
        expected = {
            StrategyType.DELAY,
            StrategyType.BARRIER_TIMEOUT,
            StrategyType.BARRIER_SKIP,
            StrategyType.ASYNC_DESYNC,
        }
        assert strategies == expected


class TestBarrierProxyBasics:
    """Tests for basic BarrierProxy functionality."""

    def test_get_layer_returns_l1(self):
        """_get_layer returns 'L1'."""
        proxy = BarrierProxy(lambda: None)
        assert proxy._get_layer() == "L1"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        result = proxy()
        original.assert_called_once_with()
        assert result is None

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once_with()
        assert result is None

    def test_unsupported_strategy_raises_error(self):
        """Unsupported strategy raises ValueError."""
        proxy = BarrierProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-unsupported",
            strategy=StrategyType.CORRUPT_TENSOR,  # Not supported by BarrierProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value).lower()


class TestBarrierDelayStrategy:
    """Tests for BarrierProxy DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
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
            result = proxy()
            mock_sleep.assert_called_once_with(0.01)
            assert result is None
            original.assert_called_once()

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-default",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},  # No delay_seconds
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(10.0)  # Default is 10 seconds

    def test_delay_strategy_passes_group(self):
        """DELAY strategy passes group to original."""
        mock_group = MagicMock()
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-delay-group",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy(group=mock_group, async_op=True)
            original.assert_called_once()


class TestBarrierTimeoutStrategy:
    """Tests for BARRIER_TIMEOUT strategy."""

    def test_barrier_timeout_raises_on_missing_rank(self):
        """BARRIER_TIMEOUT raises ValueError if timeout_rank not specified."""
        original = MagicMock()
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-timeout-no-rank",
            strategy=StrategyType.BARRIER_TIMEOUT,
            trigger=trigger,
            parameters={},  # Missing timeout_rank
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy()
        assert "timeout_rank must be specified" in str(exc_info.value)

    def test_barrier_timeout_non_matching_rank_calls_original(self):
        """BARRIER_TIMEOUT with non-matching rank calls original."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-timeout-other",
            strategy=StrategyType.BARRIER_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_rank": 99},  # Different from current rank (0)
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Mock _get_current_rank to return 0
        proxy._get_current_rank = MagicMock(return_value=0)

        result = proxy()
        original.assert_called_once()
        assert result is None

    def test_barrier_timeout_matching_rank_hangs(self):
        """BARRIER_TIMEOUT with matching rank enters infinite loop."""
        original = MagicMock()
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-timeout-match",
            strategy=StrategyType.BARRIER_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_rank": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Mock _get_current_rank to return 0 (matching)
        proxy._get_current_rank = MagicMock(return_value=0)

        # Mock time.sleep to raise an exception after first call
        # to break out of the infinite loop
        with patch("time.sleep", side_effect=InterruptedError("Test break")):
            with pytest.raises(InterruptedError):
                proxy()

        # Original should not be called for timeout rank
        original.assert_not_called()

    def test_barrier_timeout_passes_kwargs_to_non_timeout_rank(self):
        """BARRIER_TIMEOUT passes kwargs to original for non-timeout ranks."""
        mock_group = MagicMock()
        mock_device_ids = [0, 1]
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-timeout-kwargs",
            strategy=StrategyType.BARRIER_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_rank": 99},  # Not current rank
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy._get_current_rank = MagicMock(return_value=0)

        proxy(group=mock_group, async_op=True, device_ids=mock_device_ids)
        call_kwargs = original.call_args[1]
        assert call_kwargs["group"] is mock_group
        assert call_kwargs["async_op"] is True
        assert call_kwargs["device_ids"] is mock_device_ids


class TestBarrierSkipStrategy:
    """Tests for BARRIER_SKIP strategy."""

    def test_barrier_skip_returns_without_calling_original(self):
        """BARRIER_SKIP returns None without calling original."""
        original = MagicMock()
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip",
            strategy=StrategyType.BARRIER_SKIP,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_barrier_skip_ignores_all_kwargs(self):
        """BARRIER_SKIP ignores all kwargs and returns None."""
        mock_group = MagicMock()
        original = MagicMock()
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip-kwargs",
            strategy=StrategyType.BARRIER_SKIP,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(group=mock_group, async_op=True, device_ids=[0])

        original.assert_not_called()
        assert result is None


class TestAsyncDesyncStrategy:
    """Tests for ASYNC_DESYNC strategy."""

    def test_async_desync_adds_random_delay(self):
        """ASYNC_DESYNC adds a random delay before calling original."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-desync",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once()
            # Delay should be between 0 and max_delay
            delay_arg = mock_sleep.call_args[0][0]
            assert 0 <= delay_arg <= 1.0
            original.assert_called_once()

    def test_async_desync_default_max_delay(self):
        """ASYNC_DESYNC uses default max_delay of 5.0."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-desync-default",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={},  # No max_delay
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once()
            delay_arg = mock_sleep.call_args[0][0]
            assert 0 <= delay_arg <= 5.0  # Default max is 5.0

    def test_async_desync_with_seed_is_deterministic(self):
        """ASYNC_DESYNC with seed produces deterministic delays per rank."""
        original = MagicMock(return_value=None)

        # First call with seed
        proxy1 = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config1 = FaultConfig(
            id="test-desync-seed1",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 10.0, "seed": 42},
        )
        proxy1.set_config(config1)
        proxy1.set_step(0)
        proxy1._get_current_rank = MagicMock(return_value=0)

        with patch("time.sleep") as mock_sleep1:
            proxy1()
            delay1 = mock_sleep1.call_args[0][0]

        # Second call with same seed should give same delay
        proxy2 = BarrierProxy(original)
        config2 = FaultConfig(
            id="test-desync-seed2",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 10.0, "seed": 42},
        )
        proxy2.set_config(config2)
        proxy2.set_step(0)
        proxy2._get_current_rank = MagicMock(return_value=0)

        with patch("time.sleep") as mock_sleep2:
            proxy2()
            delay2 = mock_sleep2.call_args[0][0]

        assert delay1 == delay2

    def test_async_desync_different_ranks_get_different_delays(self):
        """ASYNC_DESYNC produces different delays for different ranks with same seed."""
        original = MagicMock(return_value=None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)

        # Rank 0
        proxy1 = BarrierProxy(original)
        config1 = FaultConfig(
            id="test-desync-rank0",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 10.0, "seed": 100},
        )
        proxy1.set_config(config1)
        proxy1.set_step(0)
        proxy1._get_current_rank = MagicMock(return_value=0)

        with patch("time.sleep") as mock_sleep1:
            proxy1()
            delay_rank0 = mock_sleep1.call_args[0][0]

        # Rank 1
        proxy2 = BarrierProxy(original)
        config2 = FaultConfig(
            id="test-desync-rank1",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 10.0, "seed": 100},
        )
        proxy2.set_config(config2)
        proxy2.set_step(0)
        proxy2._get_current_rank = MagicMock(return_value=1)

        with patch("time.sleep") as mock_sleep2:
            proxy2()
            delay_rank1 = mock_sleep2.call_args[0][0]

        # Different ranks should get different delays
        assert delay_rank0 != delay_rank1

    def test_async_desync_passes_kwargs(self):
        """ASYNC_DESYNC passes kwargs to original."""
        mock_group = MagicMock()
        mock_device_ids = [0]
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-desync-kwargs",
            strategy=StrategyType.ASYNC_DESYNC,
            trigger=trigger,
            parameters={"max_delay": 0.0},  # No delay for fast test
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy(group=mock_group, async_op=True, device_ids=mock_device_ids)
            call_kwargs = original.call_args[1]
            assert call_kwargs["group"] is mock_group
            assert call_kwargs["async_op"] is True
            assert call_kwargs["device_ids"] is mock_device_ids


class TestBarrierProxyIntegration:
    """Integration tests for BarrierProxy."""

    def test_step_based_triggering(self):
        """Strategy only triggers within configured step range."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=7)
        config = FaultConfig(
            id="test-step-based",
            strategy=StrategyType.BARRIER_SKIP,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)

        skipped_steps = []
        for step in range(10):
            proxy.set_step(step)
            original.reset_mock()
            proxy()
            if not original.called:
                skipped_steps.append(step)

        assert skipped_steps == [5, 6, 7]

    def test_periodic_triggering(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.BARRIER_SKIP,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)

        skipped_steps = []
        for step in range(10):
            proxy.set_step(step)
            original.reset_mock()
            proxy()
            if not original.called:
                skipped_steps.append(step)

        assert skipped_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-barrier"
        original = MagicMock(return_value=None)
        proxy = BarrierProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.BARRIER_SKIP,
            trigger=trigger,
            parameters={},
            severity="high",
            expected_behavior="Should skip barrier synchronization",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "barrier_skip"
        assert call_kwargs["target_layer"] == "L1"
        assert call_kwargs["severity"] == "high"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-barrier"
        assert collector.record_fault_outcome.call_args[0][1] == "success"

    def test_collector_records_failure(self):
        """Collector records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-err"
        original = MagicMock()
        proxy = BarrierProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-failure-recording",
            strategy=StrategyType.BARRIER_TIMEOUT,
            trigger=trigger,
            parameters={},  # Missing required 'timeout_rank'
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError):
            proxy()

        # Should record failure
        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-err"
        assert collector.record_fault_outcome.call_args[0][1] == "failure"

    def test_get_current_rank_without_distributed(self):
        """_get_current_rank returns 0 when distributed not initialized."""
        proxy = BarrierProxy(lambda: None)
        rank = proxy._get_current_rank()
        assert rank == 0
