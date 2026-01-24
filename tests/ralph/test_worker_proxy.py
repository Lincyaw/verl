"""
Unit tests for Worker proxies.

Tests cover ComputeValuesProxy with all 5 supported strategies:
- DELAY: Adds delay before computing values
- WRONG_VALUES: Adds noise to computed values
- CONSTANT_VALUES: Returns constant value for all predictions
- NAN_VALUES: Injects NaN into value predictions
- INVERTED_VALUES: Negates the computed values
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.worker import ComputeValuesProxy


class TestComputeValuesProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_compute_values_target(self):
        """ComputeValuesProxy is registered for 'CriticWorker.compute_values' target."""
        assert ProxyRegistry.is_registered("CriticWorker.compute_values")
        assert ProxyRegistry.get_proxy("CriticWorker.compute_values") is ComputeValuesProxy

    def test_supported_strategies(self):
        """ComputeValuesProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("CriticWorker.compute_values")
        expected = {
            StrategyType.DELAY,
            StrategyType.WRONG_VALUES,
            StrategyType.CONSTANT_VALUES,
            StrategyType.NAN_VALUES,
            StrategyType.INVERTED_VALUES,
        }
        assert strategies == expected


class TestComputeValuesProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_worker(self):
        """_get_layer returns 'Worker'."""
        proxy = ComputeValuesProxy(lambda: None)
        assert proxy._get_layer() == "Worker"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        result = proxy()
        original.assert_called_once()
        assert "values" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once()
        torch.testing.assert_close(result["values"], torch.ones(5))

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = ComputeValuesProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by ComputeValuesProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_adds_delay(self):
        """DELAY strategy adds delay before computing values."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},  # Short delay for testing
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy()
            mock_sleep.assert_called_once_with(0.01)

        original.assert_called_once()
        assert "values" in result

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay of 10.0 seconds."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},  # No delay_seconds specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(10.0)


class TestWrongValuesStrategy:
    """Tests for WRONG_VALUES strategy."""

    def test_wrong_values_adds_noise(self):
        """WRONG_VALUES strategy adds noise to value predictions."""
        original_values = torch.ones(5, 10)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Result should be different from original (noise added)
        assert not torch.allclose(result["values"], original_values)
        # Shape should be preserved
        assert result["values"].shape == original_values.shape

    def test_wrong_values_default_noise_scale(self):
        """WRONG_VALUES uses default noise_scale of 0.1."""
        original_values = torch.ones(10)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
            parameters={},  # No noise_scale specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Should have changed values (noise applied)
        assert not torch.allclose(result["values"], original_values)
        # With noise_scale=0.1, changes should be relatively small
        diff = (result["values"] - original_values).abs().mean()
        assert diff < 0.5  # Noise should be moderate

    def test_wrong_values_with_multiple_value_keys(self):
        """WRONG_VALUES handles multiple value-related keys."""
        original = MagicMock(return_value={
            "values": torch.ones(5),
            "critic_values": torch.ones(5) * 2,
            "other_key": torch.ones(5) * 3,
        })
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Value keys should have noise added
        assert not torch.allclose(result["values"], torch.ones(5))
        assert not torch.allclose(result["critic_values"], torch.ones(5) * 2)
        # Other keys should be unchanged
        torch.testing.assert_close(result["other_key"], torch.ones(5) * 3)

    def test_wrong_values_preserves_shape(self):
        """WRONG_VALUES preserves tensor shapes."""
        original_values = torch.ones(3, 4, 5)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result["values"].shape == torch.Size([3, 4, 5])


class TestConstantValuesStrategy:
    """Tests for CONSTANT_VALUES strategy."""

    def test_constant_values_returns_constant(self):
        """CONSTANT_VALUES returns configured constant for all values."""
        original_values = torch.ones(5, 10) * 3.14
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={"constant_value": 42.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # All values should be the constant
        expected = torch.full_like(original_values, 42.0)
        torch.testing.assert_close(result["values"], expected)

    def test_constant_values_default_zero(self):
        """CONSTANT_VALUES uses default constant_value of 0.0."""
        original_values = torch.ones(10) * 5
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={},  # No constant_value specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # All values should be zero
        torch.testing.assert_close(result["values"], torch.zeros(10))

    def test_constant_values_negative_constant(self):
        """CONSTANT_VALUES handles negative constant values."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={"constant_value": -100.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.full((5,), -100.0)
        torch.testing.assert_close(result["values"], expected)

    def test_constant_values_preserves_shape(self):
        """CONSTANT_VALUES preserves tensor shapes."""
        original_values = torch.ones(3, 4, 5)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={"constant_value": 1.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result["values"].shape == torch.Size([3, 4, 5])
        assert (result["values"] == 1.5).all()

    def test_constant_values_preserves_dtype(self):
        """CONSTANT_VALUES preserves tensor dtype."""
        original_values = torch.ones(5, dtype=torch.float16)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={"constant_value": 2.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result["values"].dtype == torch.float16


class TestNanValuesStrategy:
    """Tests for NAN_VALUES strategy."""

    def test_nan_values_injects_nan(self):
        """NAN_VALUES injects NaN into value predictions."""
        original_values = torch.ones(100)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_VALUES,
            trigger=trigger,
            parameters={"nan_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Some values should be NaN
        nan_count = result["values"].isnan().sum().item()
        assert nan_count > 0
        # Roughly half should be NaN
        assert 30 <= nan_count <= 70

    def test_nan_values_default_ratio(self):
        """NAN_VALUES uses default nan_ratio of 0.1."""
        original_values = torch.ones(1000)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_VALUES,
            trigger=trigger,
            parameters={},  # No nan_ratio specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Roughly 10% should be NaN
        nan_count = result["values"].isnan().sum().item()
        assert 50 <= nan_count <= 150  # 10% of 1000 with variance

    def test_nan_values_preserves_shape(self):
        """NAN_VALUES preserves tensor shapes."""
        original_values = torch.ones(3, 4, 5)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_VALUES,
            trigger=trigger,
            parameters={"nan_ratio": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result["values"].shape == torch.Size([3, 4, 5])

    def test_nan_values_zero_ratio(self):
        """NAN_VALUES with ratio 0 doesn't inject any NaN."""
        original_values = torch.ones(100)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_VALUES,
            trigger=trigger,
            parameters={"nan_ratio": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        nan_count = result["values"].isnan().sum().item()
        assert nan_count == 0

    def test_nan_values_full_ratio(self):
        """NAN_VALUES with ratio 1 sets all values to NaN."""
        original_values = torch.ones(10)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_VALUES,
            trigger=trigger,
            parameters={"nan_ratio": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        nan_count = result["values"].isnan().sum().item()
        assert nan_count == 10


class TestInvertedValuesStrategy:
    """Tests for INVERTED_VALUES strategy."""

    def test_inverted_values_negates_values(self):
        """INVERTED_VALUES negates value predictions."""
        original_values = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0])
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.tensor([-1.0, -2.0, -3.0, 1.0, 2.0])
        torch.testing.assert_close(result["values"], expected)

    def test_inverted_values_preserves_shape(self):
        """INVERTED_VALUES preserves tensor shapes."""
        original_values = torch.ones(3, 4, 5)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result["values"].shape == torch.Size([3, 4, 5])
        torch.testing.assert_close(result["values"], -torch.ones(3, 4, 5))

    def test_inverted_values_with_multiple_keys(self):
        """INVERTED_VALUES handles multiple value-related keys."""
        original = MagicMock(return_value={
            "values": torch.ones(5),
            "v": torch.ones(5) * 2,
            "other_key": torch.ones(5) * 3,
        })
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Value keys should be negated
        torch.testing.assert_close(result["values"], -torch.ones(5))
        torch.testing.assert_close(result["v"], -torch.ones(5) * 2)
        # Other keys should be unchanged
        torch.testing.assert_close(result["other_key"], torch.ones(5) * 3)

    def test_inverted_values_with_scalars(self):
        """INVERTED_VALUES handles scalar values correctly."""
        original = MagicMock(return_value={"value": 5.0, "other": 10.0})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["value"] == -5.0
        assert result["other"] == 10.0  # Not a value key


class TestComputeValuesProxyIntegration:
    """Integration tests for ComputeValuesProxy."""

    def test_step_based_trigger(self):
        """Proxy respects step-based trigger conditions."""
        original_values = torch.ones(5)
        original = MagicMock(return_value={"values": original_values.clone()})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CONSTANT_VALUES,
            trigger=trigger,
            parameters={"constant_value": 42.0},
        )
        proxy.set_config(config)

        # Before trigger range - should return original
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["values"], original_values)

        # Within trigger range - should return constant
        original.reset_mock()
        original.return_value = {"values": original_values.clone()}
        proxy.set_step(7)
        result = proxy()
        torch.testing.assert_close(result["values"], torch.full((5,), 42.0))

        # After trigger range - should return original
        original.reset_mock()
        original.return_value = {"values": original_values.clone()}
        proxy.set_step(15)
        result = proxy()
        torch.testing.assert_close(result["values"], original_values)

    def test_periodic_trigger(self):
        """Proxy respects periodic trigger conditions."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Step 0 (0 % 3 == 0) - should trigger
        proxy.set_step(0)
        result = proxy()
        torch.testing.assert_close(result["values"], -torch.ones(5))

        # Step 1 - should not trigger
        original.reset_mock()
        original.return_value = {"values": torch.ones(5)}
        proxy.set_step(1)
        result = proxy()
        torch.testing.assert_close(result["values"], torch.ones(5))

        # Step 3 - should trigger
        original.reset_mock()
        original.return_value = {"values": torch.ones(5)}
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["values"], -torch.ones(5))

    def test_collector_recording(self):
        """Proxy records fault injection to collector."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
            expected_behavior="Values should be negated",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        # Should have recorded start and end
        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()

        # Check outcome was "success"
        call_args = collector.record_fault_outcome.call_args
        assert call_args[0][0] == "fault-123"
        assert call_args[0][1] == "success"

    def test_collector_records_failure_on_exception(self):
        """Proxy records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-456"

        def failing_original(*args, **kwargs):
            raise RuntimeError("Test error")

        proxy = ComputeValuesProxy(failing_original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError, match="Test error"):
            proxy()

        # Should have recorded failure
        call_args = collector.record_fault_outcome.call_args
        assert call_args[0][0] == "fault-456"
        assert call_args[0][1] == "failure"

    def test_args_passing(self):
        """Proxy correctly passes args and kwargs to original."""
        original = MagicMock(return_value={"values": torch.ones(5)})
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("arg1", "arg2", key1="value1", key2="value2")

        original.assert_called_once_with("arg1", "arg2", key1="value1", key2="value2")

    def test_tuple_output_handling(self):
        """Proxy handles tuple outputs correctly."""
        original = MagicMock(return_value=(torch.ones(5), torch.ones(5) * 2))
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Tuple items should be negated
        assert isinstance(result, tuple)
        torch.testing.assert_close(result[0], -torch.ones(5))
        torch.testing.assert_close(result[1], -torch.ones(5) * 2)

    def test_list_output_handling(self):
        """Proxy handles list outputs correctly."""
        original = MagicMock(return_value=[torch.ones(5), torch.ones(5) * 2])
        proxy = ComputeValuesProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_VALUES,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # List items should be negated
        assert isinstance(result, list)
        torch.testing.assert_close(result[0], -torch.ones(5))
        torch.testing.assert_close(result[1], -torch.ones(5) * 2)
