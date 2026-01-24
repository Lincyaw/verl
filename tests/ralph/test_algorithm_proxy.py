"""
Unit tests for Algorithm proxies.

Tests cover GAEProxy with all 5 supported strategies:
- WRONG_ADVANTAGE: Returns random noise instead of computed advantages
- ZERO_ADVANTAGE: Returns zeros for all advantages
- INVERTED_ADVANTAGE: Negates the computed advantages
- SCALED_ADVANTAGE: Multiplies advantages by a scale factor
- DELAYED_ADVANTAGE: Adds delay before computing advantages
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.algorithm import GAEProxy


class TestGAEProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_gae_target(self):
        """GAEProxy is registered for 'compute_gae_advantage_return' target."""
        assert ProxyRegistry.is_registered("compute_gae_advantage_return")
        assert ProxyRegistry.get_proxy("compute_gae_advantage_return") is GAEProxy

    def test_supported_strategies(self):
        """GAEProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("compute_gae_advantage_return")
        expected = {
            StrategyType.WRONG_ADVANTAGE,
            StrategyType.ZERO_ADVANTAGE,
            StrategyType.INVERTED_ADVANTAGE,
            StrategyType.SCALED_ADVANTAGE,
            StrategyType.DELAYED_ADVANTAGE,
        }
        assert strategies == expected


class TestGAEProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_algorithm(self):
        """_get_layer returns 'Algorithm'."""
        proxy = GAEProxy(lambda: None)
        assert proxy._get_layer() == "Algorithm"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"advantages": torch.ones(5), "returns": torch.ones(5)})
        proxy = GAEProxy(original)
        result = proxy()
        original.assert_called_once()
        assert "advantages" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ADVANTAGE,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once()
        torch.testing.assert_close(result["advantages"], torch.ones(5))

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = GAEProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by GAEProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestWrongAdvantageStrategy:
    """Tests for WRONG_ADVANTAGE strategy."""

    def test_wrong_advantage_replaces_with_noise(self):
        """WRONG_ADVANTAGE strategy replaces advantages with random noise."""
        original_adv = torch.ones(5, 10)
        original = MagicMock(return_value={"advantages": original_adv.clone()})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ADVANTAGE,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Result should be different from original (random noise)
        assert not torch.allclose(result["advantages"], original_adv)
        # Shape should be preserved
        assert result["advantages"].shape == original_adv.shape

    def test_wrong_advantage_default_noise_scale(self):
        """WRONG_ADVANTAGE uses default noise_scale of 1.0."""
        original = MagicMock(return_value={"advantages": torch.ones(10)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ADVANTAGE,
            trigger=trigger,
            parameters={},  # No noise_scale specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Should have changed values (noise applied)
        assert not torch.allclose(result["advantages"], torch.ones(10))

    def test_wrong_advantage_with_returns(self):
        """WRONG_ADVANTAGE also corrupts returns if present."""
        original = MagicMock(return_value={
            "advantages": torch.ones(5),
            "returns": torch.ones(5) * 2,
        })
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ADVANTAGE,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        torch.manual_seed(42)
        result = proxy()

        # Both should be corrupted
        assert not torch.allclose(result["advantages"], torch.ones(5))
        assert not torch.allclose(result["returns"], torch.ones(5) * 2)

    def test_wrong_advantage_preserves_other_keys(self):
        """WRONG_ADVANTAGE preserves non-advantage keys."""
        original = MagicMock(return_value={
            "advantages": torch.ones(5),
            "other_data": "should_be_preserved",
            "metadata": {"key": "value"},
        })
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["other_data"] == "should_be_preserved"
        assert result["metadata"] == {"key": "value"}


class TestZeroAdvantageStrategy:
    """Tests for ZERO_ADVANTAGE strategy."""

    def test_zero_advantage_returns_zeros(self):
        """ZERO_ADVANTAGE strategy replaces advantages with zeros."""
        original = MagicMock(return_value={"advantages": torch.ones(5, 10) * 5})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.zeros(5, 10))

    def test_zero_advantage_with_returns(self):
        """ZERO_ADVANTAGE also zeros returns if present."""
        original = MagicMock(return_value={
            "advantages": torch.ones(5) * 3,
            "returns": torch.ones(5) * 7,
        })
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.zeros(5))
        torch.testing.assert_close(result["returns"], torch.zeros(5))

    def test_zero_advantage_preserves_shape(self):
        """ZERO_ADVANTAGE preserves tensor shape."""
        shapes = [(5,), (10, 20), (3, 4, 5)]
        for shape in shapes:
            original = MagicMock(return_value={"advantages": torch.randn(*shape)})
            proxy = GAEProxy(original)
            trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
            config = FaultConfig(
                id="test",
                strategy=StrategyType.ZERO_ADVANTAGE,
                trigger=trigger,
            )
            proxy.set_config(config)
            proxy.set_step(0)

            result = proxy()

            assert result["advantages"].shape == shape
            torch.testing.assert_close(result["advantages"], torch.zeros(*shape))

    def test_zero_advantage_preserves_dtype(self):
        """ZERO_ADVANTAGE preserves tensor dtype."""
        original = MagicMock(return_value={"advantages": torch.ones(5, dtype=torch.float16)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["advantages"].dtype == torch.float16


class TestInvertedAdvantageStrategy:
    """Tests for INVERTED_ADVANTAGE strategy."""

    def test_inverted_advantage_negates_values(self):
        """INVERTED_ADVANTAGE strategy negates all advantage values."""
        original_adv = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        original = MagicMock(return_value={"advantages": original_adv.clone()})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
        torch.testing.assert_close(result["advantages"], expected)

    def test_inverted_advantage_with_returns(self):
        """INVERTED_ADVANTAGE also negates returns."""
        original = MagicMock(return_value={
            "advantages": torch.tensor([1.0, 2.0]),
            "returns": torch.tensor([3.0, 4.0]),
        })
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.tensor([-1.0, -2.0]))
        torch.testing.assert_close(result["returns"], torch.tensor([-3.0, -4.0]))

    def test_inverted_advantage_preserves_shape(self):
        """INVERTED_ADVANTAGE preserves tensor shape."""
        original = MagicMock(return_value={"advantages": torch.ones(5, 10)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["advantages"].shape == (5, 10)
        torch.testing.assert_close(result["advantages"], -torch.ones(5, 10))


class TestScaledAdvantageStrategy:
    """Tests for SCALED_ADVANTAGE strategy."""

    def test_scaled_advantage_with_zero_scale(self):
        """SCALED_ADVANTAGE with scale=0 returns zeros."""
        original = MagicMock(return_value={"advantages": torch.ones(5) * 10})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={"scale_factor": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.zeros(5))

    def test_scaled_advantage_with_positive_scale(self):
        """SCALED_ADVANTAGE with positive scale multiplies values."""
        original = MagicMock(return_value={"advantages": torch.ones(5) * 2})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={"scale_factor": 3.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.ones(5) * 6)

    def test_scaled_advantage_with_negative_scale(self):
        """SCALED_ADVANTAGE with negative scale inverts values."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={"scale_factor": -2.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.ones(5) * -2)

    def test_scaled_advantage_with_fractional_scale(self):
        """SCALED_ADVANTAGE with fractional scale reduces values."""
        original = MagicMock(return_value={"advantages": torch.ones(5) * 10})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={"scale_factor": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.ones(5) * 5)

    def test_scaled_advantage_default_scale_is_zero(self):
        """SCALED_ADVANTAGE defaults to scale_factor of 0.0."""
        original = MagicMock(return_value={"advantages": torch.ones(5) * 10})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={},  # No scale_factor specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.zeros(5))

    def test_scaled_advantage_with_returns(self):
        """SCALED_ADVANTAGE also scales returns."""
        original = MagicMock(return_value={
            "advantages": torch.ones(5),
            "returns": torch.ones(5) * 2,
        })
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SCALED_ADVANTAGE,
            trigger=trigger,
            parameters={"scale_factor": 3.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], torch.ones(5) * 3)
        torch.testing.assert_close(result["returns"], torch.ones(5) * 6)


class TestDelayedAdvantageStrategy:
    """Tests for DELAYED_ADVANTAGE strategy."""

    def test_delayed_advantage_with_config(self):
        """DELAYED_ADVANTAGE strategy applies configured delay."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAYED_ADVANTAGE,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},  # Short delay for testing
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch.object(proxy, "_apply_delay") as mock_delay:
            result = proxy()
            mock_delay.assert_called_once_with(0.01)
        original.assert_called_once()
        torch.testing.assert_close(result["advantages"], torch.ones(5))

    def test_delayed_advantage_default_delay(self):
        """DELAYED_ADVANTAGE uses default delay of 10.0 seconds."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAYED_ADVANTAGE,
            trigger=trigger,
            parameters={},  # No delay_seconds specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch.object(proxy, "_apply_delay") as mock_delay:
            result = proxy()
            mock_delay.assert_called_once_with(10.0)

    def test_delayed_advantage_returns_original_result(self):
        """DELAYED_ADVANTAGE returns the original result unchanged."""
        original_result = {
            "advantages": torch.tensor([1.0, 2.0, 3.0]),
            "returns": torch.tensor([4.0, 5.0, 6.0]),
            "metadata": "test",
        }
        original = MagicMock(return_value=original_result)
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAYED_ADVANTAGE,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["advantages"], original_result["advantages"])
        torch.testing.assert_close(result["returns"], original_result["returns"])
        assert result["metadata"] == "test"


class TestGAEProxyIntegration:
    """Integration tests for GAEProxy."""

    def test_step_based_trigger(self):
        """Step-based trigger activates within range."""
        original = MagicMock(return_value={"advantages": torch.ones(5) * 10})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Before range - original
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["advantages"], torch.ones(5) * 10)

        # In range - zeros
        proxy.set_step(7)
        result = proxy()
        torch.testing.assert_close(result["advantages"], torch.zeros(5))

        # After range - original
        proxy.set_step(12)
        result = proxy()
        torch.testing.assert_close(result["advantages"], torch.ones(5) * 10)

    def test_periodic_trigger(self):
        """Periodic trigger activates every N steps."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=3,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Step 0 - periodic triggers (0 % 3 == 0)
        proxy.set_step(0)
        result = proxy()
        torch.testing.assert_close(result["advantages"], -torch.ones(5))

        # Step 1 - no trigger
        proxy.set_step(1)
        result = proxy()
        torch.testing.assert_close(result["advantages"], torch.ones(5))

        # Step 2 - no trigger
        proxy.set_step(2)
        result = proxy()
        torch.testing.assert_close(result["advantages"], torch.ones(5))

        # Step 3 - periodic triggers
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["advantages"], -torch.ones(5))

    def test_collector_recording(self):
        """Collector records fault injection events."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = GAEProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
            severity="high",
            expected_behavior="Zero advantages",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "zero_advantage"
        assert call_kwargs["target_layer"] == "Algorithm"
        assert call_kwargs["severity"] == "high"
        collector.record_fault_outcome.assert_called_once_with("fault123", "success", 0)

    def test_failure_recording(self):
        """Collector records failure when strategy raises exception."""
        def failing_original():
            return {"advantages": torch.ones(5)}

        original = MagicMock(side_effect=RuntimeError("Original failed"))
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = GAEProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,  # This calls original first
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError, match="Original failed"):
            proxy()

        collector.record_fault_outcome.assert_called_once()
        call_args = collector.record_fault_outcome.call_args[0]
        assert call_args[0] == "fault123"
        assert "exception" in call_args[1]
        assert "RuntimeError" in call_args[1]

    def test_tuple_output_handling(self):
        """GAEProxy handles tuple outputs correctly."""
        adv = torch.ones(5)
        ret = torch.ones(5) * 2
        original = MagicMock(return_value=(adv.clone(), ret.clone()))
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.INVERTED_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert isinstance(result, tuple)
        assert len(result) == 2
        torch.testing.assert_close(result[0], -adv)
        torch.testing.assert_close(result[1], -ret)

    def test_list_output_handling(self):
        """GAEProxy handles list outputs correctly."""
        adv = torch.ones(5)
        ret = torch.ones(5) * 2
        original = MagicMock(return_value=[adv.clone(), ret.clone()])
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert isinstance(result, list)
        assert len(result) == 2
        torch.testing.assert_close(result[0], torch.zeros(5))
        torch.testing.assert_close(result[1], torch.zeros(5))

    def test_passes_args_to_original(self):
        """GAEProxy passes arguments to original function."""
        original = MagicMock(return_value={"advantages": torch.ones(5)})
        proxy = GAEProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_ADVANTAGE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        values = torch.randn(5)
        rewards = torch.randn(5)
        proxy(values, rewards, gamma=0.99, lam=0.95)

        original.assert_called_once_with(values, rewards, gamma=0.99, lam=0.95)
