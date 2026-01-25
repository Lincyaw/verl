"""
Unit tests for Algorithm proxies.

Tests cover GAEProxy with all 5 supported strategies:
- WRONG_ADVANTAGE: Returns random noise instead of computed advantages
- ZERO_ADVANTAGE: Returns zeros for all advantages
- INVERTED_ADVANTAGE: Negates the computed advantages
- SCALED_ADVANTAGE: Multiplies advantages by a scale factor
- DELAYED_ADVANTAGE: Adds delay before computing advantages

Tests also cover KLPenaltyProxy with all 5 supported strategies:
- DELAY: Adds delay before computing KL penalty
- WRONG_KL: Scales KL penalty by a configurable factor
- ZERO_KL: Returns zero KL penalty
- EXTREME_KL: Returns very large KL value
- NEGATIVE_KL: Returns negative KL penalty

Tests also cover GRPOProxy with all 4 supported strategies:
- WRONG_GROUPING: Uses shuffled or random group indices
- WRONG_NORMALIZATION: Applies incorrect mean/std computation
- SKIP_NORMALIZATION: Skips normalization, returns raw scores
- SINGLE_SAMPLE_GROUPS: Treats each sample as its own group
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.algorithm import GAEProxy, GRPOProxy, KLPenaltyProxy


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
        original = MagicMock(
            return_value={
                "advantages": torch.ones(5),
                "returns": torch.ones(5) * 2,
            }
        )
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
        original = MagicMock(
            return_value={
                "advantages": torch.ones(5),
                "other_data": "should_be_preserved",
                "metadata": {"key": "value"},
            }
        )
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
        original = MagicMock(
            return_value={
                "advantages": torch.ones(5) * 3,
                "returns": torch.ones(5) * 7,
            }
        )
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
        original = MagicMock(
            return_value={
                "advantages": torch.tensor([1.0, 2.0]),
                "returns": torch.tensor([3.0, 4.0]),
            }
        )
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
        original = MagicMock(
            return_value={
                "advantages": torch.ones(5),
                "returns": torch.ones(5) * 2,
            }
        )
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
            proxy()
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


# ============================================================================
# KLPenaltyProxy Tests
# ============================================================================


class TestKLPenaltyProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_kl_penalty_target(self):
        """KLPenaltyProxy is registered for 'apply_kl_penalty' target."""
        assert ProxyRegistry.is_registered("apply_kl_penalty")
        assert ProxyRegistry.get_proxy("apply_kl_penalty") is KLPenaltyProxy

    def test_supported_strategies(self):
        """KLPenaltyProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("apply_kl_penalty")
        expected = {
            StrategyType.DELAY,
            StrategyType.WRONG_KL,
            StrategyType.ZERO_KL,
            StrategyType.EXTREME_KL,
            StrategyType.NEGATIVE_KL,
        }
        assert strategies == expected


class TestKLPenaltyProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_algorithm(self):
        """_get_layer returns 'Algorithm'."""
        proxy = KLPenaltyProxy(lambda: None)
        assert proxy._get_layer() == "Algorithm"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"kl": torch.ones(5), "kl_penalty": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        result = proxy()
        original.assert_called_once()
        assert "kl" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once()
        torch.testing.assert_close(result["kl"], torch.ones(5))

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = KLPenaltyProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by KLPenaltyProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestKLPenaltyDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch.object(proxy, "_apply_delay") as mock_delay:
            result = proxy()
            mock_delay.assert_called_once_with(0.01)
        original.assert_called_once()
        torch.testing.assert_close(result["kl"], torch.ones(5))

    def test_delay_default(self):
        """DELAY uses default of 10.0 seconds."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch.object(proxy, "_apply_delay") as mock_delay:
            proxy()
            mock_delay.assert_called_once_with(10.0)


class TestWrongKLStrategy:
    """Tests for WRONG_KL strategy."""

    def test_wrong_kl_scales_values(self):
        """WRONG_KL strategy scales KL values by factor."""
        original_kl = torch.ones(5, 10) * 2
        original = MagicMock(return_value={"kl": original_kl.clone()})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            parameters={"scale_factor": 5.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.ones(5, 10) * 10  # 2 * 5 = 10
        torch.testing.assert_close(result["kl"], expected)

    def test_wrong_kl_default_scale(self):
        """WRONG_KL uses default scale_factor of 10.0."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.ones(5) * 10)

    def test_wrong_kl_with_kl_penalty_key(self):
        """WRONG_KL also scales kl_penalty if present."""
        original = MagicMock(
            return_value={
                "kl": torch.ones(5),
                "kl_penalty": torch.ones(5) * 2,
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            parameters={"scale_factor": 3.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.ones(5) * 3)
        torch.testing.assert_close(result["kl_penalty"], torch.ones(5) * 6)

    def test_wrong_kl_preserves_other_keys(self):
        """WRONG_KL preserves non-KL keys."""
        original = MagicMock(
            return_value={
                "kl": torch.ones(5),
                "other_data": "should_be_preserved",
                "metadata": {"key": "value"},
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            parameters={"scale_factor": 2.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["other_data"] == "should_be_preserved"
        assert result["metadata"] == {"key": "value"}

    def test_wrong_kl_scalar_float(self):
        """WRONG_KL handles scalar float values."""
        original = MagicMock(return_value={"kl": 1.5})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_KL,
            trigger=trigger,
            parameters={"scale_factor": 4.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"] == 6.0
        assert isinstance(result["kl"], float)


class TestZeroKLStrategy:
    """Tests for ZERO_KL strategy."""

    def test_zero_kl_returns_zeros(self):
        """ZERO_KL strategy replaces KL values with zeros."""
        original = MagicMock(return_value={"kl": torch.ones(5, 10) * 5})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.zeros(5, 10))

    def test_zero_kl_with_kl_penalty(self):
        """ZERO_KL also zeros kl_penalty if present."""
        original = MagicMock(
            return_value={
                "kl": torch.ones(5) * 3,
                "kl_penalty": torch.ones(5) * 7,
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.zeros(5))
        torch.testing.assert_close(result["kl_penalty"], torch.zeros(5))

    def test_zero_kl_preserves_shape(self):
        """ZERO_KL preserves tensor shape."""
        shapes = [(5,), (10, 20), (3, 4, 5)]
        for shape in shapes:
            original = MagicMock(return_value={"kl": torch.randn(*shape)})
            proxy = KLPenaltyProxy(original)
            trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
            config = FaultConfig(
                id="test",
                strategy=StrategyType.ZERO_KL,
                trigger=trigger,
            )
            proxy.set_config(config)
            proxy.set_step(0)

            result = proxy()

            assert result["kl"].shape == shape
            torch.testing.assert_close(result["kl"], torch.zeros(*shape))

    def test_zero_kl_preserves_dtype(self):
        """ZERO_KL preserves tensor dtype."""
        original = MagicMock(return_value={"kl": torch.ones(5, dtype=torch.float16)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"].dtype == torch.float16

    def test_zero_kl_scalar_values(self):
        """ZERO_KL handles scalar int/float values."""
        original = MagicMock(return_value={"kl": 5.5, "kl_penalty": 3})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"] == 0.0
        assert isinstance(result["kl"], float)
        assert result["kl_penalty"] == 0
        assert isinstance(result["kl_penalty"], int)


class TestExtremeKLStrategy:
    """Tests for EXTREME_KL strategy."""

    def test_extreme_kl_returns_extreme_value(self):
        """EXTREME_KL strategy replaces KL with extreme value."""
        original = MagicMock(return_value={"kl": torch.ones(5, 10)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
            parameters={"extreme_value": 1e6},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.full((5, 10), 1e6)
        torch.testing.assert_close(result["kl"], expected)

    def test_extreme_kl_default_value(self):
        """EXTREME_KL uses default extreme_value of 1e6."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.full((5,), 1e6)
        torch.testing.assert_close(result["kl"], expected)

    def test_extreme_kl_custom_value(self):
        """EXTREME_KL uses custom extreme value from config."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
            parameters={"extreme_value": 999.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.full((5,), 999.0))

    def test_extreme_kl_preserves_shape(self):
        """EXTREME_KL preserves tensor shape."""
        original = MagicMock(return_value={"kl": torch.ones(3, 4, 5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"].shape == (3, 4, 5)

    def test_extreme_kl_with_kl_penalty(self):
        """EXTREME_KL also sets kl_penalty to extreme value."""
        original = MagicMock(
            return_value={
                "kl": torch.ones(5),
                "kl_penalty": torch.ones(5) * 2,
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
            parameters={"extreme_value": 1e8},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.full((5,), 1e8))
        torch.testing.assert_close(result["kl_penalty"], torch.full((5,), 1e8))

    def test_extreme_kl_scalar_values(self):
        """EXTREME_KL handles scalar int/float values."""
        original = MagicMock(return_value={"kl": 1.5})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTREME_KL,
            trigger=trigger,
            parameters={"extreme_value": 1e6},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"] == 1e6
        assert isinstance(result["kl"], float)


class TestNegativeKLStrategy:
    """Tests for NEGATIVE_KL strategy."""

    def test_negative_kl_negates_values(self):
        """NEGATIVE_KL strategy negates all KL values."""
        original_kl = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        original = MagicMock(return_value={"kl": original_kl.clone()})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        expected = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
        torch.testing.assert_close(result["kl"], expected)

    def test_negative_kl_with_kl_penalty(self):
        """NEGATIVE_KL also negates kl_penalty."""
        original = MagicMock(
            return_value={
                "kl": torch.tensor([1.0, 2.0]),
                "kl_penalty": torch.tensor([3.0, 4.0]),
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], torch.tensor([-1.0, -2.0]))
        torch.testing.assert_close(result["kl_penalty"], torch.tensor([-3.0, -4.0]))

    def test_negative_kl_preserves_shape(self):
        """NEGATIVE_KL preserves tensor shape."""
        original = MagicMock(return_value={"kl": torch.ones(5, 10)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"].shape == (5, 10)
        torch.testing.assert_close(result["kl"], -torch.ones(5, 10))

    def test_negative_kl_preserves_other_keys(self):
        """NEGATIVE_KL preserves non-KL keys."""
        original = MagicMock(
            return_value={
                "kl": torch.ones(5),
                "loss": torch.ones(5) * 2,
                "metadata": "should_preserve",
            }
        )
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        torch.testing.assert_close(result["kl"], -torch.ones(5))
        torch.testing.assert_close(result["loss"], torch.ones(5) * 2)  # Unchanged
        assert result["metadata"] == "should_preserve"

    def test_negative_kl_scalar_values(self):
        """NEGATIVE_KL handles scalar int/float values."""
        original = MagicMock(return_value={"kl": 5.5, "kl_penalty": -3})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["kl"] == -5.5
        assert isinstance(result["kl"], float)
        assert result["kl_penalty"] == 3
        assert isinstance(result["kl_penalty"], int)


class TestKLPenaltyProxyIntegration:
    """Integration tests for KLPenaltyProxy."""

    def test_step_based_trigger(self):
        """Step-based trigger activates within range."""
        original = MagicMock(return_value={"kl": torch.ones(5) * 10})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Before range - original
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["kl"], torch.ones(5) * 10)

        # In range - zeros
        proxy.set_step(7)
        result = proxy()
        torch.testing.assert_close(result["kl"], torch.zeros(5))

        # After range - original
        proxy.set_step(12)
        result = proxy()
        torch.testing.assert_close(result["kl"], torch.ones(5) * 10)

    def test_periodic_trigger(self):
        """Periodic trigger activates every N steps."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=3,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Step 0 - periodic triggers (0 % 3 == 0)
        proxy.set_step(0)
        result = proxy()
        torch.testing.assert_close(result["kl"], -torch.ones(5))

        # Step 1 - no trigger
        proxy.set_step(1)
        result = proxy()
        torch.testing.assert_close(result["kl"], torch.ones(5))

        # Step 2 - no trigger
        proxy.set_step(2)
        result = proxy()
        torch.testing.assert_close(result["kl"], torch.ones(5))

        # Step 3 - periodic triggers
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["kl"], -torch.ones(5))

    def test_collector_recording(self):
        """Collector records fault injection events."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = KLPenaltyProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
            severity="high",
            expected_behavior="Zero KL penalty",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "zero_kl"
        assert call_kwargs["target_layer"] == "Algorithm"
        assert call_kwargs["severity"] == "high"
        collector.record_fault_outcome.assert_called_once_with("fault123", "success", 0)

    def test_failure_recording(self):
        """Collector records failure when strategy raises exception."""
        original = MagicMock(side_effect=RuntimeError("KL computation failed"))
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = KLPenaltyProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError, match="KL computation failed"):
            proxy()

        collector.record_fault_outcome.assert_called_once()
        call_args = collector.record_fault_outcome.call_args[0]
        assert call_args[0] == "fault123"
        assert "exception" in call_args[1]
        assert "RuntimeError" in call_args[1]

    def test_tuple_output_handling(self):
        """KLPenaltyProxy handles tuple outputs correctly."""
        kl = torch.ones(5)
        penalty = torch.ones(5) * 2
        original = MagicMock(return_value=(kl.clone(), penalty.clone()))
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.NEGATIVE_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert isinstance(result, tuple)
        assert len(result) == 2
        torch.testing.assert_close(result[0], -kl)
        torch.testing.assert_close(result[1], -penalty)

    def test_list_output_handling(self):
        """KLPenaltyProxy handles list outputs correctly."""
        kl = torch.ones(5)
        penalty = torch.ones(5) * 2
        original = MagicMock(return_value=[kl.clone(), penalty.clone()])
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
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
        """KLPenaltyProxy passes arguments to original function."""
        original = MagicMock(return_value={"kl": torch.ones(5)})
        proxy = KLPenaltyProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.ZERO_KL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        policy_logprobs = torch.randn(5)
        ref_logprobs = torch.randn(5)
        proxy(policy_logprobs, ref_logprobs, kl_coef=0.1)

        original.assert_called_once_with(policy_logprobs, ref_logprobs, kl_coef=0.1)


# =============================================================================
# GRPOProxy Tests
# =============================================================================


class TestGRPOProxyRegistration:
    """Test GRPOProxy registration in ProxyRegistry."""

    def test_proxy_is_registered(self):
        """GRPOProxy should be registered for compute_grpo_outcome_advantage."""
        assert ProxyRegistry.is_registered("compute_grpo_outcome_advantage")

    def test_supported_strategies(self):
        """GRPOProxy should support correct strategies."""
        strategies = ProxyRegistry.get_supported_strategies("compute_grpo_outcome_advantage")
        assert StrategyType.WRONG_GROUPING in strategies
        assert StrategyType.WRONG_NORMALIZATION in strategies
        assert StrategyType.SKIP_NORMALIZATION in strategies
        assert StrategyType.SINGLE_SAMPLE_GROUPS in strategies
        assert len(strategies) == 4


class TestGRPOProxyBasics:
    """Test basic GRPOProxy functionality."""

    def test_get_layer_returns_algorithm(self):
        """_get_layer should return 'Algorithm'."""
        original = MagicMock()
        proxy = GRPOProxy(original)
        assert proxy._get_layer() == "Algorithm"

    def test_call_without_config_calls_original(self):
        """Calling proxy without config should call original function."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))
        proxy = GRPOProxy(original)

        result = proxy()

        original.assert_called_once()
        assert result[0] is advantages
        assert result[1] is returns

    def test_disabled_config_calls_original(self):
        """Disabled config should not inject fault."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))
        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_GROUPING,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        original.assert_called_once()
        assert result[0] is advantages

    def test_unsupported_strategy_raises_error(self):
        """Setting unsupported strategy should raise ValueError."""
        original = MagicMock()
        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by GRPOProxy
            trigger=trigger,
        )

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestWrongGroupingStrategy:
    """Test WRONG_GROUPING strategy."""

    def test_shuffles_index_array(self):
        """WRONG_GROUPING should shuffle the index array."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        original_index = np.array([0, 0, 1, 1])

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_GROUPING,
            trigger=trigger,
            parameters={"shuffle_seed": 42},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(token_rewards, response_mask, original_index.copy())

        # Check that original was called with a modified index
        original.assert_called_once()
        call_args = original.call_args[0]
        # The index should have been shuffled
        assert len(call_args) >= 3

    def test_with_index_in_kwargs(self):
        """WRONG_GROUPING handles index passed as kwarg."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        original_index = np.array([0, 0, 1, 1])

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_GROUPING,
            trigger=trigger,
            parameters={"shuffle_seed": 42},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(token_rewards, response_mask, index=original_index.copy())

        original.assert_called_once()
        call_kwargs = original.call_args[1]
        assert "index" in call_kwargs

    def test_randomize_option(self):
        """WRONG_GROUPING with randomize=True uses random indices."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        original_index = np.array([0, 0, 1, 1])

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_GROUPING,
            trigger=trigger,
            parameters={"randomize": True, "shuffle_seed": 42},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(token_rewards, response_mask, original_index.copy())

        original.assert_called_once()

    def test_no_index_parameter_calls_original(self):
        """When no index parameter found, calls original unchanged."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_GROUPING,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Call with only 2 args (no index)
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        proxy(token_rewards, response_mask)

        original.assert_called_once()


class TestWrongNormalizationStrategy:
    """Test WRONG_NORMALIZATION strategy."""

    def test_scales_result(self):
        """WRONG_NORMALIZATION scales the result."""
        advantages = torch.ones(4, 10)
        returns = torch.ones(4, 10)
        original = MagicMock(return_value=(advantages.clone(), returns.clone()))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
            parameters={"mean_scale": 2.0, "std_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Result should be scaled by mean_scale * std_scale = 2.0 * 0.5 = 1.0
        expected = advantages * 2.0 * 0.5
        torch.testing.assert_close(result[0], expected)

    def test_default_scales(self):
        """WRONG_NORMALIZATION uses default scales if not specified."""
        advantages = torch.ones(4, 10)
        returns = torch.ones(4, 10)
        original = MagicMock(return_value=(advantages.clone(), returns.clone()))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Default mean_scale=2.0, std_scale=0.5
        expected = advantages * 2.0 * 0.5
        torch.testing.assert_close(result[0], expected)

    def test_custom_scales(self):
        """WRONG_NORMALIZATION uses custom scale values."""
        advantages = torch.ones(4, 10) * 2
        returns = torch.ones(4, 10)
        original = MagicMock(return_value=(advantages.clone(), returns.clone()))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
            parameters={"mean_scale": 3.0, "std_scale": 2.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Result scaled by 3.0 * 2.0 = 6.0
        expected = advantages * 6.0
        torch.testing.assert_close(result[0], expected)

    def test_handles_dict_result(self):
        """WRONG_NORMALIZATION handles dict results."""
        original_result = {
            "advantages": torch.ones(4, 10),
            "returns": torch.ones(4, 10) * 2,
        }
        original = MagicMock(return_value=original_result)

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
            parameters={"mean_scale": 2.0, "std_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert isinstance(result, dict)
        torch.testing.assert_close(result["advantages"], torch.ones(4, 10))
        torch.testing.assert_close(result["returns"], torch.ones(4, 10) * 2)


class TestSkipNormalizationStrategy:
    """Test SKIP_NORMALIZATION strategy."""

    def test_returns_raw_scores(self):
        """SKIP_NORMALIZATION returns unnormalized scores."""
        token_rewards = torch.ones(4, 10)  # Sum will be 10 for each
        response_mask = torch.ones(4, 10)

        original = MagicMock()
        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(token_rewards, response_mask)

        # Should NOT call original
        original.assert_not_called()
        # Result should be raw scores broadcasted
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Each row should have sum=10 broadcasted across response length
        expected_score = torch.ones(4, 10) * 10
        torch.testing.assert_close(result[0], expected_score)

    def test_different_rewards(self):
        """SKIP_NORMALIZATION handles different reward values."""
        token_rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Sums: 3, 7
        response_mask = torch.ones(2, 2)

        original = MagicMock()
        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(token_rewards, response_mask)

        expected = torch.tensor([[3.0, 3.0], [7.0, 7.0]])
        torch.testing.assert_close(result[0], expected)

    def test_with_kwargs(self):
        """SKIP_NORMALIZATION handles kwargs input."""
        token_rewards = torch.ones(4, 10)
        response_mask = torch.ones(4, 10)

        original = MagicMock()
        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(token_level_rewards=token_rewards, response_mask=response_mask)

        assert isinstance(result, tuple)
        original.assert_not_called()

    def test_fallback_on_missing_args(self):
        """SKIP_NORMALIZATION falls back to original if args missing."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Call with no args
        proxy()

        original.assert_called_once()


class TestSingleSampleGroupsStrategy:
    """Test SINGLE_SAMPLE_GROUPS strategy."""

    def test_creates_unique_indices(self):
        """SINGLE_SAMPLE_GROUPS creates unique index for each sample."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        original_index = np.array([0, 0, 1, 1])

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SINGLE_SAMPLE_GROUPS,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(token_rewards, response_mask, original_index.copy())

        original.assert_called_once()
        call_args = original.call_args[0]
        # Third argument should be unique indices
        modified_index = call_args[2]
        assert len(np.unique(modified_index)) == 4  # Each sample in own group

    def test_with_index_in_kwargs(self):
        """SINGLE_SAMPLE_GROUPS handles index in kwargs."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        original_index = np.array([0, 0, 1, 1])

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SINGLE_SAMPLE_GROUPS,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(token_rewards, response_mask, index=original_index)

        original.assert_called_once()
        call_kwargs = original.call_args[1]
        assert "index" in call_kwargs
        assert len(np.unique(call_kwargs["index"])) == 4

    def test_adds_index_if_missing(self):
        """SINGLE_SAMPLE_GROUPS adds index to kwargs if not in args."""
        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)

        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SINGLE_SAMPLE_GROUPS,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Only pass 2 args (no index in position 3)
        proxy(token_rewards, response_mask)

        original.assert_called_once()
        call_kwargs = original.call_args[1]
        assert "index" in call_kwargs
        np.testing.assert_array_equal(call_kwargs["index"], np.arange(4))

    def test_fallback_when_no_rewards(self):
        """SINGLE_SAMPLE_GROUPS falls back when can't determine batch size."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.SINGLE_SAMPLE_GROUPS,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Call with no args
        result = proxy()

        original.assert_called_once()
        assert result[0] is advantages


class TestGRPOProxyIntegration:
    """Integration tests for GRPOProxy."""

    def test_step_based_trigger(self):
        """GRPOProxy respects step-based trigger."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=10)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Before range - should not inject
        proxy.set_step(3)
        result = proxy()
        assert result[0] is advantages  # Original unchanged

        # In range - should inject
        proxy.set_step(7)
        result = proxy()
        assert result[0] is not advantages  # Modified

    def test_periodic_trigger(self):
        """GRPOProxy respects periodic trigger."""
        original = MagicMock(return_value=(torch.ones(4, 10), torch.ones(4, 10)))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)

        results = []
        for step in range(6):
            proxy.set_step(step)
            original.reset_mock()
            original.return_value = (torch.ones(4, 10), torch.ones(4, 10))
            result = proxy()
            # Check if result was modified (scale != 1)
            is_modified = not torch.allclose(result[0], torch.ones(4, 10))
            results.append(is_modified)

        # Steps 0, 3 should trigger (every 3 steps)
        assert results[0] is True  # step 0
        assert results[1] is False  # step 1
        assert results[2] is False  # step 2
        assert results[3] is True  # step 3

    def test_collector_records_fault(self):
        """GRPOProxy records fault with collector."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = GRPOProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
            severity="medium",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "wrong_normalization"
        assert call_kwargs["target_layer"] == "Algorithm"
        assert call_kwargs["severity"] == "medium"
        collector.record_fault_outcome.assert_called_once_with("fault123", "success", 0)

    def test_failure_recording(self):
        """GRPOProxy records failure when strategy raises exception."""
        original = MagicMock(side_effect=RuntimeError("GRPO computation failed"))
        collector = MagicMock()
        collector.record_fault_injection = MagicMock(return_value="fault123")
        collector.record_fault_outcome = MagicMock()

        proxy = GRPOProxy(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError, match="GRPO computation failed"):
            proxy()

        collector.record_fault_outcome.assert_called_once()
        call_args = collector.record_fault_outcome.call_args[0]
        assert call_args[0] == "fault123"
        assert "exception" in call_args[1]

    def test_tuple_output_preservation(self):
        """GRPOProxy preserves tuple output structure."""
        advantages = torch.ones(4, 10)
        returns = torch.ones(4, 10) * 2
        original = MagicMock(return_value=(advantages.clone(), returns.clone()))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
            parameters={"mean_scale": 1.0, "std_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_passes_all_args_to_original(self):
        """GRPOProxy passes all arguments to original function."""
        advantages = torch.randn(4, 10)
        returns = torch.randn(4, 10)
        original = MagicMock(return_value=(advantages, returns))

        proxy = GRPOProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_NORMALIZATION,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        token_rewards = torch.randn(4, 10)
        response_mask = torch.ones(4, 10)
        index = np.array([0, 0, 1, 1])
        proxy(token_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True)

        original.assert_called_once()
        call_args = original.call_args
        torch.testing.assert_close(call_args[0][0], token_rewards)
        torch.testing.assert_close(call_args[0][1], response_mask)
        assert call_args[1]["epsilon"] == 1e-6
        assert call_args[1]["norm_adv_by_std_in_grpo"] is True
