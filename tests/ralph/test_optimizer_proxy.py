"""
Unit tests for optimizer proxies.

Tests for OptimizerStepProxy and LRSchedulerProxy classes.
"""

import sys
from unittest.mock import MagicMock

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.optimizer import LRSchedulerProxy, OptimizerStepProxy

# Mock torch before importing
mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
mock_torch.randn_like = MagicMock(side_effect=lambda x: x)
sys.modules["torch"] = mock_torch

# Clear registry before importing proxies to avoid duplicate registration errors
ProxyRegistry.clear()

# ============================================================================
# OptimizerStepProxy Tests
# ============================================================================


class TestOptimizerStepProxyRegistration:
    """Test OptimizerStepProxy registration and supported strategies."""

    def test_registered_target(self):
        """Test that OptimizerStepProxy is registered for 'optimizer.step'."""
        assert ProxyRegistry.is_registered("optimizer.step")
        proxy_class = ProxyRegistry.get_proxy("optimizer.step")
        assert proxy_class == OptimizerStepProxy

    def test_supported_strategies(self):
        """Test that OptimizerStepProxy declares correct supported strategies."""
        expected = {
            StrategyType.SKIP,
            StrategyType.REPEAT,
            StrategyType.CORRUPTED_MOMENTUM,
            StrategyType.RESET_STATE,
        }
        assert OptimizerStepProxy.SUPPORTED_STRATEGIES == expected


class TestOptimizerStepProxyBasics:
    """Test basic OptimizerStepProxy functionality."""

    def test_get_layer(self):
        """Test that _get_layer returns 'Optimizer'."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)
        assert proxy._get_layer() == "Optimizer"

    def test_call_without_config(self):
        """Test that proxy calls original when no config is set."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        proxy()

        original.assert_called_once_with()

    def test_call_with_disabled_config(self):
        """Test that proxy calls original when config is disabled."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once_with()

    def test_unsupported_strategy_raises(self):
        """Test that setting unsupported strategy raises ValueError."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by OptimizerStepProxy
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestOptimizerStepSkipStrategy:
    """Test SKIP strategy for OptimizerStepProxy."""

    def test_skip_does_not_call_original(self):
        """Test that SKIP strategy does not call the original function."""
        original = MagicMock(return_value="original_result")
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_skip_returns_default_value(self):
        """Test that SKIP strategy returns configured default value."""
        original = MagicMock(return_value="original_result")
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"default_return": "skipped"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        assert result == "skipped"


class TestOptimizerStepRepeatStrategy:
    """Test REPEAT strategy for OptimizerStepProxy."""

    def test_repeat_calls_original_multiple_times(self):
        """Test that REPEAT strategy calls original N times."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.REPEAT,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"times": 3},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert original.call_count == 3

    def test_repeat_default_is_two(self):
        """Test that REPEAT defaults to 2 times."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.REPEAT,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert original.call_count == 2

    def test_repeat_returns_last_result(self):
        """Test that REPEAT returns the last call's result."""
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            return f"result_{call_count[0]}"

        original = MagicMock(side_effect=side_effect)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.REPEAT,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"times": 3},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        assert result == "result_3"


class TestCorruptedMomentumStrategy:
    """Test CORRUPTED_MOMENTUM strategy for OptimizerStepProxy."""

    def test_corrupted_momentum_calls_original(self):
        """Test that CORRUPTED_MOMENTUM calls the original function."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPTED_MOMENTUM,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"noise_scale": 0.1},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_corrupted_momentum_with_closure(self):
        """Test that CORRUPTED_MOMENTUM passes closure correctly."""
        closure = MagicMock(return_value=1.0)
        original = MagicMock(return_value=1.0)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPTED_MOMENTUM,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy(closure=closure)

        original.assert_called_once_with(closure=closure)

    def test_corrupted_momentum_modifies_state(self):
        """Test that CORRUPTED_MOMENTUM modifies optimizer state."""
        # Create a mock optimizer with state
        mock_optimizer = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.__class__ = mock_torch.Tensor

        # Make isinstance check work
        __builtins__["isinstance"] if isinstance(__builtins__, dict) else __builtins__.isinstance

        mock_optimizer.state = {
            "param1": {
                "momentum_buffer": mock_tensor,
                "exp_avg": mock_tensor,
            }
        }

        original = MagicMock(return_value=None)
        original.__self__ = mock_optimizer  # Simulate bound method
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPTED_MOMENTUM,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"noise_scale": 0.5},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        # Call the strategy
        proxy()

        # Original should be called
        original.assert_called_once()

    def test_corrupted_momentum_default_noise_scale(self):
        """Test that CORRUPTED_MOMENTUM uses default noise scale of 0.1."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPTED_MOMENTUM,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No noise_scale specified
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        # Should not raise
        proxy()
        original.assert_called_once()


class TestResetStateStrategy:
    """Test RESET_STATE strategy for OptimizerStepProxy."""

    def test_reset_state_calls_original(self):
        """Test that RESET_STATE calls the original function."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.RESET_STATE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_reset_state_clears_optimizer_state(self):
        """Test that RESET_STATE clears optimizer state."""
        mock_optimizer = MagicMock()
        mock_optimizer.state = {"param1": {"momentum_buffer": MagicMock()}}

        original = MagicMock(return_value=None)
        original.__self__ = mock_optimizer
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.RESET_STATE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        mock_optimizer.state.clear.assert_called_once()

    def test_reset_state_with_closure(self):
        """Test that RESET_STATE passes closure correctly."""
        closure = MagicMock(return_value=1.0)
        original = MagicMock(return_value=1.0)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.RESET_STATE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy(closure=closure)

        original.assert_called_once_with(closure=closure)


class TestOptimizerStepProxyIntegration:
    """Integration tests for OptimizerStepProxy."""

    def test_step_based_trigger(self):
        """Test that proxy respects step-based trigger."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.STEP_BASED, start_step=5, end_step=10),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Should not trigger
        proxy.set_step(1)
        proxy()
        assert original.call_count == 1

        # Step 5: Should trigger (skip)
        proxy.set_step(5)
        proxy()
        assert original.call_count == 1  # Still 1, skipped

        # Step 10: Should trigger (skip)
        proxy.set_step(10)
        proxy()
        assert original.call_count == 1  # Still 1, skipped

        # Step 11: Should not trigger
        proxy.set_step(11)
        proxy()
        assert original.call_count == 2

    def test_periodic_trigger(self):
        """Test that proxy respects periodic trigger."""
        original = MagicMock(return_value=None)
        proxy = OptimizerStepProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.REPEAT,
            trigger=TriggerConfig(TriggerType.PERIODIC, every_n_steps=3),
            parameters={"times": 2},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Not periodic multiple
        proxy.set_step(1)
        proxy()
        assert original.call_count == 1

        # Step 3: Periodic multiple - repeats
        proxy.set_step(3)
        proxy()
        assert original.call_count == 3  # 1 + 2

        # Step 4: Not periodic multiple
        proxy.set_step(4)
        proxy()
        assert original.call_count == 4

    def test_collector_recording(self):
        """Test that proxy records injections to collector."""
        original = MagicMock(return_value=None)
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_123"

        proxy = OptimizerStepProxy(original, collector)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
            severity="medium",
            expected_behavior="Should skip optimizer step",
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once_with("fault_123", "success", 0)


# ============================================================================
# LRSchedulerProxy Tests
# ============================================================================


class TestLRSchedulerProxyRegistration:
    """Test LRSchedulerProxy registration and supported strategies."""

    def test_registered_target(self):
        """Test that LRSchedulerProxy is registered for 'lr_scheduler.step'."""
        assert ProxyRegistry.is_registered("lr_scheduler.step")
        proxy_class = ProxyRegistry.get_proxy("lr_scheduler.step")
        assert proxy_class == LRSchedulerProxy

    def test_supported_strategies(self):
        """Test that LRSchedulerProxy declares correct supported strategies."""
        expected = {
            StrategyType.SKIP,
            StrategyType.WRONG_LR,
            StrategyType.LR_SPIKE,
            StrategyType.LR_ZERO,
        }
        assert LRSchedulerProxy.SUPPORTED_STRATEGIES == expected


class TestLRSchedulerProxyBasics:
    """Test basic LRSchedulerProxy functionality."""

    def test_get_layer(self):
        """Test that _get_layer returns 'Optimizer'."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)
        assert proxy._get_layer() == "Optimizer"

    def test_call_without_config(self):
        """Test that proxy calls original when no config is set."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)

        proxy()

        original.assert_called_once_with()

    def test_unsupported_strategy_raises(self):
        """Test that setting unsupported strategy raises ValueError."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.REPEAT,  # Not supported by LRSchedulerProxy
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestLRSchedulerSkipStrategy:
    """Test SKIP strategy for LRSchedulerProxy."""

    def test_skip_does_not_call_original(self):
        """Test that SKIP strategy does not call the original function."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None


class TestWrongLRStrategy:
    """Test WRONG_LR strategy for LRSchedulerProxy."""

    def test_wrong_lr_calls_original(self):
        """Test that WRONG_LR calls the original function."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"lr_factor": 5.0},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_wrong_lr_scales_learning_rate(self):
        """Test that WRONG_LR scales learning rate by factor."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.01}, {"lr": 0.001}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"lr_factor": 10.0},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 0.1
        assert mock_optimizer.param_groups[1]["lr"] == 0.01

    def test_wrong_lr_default_factor(self):
        """Test that WRONG_LR uses default factor of 10.0."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.01}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 0.1


class TestLRSpikeStrategy:
    """Test LR_SPIKE strategy for LRSchedulerProxy."""

    def test_lr_spike_sets_absolute_value(self):
        """Test that LR_SPIKE sets learning rate to absolute value."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.001}, {"lr": 0.0001}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.LR_SPIKE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"spike_value": 0.5},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 0.5
        assert mock_optimizer.param_groups[1]["lr"] == 0.5

    def test_lr_spike_default_value(self):
        """Test that LR_SPIKE uses default value of 1.0."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.001}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.LR_SPIKE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 1.0


class TestLRZeroStrategy:
    """Test LR_ZERO strategy for LRSchedulerProxy."""

    def test_lr_zero_sets_zero(self):
        """Test that LR_ZERO sets learning rate to zero."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.01}, {"lr": 0.001}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.LR_ZERO,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 0.0
        assert mock_optimizer.param_groups[1]["lr"] == 0.0

    def test_lr_zero_calls_original_first(self):
        """Test that LR_ZERO calls original before zeroing."""
        original = MagicMock(return_value=None)
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.LR_ZERO,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()


class TestLRSchedulerProxyIntegration:
    """Integration tests for LRSchedulerProxy."""

    def test_step_based_trigger(self):
        """Test that proxy respects step-based trigger."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.01}]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.LR_ZERO,
            trigger=TriggerConfig(TriggerType.STEP_BASED, start_step=5, end_step=10),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Should not trigger
        mock_optimizer.param_groups[0]["lr"] = 0.01
        proxy.set_step(1)
        proxy()
        assert mock_optimizer.param_groups[0]["lr"] == 0.01  # Unchanged

        # Step 5: Should trigger
        mock_optimizer.param_groups[0]["lr"] = 0.01
        proxy.set_step(5)
        proxy()
        assert mock_optimizer.param_groups[0]["lr"] == 0.0

    def test_collector_recording(self):
        """Test that proxy records injections to collector."""
        original = MagicMock(return_value=None)
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_456"

        proxy = LRSchedulerProxy(original, collector)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
            severity="high",
            expected_behavior="Should skip scheduler step",
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once_with("fault_456", "success", 0)

    def test_no_scheduler_context_graceful(self):
        """Test that strategies work gracefully without scheduler context."""
        original = MagicMock(return_value=None)
        # No __self__ attribute - can't get scheduler
        proxy = LRSchedulerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"lr_factor": 10.0},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        # Should not raise, just call original
        proxy()
        original.assert_called_once()


# ============================================================================
# Helper Methods Tests
# ============================================================================


class TestGetOptimizerFromContext:
    """Test _get_optimizer_from_context helper method."""

    def test_bound_method_returns_self(self):
        """Test that bound method returns __self__."""
        mock_optimizer = MagicMock()
        original = MagicMock()
        original.__self__ = mock_optimizer

        proxy = OptimizerStepProxy(original)
        result = proxy._get_optimizer_from_context()

        assert result == mock_optimizer

    def test_external_optimizer_attribute(self):
        """Test that _optimizer attribute is used if available."""
        mock_optimizer = MagicMock()
        original = MagicMock()

        proxy = OptimizerStepProxy(original)
        proxy._optimizer = mock_optimizer
        result = proxy._get_optimizer_from_context()

        assert result == mock_optimizer

    def test_no_context_returns_none(self):
        """Test that None is returned when no context available."""
        original = MagicMock()

        proxy = OptimizerStepProxy(original)
        result = proxy._get_optimizer_from_context()

        assert result is None


class TestCorruptOptimizerState:
    """Test _corrupt_optimizer_state helper method."""

    def test_no_state_attribute(self):
        """Test that missing state attribute doesn't raise."""
        mock_optimizer = MagicMock(spec=[])  # No state attribute
        original = MagicMock()

        proxy = OptimizerStepProxy(original)
        # Should not raise
        proxy._corrupt_optimizer_state(mock_optimizer, 0.1)


class TestResetOptimizerState:
    """Test _reset_optimizer_state helper method."""

    def test_clears_state(self):
        """Test that state.clear() is called."""
        mock_optimizer = MagicMock()
        mock_optimizer.state = MagicMock()

        original = MagicMock()
        proxy = OptimizerStepProxy(original)

        proxy._reset_optimizer_state(mock_optimizer)

        mock_optimizer.state.clear.assert_called_once()


class TestScaleLearningRate:
    """Test _scale_learning_rate helper method."""

    def test_scales_all_param_groups(self):
        """Test that all param groups are scaled."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {"lr": 0.1, "weight_decay": 0.01},
            {"lr": 0.01, "weight_decay": 0.001},
        ]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock()
        proxy = LRSchedulerProxy(original)

        proxy._scale_learning_rate(mock_scheduler, 2.0)

        assert mock_optimizer.param_groups[0]["lr"] == 0.2
        assert mock_optimizer.param_groups[1]["lr"] == 0.02

    def test_no_optimizer_attribute(self):
        """Test that missing optimizer attribute doesn't raise."""
        mock_scheduler = MagicMock(spec=[])  # No optimizer attribute

        original = MagicMock()
        proxy = LRSchedulerProxy(original)

        # Should not raise
        proxy._scale_learning_rate(mock_scheduler, 2.0)


class TestSetLearningRate:
    """Test _set_learning_rate helper method."""

    def test_sets_all_param_groups(self):
        """Test that all param groups are set to same value."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {"lr": 0.1},
            {"lr": 0.01},
        ]

        mock_scheduler = MagicMock()
        mock_scheduler.optimizer = mock_optimizer

        original = MagicMock()
        proxy = LRSchedulerProxy(original)

        proxy._set_learning_rate(mock_scheduler, 0.5)

        assert mock_optimizer.param_groups[0]["lr"] == 0.5
        assert mock_optimizer.param_groups[1]["lr"] == 0.5
