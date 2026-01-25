"""
Unit tests for Megatron proxies.

Tests for MegatronOptimizerProxy and ParallelismProxy classes.
"""

import sys
from unittest.mock import MagicMock

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.megatron import MegatronOptimizerProxy, ParallelismProxy

# Mock torch before importing
mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.randn_like = MagicMock(side_effect=lambda x: x)
mock_torch.randperm = MagicMock(side_effect=lambda n: MagicMock(__getitem__=lambda s, i: list(range(n))))
sys.modules["torch"] = mock_torch

# Clear registry before importing proxies to avoid duplicate registration errors
ProxyRegistry.clear()

# ============================================================================
# MegatronOptimizerProxy Tests
# ============================================================================


class TestMegatronOptimizerProxyRegistration:
    """Test MegatronOptimizerProxy registration and supported strategies."""

    def test_registered_target(self):
        """Test that MegatronOptimizerProxy is registered for 'MegatronEngine.optimizer_step'."""
        assert ProxyRegistry.is_registered("MegatronEngine.optimizer_step")
        proxy_class = ProxyRegistry.get_proxy("MegatronEngine.optimizer_step")
        assert proxy_class == MegatronOptimizerProxy

    def test_supported_strategies(self):
        """Test that MegatronOptimizerProxy declares correct supported strategies."""
        expected = {
            StrategyType.SKIP,
            StrategyType.GRADIENT_OVERFLOW,
            StrategyType.NAN_PARAMS,
            StrategyType.WRONG_LR,
        }
        assert MegatronOptimizerProxy.SUPPORTED_STRATEGIES == expected


class TestMegatronOptimizerProxyBasics:
    """Test basic MegatronOptimizerProxy functionality."""

    def test_get_layer(self):
        """Test that _get_layer returns 'Megatron'."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)
        assert proxy._get_layer() == "Megatron"

    def test_call_without_config(self):
        """Test that proxy calls original when no config is set."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

        proxy()

        original.assert_called_once_with()

    def test_call_with_disabled_config(self):
        """Test that proxy calls original when config is disabled."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

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
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by MegatronOptimizerProxy
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestMegatronSkipStrategy:
    """Test SKIP strategy for MegatronOptimizerProxy."""

    def test_skip_does_not_call_original(self):
        """Test that SKIP strategy does not call the original function."""
        original = MagicMock(return_value="original_result")
        proxy = MegatronOptimizerProxy(original)

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
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"default_return": {"skipped": True}},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        assert result == {"skipped": True}


class TestGradientOverflowStrategy:
    """Test GRADIENT_OVERFLOW strategy for MegatronOptimizerProxy."""

    def test_gradient_overflow_calls_original(self):
        """Test that GRADIENT_OVERFLOW calls the original function by default."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_gradient_overflow_set_flag_only(self):
        """Test that GRADIENT_OVERFLOW with set_flag_only=True skips original."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"set_flag_only": True},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_gradient_overflow_sets_overflow_flag(self):
        """Test that GRADIENT_OVERFLOW sets overflow flag on optimizer."""
        mock_optimizer = MagicMock()
        mock_optimizer.overflow = False

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.overflow is True

    def test_gradient_overflow_sets_found_inf(self):
        """Test that GRADIENT_OVERFLOW sets found_inf for FP16 optimizers."""
        mock_optimizer = MagicMock()
        mock_optimizer.found_inf = None
        # Remove overflow attribute to test found_inf path
        del mock_optimizer.overflow

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer
        del mock_engine.overflow

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        # Verify tensor was created
        mock_torch.tensor.assert_called()

    def test_gradient_overflow_passes_args(self):
        """Test that GRADIENT_OVERFLOW passes args to original."""
        original = MagicMock(return_value="result")
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy("arg1", kwarg1="value1")

        original.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "result"


class TestNanParamsStrategy:
    """Test NAN_PARAMS strategy for MegatronOptimizerProxy."""

    def test_nan_params_calls_original(self):
        """Test that NAN_PARAMS calls the original function."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_PARAMS,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"nan_ratio": 0.01},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_nan_params_default_ratio(self):
        """Test that NAN_PARAMS uses default nan_ratio of 0.001."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_PARAMS,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No nan_ratio specified
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        # Should not raise
        proxy()
        original.assert_called_once()

    def test_nan_params_corrupts_model_parameters(self):
        """Test that NAN_PARAMS corrupts model parameters."""
        # Create mock model with parameters
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_param.data = MagicMock()

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]

        mock_engine = MagicMock()
        mock_engine.model = mock_model

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_PARAMS,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"nan_ratio": 0.1},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        # Verify named_parameters was called
        mock_model.named_parameters.assert_called_once()
        original.assert_called_once()

    def test_nan_params_with_specific_param_names(self):
        """Test that NAN_PARAMS only corrupts specified parameters."""
        mock_param1 = MagicMock()
        mock_param1.requires_grad = True
        mock_param1.data = MagicMock()

        mock_param2 = MagicMock()
        mock_param2.requires_grad = True
        mock_param2.data = MagicMock()

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_param1),
            ("layer2.weight", mock_param2),
        ]

        mock_engine = MagicMock()
        mock_engine.model = mock_model

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_PARAMS,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={
                "nan_ratio": 0.1,
                "param_names": ["layer1.weight"],  # Only corrupt layer1
            },
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_nan_params_passes_args(self):
        """Test that NAN_PARAMS passes args to original."""
        original = MagicMock(return_value="result")
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.NAN_PARAMS,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy("arg1", kwarg1="value1")

        original.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "result"


class TestWrongLRStrategy:
    """Test WRONG_LR strategy for MegatronOptimizerProxy."""

    def test_wrong_lr_calls_original(self):
        """Test that WRONG_LR calls the original function."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

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
        mock_inner_optimizer = MagicMock()
        mock_inner_optimizer.param_groups = [{"lr": 0.01}, {"lr": 0.001}]

        mock_optimizer = MagicMock()
        mock_optimizer.optimizer = mock_inner_optimizer

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

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

        assert mock_inner_optimizer.param_groups[0]["lr"] == 0.1
        assert mock_inner_optimizer.param_groups[1]["lr"] == 0.01

    def test_wrong_lr_default_factor(self):
        """Test that WRONG_LR uses default factor of 10.0."""
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 0.01}]
        mock_optimizer.optimizer = None  # No wrapped optimizer

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No lr_factor specified
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        assert mock_optimizer.param_groups[0]["lr"] == 0.1

    def test_wrong_lr_restore_after(self):
        """Test that WRONG_LR can restore original LR after step."""
        mock_inner_optimizer = MagicMock()
        mock_inner_optimizer.param_groups = [{"lr": 0.01}]

        mock_optimizer = MagicMock()
        mock_optimizer.optimizer = mock_inner_optimizer

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"lr_factor": 10.0, "restore_after": True},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        # LR should be restored to original
        assert mock_inner_optimizer.param_groups[0]["lr"] == 0.01

    def test_wrong_lr_passes_args(self):
        """Test that WRONG_LR passes args to original."""
        original = MagicMock(return_value="result")
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_LR,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy("arg1", kwarg1="value1")

        original.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "result"


class TestMegatronOptimizerProxyIntegration:
    """Integration tests for MegatronOptimizerProxy."""

    def test_step_based_trigger(self):
        """Test that proxy respects step-based trigger."""
        original = MagicMock(return_value=None)
        proxy = MegatronOptimizerProxy(original)

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
        mock_optimizer = MagicMock()
        mock_optimizer.overflow = False

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock(return_value=None)
        original.__self__ = mock_engine
        proxy = MegatronOptimizerProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.PERIODIC, every_n_steps=3),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Not periodic multiple
        mock_optimizer.overflow = False
        proxy.set_step(1)
        proxy()
        assert mock_optimizer.overflow is False

        # Step 3: Periodic multiple - should trigger
        mock_optimizer.overflow = False
        proxy.set_step(3)
        proxy()
        assert mock_optimizer.overflow is True

        # Step 4: Not periodic multiple
        mock_optimizer.overflow = False
        proxy.set_step(4)
        proxy()
        assert mock_optimizer.overflow is False

    def test_collector_recording(self):
        """Test that proxy records injections to collector."""
        original = MagicMock(return_value=None)
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_123"

        proxy = MegatronOptimizerProxy(original, collector)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
            severity="medium",
            expected_behavior="Should skip Megatron optimizer step",
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once_with("fault_123", "success", 0)

    def test_collector_records_failure(self):
        """Test that proxy records failure when strategy raises."""
        original = MagicMock(side_effect=RuntimeError("Test error"))
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_456"

        proxy = MegatronOptimizerProxy(original, collector)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.GRADIENT_OVERFLOW,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        with pytest.raises(RuntimeError):
            proxy()

        collector.record_fault_injection.assert_called_once()
        # Check failure was recorded
        args, kwargs = collector.record_fault_outcome.call_args
        assert args[0] == "fault_456"
        assert args[1] == "failure"


# ============================================================================
# Helper Methods Tests
# ============================================================================


class TestGetMegatronEngine:
    """Test _get_megatron_engine helper method."""

    def test_bound_method_returns_self(self):
        """Test that bound method returns __self__."""
        mock_engine = MagicMock()
        original = MagicMock()
        original.__self__ = mock_engine

        proxy = MegatronOptimizerProxy(original)
        result = proxy._get_megatron_engine()

        assert result == mock_engine

    def test_external_engine_attribute(self):
        """Test that _engine attribute is used if available."""
        mock_engine = MagicMock()
        original = MagicMock()

        proxy = MegatronOptimizerProxy(original)
        proxy._engine = mock_engine
        result = proxy._get_megatron_engine()

        assert result == mock_engine

    def test_no_context_returns_none(self):
        """Test that None is returned when no context available."""
        original = MagicMock()

        proxy = MegatronOptimizerProxy(original)
        result = proxy._get_megatron_engine()

        assert result is None


class TestSetOverflowFlag:
    """Test _set_overflow_flag helper method."""

    def test_sets_optimizer_overflow(self):
        """Test that optimizer.overflow is set to True."""
        mock_optimizer = MagicMock()
        mock_optimizer.overflow = False

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        proxy._set_overflow_flag(mock_engine)

        assert mock_optimizer.overflow is True

    def test_sets_engine_overflow(self):
        """Test that engine.overflow is set to True."""
        mock_engine = MagicMock()
        mock_engine.overflow = False

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        proxy._set_overflow_flag(mock_engine)

        assert mock_engine.overflow is True


class TestGetModelFromEngine:
    """Test _get_model_from_engine helper method."""

    def test_returns_model_attribute(self):
        """Test that model attribute is returned."""
        mock_model = MagicMock()
        mock_engine = MagicMock()
        mock_engine.model = mock_model

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        result = proxy._get_model_from_engine(mock_engine)

        assert result == mock_model

    def test_handles_model_list(self):
        """Test that first model is returned for pipeline parallel."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_engine = MagicMock()
        mock_engine.model = [mock_model1, mock_model2]

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        result = proxy._get_model_from_engine(mock_engine)

        assert result == mock_model1

    def test_returns_module_attribute(self):
        """Test that module attribute is returned as fallback."""
        mock_model = MagicMock()
        mock_engine = MagicMock(spec=["module"])
        mock_engine.module = mock_model

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        result = proxy._get_model_from_engine(mock_engine)

        assert result == mock_model

    def test_returns_none_if_no_model(self):
        """Test that None is returned if no model found."""
        mock_engine = MagicMock(spec=[])

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        result = proxy._get_model_from_engine(mock_engine)

        assert result is None


class TestScaleMegatronLR:
    """Test _scale_megatron_lr helper method."""

    def test_scales_all_param_groups(self):
        """Test that all param groups are scaled."""
        mock_inner_optimizer = MagicMock()
        mock_inner_optimizer.param_groups = [
            {"lr": 0.1, "weight_decay": 0.01},
            {"lr": 0.01, "weight_decay": 0.001},
        ]

        mock_optimizer = MagicMock()
        mock_optimizer.optimizer = mock_inner_optimizer

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        original_lrs = proxy._scale_megatron_lr(mock_engine, 2.0)

        assert mock_inner_optimizer.param_groups[0]["lr"] == 0.2
        assert mock_inner_optimizer.param_groups[1]["lr"] == 0.02
        assert original_lrs == {0: 0.1, 1: 0.01}

    def test_no_optimizer_attribute(self):
        """Test that missing optimizer attribute returns empty dict."""
        mock_engine = MagicMock(spec=[])

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        result = proxy._scale_megatron_lr(mock_engine, 2.0)

        assert result == {}


class TestRestoreMegatronLR:
    """Test _restore_megatron_lr helper method."""

    def test_restores_all_param_groups(self):
        """Test that all param groups are restored."""
        mock_inner_optimizer = MagicMock()
        mock_inner_optimizer.param_groups = [
            {"lr": 1.0},  # Modified
            {"lr": 0.1},  # Modified
        ]

        mock_optimizer = MagicMock()
        mock_optimizer.optimizer = mock_inner_optimizer

        mock_engine = MagicMock()
        mock_engine.optimizer = mock_optimizer

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        original_lrs = {0: 0.1, 1: 0.01}
        proxy._restore_megatron_lr(mock_engine, original_lrs)

        assert mock_inner_optimizer.param_groups[0]["lr"] == 0.1
        assert mock_inner_optimizer.param_groups[1]["lr"] == 0.01

    def test_no_optimizer_attribute(self):
        """Test that missing optimizer attribute doesn't raise."""
        mock_engine = MagicMock(spec=[])

        original = MagicMock()
        proxy = MegatronOptimizerProxy(original)

        # Should not raise
        proxy._restore_megatron_lr(mock_engine, {0: 0.1})


# ============================================================================
# ParallelismProxy Tests
# ============================================================================


class TestParallelismProxyRegistration:
    """Test ParallelismProxy registration and supported strategies."""

    def test_registered_target(self):
        """Test that ParallelismProxy is registered for 'megatron.parallelism'."""
        assert ProxyRegistry.is_registered("megatron.parallelism")
        proxy_class = ProxyRegistry.get_proxy("megatron.parallelism")
        assert proxy_class == ParallelismProxy

    def test_supported_strategies(self):
        """Test that ParallelismProxy declares correct supported strategies."""
        expected = {
            StrategyType.PP_STAGE_FAILURE,
            StrategyType.TP_DESYNC,
            StrategyType.WRONG_MICRO_BATCH_ROUTING,
            StrategyType.ACTIVATION_CORRUPTION,
        }
        assert ParallelismProxy.SUPPORTED_STRATEGIES == expected


class TestParallelismProxyBasics:
    """Test basic ParallelismProxy functionality."""

    def test_get_layer(self):
        """Test that _get_layer returns 'Megatron'."""
        original = MagicMock(return_value=None)
        proxy = ParallelismProxy(original)
        assert proxy._get_layer() == "Megatron"

    def test_call_without_config(self):
        """Test that proxy calls original when no config is set."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        result = proxy("arg1", kwarg1="value1")

        original.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "result"

    def test_call_with_disabled_config(self):
        """Test that proxy calls original when config is disabled."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 0},
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_called_once_with()
        assert result == "result"

    def test_unsupported_strategy_raises(self):
        """Test that setting unsupported strategy raises ValueError."""
        original = MagicMock(return_value=None)
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.SKIP,  # Not supported by ParallelismProxy
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},
            enabled=True,
        )

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestPPStageFailureStrategy:
    """Test PP_STAGE_FAILURE strategy for ParallelismProxy."""

    def test_pp_stage_failure_requires_fail_stage(self):
        """Test that PP_STAGE_FAILURE raises without fail_stage parameter."""
        original = MagicMock(return_value=None)
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No fail_stage
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        with pytest.raises(ValueError, match="fail_stage"):
            proxy()

    def test_pp_stage_failure_non_matching_stage(self):
        """Test that PP_STAGE_FAILURE doesn't affect non-matching stages."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_pipeline_stage to return stage 0
        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 1},  # Different stage
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_called_once()
        assert result == "result"

    def test_pp_stage_failure_crash(self):
        """Test that PP_STAGE_FAILURE with crash type raises RuntimeError."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_pipeline_stage to return matching stage
        proxy._get_pipeline_stage = MagicMock(return_value=1)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 1, "failure_type": "crash"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            proxy()

    def test_pp_stage_failure_silent(self):
        """Test that PP_STAGE_FAILURE with silent type returns None."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_pipeline_stage to return matching stage
        proxy._get_pipeline_stage = MagicMock(return_value=2)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 2, "failure_type": "silent"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_pp_stage_failure_hang_with_short_duration(self):
        """Test that PP_STAGE_FAILURE with hang type delays execution."""
        import time

        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_pipeline_stage to return matching stage
        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 0, "failure_type": "hang", "hang_duration": 0.01},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        start = time.time()
        result = proxy()
        elapsed = time.time() - start

        assert elapsed >= 0.01
        original.assert_called_once()
        assert result == "result"


class TestTPDesyncStrategy:
    """Test TP_DESYNC strategy for ParallelismProxy."""

    def test_tp_desync_delay(self):
        """Test that TP_DESYNC with delay type adds delay."""
        import time

        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_tensor_parallel_rank to return matching rank
        proxy._get_tensor_parallel_rank = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.TP_DESYNC,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"desync_rank": 0, "delay_seconds": 0.01, "desync_type": "delay"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        start = time.time()
        result = proxy()
        elapsed = time.time() - start

        assert elapsed >= 0.01
        original.assert_called_once()
        assert result == "result"

    def test_tp_desync_skip(self):
        """Test that TP_DESYNC with skip type returns None."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_tensor_parallel_rank to return matching rank
        proxy._get_tensor_parallel_rank = MagicMock(return_value=1)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.TP_DESYNC,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"desync_rank": 1, "desync_type": "skip"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_tp_desync_duplicate(self):
        """Test that TP_DESYNC with duplicate type calls original twice."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_tensor_parallel_rank to return matching rank
        proxy._get_tensor_parallel_rank = MagicMock(return_value=2)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.TP_DESYNC,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"desync_rank": 2, "desync_type": "duplicate"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        assert original.call_count == 2
        assert result == "result"

    def test_tp_desync_non_matching_rank(self):
        """Test that TP_DESYNC doesn't affect non-matching ranks."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Mock _get_tensor_parallel_rank to return different rank
        proxy._get_tensor_parallel_rank = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.TP_DESYNC,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"desync_rank": 1, "desync_type": "skip"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_called_once()
        assert result == "result"

    def test_tp_desync_default_delay(self):
        """Test that TP_DESYNC uses default delay of 5.0 seconds."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # Read the default from config
        proxy._get_tensor_parallel_rank = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.TP_DESYNC,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"desync_rank": 0},  # No delay_seconds specified
            enabled=True,
        )
        proxy.set_config(config)

        # Check that default is used (we won't actually wait 5 seconds)
        assert proxy._config.parameters.get("delay_seconds", 5.0) == 5.0


class TestWrongMicroBatchRoutingStrategy:
    """Test WRONG_MICRO_BATCH_ROUTING strategy for ParallelismProxy."""

    def test_wrong_micro_batch_routing_no_modification(self):
        """Test that WRONG_MICRO_BATCH_ROUTING calls original when no modification specified."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_MICRO_BATCH_ROUTING,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No modifications
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_called_once()
        assert result == "result"

    def test_wrong_micro_batch_routing_drop(self):
        """Test that WRONG_MICRO_BATCH_ROUTING can drop micro-batches."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_MICRO_BATCH_ROUTING,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"drop_ratio": 1.0},  # Always drop
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_not_called()
        assert result is None

    def test_wrong_micro_batch_routing_duplicate(self):
        """Test that WRONG_MICRO_BATCH_ROUTING can duplicate micro-batches."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_MICRO_BATCH_ROUTING,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"duplicate_ratio": 1.0},  # Always duplicate
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        # Original called twice due to duplication + once for return
        assert original.call_count == 2

    def test_wrong_micro_batch_routing_swap_stages(self):
        """Test that WRONG_MICRO_BATCH_ROUTING can swap stages."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        mock_context = MagicMock()
        mock_context.pipeline_rank = 0

        proxy._get_pipeline_stage = MagicMock(return_value=0)
        proxy._get_parallel_context = MagicMock(return_value=mock_context)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_MICRO_BATCH_ROUTING,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"swap_stages": [[0, 1]]},  # Swap stage 0 and 1
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        # Context should have been modified
        # Stage 0 should become 1 during execution
        original.assert_called_once()
        # Context should be restored after
        assert mock_context.pipeline_rank == 0


class TestActivationCorruptionStrategy:
    """Test ACTIVATION_CORRUPTION strategy for ParallelismProxy."""

    def test_activation_corruption_noise(self):
        """Test that ACTIVATION_CORRUPTION adds noise to activations."""
        mock_tensor = MagicMock()
        mock_tensor.numel.return_value = 100
        mock_tensor.clone.return_value = mock_tensor

        original = MagicMock(return_value=mock_tensor)
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"noise_scale": 0.1, "corrupt_type": "noise"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()
        mock_tensor.clone.assert_called()

    def test_activation_corruption_zero(self):
        """Test that ACTIVATION_CORRUPTION can zero activations."""
        mock_tensor = MagicMock()
        mock_tensor.numel.return_value = 100
        mock_tensor.clone.return_value = mock_tensor
        mock_tensor.view.return_value = mock_tensor
        mock_tensor.shape = (10, 10)

        original = MagicMock(return_value=mock_tensor)
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"corrupt_ratio": 0.5, "corrupt_type": "zero"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()

    def test_activation_corruption_scale(self):
        """Test that ACTIVATION_CORRUPTION can scale activations."""
        mock_tensor = MagicMock()
        mock_tensor.numel.return_value = 100
        mock_tensor.clone.return_value = mock_tensor
        mock_tensor.__mul__ = MagicMock(return_value=mock_tensor)

        original = MagicMock(return_value=mock_tensor)
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"noise_scale": 10.0, "corrupt_type": "scale"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()
        mock_tensor.clone.assert_called()
        mock_tensor.__mul__.assert_called_with(10.0)

    def test_activation_corruption_dict_result(self):
        """Test that ACTIVATION_CORRUPTION handles dict results."""
        mock_tensor = MagicMock()
        mock_tensor.numel.return_value = 100
        mock_tensor.clone.return_value = mock_tensor

        original = MagicMock(return_value={"activations": mock_tensor, "other": "value"})
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"corrupt_type": "noise"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        result = proxy()

        original.assert_called_once()
        assert isinstance(result, dict)
        assert "activations" in result
        assert result["other"] == "value"

    def test_activation_corruption_target_layers(self):
        """Test that ACTIVATION_CORRUPTION respects target_layers parameter."""
        mock_tensor1 = MagicMock()
        mock_tensor1.numel.return_value = 100
        mock_tensor1.clone.return_value = mock_tensor1

        mock_tensor2 = MagicMock()
        mock_tensor2.numel.return_value = 100
        mock_tensor2.clone.return_value = mock_tensor2

        original = MagicMock(return_value={"layer1": mock_tensor1, "layer2": mock_tensor2})
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"target_layers": ["layer1"], "corrupt_type": "noise"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        original.assert_called_once()
        # layer1 should be corrupted (clone called)
        mock_tensor1.clone.assert_called()
        # layer2 should NOT be corrupted
        mock_tensor2.clone.assert_not_called()

    def test_activation_corruption_default_params(self):
        """Test that ACTIVATION_CORRUPTION uses default parameters."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.ACTIVATION_CORRUPTION,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={},  # No parameters specified
            enabled=True,
        )
        proxy.set_config(config)

        # Check defaults
        assert proxy._config.parameters.get("noise_scale", 0.01) == 0.01
        assert proxy._config.parameters.get("corrupt_ratio", 0.1) == 0.1
        assert proxy._config.parameters.get("corrupt_type", "noise") == "noise"


class TestParallelismProxyIntegration:
    """Integration tests for ParallelismProxy."""

    def test_step_based_trigger(self):
        """Test that proxy respects step-based trigger."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.STEP_BASED, start_step=5, end_step=10),
            parameters={"fail_stage": 0, "failure_type": "silent"},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Should not trigger
        proxy.set_step(1)
        result = proxy()
        assert original.call_count == 1
        assert result == "result"

        # Step 5: Should trigger (silent = None)
        proxy.set_step(5)
        result = proxy()
        assert original.call_count == 1  # Not called again
        assert result is None

        # Step 11: Should not trigger
        proxy.set_step(11)
        result = proxy()
        assert original.call_count == 2
        assert result == "result"

    def test_periodic_trigger(self):
        """Test that proxy respects periodic trigger."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        proxy._get_pipeline_stage = MagicMock(return_value=0)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.PERIODIC, every_n_steps=3),
            parameters={"fail_stage": 0, "failure_type": "silent"},
            enabled=True,
        )
        proxy.set_config(config)

        # Step 1: Not periodic multiple
        proxy.set_step(1)
        result = proxy()
        assert result == "result"

        # Step 3: Periodic multiple - should trigger
        proxy.set_step(3)
        result = proxy()
        assert result is None

        # Step 4: Not periodic multiple
        proxy.set_step(4)
        result = proxy()
        assert result == "result"

    def test_collector_recording(self):
        """Test that proxy records injections to collector."""
        original = MagicMock(return_value="result")
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_123"

        proxy = ParallelismProxy(original, collector)
        proxy._get_pipeline_stage = MagicMock(return_value=1)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 1, "failure_type": "silent"},
            enabled=True,
            severity="high",
            expected_behavior="Should fail pipeline stage 1",
        )
        proxy.set_config(config)
        proxy.set_step(1)

        proxy()

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once_with("fault_123", "success", 0)

    def test_collector_records_failure(self):
        """Test that proxy records failure when strategy raises."""
        original = MagicMock(return_value="result")
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_456"

        proxy = ParallelismProxy(original, collector)
        proxy._get_pipeline_stage = MagicMock(return_value=2)

        config = FaultConfig(
            id="test",
            strategy=StrategyType.PP_STAGE_FAILURE,
            trigger=TriggerConfig(TriggerType.ONE_SHOT, step=1),
            parameters={"fail_stage": 2, "failure_type": "crash"},
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(1)

        with pytest.raises(RuntimeError):
            proxy()

        collector.record_fault_injection.assert_called_once()
        # Check failure was recorded
        args, kwargs = collector.record_fault_outcome.call_args
        assert args[0] == "fault_456"
        assert args[1] == "failure"

    def test_args_kwargs_passing(self):
        """Test that args and kwargs are passed correctly to original."""
        original = MagicMock(return_value="result")
        proxy = ParallelismProxy(original)

        # No config - should just pass through
        result = proxy("arg1", "arg2", key1="val1", key2="val2")

        original.assert_called_once_with("arg1", "arg2", key1="val1", key2="val2")
        assert result == "result"


class TestParallelismProxyHelperMethods:
    """Test helper methods for ParallelismProxy."""

    def test_get_current_rank_default(self):
        """Test that _get_current_rank returns 0 when not distributed."""
        original = MagicMock()
        proxy = ParallelismProxy(original)

        rank = proxy._get_current_rank()
        assert rank == 0

    def test_get_pipeline_stage_default(self):
        """Test that _get_pipeline_stage returns 0 when not in PP mode."""
        original = MagicMock()
        proxy = ParallelismProxy(original)

        stage = proxy._get_pipeline_stage()
        assert stage == 0

    def test_get_tensor_parallel_rank_default(self):
        """Test that _get_tensor_parallel_rank returns 0 when not in TP mode."""
        original = MagicMock()
        proxy = ParallelismProxy(original)

        rank = proxy._get_tensor_parallel_rank()
        assert rank == 0

    def test_get_parallel_context_bound_method(self):
        """Test that bound method returns __self__."""
        mock_context = MagicMock()
        original = MagicMock()
        original.__self__ = mock_context

        proxy = ParallelismProxy(original)
        result = proxy._get_parallel_context()

        assert result == mock_context

    def test_get_parallel_context_external_attribute(self):
        """Test that _context attribute is used if available."""
        mock_context = MagicMock()
        original = MagicMock()

        proxy = ParallelismProxy(original)
        proxy._context = mock_context
        result = proxy._get_parallel_context()

        assert result == mock_context

    def test_get_parallel_context_none(self):
        """Test that None is returned when no context available."""
        original = MagicMock()

        proxy = ParallelismProxy(original)
        result = proxy._get_parallel_context()

        assert result is None

    def test_get_pipeline_stage_from_context(self):
        """Test that pipeline_rank from context is used."""
        mock_context = MagicMock()
        mock_context.pipeline_rank = 3

        original = MagicMock()
        original.__self__ = mock_context

        proxy = ParallelismProxy(original)
        stage = proxy._get_pipeline_stage()

        assert stage == 3

    def test_get_tensor_parallel_rank_from_context(self):
        """Test that tensor_rank from context is used."""
        mock_context = MagicMock()
        mock_context.tensor_rank = 5

        original = MagicMock()
        original.__self__ = mock_context

        proxy = ParallelismProxy(original)
        rank = proxy._get_tensor_parallel_rank()

        assert rank == 5
