"""
Unit tests for RayGetProxy class.

Tests cover all 5 supported strategies:
- DELAY: Adds delay before calling original ray.get
- RAISE_EXCEPTION: Raises a configurable exception
- CORRUPT_TENSOR: Corrupts tensor data with Gaussian noise
- OBJECT_LOST: Raises ObjectLostError simulating lost objects
- PARTIAL_FAILURE: Sets a ratio of results to None
"""

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.l0_ray import ObjectLostError, RayGetProxy


class TestRayGetProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_ray_get_target(self):
        """RayGetProxy is registered for 'ray.get' target."""
        assert ProxyRegistry.is_registered("ray.get")
        assert ProxyRegistry.get_proxy("ray.get") is RayGetProxy

    def test_supported_strategies(self):
        """RayGetProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("ray.get")
        expected = {
            StrategyType.DELAY,
            StrategyType.RAISE_EXCEPTION,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.OBJECT_LOST,
            StrategyType.PARTIAL_FAILURE,
        }
        assert strategies == expected


class TestRayGetProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_l0(self):
        """_get_layer returns 'L0'."""
        proxy = RayGetProxy(lambda x: x)
        assert proxy._get_layer() == "L0"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        result = proxy("arg1", kwarg="value")
        original.assert_called_once_with("arg1", kwarg="value")
        assert result == "result"

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("arg")
        original.assert_called_once_with("arg")
        assert result == "result"


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value="delayed_result")
        proxy = RayGetProxy(original)
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
            result = proxy("refs")
            mock_sleep.assert_called_once_with(0.01)
            assert result == "delayed_result"

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
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
            proxy("refs")
            mock_sleep.assert_called_once_with(10.0)  # Default is 10 seconds


class TestRaiseExceptionStrategy:
    """Tests for RAISE_EXCEPTION strategy."""

    def test_raise_exception_strategy(self):
        """RAISE_EXCEPTION strategy raises configured exception."""
        original = MagicMock()
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-exception",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError", "message": "Test error"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy("refs")
        assert "Test error" in str(exc_info.value)
        original.assert_not_called()

    def test_raise_exception_timeout_error(self):
        """RAISE_EXCEPTION strategy can raise TimeoutError."""
        proxy = RayGetProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-timeout",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "TimeoutError", "message": "Ray get timeout"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(TimeoutError) as exc_info:
            proxy("refs")
        assert "Ray get timeout" in str(exc_info.value)


class TestCorruptTensorStrategy:
    """Tests for CORRUPT_TENSOR strategy."""

    def test_corrupt_tensor_single_result(self):
        """CORRUPT_TENSOR strategy corrupts single tensor result."""
        tensor = torch.ones(10, 10)
        original = MagicMock(return_value=tensor.clone())
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("refs")
        # Result should be different due to noise
        assert not torch.equal(result, tensor)
        # Shape should be preserved
        assert result.shape == tensor.shape

    def test_corrupt_tensor_list_result(self):
        """CORRUPT_TENSOR strategy corrupts list of tensors."""
        tensors = [torch.ones(5), torch.zeros(5)]
        original = MagicMock(return_value=[t.clone() for t in tensors])
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-list",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(["ref1", "ref2"])
        assert len(result) == 2
        assert result[0].shape == tensors[0].shape
        assert result[1].shape == tensors[1].shape

    def test_corrupt_tensor_preserves_non_tensors(self):
        """CORRUPT_TENSOR strategy preserves non-tensor values."""
        mixed_result = {"tensor": torch.ones(5), "string": "hello", "number": 42}
        original = MagicMock(return_value=mixed_result.copy())
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-mixed",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("refs")
        assert result["string"] == "hello"
        assert result["number"] == 42
        assert result["tensor"].shape == torch.ones(5).shape


class TestObjectLostStrategy:
    """Tests for OBJECT_LOST strategy."""

    def test_object_lost_raises_error(self):
        """OBJECT_LOST strategy raises ObjectLostError."""
        original = MagicMock()
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-object-lost",
            strategy=StrategyType.OBJECT_LOST,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ObjectLostError) as exc_info:
            proxy("single_ref")
        assert "ralph_simulated" in str(exc_info.value)
        assert "ralph_fault_injection" in str(exc_info.value)
        original.assert_not_called()

    def test_object_lost_with_list_refs(self):
        """OBJECT_LOST strategy with list of refs uses first ref."""
        original = MagicMock()
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-object-lost-list",
            strategy=StrategyType.OBJECT_LOST,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        refs = ["ref1", "ref2", "ref3"]
        with pytest.raises(ObjectLostError) as exc_info:
            proxy(refs)
        # First ref should be in the error
        assert exc_info.value.object_ref == "ref1"

    def test_object_lost_error_attributes(self):
        """ObjectLostError has correct attributes."""
        error = ObjectLostError(
            object_ref="test_ref",
            owner_address="test_owner",
            call_site="test_site",
        )
        assert error.object_ref == "test_ref"
        assert error.owner_address == "test_owner"
        assert error.call_site == "test_site"


class TestPartialFailureStrategy:
    """Tests for PARTIAL_FAILURE strategy."""

    def test_partial_failure_single_ref(self):
        """PARTIAL_FAILURE with single ref may return None."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-partial",
            strategy=StrategyType.PARTIAL_FAILURE,
            trigger=trigger,
            parameters={"fail_ratio": 1.0},  # Always fail
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("single_ref")
        assert result is None

    def test_partial_failure_never_fails(self):
        """PARTIAL_FAILURE with fail_ratio=0 never fails."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-no-fail",
            strategy=StrategyType.PARTIAL_FAILURE,
            trigger=trigger,
            parameters={"fail_ratio": 0.0},  # Never fail
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("single_ref")
        assert result == "result"

    def test_partial_failure_list_results(self):
        """PARTIAL_FAILURE sets some list results to None."""
        original = MagicMock(return_value=["r1", "r2", "r3", "r4"])
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-partial-list",
            strategy=StrategyType.PARTIAL_FAILURE,
            trigger=trigger,
            parameters={"fail_ratio": 1.0},  # All fail
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(["ref1", "ref2", "ref3", "ref4"])
        assert all(r is None for r in result)

    def test_partial_failure_default_ratio(self):
        """PARTIAL_FAILURE uses default 0.5 ratio."""
        results = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"]
        original = MagicMock(return_value=results.copy())
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-default-ratio",
            strategy=StrategyType.PARTIAL_FAILURE,
            trigger=trigger,
            parameters={},  # No fail_ratio, should use 0.5
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Seed random for determinism
        random.seed(42)
        result = proxy(["ref" + str(i) for i in range(10)])

        # With 50% probability, roughly half should be None
        none_count = sum(1 for r in result if r is None)
        # Allow some variance but expect around 5 Nones
        assert 0 < none_count < 10

    def test_partial_failure_passes_kwargs(self):
        """PARTIAL_FAILURE passes timeout and other kwargs."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-kwargs",
            strategy=StrategyType.PARTIAL_FAILURE,
            trigger=trigger,
            parameters={"fail_ratio": 0.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("ref", timeout=30)
        original.assert_called_once_with("ref", timeout=30)


class TestRayGetProxyIntegration:
    """Integration tests for RayGetProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.OBJECT_LOST,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0-4 should work
        for step in range(5):
            proxy.set_step(step)
            result = proxy("ref")
            assert result == "result"

        # Step 5 should raise
        proxy.set_step(5)
        with pytest.raises(ObjectLostError):
            proxy("ref")

        # Step 6 should work again (one-shot)
        proxy.set_step(6)
        result = proxy("ref")
        assert result == "result"

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.OBJECT_LOST,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0, 3, 6 should trigger (step % 3 == 0)
        trigger_steps = []
        for step in range(10):
            proxy.set_step(step)
            try:
                proxy("ref")
            except ObjectLostError:
                trigger_steps.append(step)

        assert trigger_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"
        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
            severity="high",
            expected_behavior="Should delay execution",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy("refs")

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args
        assert call_kwargs[1]["fault_type"] == "delay"
        assert call_kwargs[1]["target_layer"] == "L0"
        assert call_kwargs[1]["severity"] == "high"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-123"
        assert collector.record_fault_outcome.call_args[0][1] == "success"
