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


# =============================================================================
# RayPutProxy Tests
# =============================================================================

from ralph.proxies.l0_ray import DummyObjectRef, ObjectStoreFullError, RayPutProxy


class TestRayPutProxyRegistration:
    """Tests for RayPutProxy registration."""

    def test_registered_with_ray_put_target(self):
        """RayPutProxy is registered for 'ray.put' target."""
        assert ProxyRegistry.is_registered("ray.put")
        assert ProxyRegistry.get_proxy("ray.put") is RayPutProxy

    def test_supported_strategies(self):
        """RayPutProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("ray.put")
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.STORE_FULL,
            StrategyType.SILENT_DROP,
        }
        assert strategies == expected


class TestRayPutProxyBasics:
    """Tests for basic RayPutProxy functionality."""

    def test_get_layer_returns_l0(self):
        """_get_layer returns 'L0'."""
        proxy = RayPutProxy(lambda x: x)
        assert proxy._get_layer() == "L0"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value="object_ref")
        proxy = RayPutProxy(original)
        result = proxy("value", kwarg="extra")
        original.assert_called_once_with("value", kwarg="extra")
        assert result == "object_ref"

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value="object_ref")
        proxy = RayPutProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("value")
        original.assert_called_once_with("value")
        assert result == "object_ref"


class TestRayPutDelayStrategy:
    """Tests for RayPutProxy DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy applies configured delay."""
        original = MagicMock(return_value="delayed_ref")
        proxy = RayPutProxy(original)
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
            result = proxy("data")
            mock_sleep.assert_called_once_with(0.01)
            assert result == "delayed_ref"

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay when not specified."""
        original = MagicMock(return_value="ref")
        proxy = RayPutProxy(original)
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
            proxy("data")
            mock_sleep.assert_called_once_with(10.0)


class TestRayPutCorruptTensorStrategy:
    """Tests for RayPutProxy CORRUPT_TENSOR strategy."""

    def test_corrupt_tensor_before_put(self):
        """CORRUPT_TENSOR strategy corrupts tensor before storing."""
        tensor = torch.ones(10, 10)
        stored_value = None

        def capture_put(value):
            nonlocal stored_value
            stored_value = value.clone() if isinstance(value, torch.Tensor) else value
            return "ref"

        proxy = RayPutProxy(capture_put)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 1.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(tensor.clone())
        assert result == "ref"
        # Stored value should be corrupted (different from original)
        assert stored_value is not None
        assert not torch.equal(stored_value, tensor)
        assert stored_value.shape == tensor.shape

    def test_corrupt_tensor_dict_value(self):
        """CORRUPT_TENSOR strategy corrupts tensors in dict values."""
        value = {"tensor": torch.ones(5), "string": "hello"}
        stored_value = None

        def capture_put(v):
            nonlocal stored_value
            stored_value = v
            return "ref"

        proxy = RayPutProxy(capture_put)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-corrupt-dict",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy({"tensor": torch.ones(5), "string": "hello"})
        assert stored_value["string"] == "hello"  # Non-tensor preserved
        assert stored_value["tensor"].shape == torch.ones(5).shape

    def test_corrupt_tensor_default_noise_scale(self):
        """CORRUPT_TENSOR strategy uses default noise scale."""
        tensor = torch.zeros(100)
        stored_value = None

        def capture_put(v):
            nonlocal stored_value
            stored_value = v.clone() if isinstance(v, torch.Tensor) else v
            return "ref"

        proxy = RayPutProxy(capture_put)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-default",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={},  # Default noise_scale = 0.1
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(tensor.clone())
        # With default 0.1 noise scale, values should be small but non-zero
        assert not torch.equal(stored_value, tensor)


class TestStoreFulStrategy:
    """Tests for RayPutProxy STORE_FULL strategy."""

    def test_store_full_raises_error(self):
        """STORE_FULL strategy raises ObjectStoreFullError."""
        original = MagicMock()
        proxy = RayPutProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-store-full",
            strategy=StrategyType.STORE_FULL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ObjectStoreFullError) as exc_info:
            proxy("data_to_store")
        assert "Object store full" in str(exc_info.value)
        original.assert_not_called()

    def test_store_full_custom_message(self):
        """STORE_FULL strategy uses custom message."""
        proxy = RayPutProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-custom-msg",
            strategy=StrategyType.STORE_FULL,
            trigger=trigger,
            parameters={"message": "Custom error: storage exhausted"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ObjectStoreFullError) as exc_info:
            proxy("data")
        assert "Custom error: storage exhausted" in str(exc_info.value)

    def test_store_full_error_attributes(self):
        """ObjectStoreFullError has correct attributes."""
        error = ObjectStoreFullError("test message")
        assert error.message == "test message"
        assert str(error) == "test message"


class TestSilentDropStrategy:
    """Tests for RayPutProxy SILENT_DROP strategy."""

    def test_silent_drop_returns_dummy_ref(self):
        """SILENT_DROP strategy returns DummyObjectRef."""
        original = MagicMock()
        proxy = RayPutProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-silent-drop",
            strategy=StrategyType.SILENT_DROP,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("data_to_drop")
        assert isinstance(result, DummyObjectRef)
        original.assert_not_called()

    def test_silent_drop_custom_object_id(self):
        """SILENT_DROP strategy uses custom object_id."""
        proxy = RayPutProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-custom-id",
            strategy=StrategyType.SILENT_DROP,
            trigger=trigger,
            parameters={"object_id": "custom_id_123"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("data")
        assert isinstance(result, DummyObjectRef)
        assert result._object_id == "custom_id_123"

    def test_silent_drop_default_object_id(self):
        """SILENT_DROP uses unique default object_id based on value id."""
        proxy = RayPutProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-default-id",
            strategy=StrategyType.SILENT_DROP,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        data = "my_data"
        result = proxy(data)
        assert isinstance(result, DummyObjectRef)
        assert result._object_id.startswith("dropped_")


class TestDummyObjectRef:
    """Tests for DummyObjectRef class."""

    def test_dummy_ref_repr(self):
        """DummyObjectRef has correct repr."""
        ref = DummyObjectRef("test_id")
        assert repr(ref) == "DummyObjectRef(test_id)"

    def test_dummy_ref_hash(self):
        """DummyObjectRef is hashable."""
        ref1 = DummyObjectRef("id1")
        ref2 = DummyObjectRef("id1")
        ref3 = DummyObjectRef("id2")
        assert hash(ref1) == hash(ref2)
        assert hash(ref1) != hash(ref3)

    def test_dummy_ref_equality(self):
        """DummyObjectRef equality works correctly."""
        ref1 = DummyObjectRef("id1")
        ref2 = DummyObjectRef("id1")
        ref3 = DummyObjectRef("id2")
        assert ref1 == ref2
        assert ref1 != ref3
        assert ref1 != "id1"  # Not equal to non-DummyObjectRef

    def test_dummy_ref_default_id(self):
        """DummyObjectRef uses default id if not provided."""
        ref = DummyObjectRef()
        assert ref._object_id == "dummy"


class TestRayPutProxyIntegration:
    """Integration tests for RayPutProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        original = MagicMock(return_value="ref")
        proxy = RayPutProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.STORE_FULL,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0-4 should work
        for step in range(5):
            proxy.set_step(step)
            result = proxy("data")
            assert result == "ref"

        # Step 5 should raise
        proxy.set_step(5)
        with pytest.raises(ObjectStoreFullError):
            proxy("data")

        # Step 6 should work again (one-shot)
        proxy.set_step(6)
        result = proxy("data")
        assert result == "ref"

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value="ref")
        proxy = RayPutProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.STORE_FULL,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Steps 0, 3, 6 should trigger (step % 3 == 0)
        trigger_steps = []
        for step in range(10):
            proxy.set_step(step)
            try:
                proxy("data")
            except ObjectStoreFullError:
                trigger_steps.append(step)

        assert trigger_steps == [0, 3, 6, 9]

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-456"
        original = MagicMock(return_value="ref")
        proxy = RayPutProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.0},
            severity="medium",
            expected_behavior="Should delay put operation",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            proxy("data")

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args
        assert call_kwargs[1]["fault_type"] == "delay"
        assert call_kwargs[1]["target_layer"] == "L0"
        assert call_kwargs[1]["severity"] == "medium"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-456"
        assert collector.record_fault_outcome.call_args[0][1] == "success"

    def test_collector_records_failure(self):
        """Collector records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-789"
        proxy = RayPutProxy(MagicMock(), collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-failure",
            strategy=StrategyType.STORE_FULL,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ObjectStoreFullError):
            proxy("data")

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()
        # Check outcome is "raised" for exception
        assert collector.record_fault_outcome.call_args[0][1] == "raised"


# =============================================================================
# ExecuteAllProxy Tests
# =============================================================================

from ralph.proxies.l0_ray import ExecuteAllProxy, WorkerDeathError


class TestExecuteAllProxyRegistration:
    """Tests for ExecuteAllProxy registration."""

    def test_registered_with_execute_all_sync_target(self):
        """ExecuteAllProxy is registered for 'execute_all_sync' target."""
        assert ProxyRegistry.is_registered("execute_all_sync")
        assert ProxyRegistry.get_proxy("execute_all_sync") is ExecuteAllProxy

    def test_supported_strategies(self):
        """ExecuteAllProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("execute_all_sync")
        expected = {
            StrategyType.WORKER_DEATH,
            StrategyType.STRAGGLER,
            StrategyType.SKIP_WORKER,
            StrategyType.DUPLICATE_CALL,
        }
        assert strategies == expected


class TestExecuteAllProxyBasics:
    """Tests for basic ExecuteAllProxy functionality."""

    def test_get_layer_returns_l0(self):
        """_get_layer returns 'L0'."""
        proxy = ExecuteAllProxy(lambda method, *args: [])
        assert proxy._get_layer() == "L0"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value=["result1", "result2"])
        proxy = ExecuteAllProxy(original)
        result = proxy("test_method", arg1="value")
        original.assert_called_once_with("test_method", arg1="value")
        assert result == ["result1", "result2"]

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original = MagicMock(return_value=["result"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WORKER_DEATH,
            trigger=trigger,
            enabled=False,
            parameters={"kill_worker_idx": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("method")
        original.assert_called_once_with("method")
        assert result == ["result"]

    def test_unsupported_strategy_raises_error(self):
        """Setting unsupported strategy raises ValueError."""
        proxy = ExecuteAllProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by ExecuteAllProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestWorkerDeathStrategy:
    """Tests for WORKER_DEATH strategy."""

    def test_worker_death_raises_error_without_ray(self):
        """WORKER_DEATH strategy raises WorkerDeathError when Ray not available."""
        original = MagicMock()
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-worker-death",
            strategy=StrategyType.WORKER_DEATH,
            trigger=trigger,
            parameters={"kill_worker_idx": 1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Without actual Ray, it raises WorkerDeathError
        with pytest.raises(WorkerDeathError) as exc_info:
            proxy("generate_sequences")
        assert exc_info.value.worker_index == 1
        assert "Worker 1 death" in str(exc_info.value)

    def test_worker_death_requires_kill_worker_idx(self):
        """WORKER_DEATH strategy requires kill_worker_idx parameter."""
        proxy = ExecuteAllProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-idx",
            strategy=StrategyType.WORKER_DEATH,
            trigger=trigger,
            parameters={},  # Missing kill_worker_idx
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("method")
        assert "kill_worker_idx" in str(exc_info.value)


class TestStragglerStrategy:
    """Tests for STRAGGLER strategy."""

    def test_straggler_adds_delay(self):
        """STRAGGLER strategy adds delay before execution."""
        original = MagicMock(return_value=["result1", "result2"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-straggler",
            strategy=StrategyType.STRAGGLER,
            trigger=trigger,
            parameters={"straggler_idx": 0, "delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy("method")
            mock_sleep.assert_called_once_with(0.01)
            assert result == ["result1", "result2"]

    def test_straggler_requires_straggler_idx(self):
        """STRAGGLER strategy requires straggler_idx parameter."""
        proxy = ExecuteAllProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-idx",
            strategy=StrategyType.STRAGGLER,
            trigger=trigger,
            parameters={"delay_seconds": 10},  # Missing straggler_idx
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("method")
        assert "straggler_idx" in str(exc_info.value)

    def test_straggler_default_delay(self):
        """STRAGGLER strategy uses default 60s delay."""
        original = MagicMock(return_value=["result"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-default-delay",
            strategy=StrategyType.STRAGGLER,
            trigger=trigger,
            parameters={"straggler_idx": 0},  # No delay_seconds
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy("method")
            mock_sleep.assert_called_once_with(60.0)


class TestSkipWorkerStrategy:
    """Tests for SKIP_WORKER strategy."""

    def test_skip_worker_removes_result(self):
        """SKIP_WORKER strategy removes result at specified index."""
        original = MagicMock(return_value=["r0", "r1", "r2", "r3"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        # Result at index 2 ("r2") should be removed
        assert result == ["r0", "r1", "r3"]
        assert len(result) == 3

    def test_skip_worker_first_index(self):
        """SKIP_WORKER can skip first worker (index 0)."""
        original = MagicMock(return_value=["first", "second", "third"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip-first",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["second", "third"]

    def test_skip_worker_last_index(self):
        """SKIP_WORKER can skip last worker."""
        original = MagicMock(return_value=["first", "second", "third"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-skip-last",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["first", "second"]

    def test_skip_worker_requires_skip_idx(self):
        """SKIP_WORKER strategy requires skip_idx parameter."""
        proxy = ExecuteAllProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-idx",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("method")
        assert "skip_idx" in str(exc_info.value)

    def test_skip_worker_out_of_range_does_nothing(self):
        """SKIP_WORKER with out of range index leaves results unchanged."""
        original = MagicMock(return_value=["r0", "r1"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-out-of-range",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 10},  # Out of range
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["r0", "r1"]  # Unchanged


class TestDuplicateCallStrategy:
    """Tests for DUPLICATE_CALL strategy."""

    def test_duplicate_call_duplicates_result(self):
        """DUPLICATE_CALL strategy duplicates result at specified index."""
        original = MagicMock(return_value=["r0", "r1", "r2"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-duplicate",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={"duplicate_idx": 1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        # Result at index 1 should be duplicated
        assert result == ["r0", "r1", "r1", "r2"]
        assert len(result) == 4

    def test_duplicate_call_first_index(self):
        """DUPLICATE_CALL can duplicate first worker result."""
        original = MagicMock(return_value=["first", "second"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-dup-first",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={"duplicate_idx": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["first", "first", "second"]

    def test_duplicate_call_last_index(self):
        """DUPLICATE_CALL can duplicate last worker result."""
        original = MagicMock(return_value=["first", "second", "third"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-dup-last",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={"duplicate_idx": 2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["first", "second", "third", "third"]

    def test_duplicate_call_requires_duplicate_idx(self):
        """DUPLICATE_CALL strategy requires duplicate_idx parameter."""
        proxy = ExecuteAllProxy(MagicMock())
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-missing-idx",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("method")
        assert "duplicate_idx" in str(exc_info.value)

    def test_duplicate_call_out_of_range_does_nothing(self):
        """DUPLICATE_CALL with out of range index leaves results unchanged."""
        original = MagicMock(return_value=["r0", "r1"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-out-of-range",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={"duplicate_idx": 10},  # Out of range
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("method")
        assert result == ["r0", "r1"]  # Unchanged


class TestWorkerDeathErrorClass:
    """Tests for WorkerDeathError exception class."""

    def test_error_attributes(self):
        """WorkerDeathError has correct attributes."""
        error = WorkerDeathError(worker_index=2, message="Test kill")
        assert error.worker_index == 2
        assert error.message == "Test kill"
        assert "Worker 2 death" in str(error)

    def test_error_default_message(self):
        """WorkerDeathError uses default message."""
        error = WorkerDeathError(worker_index=0)
        assert error.worker_index == 0
        assert "fault injection" in error.message


class TestExecuteAllProxyIntegration:
    """Integration tests for ExecuteAllProxy."""

    def test_strategy_only_triggers_at_configured_step(self):
        """Strategy only triggers at configured step."""
        original = MagicMock(return_value=["r0", "r1", "r2"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        config = FaultConfig(
            id="test-step",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 1},
        )
        proxy.set_config(config)

        # Steps 0-4 should return full results
        for step in range(5):
            proxy.set_step(step)
            result = proxy("method")
            assert result == ["r0", "r1", "r2"]

        # Step 5 should skip worker
        proxy.set_step(5)
        result = proxy("method")
        assert result == ["r0", "r2"]  # r1 skipped

        # Step 6 should return full results again (one-shot)
        proxy.set_step(6)
        result = proxy("method")
        assert result == ["r0", "r1", "r2"]

    def test_periodic_trigger(self):
        """Periodic trigger triggers at intervals."""
        original = MagicMock(return_value=["r0", "r1", "r2"])
        proxy = ExecuteAllProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test-periodic",
            strategy=StrategyType.DUPLICATE_CALL,
            trigger=trigger,
            parameters={"duplicate_idx": 0},
        )
        proxy.set_config(config)

        # Collect results at different steps
        results = {}
        for step in range(10):
            proxy.set_step(step)
            results[step] = proxy("method")

        # Steps 0, 3, 6, 9 should have duplicate (4 items)
        for step in [0, 3, 6, 9]:
            assert len(results[step]) == 4, f"Step {step} should have duplicated result"

        # Other steps should have normal results (3 items)
        for step in [1, 2, 4, 5, 7, 8]:
            assert len(results[step]) == 3, f"Step {step} should have normal result"

    def test_collector_records_injection(self):
        """Collector records fault injection when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-exec-123"
        original = MagicMock(return_value=["r0", "r1"])
        proxy = ExecuteAllProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-recording",
            strategy=StrategyType.SKIP_WORKER,
            trigger=trigger,
            parameters={"skip_idx": 0},
            severity="high",
            expected_behavior="Should skip first worker",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("test_method")

        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args
        assert call_kwargs[1]["fault_type"] == "skip_worker"
        assert call_kwargs[1]["target_layer"] == "L0"
        assert call_kwargs[1]["severity"] == "high"

        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault-exec-123"
        assert collector.record_fault_outcome.call_args[0][1] == "success"

    def test_collector_records_failure(self):
        """Collector records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-exec-456"
        proxy = ExecuteAllProxy(MagicMock(), collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-failure",
            strategy=StrategyType.WORKER_DEATH,
            trigger=trigger,
            parameters={"kill_worker_idx": 0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(WorkerDeathError):
            proxy("method")

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][1] == "raised"
