"""
Consolidated unit tests for all Ralph mixins.

Tests DelayMixin, TensorCorruptionMixin, ExceptionMixin, SkipMixin,
and ResultModificationMixin from ralph.mixins.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import EXCEPTION_MAP, ExceptionMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.mixins.skip import SkipMixin
from ralph.mixins.tensor import TensorCorruptionMixin


# =============================================================================
# DelayMixin Tests
# =============================================================================


class TestDelayMixinApplyDelay:
    """Tests for DelayMixin._apply_delay method."""

    def test_apply_positive_delay(self):
        """Test applying a positive delay."""

        class TestClass(DelayMixin):
            pass

        obj = TestClass()

        start = time.time()
        obj._apply_delay(0.05)
        elapsed = time.time() - start

        assert elapsed >= 0.04  # Allow some tolerance

    def test_apply_zero_delay(self):
        """Test zero delay returns immediately."""

        class TestClass(DelayMixin):
            pass

        obj = TestClass()

        start = time.time()
        obj._apply_delay(0.0)
        elapsed = time.time() - start

        assert elapsed < 0.01

    def test_apply_negative_delay(self):
        """Test negative delay is treated as zero."""

        class TestClass(DelayMixin):
            pass

        obj = TestClass()

        start = time.time()
        obj._apply_delay(-1.0)
        elapsed = time.time() - start

        assert elapsed < 0.01


class TestDelayMixinStrategyDelay:
    """Tests for DelayMixin._strategy_delay method."""

    def test_strategy_delay_uses_config(self):
        """Test _strategy_delay reads delay_seconds from config."""

        class TestClass(DelayMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"delay_seconds": 0.05},
                )
                self._original = MagicMock(return_value="result")

        obj = TestClass()
        result = obj._strategy_delay("arg1", kwarg1="value1")

        assert result == "result"
        obj._original.assert_called_once_with("arg1", kwarg1="value1")

    def test_strategy_delay_default_value(self):
        """Test _strategy_delay uses default 10.0 seconds when not specified."""

        class TestClass(DelayMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={},  # No delay_seconds
                )
                self._original = MagicMock(return_value="result")

        obj = TestClass()

        with patch.object(obj, "_apply_delay") as mock_delay:
            obj._strategy_delay()
            mock_delay.assert_called_once_with(10.0)


# =============================================================================
# TensorCorruptionMixin Tests
# =============================================================================


class TestTensorCorruptionMixinCorruptTensor:
    """Tests for TensorCorruptionMixin._corrupt_tensor method."""

    def test_corrupt_tensor_adds_noise(self):
        """Test corrupt tensor adds Gaussian noise."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.zeros(10, 10)
        corrupted = obj._corrupt_tensor(tensor, noise_scale=0.1)

        assert corrupted.shape == tensor.shape
        assert not torch.allclose(corrupted, tensor)

    def test_corrupt_tensor_zero_noise_scale(self):
        """Test zero noise scale leaves tensor unchanged."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(5, 5)
        corrupted = obj._corrupt_tensor(tensor, noise_scale=0.0)

        assert torch.allclose(corrupted, tensor)


class TestTensorCorruptionMixinInjectNan:
    """Tests for TensorCorruptionMixin._inject_nan method."""

    def test_inject_nan_creates_nans(self):
        """Test inject_nan creates NaN values."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(100)
        result = obj._inject_nan(tensor, ratio=0.5)

        nan_count = torch.isnan(result).sum().item()
        assert nan_count > 0
        assert nan_count < 100

    def test_inject_nan_zero_ratio(self):
        """Test zero ratio creates no NaNs."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(100)
        result = obj._inject_nan(tensor, ratio=0.0)

        assert not torch.isnan(result).any()

    def test_inject_nan_preserves_original(self):
        """Test original tensor is not modified."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(100)
        original_clone = tensor.clone()
        obj._inject_nan(tensor, ratio=0.5)

        assert torch.allclose(tensor, original_clone)


class TestTensorCorruptionMixinInjectInf:
    """Tests for TensorCorruptionMixin._inject_inf method."""

    def test_inject_positive_inf(self):
        """Test inject positive infinity."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(100)
        result = obj._inject_inf(tensor, ratio=0.5, positive=True)

        pos_inf_count = torch.isinf(result).sum().item()
        assert pos_inf_count > 0
        # All inf values should be positive
        assert (result[torch.isinf(result)] > 0).all()

    def test_inject_negative_inf(self):
        """Test inject negative infinity."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(100)
        result = obj._inject_inf(tensor, ratio=0.5, positive=False)

        neg_inf_count = torch.isinf(result).sum().item()
        assert neg_inf_count > 0
        # All inf values should be negative
        assert (result[torch.isinf(result)] < 0).all()


class TestTensorCorruptionMixinScaleTensor:
    """Tests for TensorCorruptionMixin._scale_tensor method."""

    def test_scale_tensor_positive(self):
        """Test scaling with positive scale."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = obj._scale_tensor(tensor, scale=2.0)

        assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))

    def test_scale_tensor_zero(self):
        """Test scaling with zero scale."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = obj._scale_tensor(tensor, scale=0.0)

        assert torch.allclose(result, torch.zeros(3))


class TestTensorCorruptionMixinCorruptResultTensors:
    """Tests for TensorCorruptionMixin._corrupt_result_tensors method."""

    def test_corrupt_single_tensor(self):
        """Test corrupting a single tensor result."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        tensor = torch.zeros(10)
        result = obj._corrupt_result_tensors(tensor, noise_scale=0.1)

        assert not torch.allclose(result, tensor)

    def test_corrupt_dict_result(self):
        """Test corrupting tensors in dict."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        data = {
            "tensor": torch.zeros(10),
            "other": "string",
            "number": 42,
        }
        result = obj._corrupt_result_tensors(data, noise_scale=0.1)

        assert not torch.allclose(result["tensor"], data["tensor"])
        assert result["other"] == "string"
        assert result["number"] == 42

    def test_corrupt_nested_structures(self):
        """Test corrupting nested structures."""

        class TestClass(TensorCorruptionMixin):
            pass

        obj = TestClass()
        data = {
            "nested": {
                "tensor": torch.zeros(5),
            },
            "list": [torch.ones(3)],
        }
        result = obj._corrupt_result_tensors(data, noise_scale=0.1)

        assert not torch.allclose(
            result["nested"]["tensor"], data["nested"]["tensor"]
        )
        assert not torch.allclose(result["list"][0], data["list"][0])


# =============================================================================
# ExceptionMixin Tests
# =============================================================================


class TestExceptionMap:
    """Tests for EXCEPTION_MAP constant."""

    def test_required_exceptions_present(self):
        """Test all required exception types are in EXCEPTION_MAP."""
        required = [
            "IOError",
            "RuntimeError",
            "ValueError",
            "PermissionError",
            "TimeoutError",
            "FileNotFoundError",
            "OSError",
        ]
        for exc_name in required:
            assert exc_name in EXCEPTION_MAP


class TestExceptionMixinRaiseException:
    """Tests for ExceptionMixin._raise_exception method."""

    def test_raise_runtime_error(self):
        """Test raising RuntimeError."""

        class TestClass(ExceptionMixin):
            pass

        obj = TestClass()

        with pytest.raises(RuntimeError, match="Test error"):
            obj._raise_exception("RuntimeError", "Test error")

    def test_raise_io_error(self):
        """Test raising IOError."""

        class TestClass(ExceptionMixin):
            pass

        obj = TestClass()

        with pytest.raises(IOError, match="IO failed"):
            obj._raise_exception("IOError", "IO failed")

    def test_raise_unknown_type_raises_value_error(self):
        """Test unknown exception type raises ValueError."""

        class TestClass(ExceptionMixin):
            pass

        obj = TestClass()

        with pytest.raises(ValueError, match="Unknown exception type"):
            obj._raise_exception("UnknownError", "message")


class TestExceptionMixinStrategyRaiseException:
    """Tests for ExceptionMixin._strategy_raise_exception method."""

    def test_strategy_reads_config(self):
        """Test strategy reads exc_type and message from config."""

        class TestClass(ExceptionMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.RAISE_EXCEPTION,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={
                        "exc_type": "RuntimeError",
                        "message": "Custom error message",
                    },
                )

        obj = TestClass()

        with pytest.raises(RuntimeError, match="Custom error message"):
            obj._strategy_raise_exception()

    def test_strategy_default_message(self):
        """Test strategy uses default message if not specified."""

        class TestClass(ExceptionMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.RAISE_EXCEPTION,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"exc_type": "RuntimeError"},
                )

        obj = TestClass()

        with pytest.raises(RuntimeError, match="Fault injection: RuntimeError"):
            obj._strategy_raise_exception()

    def test_strategy_missing_exc_type_raises(self):
        """Test missing exc_type raises ValueError."""

        class TestClass(ExceptionMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.RAISE_EXCEPTION,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={},
                )

        obj = TestClass()

        with pytest.raises(ValueError, match="exc_type"):
            obj._strategy_raise_exception()


# =============================================================================
# SkipMixin Tests
# =============================================================================


class TestSkipMixinSkipOperation:
    """Tests for SkipMixin._skip_operation method."""

    def test_skip_returns_default_none(self):
        """Test skip returns None by default."""

        class TestClass(SkipMixin):
            pass

        obj = TestClass()
        result = obj._skip_operation()

        assert result is None

    def test_skip_returns_custom_default(self):
        """Test skip returns custom default value."""

        class TestClass(SkipMixin):
            pass

        obj = TestClass()

        assert obj._skip_operation(default_return=42) == 42
        assert obj._skip_operation(default_return="string") == "string"
        assert obj._skip_operation(default_return={"key": "value"}) == {"key": "value"}

    def test_skip_returns_falsy_values(self):
        """Test skip handles falsy default values."""

        class TestClass(SkipMixin):
            pass

        obj = TestClass()

        assert obj._skip_operation(default_return=0) == 0
        assert obj._skip_operation(default_return="") == ""
        assert obj._skip_operation(default_return=False) is False


class TestSkipMixinRepeatOperation:
    """Tests for SkipMixin._repeat_operation method."""

    def test_repeat_calls_original_n_times(self):
        """Test repeat calls original N times."""

        class TestClass(SkipMixin):
            def __init__(self):
                self._original = MagicMock(return_value="result")

        obj = TestClass()
        result = obj._repeat_operation(3, "arg1")

        assert result == "result"
        assert obj._original.call_count == 3

    def test_repeat_returns_last_result(self):
        """Test repeat returns last call's result."""

        class TestClass(SkipMixin):
            def __init__(self):
                self._original = MagicMock(side_effect=["a", "b", "c"])

        obj = TestClass()
        result = obj._repeat_operation(3)

        assert result == "c"

    def test_repeat_zero_times_calls_once(self):
        """Test repeat with 0 times calls once (minimum)."""

        class TestClass(SkipMixin):
            def __init__(self):
                self._original = MagicMock(return_value="result")

        obj = TestClass()
        obj._repeat_operation(0)

        assert obj._original.call_count == 1


class TestSkipMixinStrategies:
    """Tests for SkipMixin strategy methods."""

    def test_strategy_skip_uses_config(self):
        """Test _strategy_skip reads default_return from config."""

        class TestClass(SkipMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.SKIP,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"default_return": "skipped"},
                )

        obj = TestClass()
        result = obj._strategy_skip()

        assert result == "skipped"

    def test_strategy_repeat_uses_config(self):
        """Test _strategy_repeat reads times from config."""

        class TestClass(SkipMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.REPEAT,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"times": 5},
                )
                self._original = MagicMock(return_value="result")

        obj = TestClass()
        obj._strategy_repeat("arg1")

        assert obj._original.call_count == 5


# =============================================================================
# ResultModificationMixin Tests
# =============================================================================


class TestResultModificationMixinModifyResult:
    """Tests for ResultModificationMixin._modify_result method."""

    def test_modify_result_applies_function(self):
        """Test _modify_result applies modifier function."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        result = obj._modify_result(5, lambda x: x * 2)

        assert result == 10


class TestResultModificationMixinNegateResult:
    """Tests for ResultModificationMixin._negate_result method."""

    def test_negate_int(self):
        """Test negating integer."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        assert obj._negate_result(5) == -5
        assert obj._negate_result(-3) == 3

    def test_negate_float(self):
        """Test negating float."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        assert obj._negate_result(3.14) == -3.14

    def test_negate_tensor(self):
        """Test negating tensor."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = obj._negate_result(tensor)

        assert torch.allclose(result, torch.tensor([-1.0, -2.0, -3.0]))

    def test_negate_dict(self):
        """Test negating dict values."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        data = {"a": 5, "b": -3, "c": "string"}
        result = obj._negate_result(data)

        assert result["a"] == -5
        assert result["b"] == 3
        assert result["c"] == "string"  # Non-numeric unchanged

    def test_negate_nested_structures(self):
        """Test negating nested structures."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        data = {"nested": {"value": 10}, "list": [1, 2, 3]}
        result = obj._negate_result(data)

        assert result["nested"]["value"] == -10
        assert result["list"] == [-1, -2, -3]


class TestResultModificationMixinSetConstant:
    """Tests for ResultModificationMixin._set_constant method."""

    def test_set_constant_int(self):
        """Test setting constant for int."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        assert obj._set_constant(5, 0) == 0
        assert isinstance(obj._set_constant(5, 0), int)

    def test_set_constant_float(self):
        """Test setting constant for float."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        assert obj._set_constant(3.14, 0.0) == 0.0
        assert isinstance(obj._set_constant(3.14, 0.0), float)

    def test_set_constant_tensor(self):
        """Test setting constant for tensor."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = obj._set_constant(tensor, 5.0)

        assert result.shape == tensor.shape
        assert torch.allclose(result, torch.tensor([5.0, 5.0, 5.0]))

    def test_set_constant_preserves_shape(self):
        """Test set_constant preserves tensor shape."""

        class TestClass(ResultModificationMixin):
            pass

        obj = TestClass()
        tensor = torch.ones(3, 4, 5)
        result = obj._set_constant(tensor, 2.0)

        assert result.shape == (3, 4, 5)


class TestResultModificationMixinStrategies:
    """Tests for ResultModificationMixin strategy methods."""

    def test_strategy_reward_flip(self):
        """Test _strategy_reward_flip negates result."""

        class TestClass(ResultModificationMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.REWARD_FLIP,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                )
                self._original = MagicMock(return_value={"reward": 1.0})

        obj = TestClass()
        result = obj._strategy_reward_flip()

        assert result["reward"] == -1.0

    def test_strategy_constant_reward(self):
        """Test _strategy_constant_reward returns constant."""

        class TestClass(ResultModificationMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.CONSTANT_REWARD,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"constant_value": 0.5},
                )
                self._original = MagicMock(return_value={"reward": 1.0})

        obj = TestClass()
        result = obj._strategy_constant_reward()

        assert result["reward"] == 0.5

    def test_strategy_constant_reward_default_zero(self):
        """Test _strategy_constant_reward defaults to 0.0."""

        class TestClass(ResultModificationMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.CONSTANT_REWARD,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={},
                )
                self._original = MagicMock(return_value=5.0)

        obj = TestClass()
        result = obj._strategy_constant_reward()

        assert result == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMixinMultipleInheritance:
    """Test multiple inheritance with mixins."""

    def test_combine_delay_and_tensor_corruption(self):
        """Test combining DelayMixin and TensorCorruptionMixin."""

        class TestClass(DelayMixin, TensorCorruptionMixin):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                    parameters={"delay_seconds": 0.01},
                )
                self._original = MagicMock(return_value=torch.zeros(10))

        obj = TestClass()

        # Can use delay methods
        obj._apply_delay(0.01)

        # Can use tensor corruption methods
        tensor = torch.zeros(10)
        corrupted = obj._corrupt_tensor(tensor, 0.1)
        assert corrupted.shape == tensor.shape

    def test_combine_exception_and_skip(self):
        """Test combining ExceptionMixin and SkipMixin."""

        class TestClass(ExceptionMixin, SkipMixin):
            def __init__(self):
                self._original = MagicMock()

        obj = TestClass()

        # Can use exception methods
        with pytest.raises(RuntimeError):
            obj._raise_exception("RuntimeError", "error")

        # Can use skip methods
        result = obj._skip_operation(default_return="skipped")
        assert result == "skipped"

    def test_all_mixins_combined(self):
        """Test all mixins can be combined."""

        class TestClass(
            DelayMixin,
            TensorCorruptionMixin,
            ExceptionMixin,
            SkipMixin,
            ResultModificationMixin,
        ):
            def __init__(self):
                self._config = FaultConfig(
                    id="test",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                )
                self._original = MagicMock(return_value=5)

        obj = TestClass()

        # All methods should be accessible
        assert hasattr(obj, "_apply_delay")
        assert hasattr(obj, "_corrupt_tensor")
        assert hasattr(obj, "_raise_exception")
        assert hasattr(obj, "_skip_operation")
        assert hasattr(obj, "_negate_result")
