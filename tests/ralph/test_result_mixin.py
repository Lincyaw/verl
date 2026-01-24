"""
Unit tests for ResultModificationMixin.

Tests cover:
- _modify_result method applies arbitrary modifier function
- _negate_result method negates numbers, tensors, and nested structures
- _set_constant method replaces values with constants for tensors and scalars
- _strategy_modify_result reads modifier_fn from config parameters
- _strategy_reward_flip negates all numeric results
- _strategy_constant_reward sets all results to a constant value
"""
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch

from ralph.mixins.result import ResultModificationMixin


@dataclass
class MockConfig:
    """Mock config class for testing."""

    parameters: Dict[str, Any]


class MockProxy(ResultModificationMixin):
    """
    Mock proxy class that inherits ResultModificationMixin for testing.

    Simulates the expected interface that ResultModificationMixin relies on.
    """

    def __init__(self, original_fn, config: MockConfig):
        self._original = original_fn
        self._config = config


class TestModifyResult:
    """Tests for the _modify_result method."""

    def test_modify_result_applies_function(self):
        """Test that _modify_result applies the modifier function to the result."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._modify_result(5, lambda x: x * 2)

        assert result == 10

    def test_modify_result_with_string(self):
        """Test modifier function on string."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._modify_result("hello", str.upper)

        assert result == "HELLO"

    def test_modify_result_with_list(self):
        """Test modifier function on list."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._modify_result([1, 2, 3], lambda x: [i * 2 for i in x])

        assert result == [2, 4, 6]

    def test_modify_result_with_tensor(self):
        """Test modifier function on tensor."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = proxy._modify_result(tensor, lambda t: t + 1)

        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))

    def test_modify_result_with_identity(self):
        """Test that identity function returns original value."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._modify_result(42, lambda x: x)

        assert result == 42


class TestNegateResult:
    """Tests for the _negate_result method."""

    def test_negate_result_with_int(self):
        """Test negating an integer."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result(5)

        assert result == -5

    def test_negate_result_with_negative_int(self):
        """Test negating a negative integer."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result(-10)

        assert result == 10

    def test_negate_result_with_float(self):
        """Test negating a float."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result(3.14)

        assert result == pytest.approx(-3.14)

    def test_negate_result_with_zero(self):
        """Test negating zero."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result(0)

        assert result == 0

    def test_negate_result_with_tensor(self):
        """Test negating a tensor."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([1.0, -2.0, 3.0])

        result = proxy._negate_result(tensor)

        assert torch.allclose(result, torch.tensor([-1.0, 2.0, -3.0]))

    def test_negate_result_with_2d_tensor(self):
        """Test negating a 2D tensor."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = proxy._negate_result(tensor)

        expected = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        assert torch.allclose(result, expected)

    def test_negate_result_with_dict(self):
        """Test negating values in a dict."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result({"a": 5, "b": -3.5})

        assert result == {"a": -5, "b": pytest.approx(3.5)}

    def test_negate_result_with_nested_dict(self):
        """Test negating values in a nested dict."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result({"outer": {"inner": 10}, "value": 2})

        assert result == {"outer": {"inner": -10}, "value": -2}

    def test_negate_result_with_list(self):
        """Test negating values in a list."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result([1, 2, -3])

        assert result == [-1, -2, 3]

    def test_negate_result_with_tuple(self):
        """Test negating values in a tuple."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result((1.0, 2.0, 3.0))

        assert result == (-1.0, -2.0, -3.0)
        assert isinstance(result, tuple)

    def test_negate_result_with_mixed_nested(self):
        """Test negating values in mixed nested structures."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result({
            "list": [1, 2],
            "tuple": (3, 4),
            "nested": {"value": 5}
        })

        assert result == {
            "list": [-1, -2],
            "tuple": (-3, -4),
            "nested": {"value": -5}
        }

    def test_negate_result_with_string_unchanged(self):
        """Test that strings are returned unchanged."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result("hello")

        assert result == "hello"

    def test_negate_result_with_none_unchanged(self):
        """Test that None is returned unchanged."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._negate_result(None)

        assert result is None

    def test_negate_result_with_dict_containing_tensor(self):
        """Test negating dict containing tensors."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([1.0, 2.0])

        result = proxy._negate_result({"reward": tensor, "scalar": 5})

        assert torch.allclose(result["reward"], torch.tensor([-1.0, -2.0]))
        assert result["scalar"] == -5


class TestSetConstant:
    """Tests for the _set_constant method."""

    def test_set_constant_with_int(self):
        """Test setting an integer to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant(5, 0.0)

        assert result == 0
        assert isinstance(result, int)

    def test_set_constant_with_float(self):
        """Test setting a float to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant(3.14, 1.0)

        assert result == pytest.approx(1.0)
        assert isinstance(result, float)

    def test_set_constant_with_tensor(self):
        """Test setting a tensor to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = proxy._set_constant(tensor, 5.0)

        expected = torch.tensor([5.0, 5.0, 5.0])
        assert torch.allclose(result, expected)

    def test_set_constant_with_2d_tensor(self):
        """Test setting a 2D tensor to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = proxy._set_constant(tensor, -1.0)

        expected = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        assert torch.allclose(result, expected)

    def test_set_constant_preserves_tensor_shape(self):
        """Test that set_constant preserves tensor shape."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.randn(3, 4, 5)

        result = proxy._set_constant(tensor, 2.0)

        assert result.shape == tensor.shape

    def test_set_constant_with_dict(self):
        """Test setting values in a dict to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant({"a": 5, "b": 10}, 0.0)

        assert result == {"a": 0, "b": 0}

    def test_set_constant_with_nested_dict(self):
        """Test setting values in a nested dict to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant({"outer": {"inner": 10}, "value": 2}, 1.0)

        assert result == {"outer": {"inner": 1}, "value": 1}

    def test_set_constant_with_list(self):
        """Test setting values in a list to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant([1, 2, 3], 0.0)

        assert result == [0, 0, 0]

    def test_set_constant_with_tuple(self):
        """Test setting values in a tuple to a constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant((1.0, 2.0, 3.0), 5.0)

        assert result == (5.0, 5.0, 5.0)
        assert isinstance(result, tuple)

    def test_set_constant_with_string_unchanged(self):
        """Test that strings are returned unchanged."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant("hello", 0.0)

        assert result == "hello"

    def test_set_constant_with_none_unchanged(self):
        """Test that None is returned unchanged."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        result = proxy._set_constant(None, 0.0)

        assert result is None

    def test_set_constant_with_dict_containing_tensor(self):
        """Test setting dict containing tensors to constant."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))
        tensor = torch.tensor([1.0, 2.0])

        result = proxy._set_constant({"reward": tensor, "scalar": 5}, 0.5)

        expected_tensor = torch.tensor([0.5, 0.5])
        assert torch.allclose(result["reward"], expected_tensor)
        assert result["scalar"] == 0


class TestStrategyModifyResult:
    """Tests for the _strategy_modify_result method."""

    def test_strategy_modify_result_calls_original(self):
        """Test that _strategy_modify_result calls the original function."""
        mock_original = MagicMock(return_value=10)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_modify_result("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")
        assert result == 10

    def test_strategy_modify_result_applies_modifier(self):
        """Test that _strategy_modify_result applies modifier_fn from config."""
        mock_original = MagicMock(return_value=5)
        config = MockConfig(parameters={"modifier_fn": lambda x: x * 3})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_modify_result()

        assert result == 15

    def test_strategy_modify_result_no_modifier(self):
        """Test that _strategy_modify_result returns original result when no modifier."""
        mock_original = MagicMock(return_value="unchanged")
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_modify_result()

        assert result == "unchanged"

    def test_strategy_modify_result_with_tensor_modifier(self):
        """Test modifier function on tensor result."""
        mock_original = MagicMock(return_value=torch.tensor([1.0, 2.0]))
        config = MockConfig(parameters={"modifier_fn": lambda t: t * 10})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_modify_result()

        expected = torch.tensor([10.0, 20.0])
        assert torch.allclose(result, expected)


class TestStrategyRewardFlip:
    """Tests for the _strategy_reward_flip method."""

    def test_strategy_reward_flip_calls_original(self):
        """Test that _strategy_reward_flip calls the original function."""
        mock_original = MagicMock(return_value=5)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        proxy._strategy_reward_flip("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_reward_flip_negates_int(self):
        """Test that _strategy_reward_flip negates integer result."""
        mock_original = MagicMock(return_value=10)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_reward_flip()

        assert result == -10

    def test_strategy_reward_flip_negates_float(self):
        """Test that _strategy_reward_flip negates float result."""
        mock_original = MagicMock(return_value=3.14)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_reward_flip()

        assert result == pytest.approx(-3.14)

    def test_strategy_reward_flip_negates_tensor(self):
        """Test that _strategy_reward_flip negates tensor result."""
        mock_original = MagicMock(return_value=torch.tensor([1.0, -2.0, 3.0]))
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_reward_flip()

        expected = torch.tensor([-1.0, 2.0, -3.0])
        assert torch.allclose(result, expected)

    def test_strategy_reward_flip_negates_dict(self):
        """Test that _strategy_reward_flip negates dict values."""
        mock_original = MagicMock(return_value={"reward": 5.0, "bonus": -2.0})
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_reward_flip()

        assert result == {"reward": -5.0, "bonus": 2.0}

    def test_strategy_reward_flip_with_nested_structure(self):
        """Test reward flip with nested structure containing tensors."""
        tensor = torch.tensor([1.0, 2.0])
        mock_original = MagicMock(return_value={
            "rewards": tensor,
            "meta": {"score": 10}
        })
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_reward_flip()

        assert torch.allclose(result["rewards"], torch.tensor([-1.0, -2.0]))
        assert result["meta"]["score"] == -10


class TestStrategyConstantReward:
    """Tests for the _strategy_constant_reward method."""

    def test_strategy_constant_reward_calls_original(self):
        """Test that _strategy_constant_reward calls the original function."""
        mock_original = MagicMock(return_value=5)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        proxy._strategy_constant_reward("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_constant_reward_default_zero(self):
        """Test that _strategy_constant_reward defaults to 0.0."""
        mock_original = MagicMock(return_value=10)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        assert result == 0

    def test_strategy_constant_reward_uses_config_value(self):
        """Test that _strategy_constant_reward uses constant_value from config."""
        mock_original = MagicMock(return_value=10.0)  # Use float to preserve precision
        config = MockConfig(parameters={"constant_value": 5.5})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        assert result == pytest.approx(5.5)

    def test_strategy_constant_reward_with_tensor(self):
        """Test constant reward with tensor result."""
        mock_original = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))
        config = MockConfig(parameters={"constant_value": 1.0})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_strategy_constant_reward_with_dict(self):
        """Test constant reward with dict result."""
        mock_original = MagicMock(return_value={"reward": 5.0, "bonus": -2.0})
        config = MockConfig(parameters={"constant_value": 0.0})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        assert result == {"reward": 0.0, "bonus": 0.0}

    def test_strategy_constant_reward_with_negative_constant(self):
        """Test constant reward with negative constant value."""
        mock_original = MagicMock(return_value=10.0)
        config = MockConfig(parameters={"constant_value": -1.0})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        assert result == -1.0

    def test_strategy_constant_reward_with_nested_structure(self):
        """Test constant reward with nested structure containing tensors."""
        tensor = torch.tensor([1.0, 2.0])
        mock_original = MagicMock(return_value={
            "rewards": tensor,
            "meta": {"score": 10}
        })
        config = MockConfig(parameters={"constant_value": 0.5})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_constant_reward()

        expected_tensor = torch.tensor([0.5, 0.5])
        assert torch.allclose(result["rewards"], expected_tensor)
        assert result["meta"]["score"] == 0


class TestResultModificationMixinIntegration:
    """Integration tests for ResultModificationMixin."""

    def test_mixin_can_be_combined_with_other_classes(self):
        """Test that ResultModificationMixin works correctly in multiple inheritance."""

        class OtherMixin:
            def other_method(self):
                return "other"

        class CombinedProxy(ResultModificationMixin, OtherMixin):
            def __init__(self, original_fn, config):
                self._original = original_fn
                self._config = config

        mock_original = MagicMock(return_value=10)
        proxy = CombinedProxy(mock_original, MockConfig(parameters={"constant_value": 0.0}))

        flip_result = proxy._strategy_reward_flip()
        constant_result = proxy._strategy_constant_reward()

        assert flip_result == -10
        assert constant_result == 0
        assert proxy.other_method() == "other"

    def test_strategies_in_sequence(self):
        """Test using different strategies in sequence."""
        call_count = [0]

        def counting_fn():
            call_count[0] += 1
            return call_count[0] * 10.0

        proxy = MockProxy(counting_fn, MockConfig(parameters={"constant_value": 1.0}))

        # First call: reward flip
        flip_result = proxy._strategy_reward_flip()
        assert flip_result == -10.0
        assert call_count[0] == 1

        # Second call: constant reward
        constant_result = proxy._strategy_constant_reward()
        assert constant_result == 1.0
        assert call_count[0] == 2

    def test_tensor_dtype_preserved_in_negate(self):
        """Test that tensor dtype is preserved after negation."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        int_tensor = torch.tensor([1, 2, 3])
        float_tensor = torch.tensor([1.0, 2.0, 3.0])

        int_result = proxy._negate_result(int_tensor)
        float_result = proxy._negate_result(float_tensor)

        assert int_result.dtype == int_tensor.dtype
        assert float_result.dtype == float_tensor.dtype

    def test_complex_nested_structure_handling(self):
        """Test handling of complex nested structures with mixed types."""
        proxy = MockProxy(MagicMock(), MockConfig(parameters={}))

        complex_structure = {
            "scalars": [1, 2.5, -3],
            "tensors": {
                "small": torch.tensor([1.0, 2.0]),
                "large": torch.randn(10, 10)
            },
            "strings": ["a", "b"],  # Should be unchanged
            "nested": [
                {"value": 5},
                (1, 2, 3)
            ]
        }

        result = proxy._negate_result(complex_structure)

        assert result["scalars"] == [-1, -2.5, 3]
        assert torch.allclose(result["tensors"]["small"], torch.tensor([-1.0, -2.0]))
        assert result["tensors"]["large"].shape == torch.Size([10, 10])
        assert result["strings"] == ["a", "b"]  # Unchanged
        assert result["nested"][0]["value"] == -5
        assert result["nested"][1] == (-1, -2, -3)
