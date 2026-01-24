"""
Unit tests for TensorCorruptionMixin.

Tests cover:
- _corrupt_tensor method adds Gaussian noise correctly
- _inject_nan method sets elements to NaN
- _inject_inf method sets elements to Inf (positive and negative)
- _scale_tensor method scales tensor by constant
- _corrupt_result_tensors recursively corrupts tensors in nested structures
- Strategy methods (_strategy_corrupt_tensor, _strategy_inject_nan, _strategy_inject_inf)
"""
import math
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch

from ralph.mixins.tensor import TensorCorruptionMixin


@dataclass
class MockConfig:
    """Mock config class for testing."""

    parameters: Dict[str, Any]


class MockProxy(TensorCorruptionMixin):
    """
    Mock proxy class that inherits TensorCorruptionMixin for testing.

    Simulates the expected interface that TensorCorruptionMixin relies on.
    """

    def __init__(self, original_fn, config: MockConfig):
        self._original = original_fn
        self._config = config


class TestCorruptTensor:
    """Tests for the _corrupt_tensor method."""

    def test_corrupt_tensor_adds_noise(self):
        """Test that _corrupt_tensor adds Gaussian noise to the tensor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.zeros(100, 100)
        noise_scale = 1.0

        corrupted = proxy._corrupt_tensor(tensor, noise_scale)

        # Result should be different from original
        assert not torch.equal(corrupted, tensor)
        # Mean should be approximately 0 (for large tensor with noise centered at 0)
        assert abs(corrupted.mean().item()) < 0.5
        # Std should be approximately noise_scale
        assert abs(corrupted.std().item() - noise_scale) < 0.5

    def test_corrupt_tensor_preserves_shape(self):
        """Test that _corrupt_tensor preserves tensor shape."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        shapes = [(10,), (10, 20), (5, 10, 15), (2, 3, 4, 5)]

        for shape in shapes:
            tensor = torch.randn(*shape)
            corrupted = proxy._corrupt_tensor(tensor, 0.1)
            assert corrupted.shape == tensor.shape

    def test_corrupt_tensor_with_zero_noise_scale(self):
        """Test that _corrupt_tensor with noise_scale=0 returns original values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(10, 10) * 5.0

        corrupted = proxy._corrupt_tensor(tensor, 0.0)

        assert torch.allclose(corrupted, tensor)

    def test_corrupt_tensor_different_dtypes(self):
        """Test that _corrupt_tensor works with different tensor dtypes."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            tensor = torch.zeros(10, dtype=dtype)
            corrupted = proxy._corrupt_tensor(tensor, 1.0)
            assert corrupted.dtype == dtype

    def test_corrupt_tensor_large_noise_scale(self):
        """Test that _corrupt_tensor works with large noise scale."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.zeros(100, 100)
        noise_scale = 100.0

        corrupted = proxy._corrupt_tensor(tensor, noise_scale)

        # Std should be approximately noise_scale
        assert abs(corrupted.std().item() - noise_scale) < 20


class TestInjectNan:
    """Tests for the _inject_nan method."""

    def test_inject_nan_injects_nan_values(self):
        """Test that _inject_nan injects NaN values into tensor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(1000)
        ratio = 0.1

        corrupted = proxy._inject_nan(tensor, ratio)

        nan_count = torch.isnan(corrupted).sum().item()
        # Should be approximately ratio of elements
        expected = 1000 * ratio
        assert abs(nan_count - expected) < 50  # Allow some variance

    def test_inject_nan_with_zero_ratio(self):
        """Test that _inject_nan with ratio=0 injects no NaN values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(100)

        corrupted = proxy._inject_nan(tensor, 0.0)

        assert not torch.isnan(corrupted).any()

    def test_inject_nan_with_full_ratio(self):
        """Test that _inject_nan with ratio=1 injects all NaN values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(100)

        corrupted = proxy._inject_nan(tensor, 1.0)

        assert torch.isnan(corrupted).all()

    def test_inject_nan_preserves_shape(self):
        """Test that _inject_nan preserves tensor shape."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.randn(10, 20, 30)

        corrupted = proxy._inject_nan(tensor, 0.1)

        assert corrupted.shape == tensor.shape

    def test_inject_nan_does_not_modify_original(self):
        """Test that _inject_nan does not modify the original tensor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(100)
        original_values = tensor.clone()

        proxy._inject_nan(tensor, 0.5)

        assert torch.equal(tensor, original_values)


class TestInjectInf:
    """Tests for the _inject_inf method."""

    def test_inject_inf_positive(self):
        """Test that _inject_inf injects positive Inf values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(1000)
        ratio = 0.1

        corrupted = proxy._inject_inf(tensor, ratio, positive=True)

        inf_count = torch.isinf(corrupted).sum().item()
        expected = 1000 * ratio
        assert abs(inf_count - expected) < 50
        # All Inf values should be positive
        inf_mask = torch.isinf(corrupted)
        if inf_mask.any():
            assert (corrupted[inf_mask] > 0).all()

    def test_inject_inf_negative(self):
        """Test that _inject_inf injects negative Inf values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(1000)
        ratio = 0.1

        corrupted = proxy._inject_inf(tensor, ratio, positive=False)

        inf_count = torch.isinf(corrupted).sum().item()
        expected = 1000 * ratio
        assert abs(inf_count - expected) < 50
        # All Inf values should be negative
        inf_mask = torch.isinf(corrupted)
        if inf_mask.any():
            assert (corrupted[inf_mask] < 0).all()

    def test_inject_inf_with_zero_ratio(self):
        """Test that _inject_inf with ratio=0 injects no Inf values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(100)

        corrupted = proxy._inject_inf(tensor, 0.0)

        assert not torch.isinf(corrupted).any()

    def test_inject_inf_with_full_ratio(self):
        """Test that _inject_inf with ratio=1 injects all Inf values."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(100)

        corrupted = proxy._inject_inf(tensor, 1.0)

        assert torch.isinf(corrupted).all()

    def test_inject_inf_preserves_shape(self):
        """Test that _inject_inf preserves tensor shape."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.randn(5, 10, 15)

        corrupted = proxy._inject_inf(tensor, 0.1)

        assert corrupted.shape == tensor.shape


class TestScaleTensor:
    """Tests for the _scale_tensor method."""

    def test_scale_tensor_positive_scale(self):
        """Test that _scale_tensor scales by positive factor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(10) * 2.0
        scale = 3.0

        scaled = proxy._scale_tensor(tensor, scale)

        assert torch.allclose(scaled, torch.ones(10) * 6.0)

    def test_scale_tensor_negative_scale(self):
        """Test that _scale_tensor scales by negative factor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(10) * 2.0
        scale = -1.0

        scaled = proxy._scale_tensor(tensor, scale)

        assert torch.allclose(scaled, torch.ones(10) * -2.0)

    def test_scale_tensor_zero_scale(self):
        """Test that _scale_tensor with scale=0 returns zeros."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.randn(10, 10)

        scaled = proxy._scale_tensor(tensor, 0.0)

        assert torch.allclose(scaled, torch.zeros(10, 10))

    def test_scale_tensor_preserves_shape(self):
        """Test that _scale_tensor preserves tensor shape."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.randn(3, 4, 5)

        scaled = proxy._scale_tensor(tensor, 2.0)

        assert scaled.shape == tensor.shape


class TestCorruptResultTensors:
    """Tests for the _corrupt_result_tensors method."""

    def test_corrupt_single_tensor(self):
        """Test corrupting a single tensor."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.zeros(100)

        corrupted = proxy._corrupt_result_tensors(tensor, 1.0)

        assert isinstance(corrupted, torch.Tensor)
        assert not torch.equal(corrupted, tensor)

    def test_corrupt_dict_of_tensors(self):
        """Test corrupting a dict containing tensors."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        result = {
            "tensor_a": torch.zeros(50),
            "tensor_b": torch.ones(50),
            "scalar": 42,
        }

        corrupted = proxy._corrupt_result_tensors(result, 1.0)

        assert isinstance(corrupted, dict)
        assert not torch.equal(corrupted["tensor_a"], result["tensor_a"])
        assert not torch.equal(corrupted["tensor_b"], result["tensor_b"])
        assert corrupted["scalar"] == 42  # Non-tensor unchanged

    def test_corrupt_list_of_tensors(self):
        """Test corrupting a list containing tensors."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        result = [torch.zeros(50), torch.ones(50), "string_value"]

        corrupted = proxy._corrupt_result_tensors(result, 1.0)

        assert isinstance(corrupted, list)
        assert len(corrupted) == 3
        assert not torch.equal(corrupted[0], result[0])
        assert not torch.equal(corrupted[1], result[1])
        assert corrupted[2] == "string_value"

    def test_corrupt_tuple_of_tensors(self):
        """Test corrupting a tuple containing tensors."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        result = (torch.zeros(50), torch.ones(50))

        corrupted = proxy._corrupt_result_tensors(result, 1.0)

        assert isinstance(corrupted, tuple)
        assert len(corrupted) == 2
        assert not torch.equal(corrupted[0], result[0])
        assert not torch.equal(corrupted[1], result[1])

    def test_corrupt_nested_structure(self):
        """Test corrupting a deeply nested structure."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        result = {
            "level1": {
                "level2": {
                    "tensor": torch.zeros(50),
                },
                "list": [torch.zeros(30), {"nested_tensor": torch.ones(20)}],
            },
            "tuple": (torch.zeros(10),),
        }

        corrupted = proxy._corrupt_result_tensors(result, 1.0)

        # Check all tensors were corrupted
        assert not torch.equal(
            corrupted["level1"]["level2"]["tensor"],
            result["level1"]["level2"]["tensor"],
        )
        assert not torch.equal(
            corrupted["level1"]["list"][0], result["level1"]["list"][0]
        )
        assert not torch.equal(
            corrupted["level1"]["list"][1]["nested_tensor"],
            result["level1"]["list"][1]["nested_tensor"],
        )
        assert not torch.equal(corrupted["tuple"][0], result["tuple"][0])

    def test_corrupt_non_tensor_passthrough(self):
        """Test that non-tensor values pass through unchanged."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        values = [42, 3.14, "string", None, True]

        for value in values:
            corrupted = proxy._corrupt_result_tensors(value, 1.0)
            assert corrupted == value


class TestStrategyCurruptTensor:
    """Tests for the _strategy_corrupt_tensor method."""

    def test_strategy_corrupt_tensor_calls_original(self):
        """Test that _strategy_corrupt_tensor calls the original function."""
        mock_original = MagicMock(return_value=torch.zeros(50))
        config = MockConfig(parameters={"noise_scale": 0.1})
        proxy = MockProxy(mock_original, config)

        proxy._strategy_corrupt_tensor("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_corrupt_tensor_uses_config_noise_scale(self):
        """Test that noise_scale is read from config parameters."""
        tensor = torch.zeros(100, 100)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={"noise_scale": 5.0})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_corrupt_tensor()

        # Std should be approximately noise_scale
        assert abs(corrupted.std().item() - 5.0) < 2.0

    def test_strategy_corrupt_tensor_default_noise_scale(self):
        """Test that default noise_scale is 0.1."""
        tensor = torch.zeros(1000, 1000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_corrupt_tensor()

        # Std should be approximately 0.1 (default)
        assert abs(corrupted.std().item() - 0.1) < 0.05

    def test_strategy_corrupt_tensor_with_dict_result(self):
        """Test corrupting dict result from original function."""
        result = {"output": torch.zeros(50), "other": torch.ones(50)}
        mock_original = MagicMock(return_value=result)
        config = MockConfig(parameters={"noise_scale": 1.0})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_corrupt_tensor()

        assert not torch.equal(corrupted["output"], result["output"])
        assert not torch.equal(corrupted["other"], result["other"])


class TestStrategyInjectNan:
    """Tests for the _strategy_inject_nan method."""

    def test_strategy_inject_nan_calls_original(self):
        """Test that _strategy_inject_nan calls the original function."""
        mock_original = MagicMock(return_value=torch.ones(50))
        config = MockConfig(parameters={"nan_ratio": 0.1})
        proxy = MockProxy(mock_original, config)

        proxy._strategy_inject_nan("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_inject_nan_uses_config_ratio(self):
        """Test that nan_ratio is read from config parameters."""
        tensor = torch.ones(10000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={"nan_ratio": 0.5})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_nan()

        nan_ratio = torch.isnan(corrupted).float().mean().item()
        assert abs(nan_ratio - 0.5) < 0.1

    def test_strategy_inject_nan_default_ratio(self):
        """Test that default nan_ratio is 0.001."""
        tensor = torch.ones(100000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_nan()

        nan_ratio = torch.isnan(corrupted).float().mean().item()
        assert abs(nan_ratio - 0.001) < 0.002

    def test_strategy_inject_nan_with_nested_result(self):
        """Test NaN injection into nested structure."""
        result = {"data": [torch.ones(1000), torch.ones(1000)]}
        mock_original = MagicMock(return_value=result)
        config = MockConfig(parameters={"nan_ratio": 0.1})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_nan()

        # Check both tensors have NaN
        assert torch.isnan(corrupted["data"][0]).any()
        assert torch.isnan(corrupted["data"][1]).any()


class TestStrategyInjectInf:
    """Tests for the _strategy_inject_inf method."""

    def test_strategy_inject_inf_calls_original(self):
        """Test that _strategy_inject_inf calls the original function."""
        mock_original = MagicMock(return_value=torch.ones(50))
        config = MockConfig(parameters={"inf_ratio": 0.1})
        proxy = MockProxy(mock_original, config)

        proxy._strategy_inject_inf("arg1", key="value")

        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_inject_inf_uses_config_ratio(self):
        """Test that inf_ratio is read from config parameters."""
        tensor = torch.ones(10000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={"inf_ratio": 0.5})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_inf()

        inf_ratio = torch.isinf(corrupted).float().mean().item()
        assert abs(inf_ratio - 0.5) < 0.1

    def test_strategy_inject_inf_default_ratio(self):
        """Test that default inf_ratio is 0.001."""
        tensor = torch.ones(100000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_inf()

        inf_ratio = torch.isinf(corrupted).float().mean().item()
        assert abs(inf_ratio - 0.001) < 0.002

    def test_strategy_inject_inf_positive_default(self):
        """Test that default is positive Inf."""
        tensor = torch.ones(1000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={"inf_ratio": 0.1})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_inf()

        inf_mask = torch.isinf(corrupted)
        if inf_mask.any():
            assert (corrupted[inf_mask] > 0).all()

    def test_strategy_inject_inf_negative_from_config(self):
        """Test that positive=False from config injects negative Inf."""
        tensor = torch.ones(1000)
        mock_original = MagicMock(return_value=tensor)
        config = MockConfig(parameters={"inf_ratio": 0.1, "positive": False})
        proxy = MockProxy(mock_original, config)

        corrupted = proxy._strategy_inject_inf()

        inf_mask = torch.isinf(corrupted)
        if inf_mask.any():
            assert (corrupted[inf_mask] < 0).all()


class TestTensorCorruptionMixinIntegration:
    """Integration tests for TensorCorruptionMixin."""

    def test_mixin_can_be_combined_with_other_classes(self):
        """Test that TensorCorruptionMixin works in multiple inheritance."""

        class OtherMixin:
            def other_method(self):
                return "other"

        class CombinedProxy(TensorCorruptionMixin, OtherMixin):
            def __init__(self, original_fn, config):
                self._original = original_fn
                self._config = config

        proxy = CombinedProxy(
            lambda: torch.zeros(50), MockConfig(parameters={"noise_scale": 1.0})
        )

        corrupted = proxy._strategy_corrupt_tensor()
        assert isinstance(corrupted, torch.Tensor)
        assert proxy.other_method() == "other"

    def test_all_methods_preserve_tensor_device(self):
        """Test that methods preserve tensor device (CPU)."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        tensor = torch.ones(10, device="cpu")

        corrupted = proxy._corrupt_tensor(tensor, 1.0)
        assert corrupted.device == tensor.device

        nan_injected = proxy._inject_nan(tensor, 0.1)
        assert nan_injected.device == tensor.device

        inf_injected = proxy._inject_inf(tensor, 0.1)
        assert inf_injected.device == tensor.device

        scaled = proxy._scale_tensor(tensor, 2.0)
        assert scaled.device == tensor.device
