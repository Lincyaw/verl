"""
Unit tests for SkipMixin.

Tests cover:
- _skip_operation method returns default value without calling original
- _repeat_operation method calls original N times and returns last result
- _strategy_skip reads default_return from config parameters
- _strategy_repeat reads times from config parameters
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from ralph.mixins.skip import SkipMixin


@dataclass
class MockConfig:
    """Mock config class for testing."""

    parameters: dict[str, Any]


class MockProxy(SkipMixin):
    """
    Mock proxy class that inherits SkipMixin for testing.

    Simulates the expected interface that SkipMixin relies on.
    """

    def __init__(self, original_fn, config: MockConfig):
        self._original = original_fn
        self._config = config


class TestSkipOperation:
    """Tests for the _skip_operation method."""

    def test_skip_operation_returns_none_by_default(self):
        """Test that _skip_operation returns None when no default provided."""
        mock_original = MagicMock(return_value="should not be called")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._skip_operation()

        assert result is None
        mock_original.assert_not_called()

    def test_skip_operation_returns_provided_default(self):
        """Test that _skip_operation returns the provided default value."""
        mock_original = MagicMock(return_value="should not be called")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._skip_operation(default_return="custom_value")

        assert result == "custom_value"
        mock_original.assert_not_called()

    def test_skip_operation_with_dict_default(self):
        """Test that _skip_operation can return complex types like dict."""
        mock_original = MagicMock()
        proxy = MockProxy(mock_original, MockConfig(parameters={}))
        expected = {"key": "value", "nested": {"inner": 123}}

        result = proxy._skip_operation(default_return=expected)

        assert result == expected
        mock_original.assert_not_called()

    def test_skip_operation_with_list_default(self):
        """Test that _skip_operation can return a list."""
        mock_original = MagicMock()
        proxy = MockProxy(mock_original, MockConfig(parameters={}))
        expected = [1, 2, 3, "four"]

        result = proxy._skip_operation(default_return=expected)

        assert result == expected
        mock_original.assert_not_called()

    def test_skip_operation_with_zero_default(self):
        """Test that _skip_operation can return zero (falsy value)."""
        mock_original = MagicMock()
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._skip_operation(default_return=0)

        assert result == 0
        mock_original.assert_not_called()

    def test_skip_operation_with_empty_string_default(self):
        """Test that _skip_operation can return empty string (falsy value)."""
        mock_original = MagicMock()
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._skip_operation(default_return="")

        assert result == ""
        mock_original.assert_not_called()

    def test_skip_operation_with_false_default(self):
        """Test that _skip_operation can return False (falsy value)."""
        mock_original = MagicMock()
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._skip_operation(default_return=False)

        assert result is False
        mock_original.assert_not_called()


class TestRepeatOperation:
    """Tests for the _repeat_operation method."""

    def test_repeat_operation_calls_original_n_times(self):
        """Test that _repeat_operation calls original the specified number of times."""
        mock_original = MagicMock(return_value="result")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        proxy._repeat_operation(3, "arg1", key="value")

        assert mock_original.call_count == 3
        mock_original.assert_has_calls(
            [
                call("arg1", key="value"),
                call("arg1", key="value"),
                call("arg1", key="value"),
            ]
        )

    def test_repeat_operation_returns_last_result(self):
        """Test that _repeat_operation returns the result of the last call."""
        call_count = [0]

        def counting_fn():
            call_count[0] += 1
            return f"result_{call_count[0]}"

        proxy = MockProxy(counting_fn, MockConfig(parameters={}))

        result = proxy._repeat_operation(3)

        assert result == "result_3"
        assert call_count[0] == 3

    def test_repeat_operation_with_one_time(self):
        """Test that _repeat_operation with times=1 calls original once."""
        mock_original = MagicMock(return_value="single_result")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._repeat_operation(1, "arg")

        assert result == "single_result"
        mock_original.assert_called_once_with("arg")

    def test_repeat_operation_with_zero_times(self):
        """Test that _repeat_operation with times=0 defaults to 1 call."""
        mock_original = MagicMock(return_value="result")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._repeat_operation(0)

        assert result == "result"
        mock_original.assert_called_once()

    def test_repeat_operation_with_negative_times(self):
        """Test that _repeat_operation with negative times defaults to 1 call."""
        mock_original = MagicMock(return_value="result")
        proxy = MockProxy(mock_original, MockConfig(parameters={}))

        result = proxy._repeat_operation(-5)

        assert result == "result"
        mock_original.assert_called_once()

    def test_repeat_operation_passes_args_correctly(self):
        """Test that all arguments are passed correctly each time."""
        captured_args = []

        def capturing_fn(*args, **kwargs):
            captured_args.append((args, kwargs))
            return len(captured_args)

        proxy = MockProxy(capturing_fn, MockConfig(parameters={}))

        proxy._repeat_operation(2, 1, 2, 3, a="x", b="y")

        assert len(captured_args) == 2
        assert captured_args[0] == ((1, 2, 3), {"a": "x", "b": "y"})
        assert captured_args[1] == ((1, 2, 3), {"a": "x", "b": "y"})

    def test_repeat_operation_propagates_exception(self):
        """Test that exceptions from original are propagated."""
        call_count = [0]

        def failing_fn():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise ValueError("fail on second call")
            return "ok"

        proxy = MockProxy(failing_fn, MockConfig(parameters={}))

        with pytest.raises(ValueError, match="fail on second call"):
            proxy._repeat_operation(3)

        # Should have been called twice (second call raises)
        assert call_count[0] == 2

    def test_repeat_operation_with_large_times(self):
        """Test that _repeat_operation works with larger repeat counts."""
        call_count = [0]

        def counting_fn():
            call_count[0] += 1
            return call_count[0]

        proxy = MockProxy(counting_fn, MockConfig(parameters={}))

        result = proxy._repeat_operation(10)

        assert result == 10
        assert call_count[0] == 10


class TestStrategySkip:
    """Tests for the _strategy_skip method."""

    def test_strategy_skip_returns_none_by_default(self):
        """Test that _strategy_skip returns None when default_return not in config."""
        mock_original = MagicMock(return_value="should not be called")
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_skip("arg1", key="value")

        assert result is None
        mock_original.assert_not_called()

    def test_strategy_skip_uses_config_default_return(self):
        """Test that _strategy_skip reads default_return from config."""
        mock_original = MagicMock(return_value="should not be called")
        config = MockConfig(parameters={"default_return": "configured_default"})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_skip()

        assert result == "configured_default"
        mock_original.assert_not_called()

    def test_strategy_skip_with_complex_default(self):
        """Test that _strategy_skip can return complex config values."""
        mock_original = MagicMock()
        expected = {"status": "skipped", "code": 42, "items": [1, 2, 3]}
        config = MockConfig(parameters={"default_return": expected})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_skip()

        assert result == expected
        mock_original.assert_not_called()

    def test_strategy_skip_with_none_config_value(self):
        """Test that explicitly configured None is returned."""
        mock_original = MagicMock()
        config = MockConfig(parameters={"default_return": None})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_skip()

        assert result is None
        mock_original.assert_not_called()

    def test_strategy_skip_ignores_all_arguments(self):
        """Test that _strategy_skip ignores all passed arguments."""
        mock_original = MagicMock()
        config = MockConfig(parameters={"default_return": "skipped"})
        proxy = MockProxy(mock_original, config)

        # Pass many arguments - they should all be ignored
        result = proxy._strategy_skip(1, 2, 3, a="x", b="y", c="z")

        assert result == "skipped"
        mock_original.assert_not_called()


class TestStrategyRepeat:
    """Tests for the _strategy_repeat method."""

    def test_strategy_repeat_default_two_times(self):
        """Test that _strategy_repeat defaults to 2 repetitions."""
        mock_original = MagicMock(return_value="result")
        config = MockConfig(parameters={})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_repeat("arg")

        assert result == "result"
        assert mock_original.call_count == 2

    def test_strategy_repeat_uses_config_times(self):
        """Test that _strategy_repeat reads times from config."""
        mock_original = MagicMock(return_value="result")
        config = MockConfig(parameters={"times": 5})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_repeat()

        assert result == "result"
        assert mock_original.call_count == 5

    def test_strategy_repeat_passes_arguments(self):
        """Test that _strategy_repeat passes arguments to original."""
        mock_original = MagicMock(return_value="done")
        config = MockConfig(parameters={"times": 3})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_repeat(1, 2, key="value")

        assert result == "done"
        mock_original.assert_has_calls(
            [
                call(1, 2, key="value"),
                call(1, 2, key="value"),
                call(1, 2, key="value"),
            ]
        )

    def test_strategy_repeat_returns_last_result(self):
        """Test that _strategy_repeat returns the last call's result."""
        results = ["first", "second", "third"]
        call_idx = [0]

        def sequenced_fn():
            result = results[call_idx[0]]
            call_idx[0] += 1
            return result

        config = MockConfig(parameters={"times": 3})
        proxy = MockProxy(sequenced_fn, config)

        result = proxy._strategy_repeat()

        assert result == "third"

    def test_strategy_repeat_with_one_time(self):
        """Test that _strategy_repeat with times=1 calls once."""
        mock_original = MagicMock(return_value="once")
        config = MockConfig(parameters={"times": 1})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_repeat()

        assert result == "once"
        mock_original.assert_called_once()


class TestSkipMixinIntegration:
    """Integration tests for SkipMixin."""

    def test_mixin_can_be_combined_with_other_classes(self):
        """Test that SkipMixin works correctly in multiple inheritance."""

        class OtherMixin:
            def other_method(self):
                return "other"

        class CombinedProxy(SkipMixin, OtherMixin):
            def __init__(self, original_fn, config):
                self._original = original_fn
                self._config = config

        mock_original = MagicMock(return_value="original_result")
        proxy = CombinedProxy(mock_original, MockConfig(parameters={"times": 2}))

        skip_result = proxy._strategy_skip()
        repeat_result = proxy._strategy_repeat()

        assert skip_result is None
        assert repeat_result == "original_result"
        assert mock_original.call_count == 2
        assert proxy.other_method() == "other"

    def test_skip_and_repeat_in_sequence(self):
        """Test using skip and repeat methods in sequence."""
        call_count = [0]

        def counting_fn():
            call_count[0] += 1
            return call_count[0]

        proxy = MockProxy(counting_fn, MockConfig(parameters={"default_return": -1, "times": 3}))

        # Skip should not call original
        skip_result = proxy._strategy_skip()
        assert skip_result == -1
        assert call_count[0] == 0

        # Repeat should call original 3 times
        repeat_result = proxy._strategy_repeat()
        assert repeat_result == 3
        assert call_count[0] == 3

    def test_repeat_with_side_effects(self):
        """Test that repeat actually causes side effects multiple times."""
        side_effects = []

        def side_effect_fn(value):
            side_effects.append(value)
            return len(side_effects)

        config = MockConfig(parameters={"times": 4})
        proxy = MockProxy(side_effect_fn, config)

        result = proxy._strategy_repeat("item")

        assert result == 4
        assert side_effects == ["item", "item", "item", "item"]

    def test_skip_preserves_original_unchanged(self):
        """Test that skip doesn't modify the original function reference."""
        original_fn = lambda: "original"
        proxy = MockProxy(original_fn, MockConfig(parameters={}))

        proxy._strategy_skip()

        # Original should still be accessible and unchanged
        assert proxy._original() == "original"
