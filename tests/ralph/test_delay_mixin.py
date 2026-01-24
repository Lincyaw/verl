"""
Unit tests for DelayMixin.

Tests cover:
- _apply_delay method delays execution by expected amount
- _strategy_delay reads delay_seconds from config parameters
- _strategy_delay calls original function with correct arguments
"""
import time
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from ralph.mixins.delay import DelayMixin


@dataclass
class MockConfig:
    """Mock config class for testing."""

    parameters: Dict[str, Any]


class MockProxy(DelayMixin):
    """
    Mock proxy class that inherits DelayMixin for testing.

    Simulates the expected interface that DelayMixin relies on.
    """

    def __init__(self, original_fn, config: MockConfig):
        self._original = original_fn
        self._config = config


class TestApplyDelay:
    """Tests for the _apply_delay method."""

    def test_apply_delay_positive_seconds(self):
        """Test that _apply_delay sleeps for the specified duration."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        delay_seconds = 0.1

        start_time = time.perf_counter()
        proxy._apply_delay(delay_seconds)
        elapsed = time.perf_counter() - start_time

        # Allow some tolerance for timing
        assert elapsed >= delay_seconds * 0.9
        assert elapsed < delay_seconds * 1.5

    def test_apply_delay_zero_seconds(self):
        """Test that _apply_delay with 0 seconds returns immediately."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        start_time = time.perf_counter()
        proxy._apply_delay(0)
        elapsed = time.perf_counter() - start_time

        # Should return essentially immediately
        assert elapsed < 0.05

    def test_apply_delay_negative_seconds(self):
        """Test that _apply_delay with negative seconds returns immediately."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        start_time = time.perf_counter()
        proxy._apply_delay(-1.0)
        elapsed = time.perf_counter() - start_time

        # Should return immediately (guard against negative sleep)
        assert elapsed < 0.05

    def test_apply_delay_fractional_seconds(self):
        """Test that _apply_delay works with fractional seconds."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        delay_seconds = 0.05

        start_time = time.perf_counter()
        proxy._apply_delay(delay_seconds)
        elapsed = time.perf_counter() - start_time

        assert elapsed >= delay_seconds * 0.8
        assert elapsed < delay_seconds * 2.0


class TestStrategyDelay:
    """Tests for the _strategy_delay method."""

    def test_strategy_delay_uses_config_delay_seconds(self):
        """Test that _strategy_delay reads delay_seconds from config."""
        mock_original = MagicMock(return_value="result")
        config = MockConfig(parameters={"delay_seconds": 0.1})
        proxy = MockProxy(mock_original, config)

        start_time = time.perf_counter()
        result = proxy._strategy_delay("arg1", key="value")
        elapsed = time.perf_counter() - start_time

        assert elapsed >= 0.09  # Allow for some timing variance
        assert result == "result"
        mock_original.assert_called_once_with("arg1", key="value")

    def test_strategy_delay_default_delay_seconds(self):
        """Test that _strategy_delay uses default 10 seconds when not specified."""
        mock_original = MagicMock(return_value="result")
        config = MockConfig(parameters={})  # No delay_seconds specified
        proxy = MockProxy(mock_original, config)

        # We don't want to wait 10 seconds in a test, so we'll verify the
        # behavior by checking that the default is accessible
        # Instead, we test by patching time.sleep
        import unittest.mock as mock

        with mock.patch("time.sleep") as mock_sleep:
            proxy._strategy_delay()
            mock_sleep.assert_called_once_with(10.0)

    def test_strategy_delay_passes_args_to_original(self):
        """Test that arguments are correctly passed to the original function."""
        mock_original = MagicMock(return_value=42)
        config = MockConfig(parameters={"delay_seconds": 0.01})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_delay(1, 2, 3, a="x", b="y")

        mock_original.assert_called_once_with(1, 2, 3, a="x", b="y")
        assert result == 42

    def test_strategy_delay_returns_original_result(self):
        """Test that the return value from original is preserved."""
        expected_result = {"key": "value", "number": 123}
        mock_original = MagicMock(return_value=expected_result)
        config = MockConfig(parameters={"delay_seconds": 0.01})
        proxy = MockProxy(mock_original, config)

        result = proxy._strategy_delay()

        assert result == expected_result

    def test_strategy_delay_with_exception(self):
        """Test that exceptions from original are propagated."""
        mock_original = MagicMock(side_effect=ValueError("test error"))
        config = MockConfig(parameters={"delay_seconds": 0.01})
        proxy = MockProxy(mock_original, config)

        with pytest.raises(ValueError, match="test error"):
            proxy._strategy_delay()

    def test_strategy_delay_delays_before_calling_original(self):
        """Test that delay happens before the original function is called."""
        call_times = []

        def record_time():
            call_times.append(time.perf_counter())
            return "done"

        config = MockConfig(parameters={"delay_seconds": 0.1})
        proxy = MockProxy(record_time, config)

        start_time = time.perf_counter()
        proxy._strategy_delay()

        # The original should have been called after the delay
        assert len(call_times) == 1
        time_to_original = call_times[0] - start_time
        assert time_to_original >= 0.09  # Delay happened before calling original


class TestDelayMixinIntegration:
    """Integration tests for DelayMixin."""

    def test_mixin_can_be_combined_with_other_classes(self):
        """Test that DelayMixin works correctly in multiple inheritance."""

        class OtherMixin:
            def other_method(self):
                return "other"

        class CombinedProxy(DelayMixin, OtherMixin):
            def __init__(self, original_fn, config):
                self._original = original_fn
                self._config = config

        proxy = CombinedProxy(lambda: "result", MockConfig(parameters={"delay_seconds": 0.01}))

        assert proxy._strategy_delay() == "result"
        assert proxy.other_method() == "other"

    def test_delay_with_real_timing(self):
        """Test actual delay timing with a real function."""
        execution_order = []

        def original_fn():
            execution_order.append("original")
            return "completed"

        config = MockConfig(parameters={"delay_seconds": 0.05})
        proxy = MockProxy(original_fn, config)

        start = time.perf_counter()
        result = proxy._strategy_delay()
        elapsed = time.perf_counter() - start

        assert result == "completed"
        assert elapsed >= 0.04
        assert execution_order == ["original"]
