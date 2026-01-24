"""
Unit tests for ExceptionMixin.

Tests cover:
- EXCEPTION_MAP contains all required exception types
- _raise_exception raises correct exception type with message
- _strategy_raise_exception reads exc_type and message from config
- Error handling for unknown exception types
"""
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from ralph.mixins.exception import EXCEPTION_MAP, ExceptionMixin


@dataclass
class MockConfig:
    """Mock config class for testing."""

    parameters: Dict[str, Any]


class MockProxy(ExceptionMixin):
    """
    Mock proxy class that inherits ExceptionMixin for testing.

    Simulates the expected interface that ExceptionMixin relies on.
    """

    def __init__(self, original_fn, config: MockConfig):
        self._original = original_fn
        self._config = config


class TestExceptionMap:
    """Tests for the EXCEPTION_MAP constant."""

    def test_exception_map_contains_ioerror(self):
        """Test that IOError is in the exception map."""
        assert "IOError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["IOError"] is IOError

    def test_exception_map_contains_runtimeerror(self):
        """Test that RuntimeError is in the exception map."""
        assert "RuntimeError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["RuntimeError"] is RuntimeError

    def test_exception_map_contains_valueerror(self):
        """Test that ValueError is in the exception map."""
        assert "ValueError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["ValueError"] is ValueError

    def test_exception_map_contains_permissionerror(self):
        """Test that PermissionError is in the exception map."""
        assert "PermissionError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["PermissionError"] is PermissionError

    def test_exception_map_contains_timeouterror(self):
        """Test that TimeoutError is in the exception map."""
        assert "TimeoutError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["TimeoutError"] is TimeoutError

    def test_exception_map_contains_filenotfounderror(self):
        """Test that FileNotFoundError is in the exception map."""
        assert "FileNotFoundError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["FileNotFoundError"] is FileNotFoundError

    def test_exception_map_contains_oserror(self):
        """Test that OSError is in the exception map."""
        assert "OSError" in EXCEPTION_MAP
        assert EXCEPTION_MAP["OSError"] is OSError

    def test_exception_map_has_required_minimum_exceptions(self):
        """Test that EXCEPTION_MAP has at least the 7 required exception types."""
        required_exceptions = [
            "IOError",
            "RuntimeError",
            "ValueError",
            "PermissionError",
            "TimeoutError",
            "FileNotFoundError",
            "OSError",
        ]
        for exc_name in required_exceptions:
            assert exc_name in EXCEPTION_MAP, f"Missing required exception: {exc_name}"


class TestRaiseException:
    """Tests for the _raise_exception method."""

    def test_raise_exception_ioerror(self):
        """Test that _raise_exception raises IOError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(IOError, match="disk full"):
            proxy._raise_exception("IOError", "disk full")

    def test_raise_exception_runtimeerror(self):
        """Test that _raise_exception raises RuntimeError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(RuntimeError, match="unexpected failure"):
            proxy._raise_exception("RuntimeError", "unexpected failure")

    def test_raise_exception_valueerror(self):
        """Test that _raise_exception raises ValueError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(ValueError, match="invalid input"):
            proxy._raise_exception("ValueError", "invalid input")

    def test_raise_exception_permissionerror(self):
        """Test that _raise_exception raises PermissionError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(PermissionError, match="access denied"):
            proxy._raise_exception("PermissionError", "access denied")

    def test_raise_exception_timeouterror(self):
        """Test that _raise_exception raises TimeoutError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(TimeoutError, match="operation timed out"):
            proxy._raise_exception("TimeoutError", "operation timed out")

    def test_raise_exception_filenotfounderror(self):
        """Test that _raise_exception raises FileNotFoundError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(FileNotFoundError, match="file missing"):
            proxy._raise_exception("FileNotFoundError", "file missing")

    def test_raise_exception_oserror(self):
        """Test that _raise_exception raises OSError correctly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(OSError, match="system error"):
            proxy._raise_exception("OSError", "system error")

    def test_raise_exception_unknown_type(self):
        """Test that _raise_exception raises ValueError for unknown exception types."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(ValueError, match="Unknown exception type: FakeError"):
            proxy._raise_exception("FakeError", "this should not work")

    def test_raise_exception_unknown_type_shows_available(self):
        """Test that error message for unknown type lists available types."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(ValueError) as exc_info:
            proxy._raise_exception("FakeError", "message")

        error_message = str(exc_info.value)
        assert "Available types:" in error_message

    def test_raise_exception_preserves_message(self):
        """Test that the exception message is preserved exactly."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))
        exact_message = "This is a very specific error message with special chars: !@#$%"

        with pytest.raises(RuntimeError) as exc_info:
            proxy._raise_exception("RuntimeError", exact_message)

        assert str(exc_info.value) == exact_message

    def test_raise_exception_empty_message(self):
        """Test that _raise_exception works with empty message."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        with pytest.raises(IOError) as exc_info:
            proxy._raise_exception("IOError", "")

        assert str(exc_info.value) == ""

    def test_raise_exception_kwargs_are_accepted(self):
        """Test that _raise_exception accepts additional kwargs (for future use)."""
        proxy = MockProxy(lambda: None, MockConfig(parameters={}))

        # Should not raise an error about unexpected kwargs
        with pytest.raises(RuntimeError):
            proxy._raise_exception("RuntimeError", "test", extra_param="value", another=123)


class TestStrategyRaiseException:
    """Tests for the _strategy_raise_exception method."""

    def test_strategy_raises_configured_exception(self):
        """Test that _strategy_raise_exception raises the exception from config."""
        config = MockConfig(parameters={"exc_type": "IOError", "message": "test failure"})
        proxy = MockProxy(lambda: None, config)

        with pytest.raises(IOError, match="test failure"):
            proxy._strategy_raise_exception()

    def test_strategy_uses_default_message(self):
        """Test that _strategy_raise_exception uses default message when not specified."""
        config = MockConfig(parameters={"exc_type": "RuntimeError"})
        proxy = MockProxy(lambda: None, config)

        with pytest.raises(RuntimeError, match="Fault injection: RuntimeError"):
            proxy._strategy_raise_exception()

    def test_strategy_raises_valueerror_when_no_exc_type(self):
        """Test that _strategy_raise_exception raises ValueError when exc_type missing."""
        config = MockConfig(parameters={"message": "some message"})
        proxy = MockProxy(lambda: None, config)

        with pytest.raises(ValueError, match="exc_type must be specified"):
            proxy._strategy_raise_exception()

    def test_strategy_raises_valueerror_for_unknown_exc_type(self):
        """Test that _strategy_raise_exception raises ValueError for unknown type."""
        config = MockConfig(parameters={"exc_type": "UnknownError", "message": "test"})
        proxy = MockProxy(lambda: None, config)

        with pytest.raises(ValueError, match="Unknown exception type"):
            proxy._strategy_raise_exception()

    def test_strategy_ignores_args(self):
        """Test that _strategy_raise_exception ignores positional arguments."""
        config = MockConfig(parameters={"exc_type": "RuntimeError", "message": "test"})
        proxy = MockProxy(lambda: "should not be called", config)

        with pytest.raises(RuntimeError, match="test"):
            proxy._strategy_raise_exception("arg1", "arg2", "arg3")

    def test_strategy_ignores_kwargs(self):
        """Test that _strategy_raise_exception ignores keyword arguments."""
        config = MockConfig(parameters={"exc_type": "TimeoutError", "message": "timeout"})
        proxy = MockProxy(lambda: "should not be called", config)

        with pytest.raises(TimeoutError, match="timeout"):
            proxy._strategy_raise_exception(key1="value1", key2="value2")

    def test_strategy_does_not_call_original(self):
        """Test that _strategy_raise_exception never calls the original function."""
        call_count = [0]

        def original_fn():
            call_count[0] += 1
            return "result"

        config = MockConfig(parameters={"exc_type": "IOError", "message": "error"})
        proxy = MockProxy(original_fn, config)

        with pytest.raises(IOError):
            proxy._strategy_raise_exception()

        assert call_count[0] == 0, "Original function should not have been called"

    def test_strategy_with_all_exception_types(self):
        """Test that _strategy_raise_exception works with all mapped exception types."""
        for exc_name, exc_class in EXCEPTION_MAP.items():
            config = MockConfig(parameters={"exc_type": exc_name, "message": f"test {exc_name}"})
            proxy = MockProxy(lambda: None, config)

            with pytest.raises(exc_class, match=f"test {exc_name}"):
                proxy._strategy_raise_exception()


class TestExceptionMixinIntegration:
    """Integration tests for ExceptionMixin."""

    def test_mixin_can_be_combined_with_other_classes(self):
        """Test that ExceptionMixin works correctly in multiple inheritance."""

        class OtherMixin:
            def other_method(self):
                return "other"

        class CombinedProxy(ExceptionMixin, OtherMixin):
            def __init__(self, original_fn, config):
                self._original = original_fn
                self._config = config

        config = MockConfig(parameters={"exc_type": "RuntimeError", "message": "combined"})
        proxy = CombinedProxy(lambda: "result", config)

        assert proxy.other_method() == "other"

        with pytest.raises(RuntimeError, match="combined"):
            proxy._strategy_raise_exception()

    def test_exception_type_is_exact_class(self):
        """Test that raised exceptions are exactly the expected class."""
        config = MockConfig(parameters={"exc_type": "FileNotFoundError", "message": "test"})
        proxy = MockProxy(lambda: None, config)

        try:
            proxy._strategy_raise_exception()
        except FileNotFoundError as e:
            assert type(e) is FileNotFoundError
            assert str(e) == "test"
        except Exception:
            pytest.fail("Expected FileNotFoundError to be raised")

    def test_can_catch_by_parent_exception(self):
        """Test that raised exceptions can be caught by parent class."""
        config = MockConfig(parameters={"exc_type": "FileNotFoundError", "message": "test"})
        proxy = MockProxy(lambda: None, config)

        # FileNotFoundError is a subclass of OSError
        with pytest.raises(OSError):
            proxy._strategy_raise_exception()
