"""
Unit tests for Agent proxies.

Tests cover CallToolProxy with all 6 supported strategies:
- DELAY: Standard delay before tool call
- TOOL_TIMEOUT: Very long delay simulating hung tool call
- TOOL_EXCEPTION: Raises an exception during tool call
- WRONG_RESULT: Returns incorrect/unexpected result from tool
- EMPTY_RESULT: Returns empty/None result from tool
- MALFORMED_RESULT: Returns structurally invalid result
"""

from unittest.mock import MagicMock, patch

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.agent import CallToolProxy


class TestCallToolProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_tool_call_target(self):
        """CallToolProxy is registered for 'ToolAgentLoop._call_tool' target."""
        assert ProxyRegistry.is_registered("ToolAgentLoop._call_tool")
        assert ProxyRegistry.get_proxy("ToolAgentLoop._call_tool") is CallToolProxy

    def test_supported_strategies(self):
        """CallToolProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("ToolAgentLoop._call_tool")
        expected = {
            StrategyType.DELAY,
            StrategyType.TOOL_TIMEOUT,
            StrategyType.TOOL_EXCEPTION,
            StrategyType.WRONG_RESULT,
            StrategyType.EMPTY_RESULT,
            StrategyType.MALFORMED_RESULT,
        }
        assert strategies == expected


class TestCallToolProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_agent(self):
        """_get_layer returns 'Agent'."""
        proxy = CallToolProxy(lambda: None)
        assert proxy._get_layer() == "Agent"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"result": "success"})
        proxy = CallToolProxy(original)
        result = proxy("test_tool", {"arg": "value"})
        original.assert_called_once_with("test_tool", {"arg": "value"})
        assert result == {"result": "success"}

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original_result = {"result": "success", "data": [1, 2, 3]}
        original = MagicMock(return_value=original_result)
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("my_tool")
        original.assert_called_once()
        assert result == original_result

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = CallToolProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by CallToolProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy uses delay_seconds from config."""
        original = MagicMock(return_value={"status": "ok"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy("tool_name", {"arg": 1})
            mock_sleep.assert_called_once_with(0.01)
            original.assert_called_once()
            assert result == {"status": "ok"}

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay_seconds of 10.0."""
        original = MagicMock(return_value="result")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(10.0)


class TestToolTimeoutStrategy:
    """Tests for TOOL_TIMEOUT strategy."""

    def test_tool_timeout_with_config(self):
        """TOOL_TIMEOUT uses timeout_seconds from config."""
        original = MagicMock(return_value={"result": "done"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TOOL_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_seconds": 60.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy("my_tool")
            mock_sleep.assert_called_once_with(60.0)
            original.assert_called_once()
            assert result == {"result": "done"}

    def test_tool_timeout_default_timeout(self):
        """TOOL_TIMEOUT uses default timeout_seconds of 300.0."""
        original = MagicMock(return_value="ok")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TOOL_TIMEOUT,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(300.0)


class TestToolExceptionStrategy:
    """Tests for TOOL_EXCEPTION strategy."""

    def test_tool_exception_raises_error(self):
        """TOOL_EXCEPTION raises the specified exception."""
        original = MagicMock(return_value="should not be called")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TOOL_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError", "message": "Tool failed"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy("failing_tool")
        assert "Tool failed" in str(exc_info.value)
        original.assert_not_called()

    def test_tool_exception_various_types(self):
        """TOOL_EXCEPTION supports various exception types."""
        proxy = CallToolProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)

        for exc_type, exc_class in [
            ("IOError", IOError),
            ("ValueError", ValueError),
            ("TimeoutError", TimeoutError),
            ("FileNotFoundError", FileNotFoundError),
        ]:
            config = FaultConfig(
                id=f"test_{exc_type}",
                strategy=StrategyType.TOOL_EXCEPTION,
                trigger=trigger,
                parameters={"exc_type": exc_type, "message": f"Test {exc_type}"},
            )
            proxy.set_config(config)
            proxy.set_step(0)

            with pytest.raises(exc_class):
                proxy()

    def test_tool_exception_default_message(self):
        """TOOL_EXCEPTION uses default message with tool name."""
        original = MagicMock()
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TOOL_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError"},  # No message
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy("my_special_tool")
        assert "my_special_tool" in str(exc_info.value)
        assert "RuntimeError" in str(exc_info.value)

    def test_tool_exception_requires_exc_type(self):
        """TOOL_EXCEPTION raises ValueError if exc_type not specified."""
        proxy = CallToolProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TOOL_EXCEPTION,
            trigger=trigger,
            parameters={},  # Missing exc_type
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy()
        assert "exc_type is required" in str(exc_info.value)

    def test_extract_tool_name_from_args(self):
        """Tool name is extracted from first positional arg."""
        proxy = CallToolProxy(lambda: None)
        tool_name = proxy._extract_tool_name(("test_tool", {"arg": 1}), {})
        assert tool_name == "test_tool"

    def test_extract_tool_name_from_kwargs(self):
        """Tool name is extracted from kwargs."""
        proxy = CallToolProxy(lambda: None)
        tool_name = proxy._extract_tool_name((), {"tool_name": "kwarg_tool"})
        assert tool_name == "kwarg_tool"

    def test_extract_tool_name_returns_unknown(self):
        """Returns 'unknown' when tool name cannot be extracted."""
        proxy = CallToolProxy(lambda: None)
        tool_name = proxy._extract_tool_name((), {})
        assert tool_name == "unknown"


class TestWrongResultStrategy:
    """Tests for WRONG_RESULT strategy."""

    def test_wrong_result_with_specific_value(self):
        """WRONG_RESULT returns specific wrong_value if provided."""
        original = MagicMock(return_value={"correct": True})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"wrong_value": {"wrong": True}},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("tool")
        assert result == {"wrong": True}
        original.assert_called_once()

    def test_wrong_result_negate_bool(self):
        """WRONG_RESULT with negate inverts boolean."""
        original = MagicMock(return_value=True)
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"corruption_type": "negate"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result is False

    def test_wrong_result_negate_number(self):
        """WRONG_RESULT with negate negates number."""
        original = MagicMock(return_value=42)
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"corruption_type": "negate"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == -42

    def test_wrong_result_invert_list(self):
        """WRONG_RESULT with invert reverses list."""
        original = MagicMock(return_value=[1, 2, 3])
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"corruption_type": "invert"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == [3, 2, 1]

    def test_wrong_result_swap_type(self):
        """WRONG_RESULT with swap_type changes type."""
        original = MagicMock(return_value=True)
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"corruption_type": "swap_type"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == 1  # bool True -> int 1

    def test_wrong_result_randomize(self):
        """WRONG_RESULT with randomize returns random value."""
        original = MagicMock(return_value="hello")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"corruption_type": "randomize"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        # Result should be different from original
        assert isinstance(result, str)
        assert result != "hello"  # Very unlikely to be same

    def test_wrong_result_default_swap_type(self):
        """WRONG_RESULT defaults to swap_type corruption."""
        original = MagicMock(return_value={"key": "value"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={},  # No corruption_type
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        # dict should be converted to list of tuples
        assert isinstance(result, list)


class TestEmptyResultStrategy:
    """Tests for EMPTY_RESULT strategy."""

    def test_empty_result_returns_none(self):
        """EMPTY_RESULT returns None by default."""
        original = MagicMock(return_value={"data": "important"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("tool")
        assert result is None

    def test_empty_result_empty_string(self):
        """EMPTY_RESULT returns empty string when configured."""
        original = MagicMock(return_value="important data")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "empty_string"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == ""

    def test_empty_result_empty_dict(self):
        """EMPTY_RESULT returns empty dict when configured."""
        original = MagicMock(return_value={"key": "value"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "empty_dict"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == {}

    def test_empty_result_empty_list(self):
        """EMPTY_RESULT returns empty list when configured."""
        original = MagicMock(return_value=[1, 2, 3])
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "empty_list"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == []

    def test_empty_result_zero(self):
        """EMPTY_RESULT returns 0 when configured."""
        original = MagicMock(return_value=42)
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "zero"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert result == 0

    def test_empty_result_call_original(self):
        """EMPTY_RESULT calls original when call_original is True."""
        original = MagicMock(return_value={"key": "value"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"call_original": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("tool")
        original.assert_called_once()
        # Returns empty value matching original type (dict -> empty dict)
        assert result == {}

    def test_empty_result_type_matching(self):
        """EMPTY_RESULT matches type of original result when call_original=True."""
        test_cases = [
            (True, False),
            (42, 0),
            (3.14, 0.0),
            ("hello", ""),
            ([1, 2], []),
            ((1, 2), ()),
            ({"a": 1}, {}),
        ]

        for original_value, expected in test_cases:
            original = MagicMock(return_value=original_value)
            proxy = CallToolProxy(original)
            trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
            config = FaultConfig(
                id="test",
                strategy=StrategyType.EMPTY_RESULT,
                trigger=trigger,
                parameters={"call_original": True},
            )
            proxy.set_config(config)
            proxy.set_step(0)

            result = proxy()
            assert result == expected, f"Expected {expected} for {original_value}, got {result}"


class TestMalformedResultStrategy:
    """Tests for MALFORMED_RESULT strategy."""

    def test_malformed_result_truncate_string(self):
        """MALFORMED_RESULT truncates string result."""
        original = MagicMock(return_value="This is a long string result")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "truncate", "truncate_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert len(result) < len("This is a long string result")
        assert result == "This is a long"  # Half of original

    def test_malformed_result_truncate_list(self):
        """MALFORMED_RESULT truncates list result."""
        original = MagicMock(return_value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "truncate", "truncate_ratio": 0.3},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert len(result) == 3  # 30% of 10

    def test_malformed_result_add_noise_keys(self):
        """MALFORMED_RESULT adds noise keys to dict."""
        original = MagicMock(return_value={"real_key": "real_value"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "add_noise_keys", "noise_key_count": 3},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert "real_key" in result
        # Should have noise keys
        noise_keys = [k for k in result.keys() if k.startswith("__noise")]
        assert len(noise_keys) == 3

    def test_malformed_result_wrap_extra(self):
        """MALFORMED_RESULT wraps result in extra nesting."""
        original = MagicMock(return_value="simple result")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "wrap_extra"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        assert isinstance(result, dict)
        assert "result" in result
        assert "metadata" in result
        assert "extra" in result
        assert result["result"] == "simple result"

    def test_malformed_result_corrupt_structure_dict(self):
        """MALFORMED_RESULT corrupts dict structure."""
        original = MagicMock(return_value={"key1": "value1", "key2": "value2", "key3": "value3"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "corrupt_structure"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        # Should have corrupted keys
        assert "__corrupted_int" in result
        assert "__corrupted_list" in result
        # Some original keys might be missing

    def test_malformed_result_corrupt_structure_list(self):
        """MALFORMED_RESULT corrupts list structure."""
        original = MagicMock(return_value=[1, 2, 3])
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={"malform_type": "corrupt_structure"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        # Should have corruption markers
        assert {"__marker": "corruption_start"} in result
        assert None in result

    def test_malformed_result_default_corrupt_structure(self):
        """MALFORMED_RESULT defaults to corrupt_structure."""
        original = MagicMock(return_value={"key": "value"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MALFORMED_RESULT,
            trigger=trigger,
            parameters={},  # No malform_type
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()
        # Should have corrupted structure (default behavior)
        assert "__corrupted_int" in result


class TestCallToolProxyIntegration:
    """Integration tests for CallToolProxy."""

    def test_step_based_trigger(self):
        """Proxy triggers within step range."""
        original = MagicMock(return_value="result")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Before range - should call original
        proxy.set_step(3)
        result = proxy("tool")
        assert result == "result"

        # In range - should return empty
        proxy.set_step(7)
        result = proxy("tool")
        assert result is None

        # After range - should call original
        proxy.set_step(15)
        result = proxy("tool")
        assert result == "result"

    def test_periodic_trigger(self):
        """Proxy triggers every N steps."""
        original = MagicMock(return_value="original")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=3,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "empty_string"},
        )
        proxy.set_config(config)

        # Step 0 - triggers (0 % 3 == 0)
        proxy.set_step(0)
        assert proxy("tool") == ""

        # Step 1 - no trigger
        proxy.set_step(1)
        assert proxy("tool") == "original"

        # Step 3 - triggers
        proxy.set_step(3)
        assert proxy("tool") == ""

        # Step 4 - no trigger
        proxy.set_step(4)
        assert proxy("tool") == "original"

    def test_collector_recording(self):
        """Proxy records fault injection with collector."""
        original = MagicMock(return_value="result")
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"

        proxy = CallToolProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-fault",
            strategy=StrategyType.EMPTY_RESULT,
            trigger=trigger,
            parameters={"empty_type": "none"},
            severity="medium",
            expected_behavior="Returns None",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("my_tool")

        # Should record fault injection
        collector.record_fault_injection.assert_called_once()
        call_args = collector.record_fault_injection.call_args
        assert call_args[1]["fault_type"] == "empty_result"
        assert call_args[1]["target_layer"] == "Agent"
        assert call_args[1]["severity"] == "medium"

        # Should record outcome
        collector.record_fault_outcome.assert_called_once()
        outcome_call = collector.record_fault_outcome.call_args
        assert outcome_call[0][0] == "fault-123"
        assert outcome_call[0][1] == "success"

    def test_collector_records_failure(self):
        """Proxy records failure when strategy raises exception."""
        original = MagicMock()
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-456"

        proxy = CallToolProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-fault",
            strategy=StrategyType.TOOL_EXCEPTION,
            trigger=trigger,
            parameters={"exc_type": "RuntimeError", "message": "Test error"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError):
            proxy("tool")

        # Should record failure outcome
        collector.record_fault_outcome.assert_called_once()
        outcome_call = collector.record_fault_outcome.call_args
        assert outcome_call[0][0] == "fault-456"
        assert outcome_call[0][1] == "failure"

    def test_kwargs_passing(self):
        """Proxy passes kwargs to original function."""
        original = MagicMock(return_value="ok")
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):
            result = proxy(tool_name="test", arg1="value1", arg2=42)

        original.assert_called_once_with(tool_name="test", arg1="value1", arg2=42)

    def test_args_passing(self):
        """Proxy passes positional args to original function."""
        original = MagicMock(return_value={"status": "success"})
        proxy = CallToolProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_RESULT,
            trigger=trigger,
            parameters={"wrong_value": "wrong"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("tool_name", {"arg": 1}, "extra_arg")
        original.assert_called_once_with("tool_name", {"arg": 1}, "extra_arg")
        assert result == "wrong"
