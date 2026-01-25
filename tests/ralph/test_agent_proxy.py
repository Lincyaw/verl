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
from ralph.proxies.agent import CallToolProxy, ToolParserProxy


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

        proxy("my_tool")

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
            proxy(tool_name="test", arg1="value1", arg2=42)

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


# =============================================================================
# ToolParserProxy Tests
# =============================================================================


class TestToolParserProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_tool_parser_target(self):
        """ToolParserProxy is registered for 'ToolParser.parse' target."""
        assert ProxyRegistry.is_registered("ToolParser.parse")
        assert ProxyRegistry.get_proxy("ToolParser.parse") is ToolParserProxy

    def test_supported_strategies(self):
        """ToolParserProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("ToolParser.parse")
        expected = {
            StrategyType.PARSE_FAILURE,
            StrategyType.WRONG_TOOL_NAME,
            StrategyType.WRONG_ARGUMENTS,
            StrategyType.EXTRA_TOOL_CALLS,
            StrategyType.MISSING_TOOL_CALLS,
        }
        assert strategies == expected


class TestToolParserProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_agent(self):
        """_get_layer returns 'Agent'."""
        proxy = ToolParserProxy(lambda: None)
        assert proxy._get_layer() == "Agent"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"name": "test_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        result = proxy("some text with tool call")
        original.assert_called_once_with("some text with tool call")
        assert result == {"name": "test_tool", "arguments": {}}

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original_result = [{"name": "tool1"}, {"name": "tool2"}]
        original = MagicMock(return_value=original_result)
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy("input")
        original.assert_called_once()
        assert result == original_result

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = ToolParserProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by ToolParserProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestParseFailureStrategy:
    """Tests for PARSE_FAILURE strategy."""

    def test_parse_failure_raises_value_error(self):
        """PARSE_FAILURE raises ValueError by default."""
        original = MagicMock(return_value={"name": "tool"})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy("input")
        assert "Failed to parse" in str(exc_info.value)
        original.assert_not_called()

    def test_parse_failure_syntax_error(self):
        """PARSE_FAILURE with syntax error_type raises ValueError."""
        proxy = ToolParserProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={"error_type": "syntax"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy()
        assert "Invalid syntax" in str(exc_info.value)

    def test_parse_failure_json_error(self):
        """PARSE_FAILURE with json error_type raises JSONDecodeError."""
        import json

        proxy = ToolParserProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={"error_type": "json"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(json.JSONDecodeError):
            proxy()

    def test_parse_failure_timeout_error(self):
        """PARSE_FAILURE with timeout error_type raises TimeoutError."""
        proxy = ToolParserProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={"error_type": "timeout"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(TimeoutError):
            proxy()

    def test_parse_failure_custom_message(self):
        """PARSE_FAILURE uses custom error message."""
        proxy = ToolParserProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={"error_message": "Custom parse error message"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError) as exc_info:
            proxy()
        assert "Custom parse error message" in str(exc_info.value)


class TestWrongToolNameStrategy:
    """Tests for WRONG_TOOL_NAME strategy."""

    def test_wrong_tool_name_with_replacement(self):
        """WRONG_TOOL_NAME replaces with specific name."""
        original = MagicMock(return_value={"name": "correct_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"replacement_name": "wrong_tool"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert result["name"] == "wrong_tool"
        original.assert_called_once()

    def test_wrong_tool_name_typo_corruption(self):
        """WRONG_TOOL_NAME with typo corruption modifies name."""
        original = MagicMock(return_value={"name": "search_documents", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"corruption_type": "typo"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Name should be modified but similar
        assert result["name"] != "search_documents"
        original.assert_called_once()

    def test_wrong_tool_name_prefix_corruption(self):
        """WRONG_TOOL_NAME with prefix corruption adds prefix."""
        original = MagicMock(return_value={"name": "my_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"corruption_type": "prefix"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert result["name"].startswith("__wrong_")

    def test_wrong_tool_name_suffix_corruption(self):
        """WRONG_TOOL_NAME with suffix corruption adds suffix."""
        original = MagicMock(return_value={"name": "my_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"corruption_type": "suffix"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert "_v2_deprecated" in result["name"]

    def test_wrong_tool_name_list_of_calls(self):
        """WRONG_TOOL_NAME handles list of tool calls."""
        original = MagicMock(
            return_value=[
                {"name": "tool1", "arguments": {}},
                {"name": "tool2", "arguments": {}},
            ]
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"replacement_name": "wrong"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert len(result) == 2
        assert all(call["name"] == "wrong" for call in result)

    def test_wrong_tool_name_similar_corruption(self):
        """WRONG_TOOL_NAME with similar corruption uses similar name."""
        original = MagicMock(return_value={"name": "get_user", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"corruption_type": "similar"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # get -> fetch transformation
        assert "fetch" in result["name"]


class TestWrongArgumentsStrategy:
    """Tests for WRONG_ARGUMENTS strategy."""

    def test_wrong_arguments_drop(self):
        """WRONG_ARGUMENTS with drop removes some arguments."""
        original = MagicMock(
            return_value={
                "name": "tool",
                "arguments": {"arg1": "val1", "arg2": "val2", "arg3": "val3", "arg4": "val4"},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "drop", "drop_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Should have fewer arguments
        assert len(result["arguments"]) < 4

    def test_wrong_arguments_add_extra(self):
        """WRONG_ARGUMENTS with add_extra adds spurious arguments."""
        original = MagicMock(
            return_value={
                "name": "tool",
                "arguments": {"real_arg": "value"},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "add_extra"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Should have extra arguments
        assert "real_arg" in result["arguments"]
        assert "__extra_arg_1" in result["arguments"]

    def test_wrong_arguments_wrong_type(self):
        """WRONG_ARGUMENTS with wrong_type changes argument types."""
        original = MagicMock(
            return_value={
                "name": "tool",
                "arguments": {"bool_arg": True, "int_arg": 42},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "wrong_type"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Types should be changed
        assert result["arguments"]["bool_arg"] == "true"  # bool -> str
        assert result["arguments"]["int_arg"] == "42"  # int -> str

    def test_wrong_arguments_wrong_value(self):
        """WRONG_ARGUMENTS with wrong_value corrupts values but keeps types."""
        original = MagicMock(
            return_value={
                "name": "tool",
                "arguments": {"bool_arg": True, "str_arg": "hello"},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "wrong_value"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # bool should be inverted, string reversed
        assert result["arguments"]["bool_arg"] is False
        assert result["arguments"]["str_arg"] == "olleh"

    def test_wrong_arguments_shuffle(self):
        """WRONG_ARGUMENTS with shuffle rearranges argument values."""
        original = MagicMock(
            return_value={
                "name": "tool",
                "arguments": {"a": 1, "b": 2, "c": 3},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "shuffle"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Values should be shuffled between keys
        values = set(result["arguments"].values())
        assert values == {1, 2, 3}  # Same values, different order

    def test_wrong_arguments_json_string(self):
        """WRONG_ARGUMENTS handles JSON string arguments."""
        import json

        original = MagicMock(
            return_value={
                "name": "tool",
                "function": {"arguments": json.dumps({"key": "value", "num": 42})},
            }
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_ARGUMENTS,
            trigger=trigger,
            parameters={"corruption_type": "wrong_type"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Should parse JSON and corrupt
        parsed = json.loads(result["function"]["arguments"])
        assert "key" in parsed


class TestExtraToolCallsStrategy:
    """Tests for EXTRA_TOOL_CALLS strategy."""

    def test_extra_tool_calls_adds_calls(self):
        """EXTRA_TOOL_CALLS adds additional tool calls."""
        original = MagicMock(return_value={"name": "real_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={"extra_count": 2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Should return list with original + 2 extras
        assert isinstance(result, list)
        assert len(result) == 3

    def test_extra_tool_calls_position_before(self):
        """EXTRA_TOOL_CALLS adds calls before original."""
        original = MagicMock(return_value={"name": "real_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={"extra_count": 1, "position": "before"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Real tool should be last
        assert result[-1]["name"] == "real_tool"

    def test_extra_tool_calls_position_after(self):
        """EXTRA_TOOL_CALLS adds calls after original."""
        original = MagicMock(return_value={"name": "real_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={"extra_count": 1, "position": "after"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Real tool should be first
        assert result[0]["name"] == "real_tool"

    def test_extra_tool_calls_custom_tools(self):
        """EXTRA_TOOL_CALLS adds specific custom tools."""
        original = MagicMock(return_value={"name": "real_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={
                "extra_tools": [
                    {"name": "custom_extra_1", "arguments": {"x": 1}},
                    {"name": "custom_extra_2", "arguments": {"y": 2}},
                ],
            },
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        names = [call.get("name") for call in result]
        assert "custom_extra_1" in names
        assert "custom_extra_2" in names

    def test_extra_tool_calls_with_list_input(self):
        """EXTRA_TOOL_CALLS works with list of original calls."""
        original = MagicMock(
            return_value=[
                {"name": "tool1", "arguments": {}},
                {"name": "tool2", "arguments": {}},
            ]
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={"extra_count": 1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert len(result) == 3  # 2 original + 1 extra


class TestMissingToolCallsStrategy:
    """Tests for MISSING_TOOL_CALLS strategy."""

    def test_missing_tool_calls_drops_some(self):
        """MISSING_TOOL_CALLS removes some tool calls."""
        original = MagicMock(
            return_value=[
                {"name": "tool1", "arguments": {}},
                {"name": "tool2", "arguments": {}},
                {"name": "tool3", "arguments": {}},
                {"name": "tool4", "arguments": {}},
            ]
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0.5, "keep_first": False},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Should have fewer calls
        assert len(result) < 4

    def test_missing_tool_calls_keep_first(self):
        """MISSING_TOOL_CALLS keeps first call when configured."""
        original = MagicMock(
            return_value=[
                {"name": "first_tool", "arguments": {}},
                {"name": "tool2", "arguments": {}},
                {"name": "tool3", "arguments": {}},
            ]
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0.9, "keep_first": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # First tool should be kept
        assert any(call["name"] == "first_tool" for call in result)

    def test_missing_tool_calls_keep_last(self):
        """MISSING_TOOL_CALLS keeps last call when configured."""
        original = MagicMock(
            return_value=[
                {"name": "tool1", "arguments": {}},
                {"name": "tool2", "arguments": {}},
                {"name": "last_tool", "arguments": {}},
            ]
        )
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0.9, "keep_first": False, "keep_last": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # Last tool should be kept
        assert any(call["name"] == "last_tool" for call in result)

    def test_missing_tool_calls_max_keep(self):
        """MISSING_TOOL_CALLS respects max_keep limit."""
        original = MagicMock(return_value=[{"name": f"tool{i}", "arguments": {}} for i in range(10)])
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0, "max_keep": 3},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        assert len(result) <= 3

    def test_missing_tool_calls_single_dict(self):
        """MISSING_TOOL_CALLS handles single dict result."""
        original = MagicMock(return_value={"name": "single_tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0.5, "keep_first": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy("input")
        # With keep_first, single call should be kept
        assert result == {"name": "single_tool", "arguments": {}}


class TestToolParserProxyIntegration:
    """Integration tests for ToolParserProxy."""

    def test_step_based_trigger(self):
        """Proxy triggers within step range."""
        original = MagicMock(return_value={"name": "tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"replacement_name": "wrong"},
        )
        proxy.set_config(config)

        # Before range - should call original
        proxy.set_step(3)
        result = proxy("input")
        assert result["name"] == "tool"

        # In range - should corrupt
        proxy.set_step(7)
        result = proxy("input")
        assert result["name"] == "wrong"

        # After range - should call original
        proxy.set_step(15)
        result = proxy("input")
        assert result["name"] == "tool"

    def test_periodic_trigger(self):
        """Proxy triggers every N steps."""
        original = MagicMock(return_value=[{"name": "tool", "arguments": {}}])
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=2,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EXTRA_TOOL_CALLS,
            trigger=trigger,
            parameters={"extra_count": 1},
        )
        proxy.set_config(config)

        # Step 0 - triggers
        proxy.set_step(0)
        result = proxy("input")
        assert len(result) == 2  # original + extra

        # Step 1 - no trigger
        proxy.set_step(1)
        result = proxy("input")
        assert len(result) == 1  # just original

        # Step 2 - triggers
        proxy.set_step(2)
        result = proxy("input")
        assert len(result) == 2

    def test_collector_recording(self):
        """Proxy records fault injection with collector."""
        original = MagicMock(return_value={"name": "tool", "arguments": {}})
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"

        proxy = ToolParserProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-fault",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"replacement_name": "wrong"},
            severity="high",
            expected_behavior="Returns wrong tool name",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("input")

        # Should record fault injection
        collector.record_fault_injection.assert_called_once()
        call_args = collector.record_fault_injection.call_args
        assert call_args[1]["fault_type"] == "wrong_tool_name"
        assert call_args[1]["target_layer"] == "Agent"
        assert call_args[1]["severity"] == "high"

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

        proxy = ToolParserProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test-fault",
            strategy=StrategyType.PARSE_FAILURE,
            trigger=trigger,
            parameters={"error_type": "syntax"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(ValueError):
            proxy("input")

        # Should record failure outcome
        collector.record_fault_outcome.assert_called_once()
        outcome_call = collector.record_fault_outcome.call_args
        assert outcome_call[0][0] == "fault-456"
        assert outcome_call[0][1] == "failure"

    def test_kwargs_passing(self):
        """Proxy passes kwargs to original function."""
        original = MagicMock(return_value={"name": "tool", "arguments": {}})
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WRONG_TOOL_NAME,
            trigger=trigger,
            parameters={"replacement_name": "wrong"},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy(text="some input", format="json")
        original.assert_called_once_with(text="some input", format="json")

    def test_args_passing(self):
        """Proxy passes positional args to original function."""
        original = MagicMock(return_value=[{"name": "tool1"}, {"name": "tool2"}])
        proxy = ToolParserProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MISSING_TOOL_CALLS,
            trigger=trigger,
            parameters={"drop_ratio": 0.5, "keep_first": True},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("text_input", "extra_arg")
        original.assert_called_once_with("text_input", "extra_arg")
