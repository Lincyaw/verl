"""
Agent proxies for Ralph fault injection framework.

Contains proxy classes for agent-level operations like tool calling
in agentic AI workflows.
"""

from typing import Any, Dict, List, Optional, Set

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.proxies.base import BaseProxy


@ProxyRegistry.register("ToolAgentLoop._call_tool")
class CallToolProxy(BaseProxy, DelayMixin, ExceptionMixin):
    """
    Proxy for tool calling operations in agentic AI workflows.

    Supports fault injection at the tool call level for agent loops.
    This allows testing how agent pipelines handle tool call failures,
    timeouts, and corrupted results.

    Supported strategies:
    - DELAY: Standard delay before tool call (inherited from DelayMixin)
    - TOOL_TIMEOUT: Very long delay simulating a hung tool call
    - TOOL_EXCEPTION: Raises an exception during tool call
    - WRONG_RESULT: Returns incorrect/unexpected result from tool
    - EMPTY_RESULT: Returns empty/None result from tool
    - MALFORMED_RESULT: Returns structurally invalid result

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - TOOL_TIMEOUT: timeout_seconds (float, default 300.0) - very long delay
    - TOOL_EXCEPTION: exc_type (str), message (str) - exception to raise
    - WRONG_RESULT: wrong_value (Any), corruption_type (str) - how to corrupt
    - EMPTY_RESULT: empty_type (str, default "none") - type of empty value
    - MALFORMED_RESULT: malform_type (str) - how to malform the result

    Expected input/output:
    - Input: Tool name, arguments, optional context
    - Output: Tool execution result (varies by tool)
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.TOOL_TIMEOUT,
        StrategyType.TOOL_EXCEPTION,
        StrategyType.WRONG_RESULT,
        StrategyType.EMPTY_RESULT,
        StrategyType.MALFORMED_RESULT,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Agent"

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Add standard delay before tool call.

        Applies a delay before calling the original tool function.
        This simulates slow tool execution or network latency.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result from original function after delay.

        Config parameters:
            delay_seconds (float): Seconds to delay (default 10.0).
        """
        delay_seconds = self._config.parameters.get("delay_seconds", 10.0)
        self._apply_delay(delay_seconds)
        return self._original(*args, **kwargs)

    def _strategy_tool_timeout(self, *args: Any, **kwargs: Any) -> Any:
        """
        Simulate a tool timeout with very long delay.

        Applies a very long delay to simulate a hung or extremely slow
        tool call. This tests timeout handling in the agent pipeline.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Result from original function after very long delay.

        Config parameters:
            timeout_seconds (float): Very long delay in seconds (default 300.0).
        """
        timeout_seconds = self._config.parameters.get("timeout_seconds", 300.0)
        self._apply_delay(timeout_seconds)
        return self._original(*args, **kwargs)

    def _strategy_tool_exception(self, *args: Any, **kwargs: Any) -> Any:
        """
        Raise an exception during tool call.

        Raises a configured exception to simulate tool execution failures.
        This tests exception handling in the agent pipeline.

        Args:
            *args: Positional arguments (not used, exception is raised first)
            **kwargs: Keyword arguments (not used, exception is raised first)

        Raises:
            The configured exception type with the specified message.

        Config parameters:
            exc_type (str): Exception type to raise (required).
            message (str): Exception message (default: "Tool call failed: {exc_type}").
        """
        exc_type = self._config.parameters.get("exc_type")
        if exc_type is None:
            raise ValueError("exc_type is required for TOOL_EXCEPTION strategy")

        # Extract tool name from args for better error messages
        tool_name = self._extract_tool_name(args, kwargs)
        default_message = f"Tool call failed for '{tool_name}': {exc_type}"
        message = self._config.parameters.get("message", default_message)

        self._raise_exception(exc_type, message)

    def _extract_tool_name(self, args: tuple, kwargs: dict) -> str:
        """
        Extract the tool name from call arguments.

        Args:
            args: Positional arguments to the tool call
            kwargs: Keyword arguments to the tool call

        Returns:
            Tool name string, or "unknown" if not found
        """
        # Check common patterns for tool name
        if len(args) > 0 and isinstance(args[0], str):
            return args[0]
        if "tool_name" in kwargs:
            return str(kwargs["tool_name"])
        if "name" in kwargs:
            return str(kwargs["name"])
        if "tool" in kwargs:
            tool = kwargs["tool"]
            if isinstance(tool, str):
                return tool
            if hasattr(tool, "name"):
                return str(tool.name)
            if hasattr(tool, "__name__"):
                return str(tool.__name__)
        return "unknown"

    def _strategy_wrong_result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return incorrect/unexpected result from tool.

        Calls the original tool, then corrupts or replaces the result
        to simulate tool malfunction or data corruption.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Corrupted or replaced result.

        Config parameters:
            wrong_value (Any, optional): Specific wrong value to return.
            corruption_type (str): How to corrupt - "negate", "invert", "randomize", "swap_type" (default "swap_type").
        """
        result = self._original(*args, **kwargs)

        # Check if a specific wrong value is provided
        if "wrong_value" in self._config.parameters:
            return self._config.parameters["wrong_value"]

        corruption_type = self._config.parameters.get("corruption_type", "swap_type")
        return self._apply_wrong_result(result, corruption_type)

    def _apply_wrong_result(self, result: Any, corruption_type: str) -> Any:
        """
        Apply corruption to the result based on corruption_type.

        Args:
            result: Original result to corrupt
            corruption_type: Type of corruption to apply

        Returns:
            Corrupted result
        """
        if corruption_type == "negate":
            return self._negate_result(result)
        elif corruption_type == "invert":
            return self._invert_result(result)
        elif corruption_type == "randomize":
            return self._randomize_result(result)
        elif corruption_type == "swap_type":
            return self._swap_type_result(result)
        else:
            # Default: swap type
            return self._swap_type_result(result)

    def _negate_result(self, result: Any) -> Any:
        """
        Negate the result value.

        Args:
            result: Result to negate

        Returns:
            Negated result (for booleans, numbers, etc.)
        """
        if isinstance(result, bool):
            return not result
        elif isinstance(result, (int, float)):
            return -result
        elif isinstance(result, str):
            return result[::-1]  # Reverse string
        elif isinstance(result, dict):
            return {k: self._negate_result(v) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._negate_result(item) for item in result]
        elif isinstance(result, tuple):
            return tuple(self._negate_result(item) for item in result)
        else:
            return result

    def _invert_result(self, result: Any) -> Any:
        """
        Invert the result (True<->False, reverse collections, etc.).

        Args:
            result: Result to invert

        Returns:
            Inverted result
        """
        if isinstance(result, bool):
            return not result
        elif isinstance(result, (list, tuple)):
            inverted = list(reversed(result))
            return tuple(inverted) if isinstance(result, tuple) else inverted
        elif isinstance(result, dict):
            # Reverse key-value pairs
            return {v: k for k, v in result.items() if isinstance(v, (str, int, float, bool))}
        elif isinstance(result, str):
            return result[::-1]
        elif isinstance(result, (int, float)):
            return 1 / result if result != 0 else float('inf')
        else:
            return result

    def _randomize_result(self, result: Any) -> Any:
        """
        Randomize the result value.

        Args:
            result: Result to randomize

        Returns:
            Randomized result
        """
        import random

        if isinstance(result, bool):
            return random.choice([True, False])
        elif isinstance(result, int):
            return random.randint(-1000, 1000)
        elif isinstance(result, float):
            return random.uniform(-1000.0, 1000.0)
        elif isinstance(result, str):
            import string
            length = len(result) if result else 10
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif isinstance(result, dict):
            return {k: self._randomize_result(v) for k, v in result.items()}
        elif isinstance(result, list):
            random.shuffle(result := list(result))
            return result
        elif isinstance(result, tuple):
            lst = list(result)
            random.shuffle(lst)
            return tuple(lst)
        else:
            return result

    def _swap_type_result(self, result: Any) -> Any:
        """
        Swap the result to a different type.

        Args:
            result: Result to swap type of

        Returns:
            Result converted to a different type
        """
        if isinstance(result, bool):
            return int(result)
        elif isinstance(result, int):
            return str(result)
        elif isinstance(result, float):
            return int(result)
        elif isinstance(result, str):
            try:
                return int(result)
            except ValueError:
                return len(result)
        elif isinstance(result, dict):
            return list(result.items())
        elif isinstance(result, list):
            return tuple(result)
        elif isinstance(result, tuple):
            return list(result)
        elif result is None:
            return ""
        else:
            return str(result)

    def _strategy_empty_result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return empty/None result from tool.

        Calls the original tool (optionally) to understand expected type,
        then returns an empty value. This tests how agents handle tools
        that return nothing useful.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Empty value based on configuration.

        Config parameters:
            empty_type (str): Type of empty value - "none", "empty_string",
                "empty_dict", "empty_list", "zero" (default "none").
            call_original (bool): Whether to call original first (default False).
        """
        call_original = self._config.parameters.get("call_original", False)

        if call_original:
            # Call original to potentially detect expected return type
            result = self._original(*args, **kwargs)
            return self._get_empty_value_like(result)

        empty_type = self._config.parameters.get("empty_type", "none")
        return self._get_empty_value(empty_type)

    def _get_empty_value(self, empty_type: str) -> Any:
        """
        Get an empty value based on the specified type.

        Args:
            empty_type: Type of empty value to return

        Returns:
            Empty value of the specified type
        """
        empty_values = {
            "none": None,
            "empty_string": "",
            "empty_dict": {},
            "empty_list": [],
            "empty_tuple": (),
            "zero": 0,
            "false": False,
        }
        return empty_values.get(empty_type, None)

    def _get_empty_value_like(self, result: Any) -> Any:
        """
        Get an empty value matching the type of the original result.

        Args:
            result: Original result to match type of

        Returns:
            Empty value of the same type
        """
        if result is None:
            return None
        elif isinstance(result, bool):
            return False
        elif isinstance(result, int):
            return 0
        elif isinstance(result, float):
            return 0.0
        elif isinstance(result, str):
            return ""
        elif isinstance(result, dict):
            return {}
        elif isinstance(result, list):
            return []
        elif isinstance(result, tuple):
            return ()
        elif isinstance(result, set):
            return set()
        else:
            return None

    def _strategy_malformed_result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return structurally invalid result from tool.

        Calls the original tool, then malforms the result to create
        structurally invalid data. This tests how agents handle
        malformed tool outputs.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Malformed result.

        Config parameters:
            malform_type (str): How to malform - "truncate", "add_noise_keys",
                "wrap_extra", "corrupt_structure" (default "corrupt_structure").
            truncate_ratio (float): For truncate, how much to keep (default 0.5).
            noise_key_count (int): For add_noise_keys, how many (default 5).
        """
        result = self._original(*args, **kwargs)
        malform_type = self._config.parameters.get("malform_type", "corrupt_structure")
        return self._apply_malformed_result(result, malform_type)

    def _apply_malformed_result(self, result: Any, malform_type: str) -> Any:
        """
        Apply malformation to the result based on malform_type.

        Args:
            result: Original result to malform
            malform_type: Type of malformation to apply

        Returns:
            Malformed result
        """
        if malform_type == "truncate":
            return self._truncate_result(result)
        elif malform_type == "add_noise_keys":
            return self._add_noise_keys(result)
        elif malform_type == "wrap_extra":
            return self._wrap_extra(result)
        elif malform_type == "corrupt_structure":
            return self._corrupt_structure(result)
        else:
            return self._corrupt_structure(result)

    def _truncate_result(self, result: Any) -> Any:
        """
        Truncate the result to a shorter version.

        Args:
            result: Result to truncate

        Returns:
            Truncated result
        """
        truncate_ratio = self._config.parameters.get("truncate_ratio", 0.5)

        if isinstance(result, str):
            length = max(1, int(len(result) * truncate_ratio))
            return result[:length]
        elif isinstance(result, (list, tuple)):
            length = max(1, int(len(result) * truncate_ratio))
            truncated = result[:length]
            return tuple(truncated) if isinstance(result, tuple) else truncated
        elif isinstance(result, dict):
            keys = list(result.keys())
            keep_count = max(1, int(len(keys) * truncate_ratio))
            return {k: result[k] for k in keys[:keep_count]}
        elif isinstance(result, bytes):
            length = max(1, int(len(result) * truncate_ratio))
            return result[:length]
        else:
            return result

    def _add_noise_keys(self, result: Any) -> Any:
        """
        Add random noise keys to dict results.

        Args:
            result: Result to add noise keys to

        Returns:
            Result with added noise keys (if dict)
        """
        import random
        import string

        noise_key_count = self._config.parameters.get("noise_key_count", 5)

        if isinstance(result, dict):
            modified = dict(result)
            for i in range(noise_key_count):
                noise_key = f"__noise_{i}_" + ''.join(
                    random.choices(string.ascii_lowercase, k=5)
                )
                noise_value = random.choice([
                    None,
                    random.randint(-100, 100),
                    ''.join(random.choices(string.ascii_letters, k=10)),
                    [random.randint(0, 10) for _ in range(3)],
                    {"nested": "noise"},
                ])
                modified[noise_key] = noise_value
            return modified
        elif isinstance(result, list):
            # Add noise elements to list
            modified = list(result)
            for _ in range(noise_key_count):
                modified.append({"__noise": True, "value": random.randint(0, 100)})
            return modified
        else:
            return result

    def _wrap_extra(self, result: Any) -> Any:
        """
        Wrap the result in extra layers of nesting.

        Args:
            result: Result to wrap

        Returns:
            Result wrapped in extra nesting
        """
        # Wrap in multiple layers of dict/list
        return {
            "result": result,
            "metadata": {
                "wrapped": True,
                "original_type": type(result).__name__,
            },
            "extra": [result, {"nested": result}],
        }

    def _corrupt_structure(self, result: Any) -> Any:
        """
        Corrupt the structural integrity of the result.

        Args:
            result: Result to corrupt structurally

        Returns:
            Structurally corrupted result
        """
        import random

        if isinstance(result, dict):
            # Remove random keys and add corrupted ones
            keys = list(result.keys())
            modified = dict(result)

            # Randomly remove some keys
            if len(keys) > 1:
                keys_to_remove = random.sample(keys, k=max(1, len(keys) // 3))
                for k in keys_to_remove:
                    del modified[k]

            # Add keys with wrong types
            modified["__corrupted_int"] = "not_an_int"
            modified["__corrupted_list"] = {"should_be": "list"}
            modified["__partial_data"] = {"incomplete": True}

            return modified
        elif isinstance(result, list):
            # Insert unexpected types at random positions
            modified = list(result)
            modified.insert(0, {"__marker": "corruption_start"})
            modified.append(None)
            modified.append({"__marker": "corruption_end"})
            return modified
        elif isinstance(result, tuple):
            # Convert to list with corruption
            return self._corrupt_structure(list(result))
        elif isinstance(result, str):
            # Insert corruption markers
            return f"[CORRUPTED]{result[:len(result)//2]}[TRUNCATED]"
        else:
            # Wrap in error-like structure
            return {"error": "corrupted", "partial_value": result}
