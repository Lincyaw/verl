"""
Agent proxies for Ralph fault injection framework.

Contains proxy classes for agent-level operations like tool calling
in agentic AI workflows.
"""

from typing import Any, Optional

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

    SUPPORTED_STRATEGIES: set[StrategyType] = {
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
            return 1 / result if result != 0 else float("inf")
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
            return "".join(random.choices(string.ascii_letters + string.digits, k=length))
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
                noise_key = f"__noise_{i}_" + "".join(random.choices(string.ascii_lowercase, k=5))
                noise_value = random.choice(
                    [
                        None,
                        random.randint(-100, 100),
                        "".join(random.choices(string.ascii_letters, k=10)),
                        [random.randint(0, 10) for _ in range(3)],
                        {"nested": "noise"},
                    ]
                )
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
            return f"[CORRUPTED]{result[: len(result) // 2]}[TRUNCATED]"
        else:
            # Wrap in error-like structure
            return {"error": "corrupted", "partial_value": result}


@ProxyRegistry.register("ToolParser.parse")
class ToolParserProxy(BaseProxy, DelayMixin):
    """
    Proxy for tool parsing operations in agentic AI workflows.

    Supports fault injection at the tool parsing level for agent loops.
    This allows testing how agent pipelines handle parse errors, malformed
    tool calls, and incorrect argument extraction.

    Supported strategies:
    - PARSE_FAILURE: Raises a parse error simulating invalid tool syntax
    - WRONG_TOOL_NAME: Returns incorrect tool name from parsing
    - WRONG_ARGUMENTS: Returns incorrect/corrupted arguments for the tool
    - EXTRA_TOOL_CALLS: Returns additional spurious tool calls
    - MISSING_TOOL_CALLS: Returns fewer tool calls than actually parsed

    Config parameters:
    - PARSE_FAILURE: error_message (str), error_type (str) - type of parse error
    - WRONG_TOOL_NAME: replacement_name (str), corruption_type (str) - how to corrupt
    - WRONG_ARGUMENTS: corruption_type (str), drop_ratio (float), add_noise (bool)
    - EXTRA_TOOL_CALLS: extra_count (int), extra_tools (list) - tools to add
    - MISSING_TOOL_CALLS: drop_ratio (float), keep_first (bool) - which to keep

    Expected input/output:
    - Input: Raw text/message to parse for tool calls
    - Output: Parsed tool call(s) with name and arguments
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.PARSE_FAILURE,
        StrategyType.WRONG_TOOL_NAME,
        StrategyType.WRONG_ARGUMENTS,
        StrategyType.EXTRA_TOOL_CALLS,
        StrategyType.MISSING_TOOL_CALLS,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "Agent"

    def _strategy_parse_failure(self, *args: Any, **kwargs: Any) -> Any:
        """
        Simulate a parsing failure.

        Raises a parsing error to simulate invalid tool call syntax or
        malformed input that cannot be parsed.

        Args:
            *args: Positional arguments (not used, error is raised first)
            **kwargs: Keyword arguments (not used, error is raised first)

        Raises:
            ValueError or configured exception for parse failures.

        Config parameters:
            error_message (str): Custom error message (default: "Failed to parse tool call").
            error_type (str): Type of error - "syntax", "json", "schema", "timeout" (default "syntax").
        """
        error_type = self._config.parameters.get("error_type", "syntax")
        default_messages = {
            "syntax": "Failed to parse tool call: Invalid syntax",
            "json": "Failed to parse tool call: Invalid JSON format",
            "schema": "Failed to parse tool call: Schema validation failed",
            "timeout": "Failed to parse tool call: Parsing timeout exceeded",
            "encoding": "Failed to parse tool call: Invalid character encoding",
        }
        default_message = default_messages.get(error_type, default_messages["syntax"])
        error_message = self._config.parameters.get("error_message", default_message)

        # Raise appropriate exception based on error_type
        if error_type == "json":
            import json

            raise json.JSONDecodeError(error_message, "", 0)
        elif error_type == "timeout":
            raise TimeoutError(error_message)
        elif error_type == "encoding":
            raise UnicodeDecodeError("utf-8", b"", 0, 1, error_message)
        else:
            # syntax, schema, or default
            raise ValueError(error_message)

    def _strategy_wrong_tool_name(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return incorrect tool name from parsing.

        Calls the original parser, then modifies the tool name(s) in the
        result to test how agents handle incorrect tool identification.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Parsed result with corrupted tool name(s).

        Config parameters:
            replacement_name (str, optional): Specific name to replace with.
            corruption_type (str): How to corrupt - "replace", "prefix", "suffix",
                "typo", "similar" (default "typo").
        """
        result = self._original(*args, **kwargs)

        # Check for specific replacement name
        if "replacement_name" in self._config.parameters:
            return self._replace_tool_name(result, self._config.parameters["replacement_name"])

        corruption_type = self._config.parameters.get("corruption_type", "typo")
        return self._corrupt_tool_name(result, corruption_type)

    def _replace_tool_name(self, result: Any, new_name: str) -> Any:
        """
        Replace tool name(s) with a specific new name.

        Args:
            result: Parsed result containing tool call(s)
            new_name: Name to replace with

        Returns:
            Result with replaced tool name(s)
        """
        return self._apply_to_tool_names(result, lambda _: new_name)

    def _corrupt_tool_name(self, result: Any, corruption_type: str) -> Any:
        """
        Corrupt tool name(s) based on corruption type.

        Args:
            result: Parsed result containing tool call(s)
            corruption_type: Type of corruption to apply

        Returns:
            Result with corrupted tool name(s)
        """
        import random

        def corrupt(name: str) -> str:
            if corruption_type == "replace":
                fake_tools = ["unknown_tool", "deprecated_function", "internal_only", "test_stub"]
                return random.choice(fake_tools)
            elif corruption_type == "prefix":
                return f"__wrong_{name}"
            elif corruption_type == "suffix":
                return f"{name}_v2_deprecated"
            elif corruption_type == "typo":
                return self._introduce_typo(name)
            elif corruption_type == "similar":
                return self._get_similar_name(name)
            else:
                return self._introduce_typo(name)

        return self._apply_to_tool_names(result, corrupt)

    def _introduce_typo(self, name: str) -> str:
        """
        Introduce a typo into a tool name.

        Args:
            name: Original tool name

        Returns:
            Tool name with typo
        """
        import random

        if not name:
            return "unknwon"

        typo_type = random.choice(["swap", "delete", "insert", "replace"])
        chars = list(name)

        if len(chars) < 2:
            return name + "_typo"

        if typo_type == "swap" and len(chars) >= 2:
            # Swap adjacent characters
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif typo_type == "delete":
            # Delete a character
            idx = random.randint(0, len(chars) - 1)
            chars.pop(idx)
        elif typo_type == "insert":
            # Insert a random character
            idx = random.randint(0, len(chars))
            chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))
        elif typo_type == "replace":
            # Replace a character with a nearby key
            idx = random.randint(0, len(chars) - 1)
            nearby = {
                "a": "sq",
                "b": "vn",
                "c": "xv",
                "d": "sf",
                "e": "wr",
                "f": "dg",
                "g": "fh",
                "h": "gj",
                "i": "uo",
                "j": "hk",
                "k": "jl",
                "l": "k",
                "m": "n",
                "n": "bm",
                "o": "ip",
                "p": "o",
                "q": "wa",
                "r": "et",
                "s": "ad",
                "t": "ry",
                "u": "yi",
                "v": "cb",
                "w": "qe",
                "x": "zc",
                "y": "tu",
                "z": "x",
            }
            char = chars[idx].lower()
            if char in nearby:
                chars[idx] = random.choice(nearby[char])

        return "".join(chars)

    def _get_similar_name(self, name: str) -> str:
        """
        Get a similar-sounding tool name.

        Args:
            name: Original tool name

        Returns:
            Similar tool name
        """
        # Common similar tool name patterns
        similar_patterns = [
            ("get", "fetch"),
            ("set", "update"),
            ("create", "make"),
            ("delete", "remove"),
            ("read", "load"),
            ("write", "save"),
            ("search", "find"),
            ("list", "enumerate"),
            ("add", "append"),
            ("exec", "run"),
        ]

        for pattern, replacement in similar_patterns:
            if pattern in name.lower():
                return name.lower().replace(pattern, replacement)

        # If no pattern matches, add a version suffix
        return f"{name}_alt"

    def _apply_to_tool_names(self, result: Any, transform_fn) -> Any:
        """
        Apply a transformation function to all tool names in the result.

        Args:
            result: Parsed result containing tool call(s)
            transform_fn: Function to apply to each tool name

        Returns:
            Result with transformed tool names
        """
        if result is None:
            return None

        # Handle dict with tool name
        if isinstance(result, dict):
            modified = dict(result)
            for key in ["name", "tool_name", "function", "tool", "action"]:
                if key in modified and isinstance(modified[key], str):
                    modified[key] = transform_fn(modified[key])
                    break
            # Handle nested function/tool object
            if "function" in modified and isinstance(modified["function"], dict):
                if "name" in modified["function"]:
                    modified["function"] = dict(modified["function"])
                    modified["function"]["name"] = transform_fn(modified["function"]["name"])
            return modified

        # Handle list of tool calls
        if isinstance(result, list):
            return [self._apply_to_tool_names(item, transform_fn) for item in result]

        # Handle tuple
        if isinstance(result, tuple):
            return tuple(self._apply_to_tool_names(item, transform_fn) for item in result)

        # Handle object with name attribute
        if hasattr(result, "name"):
            try:
                result.name = transform_fn(result.name)
            except AttributeError:
                pass  # Read-only attribute
            return result

        return result

    def _strategy_wrong_arguments(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return incorrect/corrupted arguments for the tool.

        Calls the original parser, then modifies the arguments in the
        parsed tool call(s) to test argument validation.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Parsed result with corrupted arguments.

        Config parameters:
            corruption_type (str): How to corrupt - "drop", "add_extra", "wrong_type",
                "wrong_value", "shuffle" (default "wrong_type").
            drop_ratio (float): For drop, fraction to remove (default 0.5).
            add_noise (bool): Whether to add noise arguments (default True).
        """
        result = self._original(*args, **kwargs)
        corruption_type = self._config.parameters.get("corruption_type", "wrong_type")
        return self._corrupt_arguments(result, corruption_type)

    def _corrupt_arguments(self, result: Any, corruption_type: str) -> Any:
        """
        Corrupt arguments in the parsed result.

        Args:
            result: Parsed result containing tool call(s)
            corruption_type: Type of corruption to apply

        Returns:
            Result with corrupted arguments
        """
        if result is None:
            return None

        # Handle dict with arguments
        if isinstance(result, dict):
            modified = dict(result)

            # Find arguments in various formats
            for arg_key in ["arguments", "args", "parameters", "params", "input", "kwargs"]:
                if arg_key in modified:
                    modified[arg_key] = self._corrupt_arg_value(modified[arg_key], corruption_type)
                    break

            # Handle function.arguments pattern (OpenAI style)
            if "function" in modified and isinstance(modified["function"], dict):
                if "arguments" in modified["function"]:
                    modified["function"] = dict(modified["function"])
                    modified["function"]["arguments"] = self._corrupt_arg_value(
                        modified["function"]["arguments"], corruption_type
                    )

            return modified

        # Handle list of tool calls
        if isinstance(result, list):
            return [self._corrupt_arguments(item, corruption_type) for item in result]

        # Handle tuple
        if isinstance(result, tuple):
            return tuple(self._corrupt_arguments(item, corruption_type) for item in result)

        return result

    def _corrupt_arg_value(self, args: Any, corruption_type: str) -> Any:
        """
        Corrupt an argument value based on corruption type.

        Args:
            args: Arguments to corrupt (dict, str, list, etc.)
            corruption_type: Type of corruption

        Returns:
            Corrupted arguments
        """
        import json
        import random

        # Handle string arguments (often JSON)
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                corrupted = self._corrupt_arg_value(parsed, corruption_type)
                return json.dumps(corrupted)
            except json.JSONDecodeError:
                # Not JSON, corrupt as string
                if corruption_type == "wrong_value":
                    return args[::-1]  # Reverse
                return args

        # Handle dict arguments
        if isinstance(args, dict):
            modified = dict(args)

            if corruption_type == "drop":
                # Drop some arguments
                drop_ratio = self._config.parameters.get("drop_ratio", 0.5)
                keys = list(modified.keys())
                drop_count = max(1, int(len(keys) * drop_ratio))
                for key in random.sample(keys, min(drop_count, len(keys))):
                    del modified[key]

            elif corruption_type == "add_extra":
                # Add spurious arguments
                add_noise = self._config.parameters.get("add_noise", True)
                if add_noise:
                    modified["__extra_arg_1"] = "unexpected_value"
                    modified["__extra_arg_2"] = 999
                    modified["__debug_mode"] = True

            elif corruption_type == "wrong_type":
                # Change argument types
                for key in list(modified.keys()):
                    modified[key] = self._swap_argument_type(modified[key])

            elif corruption_type == "wrong_value":
                # Keep types but change values
                for key in list(modified.keys()):
                    modified[key] = self._corrupt_argument_value(modified[key])

            elif corruption_type == "shuffle":
                # Shuffle values between keys
                keys = list(modified.keys())
                values = list(modified.values())
                random.shuffle(values)
                modified = dict(zip(keys, values, strict=False))

            return modified

        # Handle list arguments
        if isinstance(args, list):
            if corruption_type == "drop":
                drop_ratio = self._config.parameters.get("drop_ratio", 0.5)
                keep_count = max(1, int(len(args) * (1 - drop_ratio)))
                return random.sample(args, min(keep_count, len(args)))
            elif corruption_type == "add_extra":
                return args + ["extra_item", {"noise": True}]
            elif corruption_type == "shuffle":
                result = list(args)
                random.shuffle(result)
                return result
            else:
                return [self._corrupt_arg_value(item, corruption_type) for item in args]

        return args

    def _swap_argument_type(self, value: Any) -> Any:
        """
        Swap an argument to a different type.

        Args:
            value: Original argument value

        Returns:
            Value converted to different type
        """
        if isinstance(value, bool):
            return "true" if value else "false"  # bool -> str
        elif isinstance(value, int):
            return str(value)  # int -> str
        elif isinstance(value, float):
            return int(value)  # float -> int
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return [value]  # str -> list
        elif isinstance(value, list):
            return {"items": value}  # list -> dict
        elif isinstance(value, dict):
            return list(value.values())  # dict -> list
        elif value is None:
            return "null"
        else:
            return str(value)

    def _corrupt_argument_value(self, value: Any) -> Any:
        """
        Corrupt an argument value while preserving type.

        Args:
            value: Original argument value

        Returns:
            Corrupted value of same type
        """
        import random

        if isinstance(value, bool):
            return not value
        elif isinstance(value, int):
            return value + random.randint(-100, 100)
        elif isinstance(value, float):
            return value * random.uniform(0.5, 2.0)
        elif isinstance(value, str):
            if len(value) > 0:
                return value[::-1]  # Reverse
            return "corrupted"
        elif isinstance(value, list):
            if len(value) > 0:
                return list(reversed(value))
            return ["corrupted_item"]
        elif isinstance(value, dict):
            return {k: self._corrupt_argument_value(v) for k, v in value.items()}
        else:
            return value

    def _strategy_extra_tool_calls(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return additional spurious tool calls.

        Calls the original parser, then adds extra tool calls to the
        result to test how agents handle unexpected additional calls.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Parsed result with additional tool calls.

        Config parameters:
            extra_count (int): Number of extra calls to add (default 2).
            extra_tools (list, optional): Specific tool calls to add.
            position (str): Where to add - "before", "after", "random" (default "after").
        """
        result = self._original(*args, **kwargs)
        extra_count = self._config.parameters.get("extra_count", 2)
        position = self._config.parameters.get("position", "after")

        # Check for specific extra tools
        extra_tools = self._config.parameters.get("extra_tools")
        if extra_tools is None:
            extra_tools = self._generate_extra_tool_calls(extra_count)

        return self._add_extra_tool_calls(result, extra_tools, position)

    def _generate_extra_tool_calls(self, count: int) -> list[dict[str, Any]]:
        """
        Generate spurious extra tool calls.

        Args:
            count: Number of extra calls to generate

        Returns:
            List of fake tool call dicts
        """
        fake_tools = [
            {"name": "debug_inspect", "arguments": {"target": "all", "verbose": True}},
            {"name": "log_state", "arguments": {"level": "debug"}},
            {"name": "internal_validate", "arguments": {}},
            {"name": "cache_clear", "arguments": {"force": True}},
            {"name": "metrics_emit", "arguments": {"type": "gauge", "value": 0}},
            {"name": "trace_span", "arguments": {"name": "injected", "tags": []}},
            {"name": "noop", "arguments": {}},
            {"name": "echo", "arguments": {"message": "fault_injection_marker"}},
        ]

        import random

        return random.sample(fake_tools, min(count, len(fake_tools)))

    def _add_extra_tool_calls(self, result: Any, extra_tools: list, position: str) -> Any:
        """
        Add extra tool calls to the result.

        Args:
            result: Original parsed result
            extra_tools: Extra tool calls to add
            position: Where to add ("before", "after", "random")

        Returns:
            Result with extra tool calls added
        """
        import random

        # Handle None result
        if result is None:
            return extra_tools

        # Handle single tool call (dict)
        if isinstance(result, dict):
            if position == "before":
                return extra_tools + [result]
            elif position == "after":
                return [result] + extra_tools
            else:  # random
                combined = [result] + extra_tools
                random.shuffle(combined)
                return combined

        # Handle list of tool calls
        if isinstance(result, list):
            if position == "before":
                return extra_tools + list(result)
            elif position == "after":
                return list(result) + extra_tools
            else:  # random
                combined = list(result) + extra_tools
                random.shuffle(combined)
                return combined

        # Handle tuple
        if isinstance(result, tuple):
            list_result = self._add_extra_tool_calls(list(result), extra_tools, position)
            return tuple(list_result)

        # For other types, wrap in list with extras
        return [result] + extra_tools

    def _strategy_missing_tool_calls(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return fewer tool calls than actually parsed.

        Calls the original parser, then removes some tool calls from
        the result to test handling of incomplete parsing.

        Args:
            *args: Positional arguments to pass to original function
            **kwargs: Keyword arguments to pass to original function

        Returns:
            Parsed result with some tool calls removed.

        Config parameters:
            drop_ratio (float): Fraction of calls to drop (default 0.5).
            keep_first (bool): Always keep the first call (default True).
            keep_last (bool): Always keep the last call (default False).
            max_keep (int, optional): Maximum number of calls to keep.
        """
        result = self._original(*args, **kwargs)
        drop_ratio = self._config.parameters.get("drop_ratio", 0.5)
        keep_first = self._config.parameters.get("keep_first", True)
        keep_last = self._config.parameters.get("keep_last", False)
        max_keep = self._config.parameters.get("max_keep")

        return self._remove_tool_calls(result, drop_ratio, keep_first, keep_last, max_keep)

    def _remove_tool_calls(
        self,
        result: Any,
        drop_ratio: float,
        keep_first: bool,
        keep_last: bool,
        max_keep: Optional[int],
    ) -> Any:
        """
        Remove tool calls from the result.

        Args:
            result: Original parsed result
            drop_ratio: Fraction to drop (0.0 to 1.0)
            keep_first: Whether to always keep first call
            keep_last: Whether to always keep last call
            max_keep: Maximum number to keep (None for no limit)

        Returns:
            Result with some tool calls removed
        """
        import random

        # Handle None
        if result is None:
            return None

        # Handle single tool call
        if isinstance(result, dict):
            # For single call, return None or the call based on ratio
            if random.random() < drop_ratio and not keep_first:
                return None
            return result

        # Handle list of tool calls
        if isinstance(result, list):
            if len(result) == 0:
                return []

            if len(result) == 1:
                # Single item list
                if keep_first or keep_last:
                    return result
                return [] if random.random() < drop_ratio else result

            # Determine which indices to keep
            indices_to_keep = set()

            if keep_first:
                indices_to_keep.add(0)
            if keep_last:
                indices_to_keep.add(len(result) - 1)

            # Randomly keep some of the remaining
            remaining_indices = [i for i in range(len(result)) if i not in indices_to_keep]
            keep_count = max(0, int(len(remaining_indices) * (1 - drop_ratio)))

            if max_keep is not None:
                keep_count = min(keep_count, max(0, max_keep - len(indices_to_keep)))

            indices_to_keep.update(random.sample(remaining_indices, min(keep_count, len(remaining_indices))))

            # Build result preserving order
            return [result[i] for i in sorted(indices_to_keep)]

        # Handle tuple
        if isinstance(result, tuple):
            list_result = self._remove_tool_calls(list(result), drop_ratio, keep_first, keep_last, max_keep)
            if list_result is None:
                return None
            return tuple(list_result)

        return result
