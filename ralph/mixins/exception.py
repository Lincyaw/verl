"""
ExceptionMixin provides exception raising capabilities for fault injection proxies.

This mixin enables proxies to raise configurable exceptions, simulating various
error conditions like IOError, RuntimeError, TimeoutError, etc.
"""
from typing import Any

# Map of exception type names to their corresponding Python exception classes
EXCEPTION_MAP: dict[str, type[Exception]] = {
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "PermissionError": PermissionError,
    "TimeoutError": TimeoutError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "AttributeError": AttributeError,
    "ConnectionError": ConnectionError,
    "MemoryError": MemoryError,
}


class ExceptionMixin:
    """
    Mixin class providing exception raising capabilities.

    This mixin should be inherited by proxy classes that need to support
    the RAISE_EXCEPTION strategy. It provides methods to:
    - Raise a specified exception type with a custom message
    - Execute the raise_exception strategy by reading from config parameters

    Expected attributes on the inheriting class:
    - _config: FaultConfig instance with parameters dict containing 'exc_type' and 'message'
    - _original: The original function being proxied (not used by this mixin but expected)
    """

    def _raise_exception(
        self,
        exc_type: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """
        Raise an exception of the specified type with the given message.

        Args:
            exc_type: String name of the exception type (e.g., "IOError", "RuntimeError").
                      Must be a key in EXCEPTION_MAP.
            message: The error message to include in the exception.
            **kwargs: Additional keyword arguments (reserved for future use).

        Raises:
            The specified exception type with the given message.
            ValueError: If exc_type is not found in EXCEPTION_MAP.
        """
        exception_class = EXCEPTION_MAP.get(exc_type)
        if exception_class is None:
            raise ValueError(
                f"Unknown exception type: {exc_type}. "
                f"Available types: {list(EXCEPTION_MAP.keys())}"
            )
        raise exception_class(message)

    def _strategy_raise_exception(self, *args: Any, **kwargs: Any) -> Any:
        """
        Raise exception strategy implementation.

        Reads 'exc_type' and 'message' from self._config.parameters,
        then raises the specified exception. The original function is never called.

        Args:
            *args: Positional arguments (ignored, original function not called).
            **kwargs: Keyword arguments (ignored, original function not called).

        Raises:
            The exception type specified in config with the configured message.
            ValueError: If exc_type is not specified or not found in EXCEPTION_MAP.

        Config Parameters:
            exc_type (str): The exception type name (required). Must be a key in EXCEPTION_MAP.
            message (str): The error message (optional, defaults to "Fault injection: {exc_type}").
        """
        exc_type = self._config.parameters.get("exc_type")
        if exc_type is None:
            raise ValueError("exc_type must be specified in config parameters")

        default_message = f"Fault injection: {exc_type}"
        message = self._config.parameters.get("message", default_message)

        self._raise_exception(exc_type, message)
