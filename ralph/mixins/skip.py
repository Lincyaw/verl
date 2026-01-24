"""
SkipMixin provides skip and repeat operation capabilities for fault injection proxies.

This mixin enables proxies to skip operations (returning a default value) or
repeat operations multiple times, simulating various fault conditions like
dropped operations or duplicate processing.
"""
from typing import Any, Optional


class SkipMixin:
    """
    Mixin class providing skip and repeat operation capabilities.

    This mixin should be inherited by proxy classes that need to support
    the SKIP and REPEAT strategies. It provides methods to:
    - Skip an operation entirely, returning a default value
    - Repeat an operation multiple times
    - Execute the skip/repeat strategies by reading from config parameters

    Expected attributes on the inheriting class:
    - _config: FaultConfig instance with parameters dict containing 'default_return' or 'times'
    - _original: The original function being proxied
    """

    def _skip_operation(self, default_return: Optional[Any] = None) -> Any:
        """
        Skip the original operation and return a default value.

        This method does NOT call the original function. It simply returns
        the provided default value, simulating a skipped/dropped operation.

        Args:
            default_return: The value to return instead of calling the original.
                          Defaults to None.

        Returns:
            The default_return value.
        """
        return default_return

    def _repeat_operation(self, times: int, *args: Any, **kwargs: Any) -> Any:
        """
        Repeat the original operation multiple times.

        Calls self._original the specified number of times with the same arguments.
        Returns the result of the LAST call (all intermediate results are discarded).

        Args:
            times: Number of times to call the original function. Must be >= 1.
                  If times < 1, defaults to 1 call.
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            The result of the last call to self._original.
        """
        if times < 1:
            times = 1

        result = None
        for _ in range(times):
            result = self._original(*args, **kwargs)
        return result

    def _strategy_skip(self, *args: Any, **kwargs: Any) -> Any:
        """
        Skip strategy implementation.

        Reads 'default_return' from self._config.parameters (defaults to None),
        and returns that value WITHOUT calling the original function.

        Args:
            *args: Positional arguments (ignored, original function not called).
            **kwargs: Keyword arguments (ignored, original function not called).

        Returns:
            The default_return value from config, or None if not specified.

        Config Parameters:
            default_return (Any): The value to return (optional, defaults to None).
        """
        default_return = self._config.parameters.get("default_return", None)
        return self._skip_operation(default_return)

    def _strategy_repeat(self, *args: Any, **kwargs: Any) -> Any:
        """
        Repeat strategy implementation.

        Reads 'times' from self._config.parameters (defaults to 2),
        and calls the original function that many times.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            The result of the last call to self._original.

        Config Parameters:
            times (int): Number of times to repeat the operation (optional, defaults to 2).
        """
        times = self._config.parameters.get("times", 2)
        return self._repeat_operation(times, *args, **kwargs)
