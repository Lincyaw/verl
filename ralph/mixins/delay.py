"""
DelayMixin provides delay injection capabilities for fault injection proxies.

This mixin enables proxies to introduce artificial delays before executing
the original function, simulating network latency, slow operations, etc.
"""

import time
from typing import Any


class DelayMixin:
    """
    Mixin class providing delay injection capabilities.

    This mixin should be inherited by proxy classes that need to support
    the DELAY strategy. It provides methods to:
    - Apply a delay of specified duration
    - Execute the delay strategy by reading from config parameters

    Expected attributes on the inheriting class:
    - _config: FaultConfig instance with parameters dict containing 'delay_seconds'
    - _original: The original function being proxied
    """

    def _apply_delay(self, seconds: float) -> None:
        """
        Apply a delay by sleeping for the specified duration.

        Args:
            seconds: Number of seconds to sleep. Can be fractional.
        """
        if seconds > 0:
            time.sleep(seconds)

    def _strategy_delay(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delay strategy implementation.

        Reads 'delay_seconds' from self._config.parameters (defaults to 10 seconds),
        applies the delay, then executes and returns the result of the original function.

        Args:
            *args: Positional arguments to pass to the original function.
            **kwargs: Keyword arguments to pass to the original function.

        Returns:
            The result of calling self._original with the provided arguments.
        """
        delay_seconds = self._config.parameters.get("delay_seconds", 10.0)
        self._apply_delay(delay_seconds)
        return self._original(*args, **kwargs)
