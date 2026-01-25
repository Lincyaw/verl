"""
Base proxy class for Ralph fault injection framework.

Contains the abstract BaseProxy class that all proxy implementations inherit from.
Uses wrapt library for signature-preserving function wrapping.
"""

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

import wrapt

from ralph.core.config import FaultConfig, StrategyType

if TYPE_CHECKING:
    from ralph.collectors.dual_stream import DualStreamCollector


class BaseProxy(ABC):
    """
    Abstract base class for all fault injection proxies.

    This class composes wrapt.FunctionWrapper internally to provide:
    - Signature preservation for IDE introspection
    - Proper metadata forwarding (__name__, __doc__, __module__, etc.)
    - Transparent wrapping that behaves like the original function

    Subclasses must:
    1. Inherit required Mixin classes for strategy implementations
    2. Implement the abstract _get_layer() method
    3. Declare SUPPORTED_STRATEGIES class attribute
    4. Implement specific strategy methods (_strategy_xxx) for each supported strategy

    Example:
        @ProxyRegistry.register('ray.get')
        class RayGetProxy(BaseProxy, DelayMixin, ExceptionMixin):
            SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.RAISE_EXCEPTION}

            def _get_layer(self) -> str:
                return 'L0'

    Introspection:
        proxy = RayGetProxy(ray.get, collector)
        inspect.signature(proxy)  # Returns original function signature
        proxy.__name__  # Returns 'get'
        proxy.__doc__   # Returns original docstring
    """

    # Subclasses must declare which strategies they support
    SUPPORTED_STRATEGIES: set[StrategyType] = set()

    def __init__(
        self,
        original_fn: Callable,
        collector: Optional["DualStreamCollector"] = None,
    ):
        """
        Initialize the proxy.

        Args:
            original_fn: The original function being proxied
            collector: Optional DualStreamCollector for recording fault injections.
                      If None, injection recording is disabled.
        """
        self._original = original_fn
        self._collector = collector
        self._config: Optional[FaultConfig] = None
        self._current_step = 0
        self._rng = None  # Random number generator for probabilistic triggers

        # Build strategy method mapping
        self._strategy_methods: dict[StrategyType, Callable] = {}
        self._build_strategy_map()

        # Create the wrapt wrapper that delegates to our _invoke method
        self._wrapper = self._create_wrapper(original_fn)

        # Log warning for async/generator functions (not fully supported)
        self._check_function_type(original_fn)

    def _create_wrapper(self, original_fn: Callable) -> Callable:
        """
        Create a wrapt-based wrapper around the original function.

        Args:
            original_fn: The original function to wrap

        Returns:
            A wrapped function that preserves signature and metadata
        """
        # Capture self in closure for the decorator
        proxy_self = self

        @wrapt.decorator
        def _wrapper(wrapped, instance, args, kwargs):
            return proxy_self._invoke(wrapped, instance, args, kwargs)

        return _wrapper(original_fn)

    def _check_function_type(self, fn: Callable) -> None:
        """
        Check if function is async or generator and log warning.

        Args:
            fn: The function to check
        """
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        if asyncio.iscoroutinefunction(fn):
            logger.warning(
                f"Wrapping async function {getattr(fn, '__name__', fn)} - "
                "async functions are not fully supported by Ralph proxies. "
                "Fault injection may not work correctly."
            )
        elif inspect.isgeneratorfunction(fn):
            logger.warning(
                f"Wrapping generator function {getattr(fn, '__name__', fn)} - "
                "generator functions are not fully supported by Ralph proxies. "
                "Fault injection may not work correctly."
            )

    def _build_strategy_map(self) -> None:
        """
        Build mapping from StrategyType to strategy method.

        Looks for methods named _strategy_{strategy.value} for each strategy
        in SUPPORTED_STRATEGIES. Methods can be provided by the class itself
        or by inherited Mixin classes.
        """
        for strategy in self.SUPPORTED_STRATEGIES:
            method_name = f"_strategy_{strategy.value}"
            if hasattr(self, method_name):
                self._strategy_methods[strategy] = getattr(self, method_name)

    def set_config(self, config: FaultConfig) -> None:
        """
        Set the fault configuration for this proxy.

        Args:
            config: The fault configuration to use

        Raises:
            ValueError: If the strategy in config is not in SUPPORTED_STRATEGIES
        """
        if config.strategy not in self.SUPPORTED_STRATEGIES:
            supported = ", ".join(s.value for s in sorted(self.SUPPORTED_STRATEGIES, key=lambda x: x.value))
            raise ValueError(
                f"Strategy '{config.strategy.value}' not supported by {self.__class__.__name__}. "
                f"Supported strategies: [{supported}]"
            )
        self._config = config

    def set_step(self, step: int) -> None:
        """
        Set the current training step.

        Args:
            step: The current training step number
        """
        self._current_step = step

    def set_rng(self, rng) -> None:
        """
        Set the random number generator for probabilistic triggers.

        Args:
            rng: Random number generator (should have a .random() method)
        """
        self._rng = rng

    def __call__(self, *args, **kwargs) -> Any:
        """
        Transparent proxy call entry point.

        Delegates to the internal wrapt wrapper which calls _invoke.
        """
        return self._wrapper(*args, **kwargs)

    def _invoke(
        self,
        wrapped: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Core invocation logic called by the wrapt wrapper.

        This method implements the fault injection logic:
        - Checks if fault should be injected
        - Either executes the fault strategy or the original function

        Args:
            wrapped: The wrapped (original) function
            instance: The instance for bound methods, or None for functions
            args: Positional arguments tuple
            kwargs: Keyword arguments dict

        Returns:
            Result from either the strategy method or original function
        """
        if self._should_inject():
            fault_id = self._record_injection_start()
            try:
                result = self._execute_strategy(*args, **kwargs)
                self._record_injection_end(fault_id, "success")
                return result
            except Exception as e:
                self._record_injection_end(fault_id, f"exception: {type(e).__name__}: {e}")
                raise

        # Normal execution - call the wrapped function
        return wrapped(*args, **kwargs)

    def _should_inject(self) -> bool:
        """
        Check if fault should be injected at current step.

        Returns:
            True if fault should be triggered, False otherwise
        """
        if not self._config or not self._config.enabled:
            return False
        return self._config.trigger.should_trigger(self._current_step, self._rng)

    def _execute_strategy(self, *args, **kwargs) -> Any:
        """
        Execute the configured fault injection strategy.

        Args:
            *args: Positional arguments to pass to strategy method
            **kwargs: Keyword arguments to pass to strategy method

        Returns:
            Result from the strategy method

        Raises:
            NotImplementedError: If strategy is declared but method not found
        """
        strategy = self._config.strategy

        if strategy in self._strategy_methods:
            return self._strategy_methods[strategy](*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Strategy '{strategy.value}' declared in SUPPORTED_STRATEGIES "
                f"but _strategy_{strategy.value}() not implemented in {self.__class__.__name__}"
            )

    def _record_injection_start(self) -> str:
        """
        Record the start of a fault injection.

        Returns:
            Unique fault_id for this injection, or empty string if no collector
        """
        if not self._collector:
            return ""

        return self._collector.record_fault_injection(
            fault_type=self._config.strategy.value,
            target_layer=self._get_layer(),
            target_function=self._get_target_name(),
            severity=self._config.severity,
            parameters=self._config.parameters,
            expected_behavior=self._config.expected_behavior,
        )

    def _record_injection_end(self, fault_id: str, outcome: str, duration_ms: float = 0) -> None:
        """
        Record the end of a fault injection.

        Args:
            fault_id: The unique ID returned by _record_injection_start()
            outcome: Description of the outcome (e.g., "success", "exception: ...")
            duration_ms: Duration of the injection in milliseconds
        """
        if not self._collector or not fault_id:
            return

        self._collector.record_fault_outcome(fault_id, outcome, duration_ms)

    @abstractmethod
    def _get_layer(self) -> str:
        """
        Return the layer this proxy belongs to.

        Returns:
            Layer identifier string (e.g., 'L0', 'L1', 'L2', 'L3', 'Algorithm')
        """
        raise NotImplementedError

    def _get_target_name(self) -> str:
        """
        Return the name of the target function being proxied.

        Returns:
            The function name or string representation
        """
        return getattr(self._original, "__name__", str(self._original))

    def get_original(self) -> Callable:
        """
        Get the original function being proxied.

        Returns:
            The original function
        """
        return self._original

    def get_config(self) -> Optional[FaultConfig]:
        """
        Get the current fault configuration.

        Returns:
            The current FaultConfig or None if not set
        """
        return self._config

    def get_step(self) -> int:
        """
        Get the current training step.

        Returns:
            The current step number
        """
        return self._current_step

    # =========================================================================
    # Special attribute forwarding via __getattribute__
    # =========================================================================

    # Attributes to forward to the wrapped function (bypass class docstring, etc.)
    _FORWARD_TO_WRAPPED = frozenset({"__doc__", "__annotations__"})

    def __getattribute__(self, name: str) -> Any:
        """
        Custom attribute access to forward special attributes to wrapped function.

        For __doc__ and __annotations__, Python's default lookup would return
        the class's values instead of the instance property. We intercept these
        to return the wrapped function's values.
        """
        # Check if this is one of the special forwarded attributes
        if name in object.__getattribute__(self, "_FORWARD_TO_WRAPPED"):
            # Get _original without triggering recursion
            original = object.__getattribute__(self, "_original")
            return getattr(original, name, None if name == "__doc__" else {})

        # Default attribute lookup
        return object.__getattribute__(self, name)

    # =========================================================================
    # Introspection properties - delegate to wrapped function for transparency
    # =========================================================================

    @property
    def __wrapped__(self) -> Callable:
        """Return the original wrapped function."""
        return self._original

    @property
    def __name__(self) -> str:
        """Return the original function name."""
        return getattr(self._original, "__name__", "")

    @property
    def __module__(self) -> str:
        """Return the original function module."""
        return getattr(self._original, "__module__", "")

    def get_qualname(self) -> str:
        """Return the original function qualified name.

        Note: __qualname__ cannot be a property on ABC subclasses because
        the metaclass expects it to be a string during class creation.
        Use this method instead, or access original.__qualname__ directly.
        """
        return getattr(self._original, "__qualname__", "")

    @property
    def __signature__(self) -> inspect.Signature:
        """Return the original function signature.

        This is the key property that enables IDE introspection.
        """
        return inspect.signature(self._original)

    # =========================================================================
    # Async/Generator detection helpers
    # =========================================================================

    @staticmethod
    def _is_async_function(fn: Callable) -> bool:
        """
        Check if a function is an async coroutine function.

        Args:
            fn: The function to check

        Returns:
            True if fn is an async function, False otherwise
        """
        import asyncio

        return asyncio.iscoroutinefunction(fn)

    @staticmethod
    def _is_generator_function(fn: Callable) -> bool:
        """
        Check if a function is a generator function.

        Args:
            fn: The function to check

        Returns:
            True if fn is a generator function, False otherwise
        """
        return inspect.isgeneratorfunction(fn)
