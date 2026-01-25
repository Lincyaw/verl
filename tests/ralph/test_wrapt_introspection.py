"""
Unit tests for wrapt-based BaseProxy introspection and metadata preservation.

Tests cover:
- __signature__ preservation via inspect.signature()
- __name__, __doc__, __module__ forwarding and get_qualname() method
- __wrapped__ property returning original function
- __annotations__ forwarding
- help() displays correct documentation
- Introspection works for various function types (regular, methods, decorated)
- Async and generator function detection
"""

import inspect
from typing import Any, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.proxies.base import BaseProxy


class ConcreteProxy(BaseProxy):
    """Concrete proxy for testing introspection."""

    SUPPORTED_STRATEGIES: Set[StrategyType] = {StrategyType.DELAY}

    def _get_layer(self) -> str:
        return "Test"

    def _strategy_delay(self, *args, **kwargs) -> Any:
        return self._original(*args, **kwargs)


def sample_function(a: int, b: str, c: float = 3.14, *args, **kwargs) -> str:
    """Sample function with full type annotations and docstring.

    This is a sample function used for testing introspection preservation.

    Args:
        a: An integer parameter
        b: A string parameter
        c: A float with default value
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        A string result
    """
    return f"{a}-{b}-{c}"


def simple_function(x, y):
    """Simple function without annotations."""
    return x + y


def no_doc_function(a, b):
    pass


class SampleClass:
    """Sample class for method introspection tests."""

    def instance_method(self, data: List[int], name: str = "default") -> dict:
        """Instance method with annotations."""
        return {"data": data, "name": name}

    @classmethod
    def class_method(cls, value: int) -> bool:
        """Class method example."""
        return value > 0

    @staticmethod
    def static_method(items: list) -> int:
        """Static method example."""
        return len(items)


class TestSignaturePreservation:
    """Tests for __signature__ preservation."""

    def test_signature_matches_original(self):
        """Proxy signature matches original function signature."""
        proxy = ConcreteProxy(sample_function)
        original_sig = inspect.signature(sample_function)
        proxy_sig = inspect.signature(proxy)
        assert str(proxy_sig) == str(original_sig)

    def test_signature_parameter_names(self):
        """Proxy preserves parameter names."""
        proxy = ConcreteProxy(sample_function)
        sig = inspect.signature(proxy)
        param_names = list(sig.parameters.keys())
        assert param_names == ["a", "b", "c", "args", "kwargs"]

    def test_signature_parameter_annotations(self):
        """Proxy preserves parameter annotations."""
        proxy = ConcreteProxy(sample_function)
        sig = inspect.signature(proxy)
        assert sig.parameters["a"].annotation == int
        assert sig.parameters["b"].annotation == str
        assert sig.parameters["c"].annotation == float

    def test_signature_return_annotation(self):
        """Proxy preserves return annotation."""
        proxy = ConcreteProxy(sample_function)
        sig = inspect.signature(proxy)
        assert sig.return_annotation == str

    def test_signature_default_values(self):
        """Proxy preserves default values."""
        proxy = ConcreteProxy(sample_function)
        sig = inspect.signature(proxy)
        assert sig.parameters["c"].default == 3.14

    def test_signature_simple_function(self):
        """Proxy works with simple function without annotations."""
        proxy = ConcreteProxy(simple_function)
        sig = inspect.signature(proxy)
        param_names = list(sig.parameters.keys())
        assert param_names == ["x", "y"]

    def test_signature_property_access(self):
        """__signature__ property returns Signature object."""
        proxy = ConcreteProxy(sample_function)
        sig = proxy.__signature__
        assert isinstance(sig, inspect.Signature)
        assert str(sig) == "(a: int, b: str, c: float = 3.14, *args, **kwargs) -> str"


class TestNamePreservation:
    """Tests for __name__ preservation."""

    def test_name_matches_original(self):
        """Proxy __name__ matches original function name."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__name__ == "sample_function"

    def test_name_simple_function(self):
        """Proxy __name__ works for simple functions."""
        proxy = ConcreteProxy(simple_function)
        assert proxy.__name__ == "simple_function"

    def test_name_lambda(self):
        """Proxy __name__ works for lambda functions."""
        fn = lambda x: x * 2
        proxy = ConcreteProxy(fn)
        assert proxy.__name__ == "<lambda>"

    def test_name_instance_method(self):
        """Proxy __name__ works for instance methods."""
        obj = SampleClass()
        proxy = ConcreteProxy(obj.instance_method)
        assert proxy.__name__ == "instance_method"


class TestDocPreservation:
    """Tests for __doc__ preservation."""

    def test_doc_matches_original(self):
        """Proxy __doc__ matches original function docstring."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__doc__ == sample_function.__doc__
        assert "Sample function" in proxy.__doc__

    def test_doc_includes_full_content(self):
        """Proxy __doc__ includes full docstring content."""
        proxy = ConcreteProxy(sample_function)
        assert "Args:" in proxy.__doc__
        assert "Returns:" in proxy.__doc__

    def test_doc_simple_function(self):
        """Proxy __doc__ works for simple docstrings."""
        proxy = ConcreteProxy(simple_function)
        assert proxy.__doc__ == "Simple function without annotations."

    def test_doc_none_when_no_docstring(self):
        """Proxy __doc__ is None when original has no docstring."""
        proxy = ConcreteProxy(no_doc_function)
        assert proxy.__doc__ is None


class TestModuleAndQualname:
    """Tests for __module__ and get_qualname() preservation."""

    def test_module_matches_original(self):
        """Proxy __module__ matches original function module."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__module__ == sample_function.__module__

    def test_qualname_matches_original(self):
        """Proxy get_qualname() matches original function qualname."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.get_qualname() == sample_function.__qualname__

    def test_qualname_method(self):
        """Proxy get_qualname() works for methods."""
        obj = SampleClass()
        proxy = ConcreteProxy(obj.instance_method)
        assert "instance_method" in proxy.get_qualname()


class TestWrappedProperty:
    """Tests for __wrapped__ property."""

    def test_wrapped_returns_original(self):
        """__wrapped__ returns the original function."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__wrapped__ is sample_function

    def test_wrapped_same_as_get_original(self):
        """__wrapped__ returns same as get_original()."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__wrapped__ is proxy.get_original()


class TestAnnotationsPreservation:
    """Tests for __annotations__ preservation."""

    def test_annotations_match_original(self):
        """Proxy __annotations__ matches original function annotations."""
        proxy = ConcreteProxy(sample_function)
        assert proxy.__annotations__ == sample_function.__annotations__

    def test_annotations_include_return(self):
        """Annotations include return type."""
        proxy = ConcreteProxy(sample_function)
        assert "return" in proxy.__annotations__
        assert proxy.__annotations__["return"] == str

    def test_annotations_empty_for_unannotated(self):
        """Annotations dict empty for unannotated function."""
        proxy = ConcreteProxy(simple_function)
        # simple_function has no annotations
        assert proxy.__annotations__ == simple_function.__annotations__


class TestAsyncGeneratorDetection:
    """Tests for async and generator function detection."""

    def test_is_async_function_sync(self):
        """_is_async_function returns False for sync functions."""
        assert BaseProxy._is_async_function(sample_function) is False

    def test_is_async_function_async(self):
        """_is_async_function returns True for async functions."""

        async def async_fn():
            pass

        assert BaseProxy._is_async_function(async_fn) is True

    def test_is_generator_function_regular(self):
        """_is_generator_function returns False for regular functions."""
        assert BaseProxy._is_generator_function(sample_function) is False

    def test_is_generator_function_generator(self):
        """_is_generator_function returns True for generator functions."""

        def gen_fn():
            yield 1
            yield 2

        assert BaseProxy._is_generator_function(gen_fn) is True

    def test_async_function_warning_logged(self, caplog):
        """Warning is logged when wrapping async function."""
        import logging

        async def async_fn():
            pass

        caplog.set_level(logging.WARNING)
        proxy = ConcreteProxy(async_fn)
        assert any("async function" in record.message.lower() for record in caplog.records)

    def test_generator_function_warning_logged(self, caplog):
        """Warning is logged when wrapping generator function."""
        import logging

        def gen_fn():
            yield 1

        caplog.set_level(logging.WARNING)
        proxy = ConcreteProxy(gen_fn)
        assert any("generator function" in record.message.lower() for record in caplog.records)


class TestMethodIntrospection:
    """Tests for method introspection."""

    def test_instance_method_signature(self):
        """Instance method signature preserved (without self)."""
        obj = SampleClass()
        proxy = ConcreteProxy(obj.instance_method)
        sig = inspect.signature(proxy)
        # Bound method signature doesn't include 'self'
        param_names = list(sig.parameters.keys())
        assert "data" in param_names
        assert "name" in param_names

    def test_instance_method_doc(self):
        """Instance method docstring preserved."""
        obj = SampleClass()
        proxy = ConcreteProxy(obj.instance_method)
        assert proxy.__doc__ == "Instance method with annotations."

    def test_class_method_introspection(self):
        """Class method introspection works."""
        proxy = ConcreteProxy(SampleClass.class_method)
        assert proxy.__name__ == "class_method"
        assert "Class method" in proxy.__doc__

    def test_static_method_introspection(self):
        """Static method introspection works."""
        proxy = ConcreteProxy(SampleClass.static_method)
        assert proxy.__name__ == "static_method"


class TestDecoratedFunctions:
    """Tests for decorated function introspection."""

    def test_functools_wrapped_function(self):
        """Introspection works for functools.wraps decorated functions."""
        import functools

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @decorator
        def decorated_func(a: int, b: str) -> bool:
            """Decorated function docstring."""
            return True

        proxy = ConcreteProxy(decorated_func)
        assert proxy.__name__ == "decorated_func"
        assert proxy.__doc__ == "Decorated function docstring."

    def test_multiple_decorators(self):
        """Introspection works with multiple decorators stacked."""
        import functools

        def deco1(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        def deco2(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        @deco1
        @deco2
        def multi_decorated(x: int) -> int:
            """Multi-decorated function."""
            return x * 2

        proxy = ConcreteProxy(multi_decorated)
        assert proxy.__name__ == "multi_decorated"
        assert proxy.__doc__ == "Multi-decorated function."


class TestIntrospectionWithInjection:
    """Tests that introspection works correctly during injection."""

    def test_signature_after_config(self):
        """Signature preserved after setting config."""
        proxy = ConcreteProxy(sample_function)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        sig = inspect.signature(proxy)
        assert "a" in sig.parameters

    def test_introspection_during_injection(self):
        """Introspection properties work during active injection."""
        proxy = ConcreteProxy(sample_function)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Trigger injection
        result = proxy(1, "hello", 2.5)

        # Introspection should still work
        assert proxy.__name__ == "sample_function"
        assert proxy.__doc__ is not None
        sig = inspect.signature(proxy)
        assert len(sig.parameters) == 5


class TestCallableWithoutStandardAttributes:
    """Tests for callables without standard function attributes."""

    def test_callable_class_instance(self):
        """Proxy works with callable class instance."""

        class CallableClass:
            def __call__(self, x: int) -> int:
                return x * 2

        obj = CallableClass()
        proxy = ConcreteProxy(obj)

        # Should still be callable
        assert proxy(5) == 10

        # Name might be empty or class name
        # Just ensure it doesn't crash
        _ = proxy.__name__
        _ = proxy.__doc__

    def test_builtin_function(self):
        """Proxy works with builtin functions."""
        proxy = ConcreteProxy(len)
        assert proxy.__name__ == "len"
        assert proxy.__doc__ is not None
        assert "Return the number of items" in proxy.__doc__
