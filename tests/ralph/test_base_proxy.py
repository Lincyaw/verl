"""
Unit tests for BaseProxy abstract class.

Tests cover:
- BaseProxy cannot be instantiated directly (abstract)
- SUPPORTED_STRATEGIES attribute is required
- __init__ stores original_fn and collector
- _build_strategy_map creates mapping from strategies to methods
- set_config validates strategy is supported
- set_step updates current step
- __call__ checks _should_inject and dispatches correctly
- _should_inject checks config.enabled and trigger.should_trigger
- _execute_strategy looks up and calls strategy method
- _record_injection_start and _record_injection_end call collector
- _get_layer is abstract and must be implemented
"""

from typing import Any, Set
from unittest.mock import MagicMock, patch

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.proxies.base import BaseProxy


class ConcreteProxy(BaseProxy):
    """Concrete proxy for testing BaseProxy functionality."""

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
    }

    def _get_layer(self) -> str:
        return "L0"

    def _strategy_delay(self, *args, **kwargs) -> Any:
        """Mock delay strategy that calls original."""
        return self._original(*args, **kwargs)

    def _strategy_raise_exception(self, *args, **kwargs) -> Any:
        """Mock exception strategy that raises RuntimeError."""
        raise RuntimeError("Injected fault")


class NoStrategyProxy(BaseProxy):
    """Proxy with no supported strategies for testing."""

    SUPPORTED_STRATEGIES: Set[StrategyType] = set()

    def _get_layer(self) -> str:
        return "L0"


class MissingMethodProxy(BaseProxy):
    """Proxy that declares strategies but doesn't implement methods."""

    SUPPORTED_STRATEGIES: Set[StrategyType] = {StrategyType.SKIP}

    def _get_layer(self) -> str:
        return "L0"


class TestBaseProxyAbstract:
    """Tests for BaseProxy abstract nature."""

    def test_cannot_instantiate_base_proxy(self):
        """BaseProxy cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseProxy(lambda: None)  # type: ignore
        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(exc_info.value).lower()

    def test_subclass_must_implement_get_layer(self):
        """Subclass without _get_layer cannot be instantiated."""

        class IncompleteProxy(BaseProxy):
            SUPPORTED_STRATEGIES: Set[StrategyType] = set()
            # Missing _get_layer implementation

        with pytest.raises(TypeError) as exc_info:
            IncompleteProxy(lambda: None)
        assert "_get_layer" in str(exc_info.value)


class TestBaseProxyInit:
    """Tests for __init__ method."""

    def test_stores_original_fn(self):
        """__init__ stores original function reference."""
        original = lambda x: x * 2
        proxy = ConcreteProxy(original)
        assert proxy._original is original

    def test_stores_collector(self):
        """__init__ stores collector reference."""
        collector = MagicMock()
        proxy = ConcreteProxy(lambda: None, collector)
        assert proxy._collector is collector

    def test_collector_optional(self):
        """__init__ works without collector."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._collector is None

    def test_config_starts_none(self):
        """__init__ sets _config to None."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._config is None

    def test_step_starts_zero(self):
        """__init__ sets _current_step to 0."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._current_step == 0


class TestBuildStrategyMap:
    """Tests for _build_strategy_map method."""

    def test_builds_map_for_supported_strategies(self):
        """_build_strategy_map creates entries for all supported strategies."""
        proxy = ConcreteProxy(lambda: None)
        assert StrategyType.DELAY in proxy._strategy_methods
        assert StrategyType.RAISE_EXCEPTION in proxy._strategy_methods

    def test_maps_to_correct_methods(self):
        """Strategy map points to correct methods."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._strategy_methods[StrategyType.DELAY] == proxy._strategy_delay
        assert proxy._strategy_methods[StrategyType.RAISE_EXCEPTION] == proxy._strategy_raise_exception

    def test_empty_map_for_no_strategies(self):
        """_build_strategy_map creates empty map when no strategies."""
        proxy = NoStrategyProxy(lambda: None)
        assert len(proxy._strategy_methods) == 0

    def test_missing_method_not_in_map(self):
        """Strategy with missing method is not added to map."""
        proxy = MissingMethodProxy(lambda: None)
        assert StrategyType.SKIP not in proxy._strategy_methods


class TestSetConfig:
    """Tests for set_config method."""

    def test_accepts_supported_strategy(self):
        """set_config accepts config with supported strategy."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        proxy.set_config(config)
        assert proxy._config is config

    def test_rejects_unsupported_strategy(self):
        """set_config raises ValueError for unsupported strategy."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.SKIP,  # Not in SUPPORTED_STRATEGIES
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "skip" in str(exc_info.value).lower()
        assert "not supported" in str(exc_info.value).lower()


class TestSetStep:
    """Tests for set_step method."""

    def test_updates_current_step(self):
        """set_step updates _current_step."""
        proxy = ConcreteProxy(lambda: None)
        proxy.set_step(100)
        assert proxy._current_step == 100

    def test_can_set_step_multiple_times(self):
        """set_step can be called multiple times."""
        proxy = ConcreteProxy(lambda: None)
        proxy.set_step(10)
        assert proxy._current_step == 10
        proxy.set_step(20)
        assert proxy._current_step == 20


class TestSetRng:
    """Tests for set_rng method."""

    def test_stores_rng(self):
        """set_rng stores the RNG."""
        proxy = ConcreteProxy(lambda: None)
        rng = MagicMock()
        proxy.set_rng(rng)
        assert proxy._rng is rng


class TestShouldInject:
    """Tests for _should_inject method."""

    def test_returns_false_without_config(self):
        """_should_inject returns False when no config set."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._should_inject() is False

    def test_returns_false_when_disabled(self):
        """_should_inject returns False when config.enabled is False."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(5)
        assert proxy._should_inject() is False

    def test_returns_true_when_trigger_matches(self):
        """_should_inject returns True when trigger should fire."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(5)
        assert proxy._should_inject() is True

    def test_returns_false_when_trigger_no_match(self):
        """_should_inject returns False when trigger should not fire."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(10)  # Different step
        assert proxy._should_inject() is False


class TestExecuteStrategy:
    """Tests for _execute_strategy method."""

    def test_calls_correct_strategy_method(self):
        """_execute_strategy calls the strategy method from map."""
        original = MagicMock(return_value="result")
        proxy = ConcreteProxy(original)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)

        result = proxy._execute_strategy(1, 2, key="value")

        original.assert_called_once_with(1, 2, key="value")
        assert result == "result"

    def test_raises_when_method_not_found(self):
        """_execute_strategy raises NotImplementedError when method missing."""
        proxy = MissingMethodProxy(lambda: None)
        # Force a config with SKIP even though set_config would reject it
        proxy._config = FaultConfig(
            id="test-1",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with pytest.raises(NotImplementedError) as exc_info:
            proxy._execute_strategy()
        assert "skip" in str(exc_info.value).lower()
        assert "_strategy_skip" in str(exc_info.value)


class TestCall:
    """Tests for __call__ method."""

    def test_calls_original_when_not_injecting(self):
        """__call__ calls original function when not injecting."""
        original = MagicMock(return_value="original_result")
        proxy = ConcreteProxy(original)
        # No config set, so _should_inject returns False

        result = proxy(1, 2, key="value")

        original.assert_called_once_with(1, 2, key="value")
        assert result == "original_result"

    def test_calls_strategy_when_injecting(self):
        """__call__ calls strategy method when injecting."""
        original = MagicMock(return_value="strategy_result")
        proxy = ConcreteProxy(original)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy(1, 2, key="value")

        original.assert_called_once_with(1, 2, key="value")
        assert result == "strategy_result"

    def test_propagates_exception_from_strategy(self):
        """__call__ propagates exceptions from strategy methods."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError) as exc_info:
            proxy()
        assert "Injected fault" in str(exc_info.value)


class TestRecordInjection:
    """Tests for _record_injection_start and _record_injection_end."""

    def test_record_start_calls_collector(self):
        """_record_injection_start calls collector.record_fault_injection."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"
        proxy = ConcreteProxy(lambda: None, collector)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
            severity="high",
            parameters={"delay_seconds": 5.0},
            expected_behavior="Should delay",
        )
        proxy.set_config(config)

        fault_id = proxy._record_injection_start()

        assert fault_id == "fault-123"
        collector.record_fault_injection.assert_called_once_with(
            fault_type="delay",
            target_layer="L0",
            target_function=proxy._get_target_name(),
            severity="high",
            parameters={"delay_seconds": 5.0},
            expected_behavior="Should delay",
        )

    def test_record_start_returns_empty_without_collector(self):
        """_record_injection_start returns empty string without collector."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)

        fault_id = proxy._record_injection_start()

        assert fault_id == ""

    def test_record_end_calls_collector(self):
        """_record_injection_end calls collector.record_fault_outcome."""
        collector = MagicMock()
        proxy = ConcreteProxy(lambda: None, collector)

        proxy._record_injection_end("fault-123", "success", 100.0)

        collector.record_fault_outcome.assert_called_once_with("fault-123", "success", 100.0)

    def test_record_end_does_nothing_without_collector(self):
        """_record_injection_end does nothing without collector."""
        proxy = ConcreteProxy(lambda: None)
        # Should not raise
        proxy._record_injection_end("fault-123", "success", 100.0)

    def test_record_end_does_nothing_with_empty_fault_id(self):
        """_record_injection_end does nothing with empty fault_id."""
        collector = MagicMock()
        proxy = ConcreteProxy(lambda: None, collector)

        proxy._record_injection_end("", "success", 100.0)

        collector.record_fault_outcome.assert_not_called()


class TestGetLayer:
    """Tests for _get_layer abstract method."""

    def test_returns_layer(self):
        """_get_layer returns the layer string."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._get_layer() == "L0"


class TestGetTargetName:
    """Tests for _get_target_name method."""

    def test_returns_function_name(self):
        """_get_target_name returns __name__ of function."""

        def my_function():
            pass

        proxy = ConcreteProxy(my_function)
        assert proxy._get_target_name() == "my_function"

    def test_returns_lambda_name(self):
        """_get_target_name returns '<lambda>' for lambdas."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy._get_target_name() == "<lambda>"

    def test_returns_str_for_callable_without_name(self):
        """_get_target_name returns str() for callables without __name__."""

        class CallableClass:
            def __call__(self):
                pass

        proxy = ConcreteProxy(CallableClass())
        # Should be string representation
        assert "CallableClass" in proxy._get_target_name()


class TestHelperMethods:
    """Tests for helper/getter methods."""

    def test_get_original(self):
        """get_original returns the original function."""
        original = lambda: None
        proxy = ConcreteProxy(original)
        assert proxy.get_original() is original

    def test_get_config_none_initially(self):
        """get_config returns None before set_config called."""
        proxy = ConcreteProxy(lambda: None)
        assert proxy.get_config() is None

    def test_get_config_returns_config(self):
        """get_config returns the set config."""
        proxy = ConcreteProxy(lambda: None)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        assert proxy.get_config() is config

    def test_get_step(self):
        """get_step returns current step."""
        proxy = ConcreteProxy(lambda: None)
        proxy.set_step(42)
        assert proxy.get_step() == 42


class TestIntegration:
    """Integration tests for BaseProxy."""

    def test_full_injection_flow_with_collector(self):
        """Test complete injection flow with collector."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-abc"

        original = MagicMock(return_value="result")
        proxy = ConcreteProxy(original, collector)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            enabled=True,
        )
        proxy.set_config(config)
        proxy.set_step(5)

        result = proxy(1, 2, key="value")

        # Verify injection was recorded
        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once_with("fault-abc", "success", 0)

        # Verify result
        assert result == "result"
        original.assert_called_once_with(1, 2, key="value")

    def test_full_injection_flow_exception_recorded(self):
        """Test exception outcome is recorded."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-xyz"

        proxy = ConcreteProxy(lambda: None, collector)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with pytest.raises(RuntimeError):
            proxy()

        # Verify exception outcome recorded
        collector.record_fault_outcome.assert_called_once()
        args = collector.record_fault_outcome.call_args[0]
        assert args[0] == "fault-xyz"
        assert "exception" in args[1].lower()
        assert "RuntimeError" in args[1]

    def test_multiple_calls_with_step_progression(self):
        """Test multiple calls as step progresses."""
        original = MagicMock(return_value="original")
        proxy = ConcreteProxy(original)
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=3),
        )
        proxy.set_config(config)

        # Step 0, 1, 2 - no injection
        for step in range(3):
            proxy.set_step(step)
            original.reset_mock()
            proxy()
            original.assert_called_once()

        # Step 3 - injection triggers
        proxy.set_step(3)
        original.reset_mock()
        proxy()
        original.assert_called_once()  # DELAY calls original

        # Step 4+ - no more injection (one-shot)
        proxy.set_step(4)
        original.reset_mock()
        proxy()
        original.assert_called_once()
