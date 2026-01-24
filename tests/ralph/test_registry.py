"""
Unit tests for ProxyRegistry.

Tests the registration, lookup, and management functionality
of the ProxyRegistry class from ralph.core.registry.
"""

import pytest

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry


class MockProxy:
    """Mock proxy class for testing."""

    SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.SKIP}


class MockProxy2:
    """Another mock proxy class for testing."""

    SUPPORTED_STRATEGIES = {StrategyType.CORRUPT_TENSOR}


class MockProxyNoStrategies:
    """Mock proxy with no SUPPORTED_STRATEGIES attribute."""

    pass


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    # Store original registrations
    original = ProxyRegistry.get_all_registrations().copy()

    # Clear for test
    ProxyRegistry.clear()

    yield

    # Restore original registrations
    ProxyRegistry.clear()
    for target, proxy_class in original.items():
        ProxyRegistry._registry[target] = proxy_class


class TestProxyRegistryRegister:
    """Tests for ProxyRegistry.register decorator."""

    def test_register_proxy(self):
        """Test registering a proxy class."""

        @ProxyRegistry.register("test.target")
        class TestProxy:
            SUPPORTED_STRATEGIES = {StrategyType.DELAY}

        assert ProxyRegistry.is_registered("test.target")
        assert ProxyRegistry.get_proxy("test.target") == TestProxy

    def test_register_returns_class(self):
        """Test decorator returns the class unchanged."""

        @ProxyRegistry.register("test.target2")
        class TestProxy:
            pass

        assert TestProxy is not None
        assert TestProxy.__name__ == "TestProxy"

    def test_register_duplicate_raises(self):
        """Test registering duplicate target raises ValueError."""
        ProxyRegistry._registry["duplicate.target"] = MockProxy

        with pytest.raises(ValueError, match="already registered"):

            @ProxyRegistry.register("duplicate.target")
            class AnotherProxy:
                pass

    def test_register_multiple_targets(self):
        """Test registering multiple different targets."""

        @ProxyRegistry.register("target.one")
        class Proxy1:
            pass

        @ProxyRegistry.register("target.two")
        class Proxy2:
            pass

        assert ProxyRegistry.is_registered("target.one")
        assert ProxyRegistry.is_registered("target.two")
        assert ProxyRegistry.get_proxy("target.one") == Proxy1
        assert ProxyRegistry.get_proxy("target.two") == Proxy2


class TestProxyRegistryGetProxy:
    """Tests for ProxyRegistry.get_proxy method."""

    def test_get_registered_proxy(self):
        """Test getting a registered proxy."""
        ProxyRegistry._registry["test.get"] = MockProxy

        result = ProxyRegistry.get_proxy("test.get")
        assert result == MockProxy

    def test_get_unregistered_proxy_raises(self):
        """Test getting unregistered proxy raises KeyError."""
        with pytest.raises(KeyError, match="No proxy registered"):
            ProxyRegistry.get_proxy("nonexistent.target")

    def test_get_proxy_error_shows_available(self):
        """Test KeyError message includes available targets."""
        ProxyRegistry._registry["available.target"] = MockProxy

        with pytest.raises(KeyError, match="available.target"):
            ProxyRegistry.get_proxy("missing.target")


class TestProxyRegistryGetSupportedStrategies:
    """Tests for ProxyRegistry.get_supported_strategies method."""

    def test_get_strategies_for_proxy(self):
        """Test getting supported strategies."""
        ProxyRegistry._registry["test.strategies"] = MockProxy

        strategies = ProxyRegistry.get_supported_strategies("test.strategies")
        assert StrategyType.DELAY in strategies
        assert StrategyType.SKIP in strategies

    def test_get_strategies_empty_set(self):
        """Test proxy without SUPPORTED_STRATEGIES returns empty set."""
        ProxyRegistry._registry["test.no_strategies"] = MockProxyNoStrategies

        strategies = ProxyRegistry.get_supported_strategies("test.no_strategies")
        assert strategies == set()

    def test_get_strategies_unregistered_raises(self):
        """Test getting strategies for unregistered target raises KeyError."""
        with pytest.raises(KeyError):
            ProxyRegistry.get_supported_strategies("nonexistent.target")


class TestProxyRegistryListTargets:
    """Tests for ProxyRegistry.list_targets method."""

    def test_list_empty(self):
        """Test list_targets with empty registry."""
        result = ProxyRegistry.list_targets()
        assert result == []

    def test_list_single_target(self):
        """Test list_targets with one registration."""
        ProxyRegistry._registry["single.target"] = MockProxy

        result = ProxyRegistry.list_targets()
        assert result == ["single.target"]

    def test_list_multiple_targets_sorted(self):
        """Test list_targets returns sorted list."""
        ProxyRegistry._registry["zebra.target"] = MockProxy
        ProxyRegistry._registry["apple.target"] = MockProxy
        ProxyRegistry._registry["mango.target"] = MockProxy

        result = ProxyRegistry.list_targets()
        assert result == ["apple.target", "mango.target", "zebra.target"]


class TestProxyRegistryIsRegistered:
    """Tests for ProxyRegistry.is_registered method."""

    def test_is_registered_true(self):
        """Test is_registered returns True for registered target."""
        ProxyRegistry._registry["registered.target"] = MockProxy

        assert ProxyRegistry.is_registered("registered.target") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered target."""
        assert ProxyRegistry.is_registered("unregistered.target") is False


class TestProxyRegistryUnregister:
    """Tests for ProxyRegistry.unregister method."""

    def test_unregister_existing(self):
        """Test unregistering an existing target."""
        ProxyRegistry._registry["to.unregister"] = MockProxy

        ProxyRegistry.unregister("to.unregister")
        assert not ProxyRegistry.is_registered("to.unregister")

    def test_unregister_nonexistent_raises(self):
        """Test unregistering nonexistent target raises KeyError."""
        with pytest.raises(KeyError, match="No proxy registered"):
            ProxyRegistry.unregister("nonexistent.target")


class TestProxyRegistryClear:
    """Tests for ProxyRegistry.clear method."""

    def test_clear_removes_all(self):
        """Test clear removes all registrations."""
        ProxyRegistry._registry["target.one"] = MockProxy
        ProxyRegistry._registry["target.two"] = MockProxy2

        ProxyRegistry.clear()

        assert ProxyRegistry.list_targets() == []
        assert not ProxyRegistry.is_registered("target.one")
        assert not ProxyRegistry.is_registered("target.two")


class TestProxyRegistryGetAllRegistrations:
    """Tests for ProxyRegistry.get_all_registrations method."""

    def test_get_all_empty(self):
        """Test get_all_registrations with empty registry."""
        result = ProxyRegistry.get_all_registrations()
        assert result == {}

    def test_get_all_returns_copy(self):
        """Test get_all_registrations returns a copy."""
        ProxyRegistry._registry["test.target"] = MockProxy

        result = ProxyRegistry.get_all_registrations()

        # Modifying result should not affect registry
        result["new.target"] = MockProxy2
        assert not ProxyRegistry.is_registered("new.target")

    def test_get_all_contains_registrations(self):
        """Test get_all_registrations contains all registrations."""
        ProxyRegistry._registry["target.one"] = MockProxy
        ProxyRegistry._registry["target.two"] = MockProxy2

        result = ProxyRegistry.get_all_registrations()

        assert "target.one" in result
        assert "target.two" in result
        assert result["target.one"] == MockProxy
        assert result["target.two"] == MockProxy2


class TestProxyRegistryIntegration:
    """Integration tests for ProxyRegistry."""

    def test_full_workflow(self):
        """Test full registration and lookup workflow."""
        # Register
        @ProxyRegistry.register("integration.test")
        class IntegrationProxy:
            SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.CORRUPT_TENSOR}

        # Verify registered
        assert ProxyRegistry.is_registered("integration.test")

        # Get proxy
        proxy_class = ProxyRegistry.get_proxy("integration.test")
        assert proxy_class == IntegrationProxy

        # Get strategies
        strategies = ProxyRegistry.get_supported_strategies("integration.test")
        assert StrategyType.DELAY in strategies
        assert StrategyType.CORRUPT_TENSOR in strategies

        # List targets
        assert "integration.test" in ProxyRegistry.list_targets()

        # Unregister
        ProxyRegistry.unregister("integration.test")
        assert not ProxyRegistry.is_registered("integration.test")

    def test_concurrent_registrations(self):
        """Test multiple registrations don't interfere."""

        @ProxyRegistry.register("concurrent.one")
        class Proxy1:
            SUPPORTED_STRATEGIES = {StrategyType.DELAY}

        @ProxyRegistry.register("concurrent.two")
        class Proxy2:
            SUPPORTED_STRATEGIES = {StrategyType.SKIP}

        @ProxyRegistry.register("concurrent.three")
        class Proxy3:
            SUPPORTED_STRATEGIES = {StrategyType.REPEAT}

        # Each proxy has its own strategies
        assert ProxyRegistry.get_supported_strategies("concurrent.one") == {
            StrategyType.DELAY
        }
        assert ProxyRegistry.get_supported_strategies("concurrent.two") == {
            StrategyType.SKIP
        }
        assert ProxyRegistry.get_supported_strategies("concurrent.three") == {
            StrategyType.REPEAT
        }

    def test_registry_persistence(self):
        """Test registry maintains state across calls."""
        ProxyRegistry._registry["persistent.target"] = MockProxy

        # Multiple calls should return same result
        assert ProxyRegistry.get_proxy("persistent.target") == MockProxy
        assert ProxyRegistry.get_proxy("persistent.target") == MockProxy
        assert len(ProxyRegistry.list_targets()) == 1
