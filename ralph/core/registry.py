"""
Proxy registry for Ralph fault injection framework.

Contains ProxyRegistry class for registering and retrieving proxy classes.
"""

from typing import TYPE_CHECKING

from ralph.core.config import StrategyType

if TYPE_CHECKING:
    from ralph.proxies.base import BaseProxy


class ProxyRegistry:
    """
    Registry for fault injection proxy classes.

    Provides registration and lookup of proxy classes by target function name.
    Each proxy class declares which strategies it supports.

    Example:
        @ProxyRegistry.register('ray.get')
        class RayGetProxy(BaseProxy):
            SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.RAISE_EXCEPTION}
            ...

        # Later:
        proxy_class = ProxyRegistry.get_proxy('ray.get')
        strategies = ProxyRegistry.get_supported_strategies('ray.get')
    """

    _registry: dict[str, type["BaseProxy"]] = {}

    @classmethod
    def register(cls, target: str):
        """
        Class decorator to register a proxy class for a target function.

        Args:
            target: The fully qualified name of the target function
                   (e.g., 'ray.get', 'torch.distributed.all_reduce')

        Returns:
            Decorator function that registers the class

        Example:
            @ProxyRegistry.register('ray.get')
            class RayGetProxy(BaseProxy):
                ...
        """

        def decorator(proxy_class: type["BaseProxy"]) -> type["BaseProxy"]:
            if target in cls._registry:
                raise ValueError(
                    f"Target '{target}' is already registered to {cls._registry[target].__name__}"
                )
            cls._registry[target] = proxy_class
            return proxy_class

        return decorator

    @classmethod
    def get_proxy(cls, target: str) -> type["BaseProxy"]:
        """
        Get the proxy class registered for a target function.

        Args:
            target: The fully qualified name of the target function

        Returns:
            The proxy class registered for the target

        Raises:
            KeyError: If no proxy is registered for the target
        """
        if target not in cls._registry:
            raise KeyError(
                f"No proxy registered for target '{target}'. "
                f"Available targets: {cls.list_targets()}"
            )
        return cls._registry[target]

    @classmethod
    def get_supported_strategies(cls, target: str) -> set[StrategyType]:
        """
        Get the set of strategies supported by the proxy for a target.

        Args:
            target: The fully qualified name of the target function

        Returns:
            Set of StrategyType values supported by the proxy

        Raises:
            KeyError: If no proxy is registered for the target
        """
        proxy_class = cls.get_proxy(target)
        return getattr(proxy_class, "SUPPORTED_STRATEGIES", set())

    @classmethod
    def list_targets(cls) -> list[str]:
        """
        List all registered target function names.

        Returns:
            Sorted list of registered target names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, target: str) -> bool:
        """
        Check if a target function has a registered proxy.

        Args:
            target: The fully qualified name of the target function

        Returns:
            True if a proxy is registered for the target, False otherwise
        """
        return target in cls._registry

    @classmethod
    def unregister(cls, target: str) -> None:
        """
        Remove a proxy registration.

        Primarily useful for testing purposes.

        Args:
            target: The fully qualified name of the target function

        Raises:
            KeyError: If no proxy is registered for the target
        """
        if target not in cls._registry:
            raise KeyError(f"No proxy registered for target '{target}'")
        del cls._registry[target]

    @classmethod
    def clear(cls) -> None:
        """
        Clear all proxy registrations.

        Primarily useful for testing purposes.
        """
        cls._registry.clear()

    @classmethod
    def get_all_registrations(cls) -> dict[str, type["BaseProxy"]]:
        """
        Get a copy of all current registrations.

        Returns:
            Dictionary mapping target names to proxy classes
        """
        return dict(cls._registry)
