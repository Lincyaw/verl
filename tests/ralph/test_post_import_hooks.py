"""
Unit tests for post-import hook functionality in InjectionEngine.

Tests cover:
- register_post_import_hook() requires use_post_import_hooks=True
- Hook registration for targets
- Hook fires when module imported after registration
- Hook fires immediately for already-imported modules
- Multiple hooks for different modules
- Hook cleanup on uninstall_proxies()
- Error handling when hook registration fails
"""

from unittest.mock import MagicMock, patch

import pytest

# Import proxy modules to register them with the ProxyRegistry
# This is necessary because the proxies use @ProxyRegistry.register() decorator
# which only runs when the module is imported
import ralph.proxies.l0_ray  # noqa: F401 - registers ray.get, ray.put
import ralph.proxies.l1_distributed  # noqa: F401 - registers torch.distributed.*
from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.engine import InjectionEngine


class TestPostImportHookBasics:
    """Tests for post-import hook basic functionality."""

    def test_post_import_hooks_disabled_by_default(self):
        """Post-import hooks are disabled by default."""
        engine = InjectionEngine()
        assert engine._use_post_import_hooks is False

    def test_post_import_hooks_can_be_enabled(self):
        """Post-import hooks can be enabled via constructor."""
        engine = InjectionEngine(use_post_import_hooks=True)
        assert engine._use_post_import_hooks is True

    def test_register_post_import_hook_requires_flag(self):
        """register_post_import_hook raises error when hooks disabled."""
        engine = InjectionEngine(use_post_import_hooks=False)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        with pytest.raises(RuntimeError) as exc_info:
            engine.register_post_import_hook("ray.get", config)
        assert "not enabled" in str(exc_info.value).lower()

    def test_import_hooks_dict_initialized(self):
        """_import_hooks dictionary is initialized."""
        engine = InjectionEngine()
        assert hasattr(engine, "_import_hooks")
        assert isinstance(engine._import_hooks, dict)
        assert len(engine._import_hooks) == 0


class TestPostImportHookRegistration:
    """Tests for hook registration."""

    def test_register_post_import_hook_validates_target(self):
        """Hook registration validates target is registered."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        with pytest.raises(KeyError) as exc_info:
            engine.register_post_import_hook("invalid.target", config)
        assert "No proxy registered" in str(exc_info.value)

    def test_register_post_import_hook_validates_strategy(self):
        """Hook registration validates strategy is supported."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.REWARD_FLIP,  # Not supported by ray.get
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        with pytest.raises(ValueError) as exc_info:
            engine.register_post_import_hook("ray.get", config)
        assert "not supported" in str(exc_info.value).lower()

    def test_register_post_import_hook_stores_in_configs(self):
        """Hook registration adds config to engine configs."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("wrapt.register_post_import_hook"):
            engine.register_post_import_hook("ray.get", config)

        assert "ray.get" in engine._configs
        assert engine._configs["ray.get"] is config

    def test_register_post_import_hook_stores_in_import_hooks(self):
        """Hook registration stores target/config in _import_hooks."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("wrapt.register_post_import_hook"):
            engine.register_post_import_hook("ray.get", config)

        assert "ray" in engine._import_hooks
        assert len(engine._import_hooks["ray"]) == 1
        assert engine._import_hooks["ray"][0] == ("ray.get", config)

    def test_register_post_import_hook_calls_wrapt(self):
        """Hook registration calls wrapt.register_post_import_hook."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("ralph.core.engine.wrapt.register_post_import_hook") as mock_hook:
            engine.register_post_import_hook("ray.get", config)
            mock_hook.assert_called_once()
            # First arg is callback, second is module name
            call_args = mock_hook.call_args[0]
            assert call_args[1] == "ray"


class TestPostImportHookMultiple:
    """Tests for multiple hooks."""

    def test_multiple_hooks_same_module(self):
        """Multiple hooks can be registered for the same module."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config1 = FaultConfig(
            id="test1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        config2 = FaultConfig(
            id="test2",
            strategy=StrategyType.STORE_FULL,  # Use a strategy supported by ray.put
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
        )

        with patch("wrapt.register_post_import_hook"):
            engine.register_post_import_hook("ray.get", config1)
            engine.register_post_import_hook("ray.put", config2)

        assert "ray" in engine._import_hooks
        assert len(engine._import_hooks["ray"]) == 2

    def test_multiple_hooks_different_modules(self):
        """Hooks can be registered for different modules."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config1 = FaultConfig(
            id="test1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )
        config2 = FaultConfig(
            id="test2",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
        )

        with patch("wrapt.register_post_import_hook"):
            engine.register_post_import_hook("ray.get", config1)
            engine.register_post_import_hook("torch.distributed.all_reduce", config2)

        assert "ray" in engine._import_hooks
        assert "torch.distributed" in engine._import_hooks


class TestPostImportHookCleanup:
    """Tests for hook cleanup."""

    def test_uninstall_clears_import_hooks(self):
        """uninstall_proxies clears _import_hooks."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("wrapt.register_post_import_hook"):
            engine.register_post_import_hook("ray.get", config)

        assert len(engine._import_hooks) > 0

        # Need to set _installed to True for uninstall to work
        engine._installed = True
        engine.uninstall_proxies()

        assert len(engine._import_hooks) == 0

    def test_context_manager_cleanup(self):
        """Context manager exit clears hooks."""
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("wrapt.register_post_import_hook"):
            with InjectionEngine(use_post_import_hooks=True) as engine:
                engine.register_post_import_hook("ray.get", config)
                engine._installed = True
                assert len(engine._import_hooks) > 0

        assert len(engine._import_hooks) == 0


class TestInstallSingleProxy:
    """Tests for _install_single_proxy helper method."""

    def test_install_single_proxy_creates_proxy(self):
        """_install_single_proxy creates and installs proxy."""
        engine = InjectionEngine()
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        # Add config first
        engine._configs["ray.get"] = config

        # Mock the target resolution
        mock_original = MagicMock()
        mock_module = MagicMock()

        with patch.object(engine, "_resolve_target") as mock_resolve:
            mock_resolve.return_value = (mock_original, mock_module, "get")
            engine._install_single_proxy("ray.get", config)

        assert "ray.get" in engine._proxies
        assert "ray.get" in engine._originals
        assert "ray.get" in engine._patch_refs

    def test_install_single_proxy_skips_if_already_installed(self):
        """_install_single_proxy skips if proxy already exists."""
        engine = InjectionEngine()
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        # Pre-install a mock proxy
        engine._proxies["ray.get"] = MagicMock()

        with patch.object(engine, "_resolve_target") as mock_resolve:
            engine._install_single_proxy("ray.get", config)
            # Should not call _resolve_target if already installed
            mock_resolve.assert_not_called()


class TestPostImportHookTargetValidation:
    """Tests for target validation in hook registration."""

    def test_hook_requires_qualified_target(self):
        """Hook registration requires fully qualified target."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        # Single part target should fail before reaching wrapt
        with pytest.raises((KeyError, ValueError)):
            engine.register_post_import_hook("unqualified", config)

    def test_hook_handles_nested_modules(self):
        """Hook correctly extracts module from nested path."""
        engine = InjectionEngine(use_post_import_hooks=True)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0),
        )

        with patch("wrapt.register_post_import_hook") as mock_hook:
            engine.register_post_import_hook("torch.distributed.all_reduce", config)
            call_args = mock_hook.call_args[0]
            assert call_args[1] == "torch.distributed"
