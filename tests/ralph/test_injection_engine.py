"""
Unit tests for InjectionEngine orchestrator.
"""

import os
import tempfile

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.engine import InjectionEngine
from ralph.core.registry import ProxyRegistry
from ralph.proxies.base import BaseProxy


# Mock proxy class for testing
class MockProxy(BaseProxy):
    """Mock proxy for testing InjectionEngine."""

    SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.RAISE_EXCEPTION}

    def _get_layer(self) -> str:
        return "Test"

    def _strategy_delay(self, *args, **kwargs):
        return self._original(*args, **kwargs)

    def _strategy_raise_exception(self, *args, **kwargs):
        raise RuntimeError("Mock exception")


# Create a mock module for testing
class MockModule:
    """Mock module for testing monkey-patching."""

    @staticmethod
    def target_function(x, y):
        return x + y


@pytest.fixture
def cleanup_registry():
    """Fixture to clean up registry after tests."""
    # Store original registrations
    original = ProxyRegistry.get_all_registrations().copy()

    yield

    # Clear and restore
    ProxyRegistry.clear()
    for target, proxy_class in original.items():
        try:
            ProxyRegistry.register(target)(proxy_class)
        except ValueError:
            pass  # Already registered


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestInjectionEngineInit:
    """Tests for InjectionEngine initialization."""

    def test_init_without_collector(self):
        """Test initialization without collector."""
        engine = InjectionEngine()
        assert engine._collector is None
        assert engine._configs == {}
        assert engine._proxies == {}
        assert engine._originals == {}
        assert engine._current_step == 0
        assert engine._installed is False

    def test_init_with_collector(self, temp_dir):
        """Test initialization with collector."""
        from ralph.collectors.dual_stream import DualStreamCollector

        collector = DualStreamCollector(temp_dir)
        engine = InjectionEngine(collector)
        assert engine._collector is collector
        collector.close()

    def test_init_with_seed(self):
        """Test initialization with random seed."""
        engine = InjectionEngine(seed=42)
        assert engine._rng is not None


class TestInjectionEngineAddConfig:
    """Tests for add_config method."""

    def test_add_config_success(self, cleanup_registry):
        """Test successfully adding a config."""
        # Register mock proxy
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )

        engine.add_config("test.target", config)
        assert "test.target" in engine._configs
        assert engine._configs["test.target"] == config

    def test_add_config_unregistered_target(self):
        """Test adding config for unregistered target raises KeyError."""
        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )

        with pytest.raises(KeyError, match="No proxy registered"):
            engine.add_config("unregistered.target", config)

    def test_add_config_unsupported_strategy(self, cleanup_registry):
        """Test adding config with unsupported strategy raises ValueError."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.CORRUPT_TENSOR,  # Not supported by MockProxy
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )

        with pytest.raises(ValueError, match="not supported"):
            engine.add_config("test.target", config)

    def test_add_config_after_install_raises_error(self, cleanup_registry):
        """Test adding config after proxies installed raises RuntimeError."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)

        # Mock install
        engine._installed = True

        config2 = FaultConfig(
            id="test_fault2",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=20),
        )

        with pytest.raises(RuntimeError, match="Cannot add config"):
            engine.add_config("test.target", config2)


class TestInjectionEngineRemoveConfig:
    """Tests for remove_config method."""

    def test_remove_config_success(self, cleanup_registry):
        """Test successfully removing a config."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)
        assert "test.target" in engine._configs

        engine.remove_config("test.target")
        assert "test.target" not in engine._configs

    def test_remove_config_not_found(self):
        """Test removing non-existent config raises KeyError."""
        engine = InjectionEngine()

        with pytest.raises(KeyError, match="No config for"):
            engine.remove_config("nonexistent.target")

    def test_remove_config_after_install(self, cleanup_registry):
        """Test removing config after install raises RuntimeError."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)
        engine._installed = True

        with pytest.raises(RuntimeError, match="Cannot remove config"):
            engine.remove_config("test.target")


class TestInjectionEngineGetConfig:
    """Tests for get_config method."""

    def test_get_config_exists(self, cleanup_registry):
        """Test getting existing config."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)

        result = engine.get_config("test.target")
        assert result == config

    def test_get_config_not_exists(self):
        """Test getting non-existent config returns None."""
        engine = InjectionEngine()
        assert engine.get_config("nonexistent.target") is None


class TestInjectionEngineListTargets:
    """Tests for list_targets method."""

    def test_list_targets_empty(self):
        """Test listing targets with no configs."""
        engine = InjectionEngine()
        assert engine.list_targets() == []

    def test_list_targets_sorted(self, cleanup_registry):
        """Test listing targets returns sorted list."""
        ProxyRegistry.register("z.target")(MockProxy)
        ProxyRegistry.register("a.target")(MockProxy)
        ProxyRegistry.register("m.target")(MockProxy)

        engine = InjectionEngine()
        for target in ["z.target", "a.target", "m.target"]:
            config = FaultConfig(
                id=f"fault_{target}",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config(target, config)

        result = engine.list_targets()
        assert result == ["a.target", "m.target", "z.target"]


class TestInjectionEngineInstallUninstall:
    """Tests for install_proxies and uninstall_proxies methods."""

    def test_install_proxies_monkey_patches(self, cleanup_registry):
        """Test install_proxies correctly monkey-patches targets."""
        # Create a test module
        import sys

        test_module = type(sys)("test_module")
        test_module.test_function = lambda x: x * 2
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            engine = InjectionEngine()
            config = FaultConfig(
                id="test_fault",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config("test_module.test_function", config)

            # Before install
            assert not engine.is_installed()
            original_fn = test_module.test_function

            # Install
            engine.install_proxies()

            assert engine.is_installed()
            assert test_module.test_function != original_fn
            assert isinstance(test_module.test_function, MockProxy)

        finally:
            engine.uninstall_proxies()
            del sys.modules["test_module"]

    def test_uninstall_proxies_restores_original(self, cleanup_registry):
        """Test uninstall_proxies restores original functions."""
        import sys

        test_module = type(sys)("test_module")
        original_fn = lambda x: x * 2
        test_module.test_function = original_fn
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            engine = InjectionEngine()
            config = FaultConfig(
                id="test_fault",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config("test_module.test_function", config)

            engine.install_proxies()
            assert test_module.test_function != original_fn

            engine.uninstall_proxies()
            assert test_module.test_function == original_fn
            assert not engine.is_installed()

        finally:
            del sys.modules["test_module"]

    def test_install_already_installed_raises_error(self, cleanup_registry):
        """Test installing when already installed raises RuntimeError."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        engine._installed = True

        with pytest.raises(RuntimeError, match="already installed"):
            engine.install_proxies()

    def test_uninstall_not_installed_no_error(self):
        """Test uninstalling when not installed does nothing."""
        engine = InjectionEngine()
        engine.uninstall_proxies()  # Should not raise


class TestInjectionEngineUpdateStep:
    """Tests for update_step method."""

    def test_update_step_basic(self):
        """Test basic step update."""
        engine = InjectionEngine()
        assert engine.get_step() == 0

        engine.update_step(10)
        assert engine.get_step() == 10

        engine.update_step(100)
        assert engine.get_step() == 100

    def test_update_step_propagates_to_proxies(self, cleanup_registry):
        """Test update_step propagates to installed proxies."""
        import sys

        test_module = type(sys)("test_module")
        test_module.test_function = lambda x: x * 2
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            engine = InjectionEngine()
            config = FaultConfig(
                id="test_fault",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config("test_module.test_function", config)
            engine.install_proxies()

            proxy = engine.get_proxy("test_module.test_function")
            assert proxy.get_step() == 0

            engine.update_step(50)
            assert proxy.get_step() == 50

        finally:
            engine.uninstall_proxies()
            del sys.modules["test_module"]


class TestInjectionEngineContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_uninstalls_on_exit(self, cleanup_registry):
        """Test context manager uninstalls proxies on exit."""
        import sys

        test_module = type(sys)("test_module")
        original_fn = lambda x: x * 2
        test_module.test_function = original_fn
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            with InjectionEngine() as engine:
                config = FaultConfig(
                    id="test_fault",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
                )
                engine.add_config("test_module.test_function", config)
                engine.install_proxies()
                assert engine.is_installed()

            # After context exit
            assert not engine.is_installed()
            assert test_module.test_function == original_fn

        finally:
            del sys.modules["test_module"]

    def test_context_manager_flushes_collector(self, temp_dir, cleanup_registry):
        """Test context manager flushes collector on exit."""
        from ralph.collectors.dual_stream import DualStreamCollector

        collector = DualStreamCollector(temp_dir, auto_flush=False)

        # Record some telemetry
        collector.record_telemetry("test", {"key": "value"})

        with InjectionEngine(collector):
            pass

        # After context exit, collector should be flushed
        assert os.path.exists(collector.get_telemetry_path())


class TestInjectionEngineGetActiveFaults:
    """Tests for get_active_faults method."""

    def test_get_active_faults_one_shot(self, cleanup_registry):
        """Test getting active faults with ONE_SHOT trigger."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)

        assert engine.get_active_faults() == []

        engine.update_step(10)
        assert "test_fault" in engine.get_active_faults()

        engine.update_step(11)
        assert engine.get_active_faults() == []


class TestInjectionEngineLoadConfig:
    """Tests for load_config YAML loading method."""

    def test_load_config_basic(self, temp_dir, cleanup_registry):
        """Test basic YAML config loading."""
        ProxyRegistry.register("test.target")(MockProxy)

        yaml_content = """
global:
  output_dir: /tmp/test

scenarios:
  - id: test_fault
    target: test.target
    fault_type: delay
    severity: medium
    trigger:
      type: one_shot
      at_step: 10
    parameters:
      delay_seconds: 5.0
    expected_behavior: Should delay for 5 seconds
"""
        yaml_path = os.path.join(temp_dir, "test_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        engine.load_config(yaml_path)

        assert "test.target" in engine._configs
        config = engine._configs["test.target"]
        assert config.id == "test_fault"
        assert config.strategy == StrategyType.DELAY
        assert config.parameters["delay_seconds"] == 5.0

    def test_load_config_variable_substitution(self, temp_dir, cleanup_registry):
        """Test YAML config variable substitution."""
        ProxyRegistry.register("test.target")(MockProxy)

        yaml_content = """
global:
  output_path: /custom/path

scenarios:
  - id: test_fault
    target: test.target
    fault_type: delay
    trigger:
      type: one_shot
      at_step: 10
    parameters:
      path: ${global.output_path}/subdir
"""
        yaml_path = os.path.join(temp_dir, "test_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        engine.load_config(yaml_path)

        config = engine._configs["test.target"]
        assert config.parameters["path"] == "/custom/path/subdir"

    def test_load_config_missing_required_field(self, temp_dir, cleanup_registry):
        """Test YAML config with missing required field raises ValueError."""
        yaml_content = """
scenarios:
  - id: test_fault
    # missing 'target' field
    fault_type: delay
    trigger:
      type: one_shot
      at_step: 10
"""
        yaml_path = os.path.join(temp_dir, "test_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        with pytest.raises(ValueError, match="missing required field"):
            engine.load_config(yaml_path)

    def test_load_config_empty_file(self, temp_dir):
        """Test loading empty YAML file raises ValueError."""
        yaml_path = os.path.join(temp_dir, "empty.yaml")
        with open(yaml_path, "w") as f:
            f.write("")

        engine = InjectionEngine()
        with pytest.raises(ValueError, match="Empty or invalid"):
            engine.load_config(yaml_path)

    def test_load_config_no_scenarios(self, temp_dir):
        """Test loading YAML without scenarios raises ValueError."""
        yaml_content = """
global:
  output_dir: /tmp/test
"""
        yaml_path = os.path.join(temp_dir, "no_scenarios.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        with pytest.raises(ValueError, match="No scenarios found"):
            engine.load_config(yaml_path)

    def test_load_config_file_not_found(self):
        """Test loading non-existent YAML file raises FileNotFoundError."""
        engine = InjectionEngine()
        with pytest.raises(FileNotFoundError):
            engine.load_config("/nonexistent/path.yaml")

    def test_load_config_after_install_raises_error(self):
        """Test loading config after install raises RuntimeError."""
        engine = InjectionEngine()
        engine._installed = True

        with pytest.raises(RuntimeError, match="Cannot load config"):
            engine.load_config("some_path.yaml")


class TestInjectionEngineGetters:
    """Tests for various getter methods."""

    def test_get_proxy_exists(self, cleanup_registry):
        """Test getting proxy that exists."""
        import sys

        test_module = type(sys)("test_module")
        test_module.test_function = lambda x: x * 2
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            engine = InjectionEngine()
            config = FaultConfig(
                id="test_fault",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config("test_module.test_function", config)
            engine.install_proxies()

            proxy = engine.get_proxy("test_module.test_function")
            assert proxy is not None
            assert isinstance(proxy, MockProxy)

        finally:
            engine.uninstall_proxies()
            del sys.modules["test_module"]

    def test_get_proxy_not_exists(self):
        """Test getting proxy that doesn't exist returns None."""
        engine = InjectionEngine()
        assert engine.get_proxy("nonexistent.target") is None

    def test_get_collector(self, temp_dir):
        """Test getting collector."""
        from ralph.collectors.dual_stream import DualStreamCollector

        collector = DualStreamCollector(temp_dir)
        engine = InjectionEngine(collector)
        assert engine.get_collector() is collector
        collector.close()

    def test_get_collector_none(self):
        """Test getting collector when none configured."""
        engine = InjectionEngine()
        assert engine.get_collector() is None

    def test_is_installed(self, cleanup_registry):
        """Test is_installed method."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        assert not engine.is_installed()

        engine._installed = True
        assert engine.is_installed()


class TestInjectionEngineSetSeed:
    """Tests for set_seed method."""

    def test_set_seed_updates_rng(self):
        """Test set_seed updates the RNG."""
        engine = InjectionEngine()
        engine.set_seed(42)

        # Generate some values
        val1 = engine._rng.random()

        # Reset with same seed
        engine.set_seed(42)
        val2 = engine._rng.random()

        assert val1 == val2

    def test_set_seed_propagates_to_proxies(self, cleanup_registry):
        """Test set_seed propagates to installed proxies."""
        import sys

        test_module = type(sys)("test_module")
        test_module.test_function = lambda x: x * 2
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.test_function")(MockProxy)

            engine = InjectionEngine()
            config = FaultConfig(
                id="test_fault",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )
            engine.add_config("test_module.test_function", config)
            engine.install_proxies()

            proxy = engine.get_proxy("test_module.test_function")
            old_rng = proxy._rng

            engine.set_seed(123)
            assert proxy._rng != old_rng

        finally:
            engine.uninstall_proxies()
            del sys.modules["test_module"]


class TestInjectionEngineRepr:
    """Tests for __repr__ method."""

    def test_repr_empty(self):
        """Test repr with no configs."""
        engine = InjectionEngine()
        result = repr(engine)
        assert "InjectionEngine" in result
        assert "targets=[]" in result
        assert "installed=False" in result
        assert "step=0" in result

    def test_repr_with_configs(self, cleanup_registry):
        """Test repr with configs added."""
        ProxyRegistry.register("test.target")(MockProxy)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )
        engine.add_config("test.target", config)
        engine.update_step(5)

        result = repr(engine)
        assert "test.target" in result
        assert "step=5" in result


class TestInjectionEngineIntegration:
    """Integration tests for InjectionEngine."""

    def test_full_workflow_with_delay(self, temp_dir, cleanup_registry):
        """Test full workflow: add config, install, run, uninstall."""
        import sys

        from ralph.collectors.dual_stream import DualStreamCollector

        # Create test module
        test_module = type(sys)("test_module")
        call_count = {"count": 0}

        def target_function(x):
            call_count["count"] += 1
            return x * 2

        test_module.target_function = target_function
        sys.modules["test_module"] = test_module

        try:
            ProxyRegistry.register("test_module.target_function")(MockProxy)

            collector = DualStreamCollector(temp_dir)

            with InjectionEngine(collector) as engine:
                config = FaultConfig(
                    id="delay_fault",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
                    parameters={"delay_seconds": 0.01},
                )
                engine.add_config("test_module.target_function", config)
                engine.install_proxies()

                # Run simulation
                for step in range(10):
                    engine.update_step(step)
                    result = test_module.target_function(step)
                    assert result == step * 2

            # Verify original function restored
            result = test_module.target_function(100)
            assert result == 200

            # Verify collector has data
            collector.flush()
            assert os.path.exists(collector.get_labels_path())

        finally:
            del sys.modules["test_module"]

    def test_multiple_targets(self, cleanup_registry):
        """Test engine with multiple targets."""
        import sys

        # Create test modules
        module_a = type(sys)("module_a")
        module_a.func = lambda x: x + 1
        sys.modules["module_a"] = module_a

        module_b = type(sys)("module_b")
        module_b.func = lambda x: x * 2
        sys.modules["module_b"] = module_b

        try:
            ProxyRegistry.register("module_a.func")(MockProxy)
            ProxyRegistry.register("module_b.func")(MockProxy)

            engine = InjectionEngine()

            config_a = FaultConfig(
                id="fault_a",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            )
            config_b = FaultConfig(
                id="fault_b",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            )

            engine.add_config("module_a.func", config_a)
            engine.add_config("module_b.func", config_b)

            engine.install_proxies()

            assert isinstance(module_a.func, MockProxy)
            assert isinstance(module_b.func, MockProxy)

            engine.uninstall_proxies()

            assert not isinstance(module_a.func, MockProxy)
            assert not isinstance(module_b.func, MockProxy)

        finally:
            engine.uninstall_proxies()
            del sys.modules["module_a"]
            del sys.modules["module_b"]
