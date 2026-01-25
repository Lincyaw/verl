"""
Integration tests for Ralph fault injection framework.

These tests verify the complete workflow:
1. Load YAML configuration
2. Install proxies
3. Simulate training steps
4. Collect fault injection data
5. Verify JSONL output files
6. Restore original functions
"""

import json
import os
import sys
import tempfile

import pytest

from ralph.collectors.dual_stream import DualStreamCollector
from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.engine import InjectionEngine
from ralph.core.registry import ProxyRegistry
from ralph.proxies.base import BaseProxy

# ==============================================================================
# Test Fixtures
# ==============================================================================


class IntegrationTestProxy(BaseProxy):
    """Test proxy for integration testing with all common strategies."""

    SUPPORTED_STRATEGIES = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.SKIP,
    }

    def __init__(self, original_fn, collector=None):
        super().__init__(original_fn, collector)
        self.call_count = 0
        self.injected_count = 0

    def _get_layer(self) -> str:
        return "Integration"

    def _strategy_delay(self, *args, **kwargs):
        self.injected_count += 1
        # Skip actual delay in tests
        return self._original(*args, **kwargs)

    def _strategy_raise_exception(self, *args, **kwargs):
        self.injected_count += 1
        self._config.parameters.get("exc_type", "RuntimeError")
        message = self._config.parameters.get("message", "Test exception")
        raise RuntimeError(message)

    def _strategy_corrupt_tensor(self, *args, **kwargs):
        self.injected_count += 1
        result = self._original(*args, **kwargs)
        # In real implementation this would corrupt tensors
        return result

    def _strategy_skip(self, *args, **kwargs):
        self.injected_count += 1
        return self._config.parameters.get("default_return", None)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return super().__call__(*args, **kwargs)


@pytest.fixture
def cleanup_registry():
    """Fixture to clean up registry after tests."""
    original = ProxyRegistry.get_all_registrations().copy()
    yield
    ProxyRegistry.clear()
    for target, proxy_class in original.items():
        try:
            ProxyRegistry.register(target)(proxy_class)
        except ValueError:
            pass


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_modules(cleanup_registry):
    """Fixture to create test modules and register proxies."""
    # Create test modules
    module_a = type(sys)("test_integration_module_a")
    module_a.compute = lambda x: x * 2
    module_a.transform = lambda x: x + 10
    sys.modules["test_integration_module_a"] = module_a

    module_b = type(sys)("test_integration_module_b")
    module_b.process = lambda data: {"result": data, "processed": True}
    sys.modules["test_integration_module_b"] = module_b

    # Register proxies
    ProxyRegistry.register("test_integration_module_a.compute")(IntegrationTestProxy)
    ProxyRegistry.register("test_integration_module_a.transform")(IntegrationTestProxy)
    ProxyRegistry.register("test_integration_module_b.process")(IntegrationTestProxy)

    yield {"module_a": module_a, "module_b": module_b}

    # Cleanup
    del sys.modules["test_integration_module_a"]
    del sys.modules["test_integration_module_b"]


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Fixture to create a sample YAML configuration file."""
    yaml_content = """
version: "1.0"

experiment:
  name: "integration_test"
  description: "Integration test configuration"
  seed: 42

global:
  enabled: true
  log_level: "DEBUG"
  output_dir: "{output_dir}"

data_collection:
  stream_a:
    enabled: true
    output:
      path: "${{global.output_dir}}/telemetry.jsonl"
      format: "jsonl"
  stream_b:
    enabled: true
    output:
      path: "${{global.output_dir}}/labels.jsonl"
      format: "jsonl"

scenarios:
  - id: "delay_compute"
    layer: "Integration"
    target: "test_integration_module_a.compute"
    fault_type: "delay"
    enabled: true
    severity: "medium"
    trigger:
      type: "one_shot"
      at_step: 5
    parameters:
      delay_seconds: 0.01
    expected_behavior: "Delay compute at step 5"

  - id: "skip_transform"
    layer: "Integration"
    target: "test_integration_module_a.transform"
    fault_type: "skip"
    enabled: true
    severity: "low"
    trigger:
      type: "step_based"
      start_step: 10
      end_step: 15
    parameters:
      default_return: -1
    expected_behavior: "Skip transform during steps 10-15"

  - id: "corrupt_process"
    layer: "Integration"
    target: "test_integration_module_b.process"
    fault_type: "corrupt_tensor"
    enabled: false
    severity: "high"
    trigger:
      type: "periodic"
      every_n_steps: 3
    parameters:
      noise_scale: 0.1
    expected_behavior: "Corrupt process every 3 steps (disabled)"
""".format(output_dir=temp_dir)

    yaml_path = os.path.join(temp_dir, "test_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


# ==============================================================================
# Full Workflow Integration Tests
# ==============================================================================


class TestFullWorkflow:
    """Tests for the complete load_config -> install -> run -> collect workflow."""

    def test_yaml_config_load_install_run_uninstall(self, temp_dir, test_modules, sample_yaml_config):
        """Test complete workflow from YAML config to running and cleanup."""
        module_a = test_modules["module_a"]
        module_b = test_modules["module_b"]

        # Store original functions
        original_compute = module_a.compute
        original_transform = module_a.transform
        original_process = module_b.process

        collector = DualStreamCollector(temp_dir)

        with InjectionEngine(collector) as engine:
            # Load YAML config
            engine.load_config(sample_yaml_config)

            # Verify configs loaded (only enabled ones count)
            assert "test_integration_module_a.compute" in engine._configs
            assert "test_integration_module_a.transform" in engine._configs
            # Disabled scenario should NOT be loaded
            assert "test_integration_module_b.process" not in engine._configs

            # Install proxies
            engine.install_proxies()
            assert engine.is_installed()

            # Verify proxies are installed
            assert isinstance(module_a.compute, IntegrationTestProxy)
            assert isinstance(module_a.transform, IntegrationTestProxy)
            # process should NOT be proxied (disabled)
            assert not isinstance(module_b.process, IntegrationTestProxy)

            # Simulate training loop
            results = []
            for step in range(20):
                engine.update_step(step)

                # Call proxied functions
                compute_result = module_a.compute(step)
                transform_result = module_a.transform(step)
                process_result = module_b.process(step)

                results.append(
                    {
                        "step": step,
                        "compute": compute_result,
                        "transform": transform_result,
                        "process": process_result,
                    }
                )

        # After context exit, proxies should be uninstalled
        assert not engine.is_installed()

        # Original functions should be restored
        assert module_a.compute == original_compute
        assert module_a.transform == original_transform
        assert module_b.process == original_process

        # Verify original functionality restored
        assert module_a.compute(10) == 20
        assert module_a.transform(10) == 20
        assert module_b.process(10) == {"result": 10, "processed": True}

        # Verify collector has data
        collector.close()
        assert os.path.exists(collector.get_labels_path())

    def test_full_workflow_with_data_collection(self, temp_dir, test_modules):
        """Test that fault injections are properly recorded in JSONL files."""
        module_a = test_modules["module_a"]

        collector = DualStreamCollector(
            temp_dir,
            telemetry_file="telemetry.jsonl",
            labels_file="labels.jsonl",
        )

        config = FaultConfig(
            id="test_delay",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            parameters={"delay_seconds": 0.001},
            enabled=True,
            severity="medium",
            expected_behavior="Delay at step 5",
        )

        with InjectionEngine(collector) as engine:
            engine.add_config("test_integration_module_a.compute", config)
            engine.install_proxies()

            # Run simulation triggering the fault
            for step in range(10):
                engine.update_step(step)
                module_a.compute(step)

        collector.flush()
        collector.close()

        # Verify labels file
        labels_path = collector.get_labels_path()
        assert os.path.exists(labels_path)

        with open(labels_path) as f:
            lines = f.readlines()

        # Should have at least injection start record
        assert len(lines) >= 1

        # Parse and verify JSONL content
        records = [json.loads(line) for line in lines]

        # Find injection start record
        start_records = [r for r in records if r.get("record_type") == "fault_injection_start"]
        assert len(start_records) >= 1

        start_record = start_records[0]
        assert start_record["fault_type"] == "delay"
        assert start_record["severity"] == "medium"
        assert "fault_id" in start_record
        assert "timestamp" in start_record

    def test_workflow_with_telemetry_recording(self, temp_dir, test_modules):
        """Test that telemetry events are recorded during workflow."""
        collector = DualStreamCollector(temp_dir)

        # Record some telemetry events
        collector.record_telemetry("workflow_start", {"test": "integration"})
        collector.record_telemetry("step_update", {"step": 0})
        collector.record_telemetry("workflow_end", {"success": True})

        collector.flush()
        collector.close()

        # Verify telemetry file
        telemetry_path = collector.get_telemetry_path()
        assert os.path.exists(telemetry_path)

        with open(telemetry_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        records = [json.loads(line) for line in lines]
        event_types = [r["event_type"] for r in records]
        assert "workflow_start" in event_types
        assert "step_update" in event_types
        assert "workflow_end" in event_types


# ==============================================================================
# Proxy Uninstallation Tests
# ==============================================================================


class TestProxyUninstallation:
    """Tests verifying proxy uninstallation restores original behavior."""

    def test_uninstall_restores_original_function(self, test_modules, cleanup_registry):
        """Test that uninstall_proxies restores original functions exactly."""
        module_a = test_modules["module_a"]

        # Store reference to original
        original_compute = module_a.compute
        original_id = id(original_compute)

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        engine.add_config("test_integration_module_a.compute", config)

        # Install proxy
        engine.install_proxies()
        assert module_a.compute != original_compute
        assert isinstance(module_a.compute, IntegrationTestProxy)

        # Uninstall proxy
        engine.uninstall_proxies()

        # Verify exact restoration
        assert module_a.compute == original_compute
        assert id(module_a.compute) == original_id

    def test_uninstall_preserves_module_state(self, test_modules, cleanup_registry):
        """Test that uninstall doesn't affect other module attributes."""
        module_a = test_modules["module_a"]

        # Add a custom attribute
        module_a.custom_attr = "test_value"
        module_a.custom_list = [1, 2, 3]

        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        engine.add_config("test_integration_module_a.compute", config)

        engine.install_proxies()
        engine.uninstall_proxies()

        # Custom attributes should be preserved
        assert module_a.custom_attr == "test_value"
        assert module_a.custom_list == [1, 2, 3]

    def test_uninstall_multiple_targets(self, test_modules, cleanup_registry):
        """Test uninstalling multiple proxies at once."""
        module_a = test_modules["module_a"]
        module_b = test_modules["module_b"]

        original_compute = module_a.compute
        original_transform = module_a.transform
        original_process = module_b.process

        engine = InjectionEngine()

        # Add multiple configs
        for target in ["test_integration_module_a.compute", "test_integration_module_a.transform"]:
            config = FaultConfig(
                id=f"fault_{target}",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            )
            engine.add_config(target, config)

        engine.install_proxies()

        # All targeted functions should be proxied
        assert isinstance(module_a.compute, IntegrationTestProxy)
        assert isinstance(module_a.transform, IntegrationTestProxy)
        # Not targeted
        assert module_b.process == original_process

        engine.uninstall_proxies()

        # All should be restored
        assert module_a.compute == original_compute
        assert module_a.transform == original_transform
        assert module_b.process == original_process

    def test_context_manager_guarantees_cleanup(self, test_modules, cleanup_registry):
        """Test that context manager always cleans up even on exception."""
        module_a = test_modules["module_a"]
        original_compute = module_a.compute

        try:
            with InjectionEngine() as engine:
                config = FaultConfig(
                    id="test_fault",
                    strategy=StrategyType.DELAY,
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
                )
                engine.add_config("test_integration_module_a.compute", config)
                engine.install_proxies()

                # Simulate an error during training
                raise RuntimeError("Simulated training error")
        except RuntimeError:
            pass

        # Even after exception, original should be restored
        assert module_a.compute == original_compute


# ==============================================================================
# Multiple Simultaneous Proxies Tests
# ==============================================================================


class TestMultipleSimultaneousProxies:
    """Tests for multiple proxies with different triggers active simultaneously."""

    def test_multiple_proxies_different_triggers(self, test_modules, temp_dir, cleanup_registry):
        """Test multiple proxies with different trigger types."""
        module_a = test_modules["module_a"]

        collector = DualStreamCollector(temp_dir)

        # Create configs with different trigger types
        one_shot_config = FaultConfig(
            id="one_shot_delay",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            parameters={"delay_seconds": 0.001},
        )

        step_based_config = FaultConfig(
            id="step_based_skip",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(type=TriggerType.STEP_BASED, start_step=10, end_step=15),
            parameters={"default_return": -999},
        )

        engine = InjectionEngine(collector)
        engine.add_config("test_integration_module_a.compute", one_shot_config)
        engine.add_config("test_integration_module_a.transform", step_based_config)
        engine.install_proxies()

        compute_proxy = engine.get_proxy("test_integration_module_a.compute")
        transform_proxy = engine.get_proxy("test_integration_module_a.transform")

        # Track which steps had injections
        compute_injected_steps = []
        transform_injected_steps = []

        for step in range(20):
            engine.update_step(step)

            before_compute_inject = compute_proxy.injected_count
            module_a.compute(step)
            if compute_proxy.injected_count > before_compute_inject:
                compute_injected_steps.append(step)

            before_transform_inject = transform_proxy.injected_count
            result = module_a.transform(step)
            if transform_proxy.injected_count > before_transform_inject:
                transform_injected_steps.append(step)
                assert result == -999  # Skip returns default

        engine.uninstall_proxies()
        collector.close()

        # Verify one-shot triggered exactly at step 5
        assert compute_injected_steps == [5]

        # Verify step-based triggered during steps 10-15
        assert transform_injected_steps == [10, 11, 12, 13, 14, 15]

    def test_periodic_trigger_multiple_activations(self, test_modules, cleanup_registry):
        """Test periodic trigger activates at expected intervals."""
        module_a = test_modules["module_a"]

        config = FaultConfig(
            id="periodic_delay",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5),
            parameters={"delay_seconds": 0.001},
        )

        engine = InjectionEngine()
        engine.add_config("test_integration_module_a.compute", config)
        engine.install_proxies()

        proxy = engine.get_proxy("test_integration_module_a.compute")
        injected_steps = []

        for step in range(25):
            engine.update_step(step)
            before = proxy.injected_count
            module_a.compute(step)
            if proxy.injected_count > before:
                injected_steps.append(step)

        engine.uninstall_proxies()

        # Should trigger at 0, 5, 10, 15, 20
        assert injected_steps == [0, 5, 10, 15, 20]

    def test_probabilistic_trigger_reproducibility(self, test_modules, cleanup_registry):
        """Test that probabilistic triggers are reproducible with same seed."""
        module_a = test_modules["module_a"]

        config = FaultConfig(
            id="prob_delay",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.PROBABILISTIC, probability=0.3),
            parameters={"delay_seconds": 0.001},
        )

        def run_with_seed(seed: int) -> list[int]:
            engine = InjectionEngine(seed=seed)
            engine.add_config("test_integration_module_a.compute", config)
            engine.install_proxies()

            proxy = engine.get_proxy("test_integration_module_a.compute")
            injected_steps = []

            for step in range(50):
                engine.update_step(step)
                before = proxy.injected_count
                module_a.compute(step)
                if proxy.injected_count > before:
                    injected_steps.append(step)

            engine.uninstall_proxies()
            return injected_steps

        # Run twice with same seed
        result1 = run_with_seed(42)
        result2 = run_with_seed(42)

        # Results should be identical
        assert result1 == result2

        # Run with different seed
        run_with_seed(123)

        # Results should likely be different (not guaranteed but highly probable)
        # We just check that the system works - exact comparison is probabilistic

    def test_one_shot_only_fires_once(self, test_modules, cleanup_registry):
        """Test that one-shot trigger only fires exactly once."""
        module_a = test_modules["module_a"]

        config = FaultConfig(
            id="one_shot_test",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            parameters={"delay_seconds": 0.001},
        )

        engine = InjectionEngine()
        engine.add_config("test_integration_module_a.compute", config)
        engine.install_proxies()

        proxy = engine.get_proxy("test_integration_module_a.compute")

        # Run past the trigger step multiple times
        for step in range(20):
            engine.update_step(step)
            module_a.compute(step)

        engine.uninstall_proxies()

        # Should have fired exactly once
        assert proxy.injected_count == 1


# ==============================================================================
# JSONL Output Validation Tests
# ==============================================================================


class TestJSONLOutput:
    """Tests verifying JSONL output files are valid and contain expected data."""

    def test_labels_jsonl_valid_format(self, temp_dir, test_modules, cleanup_registry):
        """Test that labels JSONL file contains valid JSON lines."""
        collector = DualStreamCollector(temp_dir)
        module_a = test_modules["module_a"]

        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            parameters={"delay_seconds": 0.001},
            enabled=True,
            severity="high",
            expected_behavior="Test delay",
        )

        with InjectionEngine(collector) as engine:
            engine.add_config("test_integration_module_a.compute", config)
            engine.install_proxies()

            for step in range(10):
                engine.update_step(step)
                module_a.compute(step)

        collector.flush()
        collector.close()

        # Read and validate JSONL
        labels_path = collector.get_labels_path()
        with open(labels_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    # Validate required fields
                    assert "timestamp" in record, f"Line {line_num}: missing timestamp"
                    assert "record_type" in record, f"Line {line_num}: missing record_type"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON at line {line_num}: {e}")

    def test_labels_jsonl_contains_expected_fields(self, temp_dir, test_modules, cleanup_registry):
        """Test that labels JSONL contains all expected fields."""
        collector = DualStreamCollector(temp_dir)
        module_a = test_modules["module_a"]

        config = FaultConfig(
            id="detailed_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            parameters={"delay_seconds": 0.001, "custom_param": "test_value"},
            enabled=True,
            severity="critical",
            expected_behavior="Test with detailed fields",
        )

        with InjectionEngine(collector) as engine:
            engine.add_config("test_integration_module_a.compute", config)
            engine.install_proxies()

            for step in range(10):
                engine.update_step(step)
                module_a.compute(step)

        collector.flush()
        collector.close()

        # Read labels file
        labels_path = collector.get_labels_path()
        with open(labels_path) as f:
            records = [json.loads(line) for line in f]

        # Find fault injection start record
        start_records = [r for r in records if r.get("record_type") == "fault_injection_start"]
        assert len(start_records) >= 1

        record = start_records[0]
        assert record["fault_type"] == "delay"
        assert record["severity"] == "critical"
        assert record["expected_behavior"] == "Test with detailed fields"
        assert "fault_id" in record
        assert "parameters" in record
        assert record["parameters"]["delay_seconds"] == 0.001
        assert record["parameters"]["custom_param"] == "test_value"

    def test_telemetry_jsonl_valid_format(self, temp_dir, cleanup_registry):
        """Test that telemetry JSONL file contains valid JSON lines."""
        collector = DualStreamCollector(temp_dir)

        # Record various telemetry events
        collector.record_telemetry("init", {"engine": "ralph"})
        collector.record_telemetry("config_loaded", {"scenarios": 3})
        collector.record_telemetry("training_start", {"total_steps": 1000})
        collector.record_telemetry("step", {"current": 50, "loss": 0.5})
        collector.record_telemetry("training_end", {"success": True})

        collector.flush()
        collector.close()

        # Read and validate JSONL
        telemetry_path = collector.get_telemetry_path()
        records = []
        with open(telemetry_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    records.append(record)
                    assert "event_id" in record
                    assert "timestamp" in record
                    assert "event_type" in record
                    assert "data" in record
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON at line {line_num}: {e}")

        assert len(records) == 5

    def test_jsonl_files_created_in_correct_location(self, temp_dir, cleanup_registry):
        """Test that JSONL files are created in the specified output directory."""
        custom_telemetry = "custom_telemetry.jsonl"
        custom_labels = "custom_labels.jsonl"

        collector = DualStreamCollector(
            temp_dir,
            telemetry_file=custom_telemetry,
            labels_file=custom_labels,
        )

        collector.record_telemetry("test", {"data": "value"})
        collector.flush()
        collector.close()

        # Verify files exist at correct paths
        expected_telemetry = os.path.join(temp_dir, custom_telemetry)
        expected_labels = os.path.join(temp_dir, custom_labels)

        assert os.path.exists(expected_telemetry)
        assert collector.get_telemetry_path() == expected_telemetry
        assert collector.get_labels_path() == expected_labels

    def test_jsonl_handles_unicode_content(self, temp_dir, cleanup_registry):
        """Test that JSONL handles unicode characters correctly."""
        collector = DualStreamCollector(temp_dir)

        # Record telemetry with unicode
        collector.record_telemetry("unicode_test", {"message": "测试中文", "emoji": "🚀✨", "special": "äöü ñ é"})

        collector.flush()
        collector.close()

        # Read and verify
        telemetry_path = collector.get_telemetry_path()
        with open(telemetry_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["data"]["message"] == "测试中文"
        assert record["data"]["emoji"] == "🚀✨"
        assert record["data"]["special"] == "äöü ñ é"

    def test_jsonl_buffer_flushing(self, temp_dir, cleanup_registry):
        """Test that collector properly buffers and flushes data."""
        collector = DualStreamCollector(temp_dir, buffer_size=5, auto_flush=True)

        # Write less than buffer size
        for i in range(3):
            collector.record_telemetry("event", {"index": i})

        # File might not exist yet (buffered)
        # Now trigger auto-flush by exceeding buffer
        for i in range(3, 10):
            collector.record_telemetry("event", {"index": i})

        collector.close()

        # All records should be written
        telemetry_path = collector.get_telemetry_path()
        with open(telemetry_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 10


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling in integration scenarios."""

    def test_empty_yaml_scenarios(self, temp_dir, cleanup_registry):
        """Test handling of YAML config with no enabled scenarios."""
        yaml_content = """
version: "1.0"
global:
  output_dir: "/tmp/test"
scenarios: []
"""
        yaml_path = os.path.join(temp_dir, "empty_scenarios.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        with pytest.raises(ValueError, match="No scenarios found"):
            engine.load_config(yaml_path)

    def test_disabled_scenarios_not_loaded(self, temp_dir, test_modules, cleanup_registry):
        """Test that disabled scenarios are not loaded."""
        yaml_content = """
version: "1.0"
global:
  output_dir: "/tmp/test"
scenarios:
  - id: "disabled_fault"
    target: "test_integration_module_a.compute"
    fault_type: "delay"
    enabled: false
    trigger:
      type: "one_shot"
      at_step: 5
    parameters:
      delay_seconds: 1.0
"""
        yaml_path = os.path.join(temp_dir, "disabled.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        engine = InjectionEngine()
        engine.load_config(yaml_path)

        # No configs should be added for disabled scenarios
        assert len(engine._configs) == 0

    def test_install_without_configs(self, cleanup_registry):
        """Test installing proxies when no configs added."""
        engine = InjectionEngine()
        engine.install_proxies()  # Should not raise

        assert engine.is_installed()
        assert len(engine._proxies) == 0

        engine.uninstall_proxies()

    def test_double_install_raises_error(self, test_modules, cleanup_registry):
        """Test that double installation raises error."""
        engine = InjectionEngine()
        config = FaultConfig(
            id="test_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        engine.add_config("test_integration_module_a.compute", config)

        engine.install_proxies()

        with pytest.raises(RuntimeError, match="already installed"):
            engine.install_proxies()

        engine.uninstall_proxies()

    def test_add_config_after_install_raises_error(self, test_modules, cleanup_registry):
        """Test adding config after install raises error."""
        engine = InjectionEngine()
        config1 = FaultConfig(
            id="test_fault1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        engine.add_config("test_integration_module_a.compute", config1)
        engine.install_proxies()

        config2 = FaultConfig(
            id="test_fault2",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )

        with pytest.raises(RuntimeError, match="Cannot add config"):
            engine.add_config("test_integration_module_a.transform", config2)

        engine.uninstall_proxies()

    def test_step_updates_before_install(self, cleanup_registry):
        """Test step updates work before proxies are installed."""
        engine = InjectionEngine()

        # Should not raise
        engine.update_step(10)
        assert engine.get_step() == 10

        engine.update_step(20)
        assert engine.get_step() == 20


# ==============================================================================
# Seed and Reproducibility Tests
# ==============================================================================


class TestReproducibility:
    """Tests for reproducibility with seed configuration."""

    def test_engine_seed_initialization(self, cleanup_registry):
        """Test engine initializes correctly with seed."""
        engine1 = InjectionEngine(seed=42)
        engine2 = InjectionEngine(seed=42)

        # Generate random values - should be identical
        val1 = engine1._rng.random()
        val2 = engine2._rng.random()

        assert val1 == val2

    def test_set_seed_resets_rng(self, cleanup_registry):
        """Test that set_seed resets the RNG."""
        engine = InjectionEngine(seed=42)

        # Generate some values
        v1 = engine._rng.random()
        v2 = engine._rng.random()

        # Reset seed
        engine.set_seed(42)

        # Should get same values again
        v3 = engine._rng.random()
        v4 = engine._rng.random()

        assert v1 == v3
        assert v2 == v4


# ==============================================================================
# Performance and Stress Tests
# ==============================================================================


class TestPerformance:
    """Tests for performance under various conditions."""

    def test_many_steps_simulation(self, test_modules, temp_dir, cleanup_registry):
        """Test simulation with many steps doesn't degrade."""
        module_a = test_modules["module_a"]
        collector = DualStreamCollector(temp_dir)

        config = FaultConfig(
            id="periodic_fault",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=100),
            parameters={"delay_seconds": 0.0001},
        )

        with InjectionEngine(collector) as engine:
            engine.add_config("test_integration_module_a.compute", config)
            engine.install_proxies()

            # Run many steps
            for step in range(10000):
                engine.update_step(step)
                module_a.compute(step)

            proxy = engine.get_proxy("test_integration_module_a.compute")

        # Should have triggered 100 times (at 0, 100, 200, ..., 9900)
        assert proxy.injected_count == 100

    def test_multiple_targets_many_steps(self, test_modules, temp_dir, cleanup_registry):
        """Test multiple proxied targets over many steps."""
        module_a = test_modules["module_a"]

        collector = DualStreamCollector(temp_dir)

        config1 = FaultConfig(
            id="fault_compute",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=50),
        )
        config2 = FaultConfig(
            id="fault_transform",
            strategy=StrategyType.SKIP,
            trigger=TriggerConfig(type=TriggerType.STEP_BASED, start_step=200, end_step=300),
            parameters={"default_return": 0},
        )

        with InjectionEngine(collector) as engine:
            engine.add_config("test_integration_module_a.compute", config1)
            engine.add_config("test_integration_module_a.transform", config2)
            engine.install_proxies()

            for step in range(500):
                engine.update_step(step)
                module_a.compute(step)
                module_a.transform(step)

            compute_proxy = engine.get_proxy("test_integration_module_a.compute")
            transform_proxy = engine.get_proxy("test_integration_module_a.transform")

        # compute: triggers at 0, 50, 100, ..., 450 = 10 times
        assert compute_proxy.injected_count == 10

        # transform: triggers at 200-300 inclusive = 101 times
        assert transform_proxy.injected_count == 101
