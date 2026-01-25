"""
Unit tests for YAML configuration loader.

Tests load_yaml_config and validate_yaml_config functions from ralph.core.config.
"""

import os
import tempfile
from typing import Generator

import pytest

from ralph.core.config import (
    DataCollectionConfig,
    ExperimentConfig,
    FaultConfig,
    GlobalConfig,
    RalphConfig,
    StrategyType,
    TriggerType,
    load_yaml_config,
    validate_yaml_config,
)


@pytest.fixture
def sample_yaml_content() -> str:
    """Sample YAML configuration matching the spec."""
    return """
version: "1.0"
experiment:
  name: "test_experiment"
  description: "Test experiment for unit tests"
  seed: 42

global:
  enabled: true
  log_level: "DEBUG"
  output_dir: "/experiments/ralph/test_run"

data_collection:
  stream_a:
    enabled: true
    sources:
      - type: "ray_logs"
        path: "/tmp/ray/session_latest/logs"
    output:
      format: "jsonl"
      path: "${global.output_dir}/telemetry.jsonl"

  stream_b:
    enabled: true
    output:
      format: "jsonl"
      path: "${global.output_dir}/labels.jsonl"

scenarios:
  - id: "test_delay"
    layer: "L0"
    target: "ray.get"
    enabled: true
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 100
    parameters:
      delay_seconds: 5.0
    expected_behavior: "Delay ray.get by 5 seconds"
    severity: "medium"

  - id: "test_corrupt_tensor"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    enabled: true
    fault_type: "corrupt_tensor"
    trigger:
      type: "step_based"
      start_step: 50
      end_step: 60
    parameters:
      noise_scale: 0.1
    expected_behavior: "Corrupt gradients with Gaussian noise"
    severity: "high"

  - id: "test_probabilistic"
    layer: "L2"
    target: "RewardManager.__call__"
    enabled: false
    fault_type: "reward_flip"
    trigger:
      type: "probabilistic"
      probability: 0.3
    parameters: {}
    expected_behavior: "Flip rewards probabilistically"
    severity: "low"

  - id: "test_periodic"
    layer: "L2"
    target: "FSDPCheckpointManager.save_checkpoint"
    enabled: true
    fault_type: "raise_exception"
    trigger:
      type: "periodic"
      every_n_steps: 10
    parameters:
      exc_type: "IOError"
      message: "Simulated IO error"
    expected_behavior: "Raise IOError every 10 steps"
    severity: "critical"
"""


@pytest.fixture
def temp_yaml_file(sample_yaml_content: str) -> Generator[str, None, None]:
    """Create a temporary YAML file with sample content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_load_valid_config(self, temp_yaml_file: str):
        """Test loading a valid YAML configuration."""
        config = load_yaml_config(temp_yaml_file)

        assert isinstance(config, RalphConfig)
        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.global_config, GlobalConfig)
        assert isinstance(config.data_collection, DataCollectionConfig)
        assert len(config.scenarios) == 4

    def test_experiment_section(self, temp_yaml_file: str):
        """Test parsing of experiment section."""
        config = load_yaml_config(temp_yaml_file)

        assert config.experiment.name == "test_experiment"
        assert config.experiment.description == "Test experiment for unit tests"
        assert config.experiment.seed == 42
        assert config.experiment.version == "1.0"

    def test_global_section(self, temp_yaml_file: str):
        """Test parsing of global section."""
        config = load_yaml_config(temp_yaml_file)

        assert config.global_config.enabled is True
        assert config.global_config.log_level == "DEBUG"
        assert config.global_config.output_dir == "/experiments/ralph/test_run"

    def test_data_collection_section(self, temp_yaml_file: str):
        """Test parsing of data_collection section with variable substitution."""
        config = load_yaml_config(temp_yaml_file)

        assert config.data_collection.stream_a_enabled is True
        assert config.data_collection.stream_b_enabled is True
        # Variable substitution should have occurred
        assert config.data_collection.stream_a_output_path == "/experiments/ralph/test_run/telemetry.jsonl"
        assert config.data_collection.stream_b_output_path == "/experiments/ralph/test_run/labels.jsonl"

    def test_scenarios_count(self, temp_yaml_file: str):
        """Test correct number of scenarios are parsed."""
        config = load_yaml_config(temp_yaml_file)
        assert len(config.scenarios) == 4

    def test_scenario_delay(self, temp_yaml_file: str):
        """Test parsing of delay scenario."""
        config = load_yaml_config(temp_yaml_file)

        delay_scenario = next(s for s in config.scenarios if s.id == "test_delay")

        assert isinstance(delay_scenario, FaultConfig)
        assert delay_scenario.strategy == StrategyType.DELAY
        assert delay_scenario.layer == "L0"
        assert delay_scenario.target == "ray.get"
        assert delay_scenario.enabled is True
        assert delay_scenario.severity == "medium"
        assert delay_scenario.parameters["delay_seconds"] == 5.0
        assert delay_scenario.trigger.type == TriggerType.ONE_SHOT
        assert delay_scenario.trigger.at_step == 100

    def test_scenario_step_based_trigger(self, temp_yaml_file: str):
        """Test parsing of step-based trigger."""
        config = load_yaml_config(temp_yaml_file)

        corrupt_scenario = next(s for s in config.scenarios if s.id == "test_corrupt_tensor")

        assert corrupt_scenario.trigger.type == TriggerType.STEP_BASED
        assert corrupt_scenario.trigger.start_step == 50
        assert corrupt_scenario.trigger.end_step == 60

    def test_scenario_probabilistic_trigger(self, temp_yaml_file: str):
        """Test parsing of probabilistic trigger."""
        config = load_yaml_config(temp_yaml_file)

        prob_scenario = next(s for s in config.scenarios if s.id == "test_probabilistic")

        assert prob_scenario.trigger.type == TriggerType.PROBABILISTIC
        assert prob_scenario.trigger.probability == 0.3
        assert prob_scenario.enabled is False

    def test_scenario_periodic_trigger(self, temp_yaml_file: str):
        """Test parsing of periodic trigger."""
        config = load_yaml_config(temp_yaml_file)

        periodic_scenario = next(s for s in config.scenarios if s.id == "test_periodic")

        assert periodic_scenario.trigger.type == TriggerType.PERIODIC
        assert periodic_scenario.trigger.every_n_steps == 10

    def test_raw_data_preserved(self, temp_yaml_file: str):
        """Test that raw YAML data is preserved."""
        config = load_yaml_config(temp_yaml_file)

        assert config.raw_data is not None
        assert "experiment" in config.raw_data
        assert "scenarios" in config.raw_data

    def test_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_empty_yaml(self):
        """Test ValueError for empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Empty or invalid"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)

    def test_no_scenarios(self):
        """Test ValueError when scenarios section is missing."""
        yaml_content = """
experiment:
  name: "test"
global:
  output_dir: "/tmp"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                with pytest.raises(ValueError, match="No scenarios found"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)

    def test_missing_required_fields(self):
        """Test ValueError when scenario is missing required fields."""
        yaml_content = """
scenarios:
  - id: "incomplete"
    fault_type: "delay"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                with pytest.raises(ValueError, match="missing required fields"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)

    def test_invalid_trigger_type(self):
        """Test ValueError for invalid trigger type."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "invalid_type"
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                with pytest.raises(ValueError, match="Invalid trigger type"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)


class TestVariableSubstitution:
    """Tests for variable substitution in YAML configs."""

    def test_global_prefix_substitution(self):
        """Test ${global.var} pattern substitution."""
        yaml_content = """
global:
  output_dir: "/data/output"
  run_id: "run_001"

scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters:
      path: "${global.output_dir}/logs"
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                test_scenario = config.scenarios[0]
                assert test_scenario.parameters["path"] == "/data/output/logs"
            finally:
                os.unlink(f.name)

    def test_simple_var_substitution(self):
        """Test ${var} pattern (without global prefix) substitution."""
        yaml_content = """
global:
  output_dir: "/data/output"

scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters:
      path: "${output_dir}/logs"
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                test_scenario = config.scenarios[0]
                assert test_scenario.parameters["path"] == "/data/output/logs"
            finally:
                os.unlink(f.name)

    def test_nested_parameters_substitution(self):
        """Test variable substitution in nested parameter structures."""
        yaml_content = """
global:
  base_path: "/data"

scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters:
      nested:
        path: "${global.base_path}/nested/path"
        items:
          - "${global.base_path}/item1"
          - "${global.base_path}/item2"
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                params = config.scenarios[0].parameters
                assert params["nested"]["path"] == "/data/nested/path"
                assert params["nested"]["items"][0] == "/data/item1"
                assert params["nested"]["items"][1] == "/data/item2"
            finally:
                os.unlink(f.name)

    def test_unknown_variable_preserved(self):
        """Test that unknown variables are preserved as-is."""
        yaml_content = """
global:
  known_var: "value"

scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters:
      unknown: "${global.unknown_var}"
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                # Unknown variables should be preserved
                assert config.scenarios[0].parameters["unknown"] == "${global.unknown_var}"
            finally:
                os.unlink(f.name)


class TestValidateYamlConfig:
    """Tests for validate_yaml_config function."""

    def test_valid_config(self, temp_yaml_file: str):
        """Test validation of valid config returns True."""
        is_valid, errors = validate_yaml_config(temp_yaml_file)
        assert is_valid is True
        assert len(errors) == 0

    def test_file_not_found(self):
        """Test validation of non-existent file."""
        is_valid, errors = validate_yaml_config("/nonexistent/path/config.yaml")
        assert is_valid is False
        assert any("not found" in e for e in errors)

    def test_invalid_yaml_syntax(self):
        """Test validation of file with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: :")
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("syntax" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)

    def test_missing_scenarios_section(self):
        """Test validation when scenarios section is missing."""
        yaml_content = """
experiment:
  name: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("scenarios" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)

    def test_empty_scenarios(self):
        """Test validation when scenarios section is empty."""
        yaml_content = """
scenarios: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("empty" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)

    def test_scenario_missing_fields(self):
        """Test validation detects missing required fields in scenarios."""
        yaml_content = """
scenarios:
  - id: "incomplete"
    fault_type: "delay"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("missing" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)


class TestStrategyAliases:
    """Tests for strategy aliases (e.g., nccl_timeout -> DELAY)."""

    def test_nccl_timeout_alias(self):
        """Test that nccl_timeout is mapped to DELAY strategy."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    fault_type: "nccl_timeout"
    trigger:
      type: "one_shot"
      at_step: 100
    parameters:
      delay_seconds: 1800
    expected_behavior: "NCCL timeout"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].strategy == StrategyType.DELAY
            finally:
                os.unlink(f.name)

    def test_gradient_corruption_alias(self):
        """Test that gradient_corruption is mapped to CORRUPT_TENSOR strategy."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    fault_type: "gradient_corruption"
    trigger:
      type: "one_shot"
      at_step: 100
    parameters:
      noise_scale: 0.1
    expected_behavior: "Gradient corruption"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].strategy == StrategyType.CORRUPT_TENSOR
            finally:
                os.unlink(f.name)

    def test_io_error_alias(self):
        """Test that io_error is mapped to RAISE_EXCEPTION strategy."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L2"
    target: "FSDPCheckpointManager.save_checkpoint"
    fault_type: "io_error"
    trigger:
      type: "one_shot"
      at_step: 100
    parameters: {}
    expected_behavior: "IO error on checkpoint"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].strategy == StrategyType.RAISE_EXCEPTION
            finally:
                os.unlink(f.name)


class TestDefaultValues:
    """Tests for default value handling."""

    def test_default_trigger_probability(self):
        """Test default probability is 1.0 for probabilistic triggers."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "probabilistic"
    parameters: {}
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].trigger.probability == 1.0
            finally:
                os.unlink(f.name)

    def test_default_enabled(self):
        """Test default enabled is True."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters: {}
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].enabled is True
            finally:
                os.unlink(f.name)

    def test_default_severity(self):
        """Test default severity is 'medium'."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters: {}
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.scenarios[0].severity == "medium"
            finally:
                os.unlink(f.name)

    def test_default_global_config(self):
        """Test defaults when global section is missing."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    parameters: {}
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.global_config.enabled is True
                assert config.global_config.log_level == "INFO"
                assert config.global_config.output_dir == ""
            finally:
                os.unlink(f.name)
