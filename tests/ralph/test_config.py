"""
Unit tests for core configuration classes.

Tests TriggerConfig, FaultConfig, TriggerType, StrategyType,
and YAML configuration loading from ralph.core.config.
"""

import os
import random
import tempfile
from unittest.mock import MagicMock

import pytest

from ralph.core.config import (
    DataCollectionConfig,
    ExperimentConfig,
    FaultConfig,
    GlobalConfig,
    RalphConfig,
    StrategyType,
    TriggerConfig,
    TriggerType,
    _parse_fault_config,
    _parse_trigger_config,
    _substitute_variables,
    load_yaml_config,
    validate_yaml_config,
)


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_one_shot_value(self):
        """Test ONE_SHOT enum value."""
        assert TriggerType.ONE_SHOT.value == "one_shot"

    def test_step_based_value(self):
        """Test STEP_BASED enum value."""
        assert TriggerType.STEP_BASED.value == "step_based"

    def test_probabilistic_value(self):
        """Test PROBABILISTIC enum value."""
        assert TriggerType.PROBABILISTIC.value == "probabilistic"

    def test_periodic_value(self):
        """Test PERIODIC enum value."""
        assert TriggerType.PERIODIC.value == "periodic"

    def test_trigger_type_from_string(self):
        """Test creating TriggerType from string value."""
        assert TriggerType("one_shot") == TriggerType.ONE_SHOT
        assert TriggerType("step_based") == TriggerType.STEP_BASED
        assert TriggerType("probabilistic") == TriggerType.PROBABILISTIC
        assert TriggerType("periodic") == TriggerType.PERIODIC

    def test_invalid_trigger_type_raises(self):
        """Test invalid trigger type raises ValueError."""
        with pytest.raises(ValueError):
            TriggerType("invalid_type")


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_common_strategies(self):
        """Test common strategy types exist."""
        assert StrategyType.DELAY.value == "delay"
        assert StrategyType.CORRUPT_TENSOR.value == "corrupt_tensor"
        assert StrategyType.INJECT_NAN.value == "inject_nan"
        assert StrategyType.INJECT_INF.value == "inject_inf"
        assert StrategyType.RAISE_EXCEPTION.value == "raise_exception"
        assert StrategyType.SKIP.value == "skip"
        assert StrategyType.REPEAT.value == "repeat"

    def test_l0_ray_strategies(self):
        """Test L0 Ray-specific strategies exist."""
        assert StrategyType.OBJECT_LOST.value == "object_lost"
        assert StrategyType.PARTIAL_FAILURE.value == "partial_failure"
        assert StrategyType.STORE_FULL.value == "store_full"
        assert StrategyType.SILENT_DROP.value == "silent_drop"
        assert StrategyType.WORKER_DEATH.value == "worker_death"
        assert StrategyType.STRAGGLER.value == "straggler"

    def test_l1_distributed_strategies(self):
        """Test L1 distributed strategies exist."""
        assert StrategyType.DEADLOCK.value == "deadlock"
        assert StrategyType.CORRUPT_GATHERED.value == "corrupt_gathered"
        assert StrategyType.MISSING_RANK.value == "missing_rank"
        assert StrategyType.BARRIER_TIMEOUT.value == "barrier_timeout"

    def test_l2_verl_strategies(self):
        """Test L2 verl-specific strategies exist."""
        assert StrategyType.REWARD_FLIP.value == "reward_flip"
        assert StrategyType.CONSTANT_REWARD.value == "constant_reward"
        assert StrategyType.FILE_NOT_FOUND.value == "file_not_found"
        assert StrategyType.CORRUPT_STATE_DICT.value == "corrupt_state_dict"

    def test_algorithm_strategies(self):
        """Test algorithm strategies exist."""
        assert StrategyType.WRONG_ADVANTAGE.value == "wrong_advantage"
        assert StrategyType.ZERO_ADVANTAGE.value == "zero_advantage"
        assert StrategyType.WRONG_KL.value == "wrong_kl"
        assert StrategyType.ZERO_KL.value == "zero_kl"

    def test_l3_gpu_memory_strategies(self):
        """Test L3 GPU memory strategies exist."""
        assert StrategyType.MEMORY_PRESSURE.value == "memory_pressure"
        assert StrategyType.OOM_SIMULATION.value == "oom_simulation"
        assert StrategyType.MEMORY_FRAGMENTATION.value == "memory_fragmentation"
        assert StrategyType.MEMORY_LEAK.value == "memory_leak"

    def test_strategy_type_from_string(self):
        """Test creating StrategyType from string value."""
        assert StrategyType("delay") == StrategyType.DELAY
        assert StrategyType("corrupt_tensor") == StrategyType.CORRUPT_TENSOR

    def test_invalid_strategy_type_raises(self):
        """Test invalid strategy type raises ValueError."""
        with pytest.raises(ValueError):
            StrategyType("invalid_strategy")


class TestTriggerConfig:
    """Tests for TriggerConfig dataclass."""

    def test_one_shot_default_values(self):
        """Test ONE_SHOT trigger with default values."""
        config = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10)
        assert config.type == TriggerType.ONE_SHOT
        assert config.at_step == 10
        assert config.start_step is None
        assert config.end_step is None
        assert config.probability == 1.0
        assert config.every_n_steps is None

    def test_step_based_config(self):
        """Test STEP_BASED trigger configuration."""
        config = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=15,
        )
        assert config.type == TriggerType.STEP_BASED
        assert config.start_step == 5
        assert config.end_step == 15

    def test_probabilistic_config(self):
        """Test PROBABILISTIC trigger configuration."""
        config = TriggerConfig(
            type=TriggerType.PROBABILISTIC,
            probability=0.3,
        )
        assert config.type == TriggerType.PROBABILISTIC
        assert config.probability == 0.3

    def test_periodic_config(self):
        """Test PERIODIC trigger configuration."""
        config = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=10,
        )
        assert config.type == TriggerType.PERIODIC
        assert config.every_n_steps == 10


class TestTriggerConfigShouldTrigger:
    """Tests for TriggerConfig.should_trigger() method."""

    def test_one_shot_triggers_at_exact_step(self):
        """Test ONE_SHOT triggers only at exact step."""
        config = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5)
        assert not config.should_trigger(4)
        assert config.should_trigger(5)
        assert not config.should_trigger(6)

    def test_one_shot_with_none_at_step(self):
        """Test ONE_SHOT with None at_step never triggers."""
        config = TriggerConfig(type=TriggerType.ONE_SHOT)
        for step in range(10):
            assert not config.should_trigger(step)

    def test_step_based_triggers_in_range(self):
        """Test STEP_BASED triggers within range."""
        config = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        assert not config.should_trigger(4)
        assert config.should_trigger(5)  # start boundary
        assert config.should_trigger(7)  # middle
        assert config.should_trigger(10)  # end boundary
        assert not config.should_trigger(11)

    def test_step_based_with_missing_bounds(self):
        """Test STEP_BASED with missing bounds returns False."""
        config = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5)
        assert not config.should_trigger(5)

        config2 = TriggerConfig(type=TriggerType.STEP_BASED, end_step=10)
        assert not config2.should_trigger(5)

    def test_probabilistic_with_zero_probability(self):
        """Test PROBABILISTIC never triggers with 0 probability."""
        config = TriggerConfig(type=TriggerType.PROBABILISTIC, probability=0.0)
        for _ in range(100):
            assert not config.should_trigger(1)

    def test_probabilistic_with_full_probability(self):
        """Test PROBABILISTIC always triggers with 1.0 probability."""
        config = TriggerConfig(type=TriggerType.PROBABILISTIC, probability=1.0)
        for _ in range(100):
            assert config.should_trigger(1)

    def test_probabilistic_with_custom_rng(self):
        """Test PROBABILISTIC uses custom RNG when provided."""
        config = TriggerConfig(type=TriggerType.PROBABILISTIC, probability=0.5)

        # Use deterministic RNG
        rng = random.Random(42)
        results = [config.should_trigger(i, rng=rng) for i in range(10)]

        # Same seed should produce same results
        rng2 = random.Random(42)
        results2 = [config.should_trigger(i, rng=rng2) for i in range(10)]

        assert results == results2

    def test_periodic_triggers_every_n_steps(self):
        """Test PERIODIC triggers every N steps."""
        config = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        assert config.should_trigger(0)  # 0 % 3 == 0
        assert not config.should_trigger(1)
        assert not config.should_trigger(2)
        assert config.should_trigger(3)  # 3 % 3 == 0
        assert config.should_trigger(6)  # 6 % 3 == 0
        assert config.should_trigger(9)

    def test_periodic_with_zero_every_n_steps(self):
        """Test PERIODIC with zero every_n_steps returns False."""
        config = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=0)
        assert not config.should_trigger(0)
        assert not config.should_trigger(1)

    def test_periodic_with_none_every_n_steps(self):
        """Test PERIODIC with None every_n_steps returns False."""
        config = TriggerConfig(type=TriggerType.PERIODIC)
        assert not config.should_trigger(0)


class TestFaultConfig:
    """Tests for FaultConfig dataclass."""

    def test_minimal_fault_config(self):
        """Test FaultConfig with minimal required fields."""
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10)
        config = FaultConfig(
            id="test-fault",
            strategy=StrategyType.DELAY,
            trigger=trigger,
        )
        assert config.id == "test-fault"
        assert config.strategy == StrategyType.DELAY
        assert config.trigger == trigger
        assert config.parameters == {}
        assert config.enabled is True
        assert config.severity == "medium"
        assert config.expected_behavior == ""
        assert config.layer == ""
        assert config.target == ""

    def test_full_fault_config(self):
        """Test FaultConfig with all fields."""
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5)
        config = FaultConfig(
            id="full-fault",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
            enabled=False,
            severity="high",
            expected_behavior="Corrupt tensors with Gaussian noise",
            layer="L1",
            target="torch.distributed.all_reduce",
        )
        assert config.id == "full-fault"
        assert config.strategy == StrategyType.CORRUPT_TENSOR
        assert config.parameters == {"noise_scale": 0.1}
        assert config.enabled is False
        assert config.severity == "high"
        assert config.expected_behavior == "Corrupt tensors with Gaussian noise"
        assert config.layer == "L1"
        assert config.target == "torch.distributed.all_reduce"

    def test_fault_config_with_complex_parameters(self):
        """Test FaultConfig with nested parameters."""
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1)
        config = FaultConfig(
            id="complex-fault",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=trigger,
            parameters={
                "exc_type": "RuntimeError",
                "message": "Test error",
                "nested": {
                    "key1": "value1",
                    "key2": [1, 2, 3],
                },
            },
        )
        assert config.parameters["exc_type"] == "RuntimeError"
        assert config.parameters["nested"]["key1"] == "value1"
        assert config.parameters["nested"]["key2"] == [1, 2, 3]


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test ExperimentConfig default values."""
        config = ExperimentConfig()
        assert config.name == ""
        assert config.description == ""
        assert config.seed is None
        assert config.version == "1.0"

    def test_custom_values(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
            seed=42,
            version="2.0",
        )
        assert config.name == "test_experiment"
        assert config.description == "Test description"
        assert config.seed == 42
        assert config.version == "2.0"


class TestGlobalConfig:
    """Tests for GlobalConfig dataclass."""

    def test_default_values(self):
        """Test GlobalConfig default values."""
        config = GlobalConfig()
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.output_dir == ""

    def test_custom_values(self):
        """Test GlobalConfig with custom values."""
        config = GlobalConfig(
            enabled=False,
            log_level="DEBUG",
            output_dir="/custom/path",
        )
        assert config.enabled is False
        assert config.log_level == "DEBUG"
        assert config.output_dir == "/custom/path"


class TestDataCollectionConfig:
    """Tests for DataCollectionConfig dataclass."""

    def test_default_values(self):
        """Test DataCollectionConfig default values."""
        config = DataCollectionConfig()
        assert config.stream_a_enabled is True
        assert config.stream_a_output_path == ""
        assert config.stream_b_enabled is True
        assert config.stream_b_output_path == ""

    def test_custom_values(self):
        """Test DataCollectionConfig with custom values."""
        config = DataCollectionConfig(
            stream_a_enabled=False,
            stream_a_output_path="/path/a.jsonl",
            stream_b_enabled=True,
            stream_b_output_path="/path/b.jsonl",
        )
        assert config.stream_a_enabled is False
        assert config.stream_a_output_path == "/path/a.jsonl"
        assert config.stream_b_enabled is True
        assert config.stream_b_output_path == "/path/b.jsonl"


class TestRalphConfig:
    """Tests for RalphConfig dataclass."""

    def test_basic_ralph_config(self):
        """Test creating basic RalphConfig."""
        experiment = ExperimentConfig(name="test")
        global_config = GlobalConfig(enabled=True)
        data_collection = DataCollectionConfig()
        scenarios = []

        config = RalphConfig(
            experiment=experiment,
            global_config=global_config,
            data_collection=data_collection,
            scenarios=scenarios,
        )

        assert config.experiment.name == "test"
        assert config.global_config.enabled is True
        assert config.scenarios == []
        assert config.raw_data == {}


class TestSubstituteVariables:
    """Tests for _substitute_variables helper function."""

    def test_substitute_global_prefix(self):
        """Test ${global.key} substitution."""
        variables = {"output_dir": "/data"}
        result = _substitute_variables("${global.output_dir}/logs", variables)
        assert result == "/data/logs"

    def test_substitute_without_prefix(self):
        """Test ${key} substitution without global prefix."""
        variables = {"output_dir": "/data"}
        result = _substitute_variables("${output_dir}/logs", variables)
        assert result == "/data/logs"

    def test_substitute_in_dict(self):
        """Test substitution in dictionary values."""
        variables = {"base": "/base"}
        data = {"path": "${global.base}/path", "other": "unchanged"}
        result = _substitute_variables(data, variables)
        assert result["path"] == "/base/path"
        assert result["other"] == "unchanged"

    def test_substitute_in_list(self):
        """Test substitution in list values."""
        variables = {"dir": "/data"}
        data = ["${global.dir}/a", "${dir}/b", "plain"]
        result = _substitute_variables(data, variables)
        assert result == ["/data/a", "/data/b", "plain"]

    def test_substitute_nested_structures(self):
        """Test substitution in nested structures."""
        variables = {"root": "/root"}
        data = {
            "nested": {
                "path": "${global.root}/nested",
                "items": ["${root}/item1", "${root}/item2"],
            }
        }
        result = _substitute_variables(data, variables)
        assert result["nested"]["path"] == "/root/nested"
        assert result["nested"]["items"] == ["/root/item1", "/root/item2"]

    def test_preserve_unknown_variables(self):
        """Test that unknown variables are preserved."""
        variables = {"known": "value"}
        result = _substitute_variables("${global.unknown}", variables)
        assert result == "${global.unknown}"

    def test_substitute_non_string_unchanged(self):
        """Test that non-string values are unchanged."""
        variables = {"key": "value"}
        assert _substitute_variables(42, variables) == 42
        assert _substitute_variables(3.14, variables) == 3.14
        assert _substitute_variables(True, variables) is True
        assert _substitute_variables(None, variables) is None


class TestParseTriggerConfig:
    """Tests for _parse_trigger_config helper function."""

    def test_parse_one_shot(self):
        """Test parsing ONE_SHOT trigger."""
        data = {"type": "one_shot", "at_step": 100}
        config = _parse_trigger_config(data)
        assert config.type == TriggerType.ONE_SHOT
        assert config.at_step == 100

    def test_parse_step_based(self):
        """Test parsing STEP_BASED trigger."""
        data = {"type": "step_based", "start_step": 10, "end_step": 20}
        config = _parse_trigger_config(data)
        assert config.type == TriggerType.STEP_BASED
        assert config.start_step == 10
        assert config.end_step == 20

    def test_parse_probabilistic(self):
        """Test parsing PROBABILISTIC trigger."""
        data = {"type": "probabilistic", "probability": 0.5}
        config = _parse_trigger_config(data)
        assert config.type == TriggerType.PROBABILISTIC
        assert config.probability == 0.5

    def test_parse_periodic(self):
        """Test parsing PERIODIC trigger."""
        data = {"type": "periodic", "every_n_steps": 5}
        config = _parse_trigger_config(data)
        assert config.type == TriggerType.PERIODIC
        assert config.every_n_steps == 5

    def test_parse_default_type(self):
        """Test parsing defaults to ONE_SHOT."""
        data = {"at_step": 1}
        config = _parse_trigger_config(data)
        assert config.type == TriggerType.ONE_SHOT

    def test_parse_default_probability(self):
        """Test default probability is 1.0."""
        data = {"type": "probabilistic"}
        config = _parse_trigger_config(data)
        assert config.probability == 1.0

    def test_parse_invalid_type_raises(self):
        """Test invalid trigger type raises ValueError."""
        data = {"type": "invalid_type"}
        with pytest.raises(ValueError, match="Invalid trigger type"):
            _parse_trigger_config(data)


class TestParseFaultConfig:
    """Tests for _parse_fault_config helper function."""

    def test_parse_valid_scenario(self):
        """Test parsing valid scenario."""
        scenario = {
            "id": "test-fault",
            "layer": "L0",
            "target": "ray.get",
            "fault_type": "delay",
            "trigger": {"type": "one_shot", "at_step": 10},
            "parameters": {"delay_seconds": 5.0},
            "expected_behavior": "Delay by 5 seconds",
        }
        config = _parse_fault_config(scenario, {})
        assert config.id == "test-fault"
        assert config.layer == "L0"
        assert config.target == "ray.get"
        assert config.strategy == StrategyType.DELAY
        assert config.parameters["delay_seconds"] == 5.0

    def test_parse_with_defaults(self):
        """Test parsing with default values."""
        scenario = {
            "id": "test",
            "layer": "L0",
            "target": "ray.get",
            "fault_type": "delay",
            "trigger": {"type": "one_shot", "at_step": 1},
            "expected_behavior": "test",
        }
        config = _parse_fault_config(scenario, {})
        assert config.enabled is True
        assert config.severity == "medium"
        assert config.parameters == {}

    def test_parse_missing_required_fields_raises(self):
        """Test missing required fields raises ValueError."""
        scenario = {"id": "incomplete", "fault_type": "delay"}
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_fault_config(scenario, {})

    def test_parse_strategy_aliases(self):
        """Test strategy aliases are recognized."""
        base_scenario = {
            "id": "test",
            "layer": "L0",
            "target": "test",
            "trigger": {"type": "one_shot", "at_step": 1},
            "expected_behavior": "test",
        }

        # Test nccl_timeout -> DELAY
        scenario = {**base_scenario, "fault_type": "nccl_timeout"}
        config = _parse_fault_config(scenario, {})
        assert config.strategy == StrategyType.DELAY

        # Test gradient_corruption -> CORRUPT_TENSOR
        scenario = {**base_scenario, "fault_type": "gradient_corruption"}
        config = _parse_fault_config(scenario, {})
        assert config.strategy == StrategyType.CORRUPT_TENSOR

        # Test io_error -> RAISE_EXCEPTION
        scenario = {**base_scenario, "fault_type": "io_error"}
        config = _parse_fault_config(scenario, {})
        assert config.strategy == StrategyType.RAISE_EXCEPTION

    def test_parse_variable_substitution(self):
        """Test variable substitution in parameters."""
        scenario = {
            "id": "test",
            "layer": "L0",
            "target": "test",
            "fault_type": "delay",
            "trigger": {"type": "one_shot", "at_step": 1},
            "parameters": {"path": "${global.output_dir}/logs"},
            "expected_behavior": "test",
        }
        config = _parse_fault_config(scenario, {"output_dir": "/data"})
        assert config.parameters["path"] == "/data/logs"


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    @pytest.fixture
    def minimal_yaml(self):
        """Create minimal valid YAML content."""
        return """
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

    def test_load_minimal_config(self, minimal_yaml):
        """Test loading minimal valid config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(minimal_yaml)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert len(config.scenarios) == 1
                assert config.scenarios[0].id == "test"
            finally:
                os.unlink(f.name)

    def test_load_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path.yaml")

    def test_load_empty_file_raises(self):
        """Test empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Empty or invalid"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_no_scenarios_raises(self):
        """Test missing scenarios raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("experiment:\n  name: test\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="No scenarios"):
                    load_yaml_config(f.name)
            finally:
                os.unlink(f.name)


class TestValidateYamlConfig:
    """Tests for validate_yaml_config function."""

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        yaml_content = """
scenarios:
  - id: "test"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 1
    expected_behavior: "test"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is True
                assert len(errors) == 0
            finally:
                os.unlink(f.name)

    def test_validate_file_not_found(self):
        """Test validation of non-existent file."""
        is_valid, errors = validate_yaml_config("/nonexistent.yaml")
        assert is_valid is False
        assert any("not found" in e.lower() for e in errors)

    def test_validate_missing_scenarios(self):
        """Test validation with missing scenarios."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("experiment:\n  name: test\n")
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("scenarios" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)

    def test_validate_empty_scenarios(self):
        """Test validation with empty scenarios."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("scenarios: []\n")
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("empty" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)

    def test_validate_scenario_missing_fields(self):
        """Test validation detects missing fields."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("scenarios:\n  - id: incomplete\n    fault_type: delay\n")
            f.flush()
            try:
                is_valid, errors = validate_yaml_config(f.name)
                assert is_valid is False
                assert any("missing" in e.lower() for e in errors)
            finally:
                os.unlink(f.name)
