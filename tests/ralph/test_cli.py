"""
Unit tests for Ralph CLI.

Tests the command-line interface including:
- Argument parsing
- --help output
- --list-targets functionality
- --validate functionality
- --config loading
- --output-dir override
- --dry-run mode
"""

import os
import sys
import tempfile
from io import StringIO
from unittest import mock

import pytest

from ralph.cli.main import create_parser, list_targets, main, validate_config


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_creation(self):
        """Parser should be created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "ralph"

    def test_parser_config_option(self):
        """Parser should accept --config option."""
        parser = create_parser()
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_parser_config_short_option(self):
        """Parser should accept -c shorthand."""
        parser = create_parser()
        args = parser.parse_args(["-c", "test.yaml"])
        assert args.config == "test.yaml"

    def test_parser_list_targets_option(self):
        """Parser should accept --list-targets option."""
        parser = create_parser()
        args = parser.parse_args(["--list-targets"])
        assert args.list_targets is True

    def test_parser_list_targets_short_option(self):
        """Parser should accept -l shorthand."""
        parser = create_parser()
        args = parser.parse_args(["-l"])
        assert args.list_targets is True

    def test_parser_validate_option(self):
        """Parser should accept --validate option."""
        parser = create_parser()
        args = parser.parse_args(["--validate"])
        assert args.validate is True

    def test_parser_validate_short_option(self):
        """Parser should accept -v shorthand."""
        parser = create_parser()
        args = parser.parse_args(["-v"])
        assert args.validate is True

    def test_parser_output_dir_option(self):
        """Parser should accept --output-dir option."""
        parser = create_parser()
        args = parser.parse_args(["--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_parser_output_dir_short_option(self):
        """Parser should accept -o shorthand."""
        parser = create_parser()
        args = parser.parse_args(["-o", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_parser_dry_run_option(self):
        """Parser should accept --dry-run option."""
        parser = create_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parser_dry_run_short_option(self):
        """Parser should accept -n shorthand."""
        parser = create_parser()
        args = parser.parse_args(["-n"])
        assert args.dry_run is True

    def test_parser_combined_options(self):
        """Parser should accept multiple options together."""
        parser = create_parser()
        args = parser.parse_args([
            "--config", "test.yaml",
            "--output-dir", "/tmp/out",
            "--dry-run",
        ])
        assert args.config == "test.yaml"
        assert args.output_dir == "/tmp/out"
        assert args.dry_run is True

    def test_parser_no_options_defaults(self):
        """Parser should have correct defaults when no options given."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.config is None
        assert args.list_targets is False
        assert args.validate is False
        assert args.output_dir is None
        assert args.dry_run is False


class TestListTargets:
    """Tests for list_targets function."""

    def test_list_targets_returns_zero(self):
        """list_targets should return 0 exit code."""
        result = list_targets()
        assert result == 0

    def test_list_targets_outputs_to_stdout(self, capsys):
        """list_targets should output to stdout."""
        list_targets()
        captured = capsys.readouterr()
        assert "Target" in captured.out or "No injection targets" in captured.out


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self, capsys):
        """validate_config should return 0 for valid config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "test"
global:
  output_dir: /tmp
scenarios:
  - id: "test1"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 10
    expected_behavior: "delay"
""")
            f.flush()
            result = validate_config(f.name)

        os.unlink(f.name)
        captured = capsys.readouterr()
        assert result == 0
        assert "valid" in captured.out.lower()

    def test_validate_missing_file(self, capsys):
        """validate_config should return 1 for missing file."""
        result = validate_config("/nonexistent/path/config.yaml")
        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.out.lower() or "error" in captured.out.lower()

    def test_validate_invalid_yaml(self, capsys):
        """validate_config should return 1 for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            f.flush()
            result = validate_config(f.name)

        os.unlink(f.name)
        captured = capsys.readouterr()
        assert result == 1

    def test_validate_missing_scenarios(self, capsys):
        """validate_config should return 1 when scenarios missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "test"
""")
            f.flush()
            result = validate_config(f.name)

        os.unlink(f.name)
        captured = capsys.readouterr()
        assert result == 1
        assert "scenario" in captured.out.lower()


class TestMain:
    """Tests for main entry point function."""

    def test_main_no_args_shows_help(self, capsys):
        """main with no args should show help and return 0."""
        result = main([])
        captured = capsys.readouterr()
        assert result == 0
        assert "usage" in captured.out.lower() or "ralph" in captured.out.lower()

    def test_main_help_option(self):
        """main --help should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_version_option(self):
        """main --version should exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_main_list_targets(self, capsys):
        """main --list-targets should return 0."""
        result = main(["--list-targets"])
        assert result == 0

    def test_main_validate_without_config(self, capsys):
        """main --validate without --config should error."""
        result = main(["--validate"])
        captured = capsys.readouterr()
        assert result == 1
        assert "requires --config" in captured.err

    def test_main_validate_with_valid_config(self):
        """main --validate with valid config should return 0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "test"
global:
  output_dir: /tmp
scenarios:
  - id: "test1"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 10
    expected_behavior: "delay"
""")
            f.flush()
            result = main(["--config", f.name, "--validate"])

        os.unlink(f.name)
        assert result == 0

    def test_main_config_dry_run(self, capsys):
        """main --config with --dry-run should show scenarios."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "test_experiment"
  description: "A test"
global:
  output_dir: /tmp
scenarios:
  - id: "test_delay"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 10
    expected_behavior: "delay"
""")
            f.flush()
            result = main(["--config", f.name, "--dry-run"])

        os.unlink(f.name)
        captured = capsys.readouterr()
        assert result == 0
        assert "test_experiment" in captured.out
        assert "test_delay" in captured.out

    def test_main_config_with_output_dir_override(self, capsys):
        """main --config with --output-dir should override output_dir."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "test"
global:
  output_dir: /original/path
scenarios:
  - id: "test1"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    trigger:
      type: "one_shot"
      at_step: 10
    expected_behavior: "delay"
""")
            f.flush()
            result = main([
                "--config", f.name,
                "--output-dir", "/new/path",
                "--dry-run",
            ])

        os.unlink(f.name)
        assert result == 0

    def test_main_config_missing_file(self, capsys):
        """main --config with missing file should return 1."""
        result = main(["--config", "/nonexistent/config.yaml"])
        captured = capsys.readouterr()
        assert result == 1
        assert "error" in captured.err.lower() or "not found" in captured.err.lower()


class TestMainIntegration:
    """Integration tests for main CLI."""

    def test_full_workflow(self, capsys):
        """Test a full CLI workflow: validate then dry-run."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
experiment:
  name: "integration_test"
  description: "Full integration test"
  seed: 42
global:
  enabled: true
  log_level: INFO
  output_dir: /tmp/ralph_test
data_collection:
  stream_a:
    enabled: true
    output:
      path: "${global.output_dir}/telemetry.jsonl"
  stream_b:
    enabled: true
    output:
      path: "${global.output_dir}/labels.jsonl"
scenarios:
  - id: "delay_scenario"
    layer: "L0"
    target: "ray.get"
    fault_type: "delay"
    severity: "medium"
    enabled: true
    trigger:
      type: "one_shot"
      at_step: 10
    parameters:
      delay_seconds: 2.0
    expected_behavior: "ray.get should be delayed by 2 seconds"
  - id: "disabled_scenario"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    fault_type: "delay"
    enabled: false
    trigger:
      type: "periodic"
      every_n_steps: 5
    expected_behavior: "Should not trigger (disabled)"
""")
            f.flush()
            config_path = f.name

        # First validate
        result = main(["--config", config_path, "--validate"])
        captured = capsys.readouterr()
        assert result == 0, f"Validation failed: {captured.out}"

        # Then dry-run
        result = main(["--config", config_path, "--dry-run"])
        captured = capsys.readouterr()
        assert result == 0, f"Dry-run failed: {captured.out}"
        assert "integration_test" in captured.out
        assert "delay_scenario" in captured.out
        assert "[enabled]" in captured.out
        assert "[disabled]" in captured.out

        os.unlink(config_path)


class TestMainModuleExecution:
    """Tests for module execution via python -m."""

    def test_module_is_importable(self):
        """ralph.cli.main should be importable."""
        import ralph.cli.main
        assert hasattr(ralph.cli.main, "main")
        assert hasattr(ralph.cli.main, "create_parser")

    def test_cli_package_exports_main(self):
        """ralph.cli should export main function."""
        from ralph.cli import main as cli_main
        assert callable(cli_main)
