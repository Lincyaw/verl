#!/usr/bin/env python3
"""Test script for UI layer fault injectors."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.fault_injection import (
    FaultInjectionConfig,
    FaultOrchestrator,
    FaultContext,
    FaultLayer,
    FaultType,
    UIFaultConfig,
)
from verl.fault_injection.injectors import (
    HydraConfigErrorInjector,
    RayInitFailureInjector,
    CLIArgErrorInjector,
    EnvVarErrorInjector,
    UIFreezeInjector,
)


def test_hydra_config_error():
    """Test Hydra configuration error injector."""
    print("\n=== Testing HydraConfigErrorInjector ===")

    config = UIFaultConfig(
        name="test_hydra_error",
        layer=FaultLayer.UI,
        type=FaultType.HYDRA_CONFIG_ERROR,
        enabled=True,
        hydra_error_type="validate",
        description="Test Hydra config validation error"
    )

    injector = HydraConfigErrorInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"Injection result: {result}")

    assert result.status.value == "FAILED"
    assert "validation failed" in result.error_message
    print("✓ Hydra config error test passed")


def test_ray_init_failure():
    """Test Ray initialization failure injector."""
    print("\n=== Testing RayInitFailureInjector ===")

    config = UIFaultConfig(
        name="test_ray_init",
        layer=FaultLayer.UI,
        type=FaultType.RAY_INIT_FAILURE,
        enabled=True,
        ray_init_error="Simulated Ray init failure",
        description="Test Ray initialization failure"
    )

    injector = RayInitFailureInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"Injection result: {result}")

    assert result.status.value == "SUCCESS"
    assert result.metadata["error_message"] == "Simulated Ray init failure"

    # Test recovery
    injector.recover(context)
    print("✓ Ray init failure test passed")


def test_cli_arg_error():
    """Test CLI argument error injector."""
    print("\n=== Testing CLIArgErrorInjector ===")

    # Save original argv
    original_argv = sys.argv.copy()

    config = UIFaultConfig(
        name="test_cli_arg",
        layer=FaultLayer.UI,
        type=FaultType.CLI_ARG_ERROR,
        enabled=True,
        cli_arg_missing="--config-path",
        description="Test missing CLI argument"
    )

    injector = CLIArgErrorInjector(config)
    context = FaultContext()

    # Add the argument first
    sys.argv.append("--config-path")
    sys.argv.append("config")

    result = injector.inject(context)
    print(f"Injection result: {result}")

    assert result.status.value == "SUCCESS"
    assert "--config-path" not in sys.argv

    # Restore original argv
    injector.recover(context)
    assert sys.argv == original_argv
    print("✓ CLI arg error test passed")


def test_env_var_error():
    """Test environment variable error injector."""
    print("\n=== Testing EnvVarErrorInjector ===")

    config = UIFaultConfig(
        name="test_env_var",
        layer=FaultLayer.UI,
        type=FaultType.ENV_VAR_ERROR,
        enabled=True,
        env_var_unset="TEST_VAR",
        description="Test unset environment variable"
    )

    # Set test variable
    os.environ["TEST_VAR"] = "test_value"

    injector = EnvVarErrorInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"Injection result: {result}")

    assert result.status.value == "SUCCESS"
    assert "TEST_VAR" not in os.environ

    # Test recovery
    injector.recover(context)
    assert os.environ.get("TEST_VAR") == "test_value"
    print("✓ Env var error test passed")


def test_ui_freeze():
    """Test UI freeze injector."""
    print("\n=== Testing UIFreezeInjector ===")

    config = UIFaultConfig(
        name="test_ui_freeze",
        layer=FaultLayer.UI,
        type=FaultType.UI_FREEZE,
        enabled=True,
        freeze_duration_seconds=2.0,
        description="Test UI freeze"
    )

    injector = UIFreezeInjector(config)
    context = FaultContext()

    start_time = time.time()
    result = injector.inject(context)
    end_time = time.time()

    print(f"Injection result: {result}")
    print(f"Freeze duration: {end_time - start_time:.2f} seconds")

    assert result.status.value == "SUCCESS"
    assert abs((end_time - start_time) - 2.0) < 0.5  # Allow some tolerance
    print("✓ UI freeze test passed")


def test_configuration_loading():
    """Test loading UI fault configuration."""
    print("\n=== Testing Configuration Loading ===")

    # Create a temporary config file
    config_yaml = """
enabled: true
global_cooldown_seconds: 10.0
max_concurrent_faults: 2
faults:
  - name: "test_ray_init"
    layer: "ui"
    type: "ray_init_failure"
    enabled: true
    ray_init_error: "Test error"
    trigger:
      type: "immediate"
    target:
      mode: "all"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    try:
        # Load configuration
        config = FaultInjectionConfig.from_yaml(config_path)
        print(f"Loaded config: {config}")

        # Create orchestrator
        orchestrator = FaultOrchestrator(config)
        print(f"Created orchestrator with {len(config.faults)} faults")

        # Test fault injection
        result = orchestrator.inject_fault("test_ray_init")
        print(f"Fault injection result: {result}")

        assert result is not None
        print("✓ Configuration loading test passed")

    finally:
        os.unlink(config_path)


def main():
    """Run all UI fault injector tests."""
    print("Running UI Layer Fault Injector Tests")
    print("=" * 50)

    try:
        test_hydra_config_error()
        test_ray_init_failure()
        test_cli_arg_error()
        test_env_var_error()
        test_ui_freeze()
        test_configuration_loading()

        print("\n" + "=" * 50)
        print("✅ All UI fault injector tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()