# Copyright 2026 Aoyang Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python3
"""Simple test for UI fault injectors without full verl imports."""

import os
import sys
import time
from pathlib import Path

# Add the fault injection module to path directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import only what we need
from verl.fault_injection.base import FaultContext, FaultStatus
from verl.fault_injection.config import FaultLayer, FaultType, UIFaultConfig
from verl.fault_injection.injectors.ui import (
    CLIArgErrorInjector,
    EnvVarErrorInjector,
    HydraConfigErrorInjector,
    RayInitFailureInjector,
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
        description="Test Hydra config validation error",
    )

    injector = HydraConfigErrorInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"Injection result status: {result.status}")
    print(f"Error message: {result.error_message}")

    assert result.status == FaultStatus.FAILED
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
        description="Test Ray initialization failure",
    )

    injector = RayInitFailureInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"Injection result status: {result.status}")
    print(f"Metadata: {result.metadata}")

    assert result.status == FaultStatus.SUCCESS
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
        description="Test missing CLI argument",
    )

    injector = CLIArgErrorInjector(config)
    context = FaultContext()

    # Add the argument first
    sys.argv.append("--config-path")
    sys.argv.append("config")

    print(f"Original argv: {sys.argv}")
    result = injector.inject(context)
    print(f"After injection argv: {sys.argv}")
    print(f"Injection result: {result}")

    assert result.status == FaultStatus.SUCCESS
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
        description="Test unset environment variable",
    )

    # Set test variable
    os.environ["TEST_VAR"] = "test_value"
    print(f"Original TEST_VAR: {os.environ.get('TEST_VAR')}")

    injector = EnvVarErrorInjector(config)
    context = FaultContext()

    result = injector.inject(context)
    print(f"After injection TEST_VAR: {os.environ.get('TEST_VAR')}")
    print(f"Injection result: {result}")

    assert result.status == FaultStatus.SUCCESS
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
        freeze_duration_seconds=1.0,  # Short duration for testing
        description="Test UI freeze",
    )

    injector = UIFreezeInjector(config)
    context = FaultContext()

    start_time = time.time()
    result = injector.inject(context)
    end_time = time.time()

    print(f"Injection result: {result}")
    print(f"Freeze duration: {end_time - start_time:.2f} seconds")

    assert result.status == FaultStatus.SUCCESS
    assert abs((end_time - start_time) - 1.0) < 0.5  # Allow some tolerance
    print("✓ UI freeze test passed")


def main():
    """Run all UI fault injector tests."""
    print("Running UI Layer Fault Injector Tests (Simplified)")
    print("=" * 60)

    try:
        test_hydra_config_error()
        test_ray_init_failure()
        test_cli_arg_error()
        test_env_var_error()
        test_ui_freeze()

        print("\n" + "=" * 60)
        print("✅ All UI fault injector tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
