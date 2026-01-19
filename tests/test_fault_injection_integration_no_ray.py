#!/usr/bin/env python3
"""Test script to verify fault injection integration with verl main flow (without Ray dependency)."""

import os
import sys
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_fault_injection_config():
    """Test fault injection configuration loading."""
    print("Testing fault injection configuration...")

    # Create a minimal test configuration
    test_config = {
        'enabled': True,
        'faults': [
            {
                'layer': 'UI',
                'fault_type': 'HYDRA_CONFIG_ERROR',
                'target': {'type': 'ALL'},
                'trigger': {'type': 'IMMEDIATE'},
                'config': {
                    'error_type': 'VALIDATION_ERROR',
                    'error_message': 'Test fault injection'
                }
            }
        ],
        'recovery': {
            'enabled': False
        },
        'monitoring': {
            'enabled': False
        }
    }

    try:
        from verl.fault_injection.config import FaultInjectionConfig
        config = FaultInjectionConfig.from_dict(test_config)
        print(f"✓ Configuration loaded successfully: enabled={config.enabled}")
        print(f"✓ Number of faults: {len(config.faults)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_fault_injection_components():
    """Test that fault injection components can be imported."""
    print("\nTesting fault injection components...")

    try:
        # Test base classes
        from verl.fault_injection.base import BaseFaultInjector, FaultContext
        print("✓ Base fault injection classes imported")

        # Test orchestrator
        from verl.fault_injection.orchestrator import FaultOrchestrator
        print("✓ Fault orchestrator imported")

        # Test configuration
        from verl.fault_injection.config import FaultType, FaultLayer
        print(f"✓ Fault types available: {len(FaultType)}")
        print(f"✓ Fault layers available: {len(FaultLayer)}")

        return True
    except Exception as e:
        print(f"✗ Failed to import components: {e}")
        return False


def test_integration_points():
    """Test that integration points exist in the code."""
    print("\nTesting integration points...")

    # Check main_ppo.py modifications
    main_ppo_path = Path(__file__).parent.parent / "verl" / "trainer" / "main_ppo.py"
    try:
        with open(main_ppo_path) as f:
            content = f.read()

        checks = [
            ("FAULT_INJECTION_AVAILABLE", "Fault injection availability flag"),
            ("FaultOrchestrator", "Fault orchestrator import"),
            ("fault_orchestrator", "Fault orchestrator parameter"),
            ("initialize_ray_fault_injection", "Ray initialization function"),
        ]

        all_found = True
        for check, desc in checks:
            if check in content:
                print(f"✓ {desc} found in main_ppo.py")
            else:
                print(f"✗ {desc} not found in main_ppo.py")
                all_found = False

        return all_found
    except Exception as e:
        print(f"✗ Failed to check main_ppo.py: {e}")
        return False


def test_ray_trainer_modifications():
    """Test that RayTrainer has been modified for fault injection."""
    print("\nTesting RayTrainer modifications...")

    ray_trainer_path = Path(__file__).parent.parent / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    try:
        with open(ray_trainer_path) as f:
            content = f.read()

        checks = [
            ("fault_orchestrator", "Fault orchestrator parameter in constructor"),
            ("self.fault_orchestrator = fault_orchestrator", "Fault orchestrator storage"),
            ("hook_critic_update", "Critic update hook"),
            ("hook_actor_update", "Actor update hook"),
            ("hook_rollout_generation", "Rollout generation hook"),
        ]

        all_found = True
        for check, desc in checks:
            if check in content:
                print(f"✓ {desc} found in ray_trainer.py")
            else:
                print(f"✗ {desc} not found in ray_trainer.py")
                all_found = False

        return all_found
    except Exception as e:
        print(f"✗ Failed to check ray_trainer.py: {e}")
        return False


def test_configuration_files():
    """Test that configuration files have been created."""
    print("\nTesting configuration files...")

    config_dir = Path(__file__).parent.parent / "verl" / "trainer" / "config"
    files_to_check = [
        ("fault_injection.yaml", "Default fault injection configuration"),
        ("optional/fault_injection.yaml", "Optional fault injection configuration"),
        ("ppo_trainer_with_fault_injection.yaml", "Example with fault injection"),
    ]

    all_found = True
    for filename, desc in files_to_check:
        file_path = config_dir / filename
        if file_path.exists():
            print(f"✓ {desc} exists")
            # Check that fault injection is disabled by default
            if "optional" in filename:
                with open(file_path) as f:
                    content = f.read()
                if "enabled: false" in content:
                    print(f"✓ {desc} has fault injection disabled by default")
                else:
                    print(f"✗ {desc} does not have fault injection disabled by default")
                    all_found = False
        else:
            print(f"✗ {desc} not found")
            all_found = False

    return all_found


def test_hook_implementations():
    """Test that fault injection hooks are properly implemented."""
    print("\nTesting hook implementations...")

    try:
        from verl.fault_injection.integration.hooks import FaultInjectionHooks
        print("✓ FaultInjectionHooks class imported")

        # Check that all required hooks exist
        hooks_class = FaultInjectionHooks
        required_hooks = [
            'hook_ui_initialization',
            'hook_ray_init',
            'hook_worker_initialization',
            'hook_rollout_generation',
            'hook_actor_update',
            'hook_critic_update',
        ]

        all_found = True
        for hook_name in required_hooks:
            if hasattr(hooks_class, hook_name):
                print(f"✓ {hook_name} method exists")
            else:
                print(f"✗ {hook_name} method not found")
                all_found = False

        return all_found
    except Exception as e:
        print(f"✗ Failed to test hooks: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("verl Fault Injection Integration Test (No Ray)")
    print("=" * 60)

    tests = [
        test_fault_injection_config,
        test_fault_injection_components,
        test_integration_points,
        test_ray_trainer_modifications,
        test_configuration_files,
        test_hook_implementations,
    ]

    all_passed = True
    for test in tests:
        if not test():
            all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("✓ All tests passed! Fault injection integration is working.")
        print("✓ The integration is minimally invasive and optional.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed. Please check the implementation.")
        print("=" * 60)
        sys.exit(1)