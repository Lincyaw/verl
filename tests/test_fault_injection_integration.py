#!/usr/bin/env python3
"""Test script to verify fault injection integration with verl main flow."""

import os
import sys
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_fault_injection_integration():
    """Test that fault injection can be integrated with verl main flow."""
    print("Testing fault injection integration...")

    # Create a minimal test configuration
    test_config = {
        'fault_injection': {
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
                'enabled': False  # Disable recovery for testing
            },
            'monitoring': {
                'enabled': False  # Disable monitoring for testing
            }
        }
    }

    # Test 1: Import the main module
    try:
        from verl.trainer.main_ppo import FAULT_INJECTION_AVAILABLE
        print(f"✓ Fault injection available: {FAULT_INJECTION_AVAILABLE}")
    except ImportError as e:
        print(f"✗ Failed to import main_ppo: {e}")
        return False

    # Test 2: Test configuration loading
    try:
        from verl.fault_injection import FaultInjectionConfig
        config = FaultInjectionConfig.from_dict(test_config['fault_injection'])
        print(f"✓ Configuration loaded successfully: {config.enabled}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

    # Test 3: Test fault injection hooks creation
    try:
        from verl.fault_injection import FaultOrchestrator, create_fault_injection_hooks
        orchestrator = FaultOrchestrator(config)
        hooks = create_fault_injection_hooks(orchestrator)
        print(f"✓ Fault injection hooks created successfully")
    except Exception as e:
        print(f"✗ Failed to create hooks: {e}")
        return False

    # Test 4: Test RayTrainer accepts fault_orchestrator parameter
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        import inspect
        sig = inspect.signature(RayPPOTrainer.__init__)
        has_fault_param = 'fault_orchestrator' in sig.parameters
        print(f"✓ RayPPOTrainer accepts fault_orchestrator: {has_fault_param}")
        if not has_fault_param:
            return False
    except Exception as e:
        print(f"✗ Failed to check RayPPOTrainer: {e}")
        return False

    print("\n✓ All integration tests passed!")
    return True


def test_minimal_invasiveness():
    """Test that the integration is minimally invasive."""
    print("\nTesting minimal invasiveness...")

    # Check that fault injection is optional
    try:
        from verl.trainer.main_ppo import run_ppo, FAULT_INJECTION_AVAILABLE
        # The code should work even if fault injection is not available
        print(f"✓ Fault injection is optional (available: {FAULT_INJECTION_AVAILABLE})")
    except Exception as e:
        print(f"✗ Fault injection is not optional: {e}")
        return False

    # Check that the default configuration has fault injection disabled
    try:
        config_path = Path(__file__).parent.parent / "verl" / "trainer" / "config" / "optional" / "fault_injection.yaml"
        with open(config_path) as f:
            content = f.read()

        if 'enabled: false' not in content:
            print("✗ Default configuration has fault injection enabled")
            return False
        print("✓ Default configuration has fault injection disabled")
    except Exception as e:
        print(f"✗ Failed to check default config: {e}")
        return False

    print("\n✓ Minimal invasiveness tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("verl Fault Injection Integration Test")
    print("=" * 60)

    success = True
    success &= test_fault_injection_integration()
    success &= test_minimal_invasiveness()

    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed! Fault injection integration is working.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed. Please check the implementation.")
        print("=" * 60)
        sys.exit(1)