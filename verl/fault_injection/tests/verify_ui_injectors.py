#!/usr/bin/env python3
"""Verify UI fault injectors are properly registered."""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import directly
from verl.fault_injection.base import FaultInjectorRegistry
from verl.fault_injection.config import FaultType

def main():
    print("Checking UI fault injector registrations...")
    print("=" * 60)

    # Get all registered fault types
    registered_types = FaultInjectorRegistry.list_fault_types()

    print(f"Total registered fault types: {len(registered_types)}")
    print("\nRegistered fault types:")
    for fault_type in sorted(registered_types, key=lambda x: x.value):
        print(f"  - {fault_type.value}")

    # Check specifically for UI fault types
    ui_fault_types = [
        FaultType.HYDRA_CONFIG_ERROR,
        FaultType.RAY_INIT_FAILURE,
        FaultType.CLI_ARG_ERROR,
        FaultType.ENV_VAR_ERROR,
        FaultType.UI_FREEZE,
        FaultType.UI_CRASH,
    ]

    print("\n" + "=" * 60)
    print("UI Layer Fault Types Registration Status:")
    print("=" * 60)

    all_ui_registered = True
    for fault_type in ui_fault_types:
        is_registered = fault_type in registered_types
        status = "✅ REGISTERED" if is_registered else "❌ NOT REGISTERED"
        print(f"{fault_type.value:<25} {status}")
        if not is_registered:
            all_ui_registered = False

    print("\n" + "=" * 60)
    if all_ui_registered:
        print("✅ All UI layer fault injectors are properly registered!")
    else:
        print("❌ Some UI layer fault injectors are missing!")
        sys.exit(1)

if __name__ == "__main__":
    main()