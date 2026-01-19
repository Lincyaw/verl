#!/usr/bin/env python3
"""CLI wrapper for fault injection.

Usage:
    python -m verl.fault_injection.cli --scenario nccl_timeout.yaml -- python train.py
    python -m verl.fault_injection.cli --list
"""

import argparse
import os
import sys

# Add parent to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(
        description="Inject faults into distributed training",
        usage="verl-fault-inject [options] -- <command>",
    )
    parser.add_argument("--scenario", "-s", help="Path to scenario YAML file")
    parser.add_argument("--list", "-l", action="store_true", help="List available fault types")
    parser.add_argument("--output", "-o", default="./fault_logs", help="Log output directory")
    parser.add_argument("command", nargs="*", help="Command to run")

    args = parser.parse_args()

    if args.list:
        _list_faults()
        return 0

    if not args.scenario:
        parser.error("--scenario is required")

    if not args.command:
        parser.error("No command specified. Use -- to separate options from command.")

    # Import directly to avoid verl/__init__.py which requires ray
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(__file__), "config.py"))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    FaultConfig = config_module.FaultConfig

    spec = importlib.util.spec_from_file_location(
        "orchestrator", os.path.join(os.path.dirname(__file__), "orchestrator.py")
    )
    orchestrator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestrator_module)
    FaultOrchestrator = orchestrator_module.FaultOrchestrator

    config = FaultConfig.from_yaml(args.scenario)
    config.log_output_dir = args.output

    orchestrator = FaultOrchestrator(config)
    return orchestrator.run(args.command)


def _list_faults():
    print("Available fault types:\n")
    categories = {
        "Network": ["network_delay", "network_loss", "network_partition", "port_block"],
        "Process": ["process_kill", "process_exit"],
        "System": ["disk_full", "fd_exhaust", "memory_pressure"],
        "Code": ["code_sleep", "code_oom", "code_nan", "env_modify"],
    }
    for cat, faults in categories.items():
        print(f"  {cat}:")
        for f in faults:
            print(f"    - {f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
