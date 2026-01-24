"""
Command-line interface for Ralph fault injection framework.

Provides entry points for running fault injection experiments,
listing available injection targets, and validating configurations.

Usage:
    python -m ralph.cli.main --help
    python -m ralph.cli.main --list-targets
    python -m ralph.cli.main --config config.yaml --validate
    python -m ralph.cli.main --config config.yaml --output-dir ./results
"""

import argparse
import sys
from typing import List, Optional


def _import_proxies() -> None:
    """
    Import proxy modules to ensure they register with ProxyRegistry.

    This function imports all proxy modules so their @ProxyRegistry.register
    decorators execute and populate the registry.
    """
    try:
        # Import proxy modules to trigger registration
        import ralph.proxies.l0_ray  # noqa: F401
        import ralph.proxies.l1_distributed  # noqa: F401
        import ralph.proxies.l2_verl  # noqa: F401
    except ImportError:
        # Some proxies may have optional dependencies (torch, ray)
        pass


def list_targets() -> int:
    """
    Print all registered injection targets and their supported strategies.

    Returns:
        Exit code (0 for success)
    """
    from ralph.core.registry import ProxyRegistry

    _import_proxies()

    targets = ProxyRegistry.list_targets()

    if not targets:
        print("No injection targets registered.")
        print("\nNote: Ensure proxy modules are importable (torch, ray may be required).")
        return 0

    print("Registered Injection Targets:")
    print("=" * 60)

    for target in targets:
        strategies = ProxyRegistry.get_supported_strategies(target)
        strategy_names = sorted([s.value for s in strategies])
        print(f"\n{target}")
        print(f"  Strategies: {', '.join(strategy_names)}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(targets)} targets registered")
    return 0


def validate_config(config_path: str) -> int:
    """
    Validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Exit code (0 for valid, 1 for invalid)
    """
    from ralph.core.config import validate_yaml_config

    print(f"Validating: {config_path}")

    is_valid, errors = validate_yaml_config(config_path)

    if is_valid:
        print("✓ Configuration is valid.")
        return 0
    else:
        print("✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
        return 1


def run_with_config(
    config_path: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """
    Load configuration and prepare the injection engine.

    Args:
        config_path: Path to the YAML configuration file
        output_dir: Optional override for output directory
        dry_run: If True, only validate and show what would be done

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from ralph.core.config import load_yaml_config

    _import_proxies()

    try:
        config = load_yaml_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Override output_dir if specified
    if output_dir:
        config.global_config.output_dir = output_dir

    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Scenarios: {len(config.scenarios)}")

    if dry_run:
        print("\nDry run - scenarios that would be configured:")
        for scenario in config.scenarios:
            status = "enabled" if scenario.enabled else "disabled"
            print(f"  [{status}] {scenario.id}: {scenario.strategy.value} on {scenario.target}")
        return 0

    # In actual usage, the user would integrate with their training loop
    # The CLI just prepares and validates the configuration
    print("\nConfiguration loaded successfully.")
    print("To use Ralph in your training script:")
    print("  from ralph.core import InjectionEngine, load_yaml_config")
    print(f"  config = load_yaml_config('{config_path}')")
    print("  engine = InjectionEngine(collector)")
    print("  for scenario in config.scenarios:")
    print("      engine.add_config(scenario.target, scenario)")
    print("  engine.install_proxies()")
    print("  # ... your training loop ...")
    print("  engine.uninstall_proxies()")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="ralph",
        description="Ralph Fault Injection Framework for RLHF Training",
        epilog=(
            "Examples:\n"
            "  %(prog)s --list-targets\n"
            "  %(prog)s --config config.yaml --validate\n"
            "  %(prog)s --config config.yaml --output-dir ./results\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        metavar="YAML_PATH",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--list-targets",
        "-l",
        action="store_true",
        help="Print all registered injection targets and exit",
    )

    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate configuration without running",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        metavar="DIR",
        help="Override output directory from config",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be done without executing",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Ralph CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --list-targets
    if args.list_targets:
        return list_targets()

    # Handle --validate (requires --config)
    if args.validate:
        if not args.config:
            print("Error: --validate requires --config", file=sys.stderr)
            return 1
        return validate_config(args.config)

    # Handle --config
    if args.config:
        return run_with_config(
            config_path=args.config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )

    # No action specified - show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
