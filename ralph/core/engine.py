"""
Injection engine orchestrator for Ralph fault injection framework.

The InjectionEngine is the central orchestrator that:
- Loads fault configurations
- Installs and manages proxy instances
- Coordinates fault injection across the training loop
"""

import importlib
import random
from typing import TYPE_CHECKING, Any, Callable, Optional

from ralph.core.config import FaultConfig
from ralph.core.registry import ProxyRegistry
from ralph.core.scheduler import TriggerScheduler

if TYPE_CHECKING:
    from ralph.proxies.base import BaseProxy

try:
    from ralph.collectors.dual_stream import DualStreamCollector
except ImportError:
    DualStreamCollector = None


class InjectionEngine:
    """
    Central orchestrator for fault injection.

    The InjectionEngine coordinates all fault injection activities:
    - Loads and validates fault configurations
    - Creates and manages proxy instances
    - Installs proxies by monkey-patching target functions
    - Coordinates step updates across all proxies
    - Cleans up by restoring original functions

    Example:
        collector = DualStreamCollector(output_dir="./logs")
        engine = InjectionEngine(collector)

        # Add fault configurations
        config = FaultConfig(
            id="delay_ray_get",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
            parameters={"delay_seconds": 5.0}
        )
        engine.add_config("ray.get", config)

        # Install and run
        with engine:
            for step in range(100):
                engine.update_step(step)
                # ... training loop ...

    Context Manager Usage:
        with InjectionEngine(collector) as engine:
            engine.add_config("ray.get", config)
            engine.install_proxies()
            # ... run training ...
    """

    def __init__(
        self,
        collector: Optional["DualStreamCollector"] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the injection engine.

        Args:
            collector: Optional DualStreamCollector for recording fault injections.
                      If None, injection recording is disabled but faults still execute.
            seed: Optional random seed for reproducible probabilistic triggering.
        """
        self._collector = collector

        # Configuration: target -> FaultConfig
        self._configs: dict[str, FaultConfig] = {}

        # Proxy instances: target -> proxy instance
        self._proxies: dict[str, BaseProxy] = {}

        # Original functions: target -> original function reference
        self._originals: dict[str, Callable] = {}

        # Monkey-patch references: target -> (module, attr_name)
        self._patch_refs: dict[str, tuple] = {}

        # Scheduler for trigger coordination
        self._scheduler = TriggerScheduler(seed=seed)

        # Random number generator for probabilistic triggers
        self._rng = random.Random(seed)

        # Current training step
        self._current_step = 0

        # Track installation state
        self._installed = False

    def add_config(self, target: str, config: FaultConfig) -> None:
        """
        Add a fault configuration for a target.

        Args:
            target: Fully qualified target function name (e.g., 'ray.get')
            config: FaultConfig specifying the fault to inject

        Raises:
            KeyError: If no proxy is registered for the target
            ValueError: If the strategy is not supported by the proxy
            RuntimeError: If proxies are already installed
        """
        if self._installed:
            raise RuntimeError(
                "Cannot add config after proxies are installed. "
                "Call uninstall_proxies() first or use a new InjectionEngine."
            )

        # Validate target has a registered proxy
        if not ProxyRegistry.is_registered(target):
            raise KeyError(
                f"No proxy registered for target '{target}'. Available targets: {ProxyRegistry.list_targets()}"
            )

        # Validate strategy is supported
        supported = ProxyRegistry.get_supported_strategies(target)
        if config.strategy not in supported:
            supported_str = ", ".join(s.value for s in sorted(supported, key=lambda x: x.value))
            raise ValueError(
                f"Strategy '{config.strategy.value}' not supported for target '{target}'. Supported: [{supported_str}]"
            )

        self._configs[target] = config
        self._scheduler.add_config(config)

    def remove_config(self, target: str) -> None:
        """
        Remove a fault configuration for a target.

        Args:
            target: Fully qualified target function name

        Raises:
            KeyError: If no config exists for the target
            RuntimeError: If proxies are already installed
        """
        if self._installed:
            raise RuntimeError("Cannot remove config after proxies are installed. Call uninstall_proxies() first.")

        if target not in self._configs:
            raise KeyError(f"No config for target '{target}'")

        config = self._configs.pop(target)
        self._scheduler.remove_config(config.id)

    def get_config(self, target: str) -> Optional[FaultConfig]:
        """
        Get the fault configuration for a target.

        Args:
            target: Fully qualified target function name

        Returns:
            FaultConfig for the target or None if not configured
        """
        return self._configs.get(target)

    def list_targets(self) -> list[str]:
        """
        List all configured targets.

        Returns:
            Sorted list of target names with configurations
        """
        return sorted(self._configs.keys())

    def load_config(self, yaml_path: str) -> None:
        """
        Load fault configurations from a YAML file.

        This method parses a YAML configuration file and creates FaultConfig
        objects for each scenario defined in the file.

        Args:
            yaml_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValueError: If the YAML format is invalid
            RuntimeError: If proxies are already installed

        Note:
            This is a placeholder that requires 5.2-yaml-config-loader to be
            implemented. For now, use add_config() to add configurations directly.
        """
        if self._installed:
            raise RuntimeError("Cannot load config after proxies are installed. Call uninstall_proxies() first.")

        # Import yaml here to avoid hard dependency
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for YAML config loading. Install with: pip install pyyaml") from None

        from ralph.core.config import StrategyType, TriggerConfig, TriggerType

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

        # Extract global variables for substitution
        global_vars = data.get("global", {})

        # Parse scenarios
        scenarios = data.get("scenarios", [])
        if not scenarios:
            raise ValueError(f"No scenarios found in {yaml_path}")

        for scenario in scenarios:
            # Validate required fields
            required_fields = ["id", "target", "fault_type", "trigger"]
            for field in required_fields:
                if field not in scenario:
                    raise ValueError(f"Scenario missing required field '{field}': {scenario}")

            # Parse trigger config
            trigger_data = scenario["trigger"]
            trigger_type_str = trigger_data.get("type", "one_shot")
            trigger_type = TriggerType(trigger_type_str)

            trigger_config = TriggerConfig(
                type=trigger_type,
                at_step=trigger_data.get("at_step"),
                start_step=trigger_data.get("start_step"),
                end_step=trigger_data.get("end_step"),
                probability=trigger_data.get("probability", 1.0),
                every_n_steps=trigger_data.get("every_n_steps"),
            )

            # Parse strategy type
            strategy_str = scenario["fault_type"]
            strategy = StrategyType(strategy_str)

            # Parse parameters with variable substitution
            parameters = scenario.get("parameters", {})
            parameters = self._substitute_variables(parameters, global_vars)

            # Create FaultConfig
            config = FaultConfig(
                id=scenario["id"],
                strategy=strategy,
                trigger=trigger_config,
                parameters=parameters,
                enabled=scenario.get("enabled", True),
                severity=scenario.get("severity", "medium"),
                expected_behavior=scenario.get("expected_behavior", ""),
            )

            # Add to engine
            target = scenario["target"]
            self.add_config(target, config)

    def _substitute_variables(self, obj: Any, variables: dict[str, Any]) -> Any:
        """
        Recursively substitute ${var} references in config objects.

        Args:
            obj: Object to process (dict, list, or scalar)
            variables: Dictionary of variable values

        Returns:
            Object with variables substituted
        """
        if isinstance(obj, str):
            # Handle ${global.key} or ${key} patterns
            import re

            pattern = r"\$\{(?:global\.)?(\w+)\}"

            def replacer(match):
                key = match.group(1)
                return str(variables.get(key, match.group(0)))

            return re.sub(pattern, replacer, obj)
        elif isinstance(obj, dict):
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_variables(item, variables) for item in obj]
        else:
            return obj

    def install_proxies(self) -> None:
        """
        Install proxy instances by monkey-patching target functions.

        For each configured target, this method:
        1. Gets the proxy class from the registry
        2. Resolves the target function from its fully qualified name
        3. Creates a proxy instance wrapping the original function
        4. Monkey-patches the original with the proxy

        Raises:
            RuntimeError: If proxies are already installed
            ImportError: If target module cannot be imported
            AttributeError: If target function doesn't exist in module
        """
        if self._installed:
            raise RuntimeError("Proxies are already installed")

        for target, config in self._configs.items():
            # Get proxy class
            proxy_class = ProxyRegistry.get_proxy(target)

            # Resolve the target function
            original_fn, module, attr_name = self._resolve_target(target)

            # Create proxy instance
            proxy = proxy_class(original_fn, self._collector)
            proxy.set_config(config)
            proxy.set_step(self._current_step)
            proxy.set_rng(self._rng)

            # Store references
            self._originals[target] = original_fn
            self._proxies[target] = proxy
            self._patch_refs[target] = (module, attr_name)

            # Monkey-patch
            setattr(module, attr_name, proxy)

        self._installed = True

    def _resolve_target(self, target: str) -> tuple:
        """
        Resolve a target string to actual function and module references.

        Args:
            target: Fully qualified target name like 'ray.get' or
                   'torch.distributed.all_reduce'

        Returns:
            Tuple of (original_function, module, attribute_name)

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If function doesn't exist
        """
        # Handle different naming patterns
        if ".__call__" in target:
            # Class method pattern: 'ClassName.__call__'
            # For now, treat as the class itself
            parts = target.replace(".__call__", "").rsplit(".", 1)
        else:
            parts = target.rsplit(".", 1)

        if len(parts) == 1:
            raise ValueError(f"Target '{target}' must be fully qualified (e.g., 'module.function')")

        module_path, attr_name = parts

        # Try to import the module
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Try parent module for nested attributes
            parent_parts = module_path.rsplit(".", 1)
            if len(parent_parts) == 2:
                parent_module_path, class_name = parent_parts
                try:
                    parent_module = importlib.import_module(parent_module_path)
                    module = getattr(parent_module, class_name)
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Cannot import target '{target}': {e}") from e
            else:
                raise ImportError(f"Cannot import module '{module_path}' for target '{target}'") from None

        # Get the original function
        if not hasattr(module, attr_name):
            raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'")

        original_fn = getattr(module, attr_name)
        return original_fn, module, attr_name

    def uninstall_proxies(self) -> None:
        """
        Uninstall proxies and restore original functions.

        This method reverses the monkey-patching performed by install_proxies(),
        restoring all target functions to their original implementations.
        """
        if not self._installed:
            return

        for target in list(self._proxies.keys()):
            if target in self._patch_refs and target in self._originals:
                module, attr_name = self._patch_refs[target]
                original_fn = self._originals[target]
                setattr(module, attr_name, original_fn)

        # Clear state
        self._proxies.clear()
        self._originals.clear()
        self._patch_refs.clear()
        self._installed = False

    def update_step(self, step: int) -> None:
        """
        Update the current training step on all proxies.

        This should be called at the beginning of each training step to
        ensure trigger conditions are evaluated correctly.

        Args:
            step: The current training step number
        """
        self._current_step = step
        self._scheduler.update_step(step)

        for proxy in self._proxies.values():
            proxy.set_step(step)

    def get_step(self) -> int:
        """
        Get the current training step.

        Returns:
            The current step number
        """
        return self._current_step

    def is_installed(self) -> bool:
        """
        Check if proxies are currently installed.

        Returns:
            True if proxies are installed, False otherwise
        """
        return self._installed

    def get_active_faults(self) -> list[str]:
        """
        Get list of faults that would trigger at the current step.

        Returns:
            List of fault config IDs that would trigger
        """
        return self._scheduler.get_active_faults()

    def get_proxy(self, target: str) -> Optional["BaseProxy"]:
        """
        Get the proxy instance for a target.

        Args:
            target: Fully qualified target name

        Returns:
            The proxy instance or None if not installed
        """
        return self._proxies.get(target)

    def get_collector(self) -> Optional["DualStreamCollector"]:
        """
        Get the collector instance.

        Returns:
            The DualStreamCollector or None if not configured
        """
        return self._collector

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducible fault injection.

        Args:
            seed: Random seed value
        """
        self._rng = random.Random(seed)
        self._scheduler.set_seed(seed)

        # Update all installed proxies
        for proxy in self._proxies.values():
            proxy.set_rng(self._rng)

    def __enter__(self) -> "InjectionEngine":
        """
        Context manager entry.

        Note: Does NOT automatically install proxies.
        Call install_proxies() explicitly after adding configs.

        Returns:
            The InjectionEngine instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit.

        Automatically uninstalls proxies and flushes the collector.
        """
        self.uninstall_proxies()

        if self._collector:
            self._collector.flush()

    def __repr__(self) -> str:
        """String representation of the engine state."""
        targets = self.list_targets()
        return f"InjectionEngine(targets={targets}, installed={self._installed}, step={self._current_step})"
