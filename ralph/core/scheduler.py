"""
TriggerScheduler for managing fault injection triggers.

Provides a higher-level interface for managing multiple fault configs
and their trigger scheduling across training steps.
"""

import random
from typing import Optional

from .config import FaultConfig, TriggerType


class TriggerScheduler:
    """
    Manages fault injection triggers across multiple configurations.

    The scheduler tracks which faults should be triggered at each step,
    handles one-shot trigger expiration, and provides a unified interface
    for checking and managing multiple fault configurations.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the TriggerScheduler.

        Args:
            seed: Optional random seed for reproducible probabilistic triggers.
        """
        self._configs: dict[str, FaultConfig] = {}
        self._current_step: int = 0
        self._triggered_one_shots: set[str] = set()
        self._rng = random.Random(seed)

    def add_config(self, config: FaultConfig) -> None:
        """
        Add a fault configuration to the scheduler.

        Args:
            config: The FaultConfig to add.

        Raises:
            ValueError: If a config with the same ID already exists.
        """
        if config.id in self._configs:
            raise ValueError(f"Config with id '{config.id}' already exists")
        self._configs[config.id] = config

    def remove_config(self, config_id: str) -> None:
        """
        Remove a fault configuration from the scheduler.

        Args:
            config_id: The ID of the config to remove.

        Raises:
            KeyError: If no config with the given ID exists.
        """
        if config_id not in self._configs:
            raise KeyError(f"No config with id '{config_id}'")
        del self._configs[config_id]
        self._triggered_one_shots.discard(config_id)

    def get_config(self, config_id: str) -> FaultConfig:
        """
        Get a fault configuration by ID.

        Args:
            config_id: The ID of the config to retrieve.

        Returns:
            The FaultConfig with the given ID.

        Raises:
            KeyError: If no config with the given ID exists.
        """
        if config_id not in self._configs:
            raise KeyError(f"No config with id '{config_id}'")
        return self._configs[config_id]

    def list_configs(self) -> list[str]:
        """
        List all registered config IDs.

        Returns:
            Sorted list of config IDs.
        """
        return sorted(self._configs.keys())

    def update_step(self, step: int) -> None:
        """
        Update the current training step.

        Args:
            step: The new current step.
        """
        self._current_step = step

    def get_step(self) -> int:
        """
        Get the current training step.

        Returns:
            The current step.
        """
        return self._current_step

    def should_trigger(self, config_id: str) -> bool:
        """
        Check if a specific fault should trigger at the current step.

        Args:
            config_id: The ID of the fault config to check.

        Returns:
            True if the fault should trigger, False otherwise.

        Raises:
            KeyError: If no config with the given ID exists.
        """
        config = self.get_config(config_id)
        return self._check_trigger(config)

    def get_active_faults(self) -> list[str]:
        """
        Get all fault IDs that should trigger at the current step.

        Returns:
            List of config IDs for faults that should trigger.
        """
        active = []
        for config_id, config in self._configs.items():
            if self._check_trigger(config):
                active.append(config_id)
        return active

    def _check_trigger(self, config: FaultConfig) -> bool:
        """
        Internal method to check if a fault should trigger.

        Handles enabled/disabled state and one-shot expiration.

        Args:
            config: The FaultConfig to check.

        Returns:
            True if the fault should trigger, False otherwise.
        """
        if not config.enabled:
            return False

        trigger = config.trigger

        # Handle one-shot triggers that have already fired
        if trigger.type == TriggerType.ONE_SHOT:
            if config.id in self._triggered_one_shots:
                return False
            if trigger.should_trigger(self._current_step, self._rng):
                self._triggered_one_shots.add(config.id)
                return True
            return False

        # For other trigger types, delegate to TriggerConfig
        return trigger.should_trigger(self._current_step, self._rng)

    def reset_one_shot(self, config_id: str) -> None:
        """
        Reset a one-shot trigger so it can fire again.

        Args:
            config_id: The ID of the config to reset.
        """
        self._triggered_one_shots.discard(config_id)

    def reset_all_one_shots(self) -> None:
        """
        Reset all one-shot triggers so they can fire again.
        """
        self._triggered_one_shots.clear()

    def clear(self) -> None:
        """
        Clear all configurations and reset scheduler state.
        """
        self._configs.clear()
        self._triggered_one_shots.clear()
        self._current_step = 0

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for probabilistic triggers.

        Args:
            seed: The random seed to use.
        """
        self._rng = random.Random(seed)
