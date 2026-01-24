"""Unit tests for TriggerScheduler."""

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.scheduler import TriggerScheduler


class TestTriggerSchedulerBasics:
    """Test basic TriggerScheduler functionality."""

    def test_init_default(self):
        """Test default initialization."""
        scheduler = TriggerScheduler()
        assert scheduler.get_step() == 0
        assert scheduler.list_configs() == []

    def test_init_with_seed(self):
        """Test initialization with seed."""
        scheduler = TriggerScheduler(seed=42)
        assert scheduler.get_step() == 0

    def test_add_config(self):
        """Test adding a config."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)
        assert "test-1" in scheduler.list_configs()

    def test_add_duplicate_config_raises(self):
        """Test adding duplicate config raises ValueError."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_config(config)

    def test_remove_config(self):
        """Test removing a config."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)
        scheduler.remove_config("test-1")
        assert "test-1" not in scheduler.list_configs()

    def test_remove_nonexistent_config_raises(self):
        """Test removing nonexistent config raises KeyError."""
        scheduler = TriggerScheduler()
        with pytest.raises(KeyError, match="No config"):
            scheduler.remove_config("nonexistent")

    def test_get_config(self):
        """Test getting a config."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)
        retrieved = scheduler.get_config("test-1")
        assert retrieved.id == "test-1"

    def test_get_nonexistent_config_raises(self):
        """Test getting nonexistent config raises KeyError."""
        scheduler = TriggerScheduler()
        with pytest.raises(KeyError, match="No config"):
            scheduler.get_config("nonexistent")

    def test_update_step(self):
        """Test updating the step."""
        scheduler = TriggerScheduler()
        scheduler.update_step(10)
        assert scheduler.get_step() == 10

    def test_clear(self):
        """Test clearing the scheduler."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="test-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)
        scheduler.update_step(10)
        scheduler.clear()
        assert scheduler.list_configs() == []
        assert scheduler.get_step() == 0


class TestOneShotTrigger:
    """Test ONE_SHOT trigger type."""

    def test_one_shot_triggers_at_exact_step(self):
        """Test ONE_SHOT triggers exactly at specified step."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="one-shot-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)

        # Before the step
        scheduler.update_step(4)
        assert not scheduler.should_trigger("one-shot-1")

        # At the step
        scheduler.update_step(5)
        assert scheduler.should_trigger("one-shot-1")

        # After the step
        scheduler.update_step(6)
        assert not scheduler.should_trigger("one-shot-1")

    def test_one_shot_triggers_only_once(self):
        """Test ONE_SHOT only triggers once."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="one-shot-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)

        scheduler.update_step(5)
        assert scheduler.should_trigger("one-shot-1")
        # Second check at same step should return False
        assert not scheduler.should_trigger("one-shot-1")

    def test_one_shot_can_be_reset(self):
        """Test ONE_SHOT can be reset to fire again."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="one-shot-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)

        scheduler.update_step(5)
        assert scheduler.should_trigger("one-shot-1")
        assert not scheduler.should_trigger("one-shot-1")

        scheduler.reset_one_shot("one-shot-1")
        assert scheduler.should_trigger("one-shot-1")


class TestStepBasedTrigger:
    """Test STEP_BASED trigger type."""

    def test_step_based_triggers_in_range(self):
        """Test STEP_BASED triggers within range."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="step-based-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(
                type=TriggerType.STEP_BASED,
                start_step=5,
                end_step=10,
            ),
        )
        scheduler.add_config(config)

        # Before range
        scheduler.update_step(4)
        assert not scheduler.should_trigger("step-based-1")

        # At start of range
        scheduler.update_step(5)
        assert scheduler.should_trigger("step-based-1")

        # In middle of range
        scheduler.update_step(7)
        assert scheduler.should_trigger("step-based-1")

        # At end of range
        scheduler.update_step(10)
        assert scheduler.should_trigger("step-based-1")

        # After range
        scheduler.update_step(11)
        assert not scheduler.should_trigger("step-based-1")


class TestPeriodicTrigger:
    """Test PERIODIC trigger type."""

    def test_periodic_triggers_every_n_steps(self):
        """Test PERIODIC triggers every N steps."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="periodic-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(
                type=TriggerType.PERIODIC,
                every_n_steps=3,
            ),
        )
        scheduler.add_config(config)

        # Step 0 (0 % 3 == 0)
        scheduler.update_step(0)
        assert scheduler.should_trigger("periodic-1")

        # Step 1 (1 % 3 != 0)
        scheduler.update_step(1)
        assert not scheduler.should_trigger("periodic-1")

        # Step 2 (2 % 3 != 0)
        scheduler.update_step(2)
        assert not scheduler.should_trigger("periodic-1")

        # Step 3 (3 % 3 == 0)
        scheduler.update_step(3)
        assert scheduler.should_trigger("periodic-1")

        # Step 6 (6 % 3 == 0)
        scheduler.update_step(6)
        assert scheduler.should_trigger("periodic-1")


class TestProbabilisticTrigger:
    """Test PROBABILISTIC trigger type."""

    def test_probabilistic_with_zero_probability(self):
        """Test PROBABILISTIC never triggers with 0 probability."""
        scheduler = TriggerScheduler(seed=42)
        config = FaultConfig(
            id="prob-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(
                type=TriggerType.PROBABILISTIC,
                probability=0.0,
            ),
        )
        scheduler.add_config(config)

        for step in range(100):
            scheduler.update_step(step)
            assert not scheduler.should_trigger("prob-1")

    def test_probabilistic_with_full_probability(self):
        """Test PROBABILISTIC always triggers with 1.0 probability."""
        scheduler = TriggerScheduler(seed=42)
        config = FaultConfig(
            id="prob-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(
                type=TriggerType.PROBABILISTIC,
                probability=1.0,
            ),
        )
        scheduler.add_config(config)

        for step in range(100):
            scheduler.update_step(step)
            assert scheduler.should_trigger("prob-1")

    def test_probabilistic_is_reproducible_with_seed(self):
        """Test PROBABILISTIC is reproducible with same seed."""
        results1 = []
        results2 = []

        for results in [results1, results2]:
            scheduler = TriggerScheduler(seed=42)
            config = FaultConfig(
                id="prob-1",
                strategy=StrategyType.DELAY,
                trigger=TriggerConfig(
                    type=TriggerType.PROBABILISTIC,
                    probability=0.5,
                ),
            )
            scheduler.add_config(config)

            for step in range(20):
                scheduler.update_step(step)
                results.append(scheduler.should_trigger("prob-1"))

        assert results1 == results2

    def test_probabilistic_fires_approximately_correct_ratio(self):
        """Test PROBABILISTIC fires approximately at the correct ratio."""
        scheduler = TriggerScheduler(seed=12345)
        config = FaultConfig(
            id="prob-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(
                type=TriggerType.PROBABILISTIC,
                probability=0.3,
            ),
        )
        scheduler.add_config(config)

        triggers = 0
        total = 1000
        for step in range(total):
            scheduler.update_step(step)
            if scheduler.should_trigger("prob-1"):
                triggers += 1

        # Allow 10% tolerance
        ratio = triggers / total
        assert 0.2 < ratio < 0.4


class TestDisabledFaults:
    """Test disabled fault handling."""

    def test_disabled_fault_never_triggers(self):
        """Test disabled faults never trigger."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="disabled-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
            enabled=False,
        )
        scheduler.add_config(config)

        scheduler.update_step(5)
        assert not scheduler.should_trigger("disabled-1")


class TestGetActiveFaults:
    """Test get_active_faults method."""

    def test_get_active_faults_returns_all_triggered(self):
        """Test get_active_faults returns all triggered faults."""
        scheduler = TriggerScheduler()

        # Add multiple configs
        config1 = FaultConfig(
            id="fault-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        config2 = FaultConfig(
            id="fault-2",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5),
        )
        config3 = FaultConfig(
            id="fault-3",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10),
        )

        scheduler.add_config(config1)
        scheduler.add_config(config2)
        scheduler.add_config(config3)

        # At step 5, fault-1 and fault-2 should trigger
        scheduler.update_step(5)
        active = scheduler.get_active_faults()
        assert "fault-1" in active
        assert "fault-2" in active
        assert "fault-3" not in active

    def test_get_active_faults_empty_when_none_trigger(self):
        """Test get_active_faults returns empty list when none trigger."""
        scheduler = TriggerScheduler()
        config = FaultConfig(
            id="fault-1",
            strategy=StrategyType.DELAY,
            trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=5),
        )
        scheduler.add_config(config)

        scheduler.update_step(1)
        assert scheduler.get_active_faults() == []
