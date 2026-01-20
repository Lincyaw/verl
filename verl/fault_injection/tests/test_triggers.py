# Copyright 2026 Individual Contributor: Aoyang Fang
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


"""
Unit tests for fault injection triggers.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from verl.fault_injection.triggers import (
    ConditionalTrigger,
    CountTrigger,
    ImmediateTrigger,
    LogPatternTrigger,
    MetricThresholdTrigger,
    ProbabilisticTrigger,
    TimedTrigger,
    TriggerManager,
)


class TestImmediateTrigger(unittest.TestCase):
    """Test cases for ImmediateTrigger."""

    def test_immediate_trigger(self):
        """Test immediate trigger always fires."""
        trigger = ImmediateTrigger()

        # Should always trigger
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())

    def test_reset(self):
        """Test reset method (no-op for immediate trigger)."""
        trigger = ImmediateTrigger()
        trigger.reset()
        self.assertTrue(trigger.should_trigger())


class TestTimedTrigger(unittest.TestCase):
    """Test cases for TimedTrigger."""

    def test_timed_trigger_delay(self):
        """Test timed trigger with delay."""
        delay = 0.1
        trigger = TimedTrigger(delay_seconds=delay)

        # Should not trigger immediately
        self.assertFalse(trigger.should_trigger())

        # Wait for delay
        time.sleep(delay)

        # Should trigger after delay
        self.assertTrue(trigger.should_trigger())

    def test_timed_trigger_with_start_time(self):
        """Test timed trigger with specific start time."""
        start_time = time.time() + 0.05  # Start in 50ms
        trigger = TimedTrigger(delay_seconds=0.1, start_time=start_time)

        # Should not trigger before start time
        self.assertFalse(trigger.should_trigger())

        # Wait until after start time but before delay
        time.sleep(0.08)
        self.assertFalse(trigger.should_trigger())

        # Wait until after delay
        time.sleep(0.05)
        self.assertTrue(trigger.should_trigger())

    def test_reset(self):
        """Test reset method."""
        trigger = TimedTrigger(delay_seconds=0.1)

        # Wait for trigger to activate
        time.sleep(0.1)
        self.assertTrue(trigger.should_trigger())

        # Reset
        trigger.reset()

        # Should not trigger immediately after reset
        self.assertFalse(trigger.should_trigger())


class TestCountTrigger(unittest.TestCase):
    """Test cases for CountTrigger."""

    def test_count_trigger_under_limit(self):
        """Test count trigger under the limit."""
        trigger = CountTrigger(max_count=3)

        # Should trigger for first 3 calls
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())

        # Should not trigger after limit
        self.assertFalse(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())

    def test_count_trigger_zero_limit(self):
        """Test count trigger with zero limit."""
        trigger = CountTrigger(max_count=0)

        # Should never trigger
        self.assertFalse(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())

    def test_reset(self):
        """Test reset method."""
        trigger = CountTrigger(max_count=2)

        # Use up the count
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())

        # Reset
        trigger.reset()

        # Should trigger again
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())


class TestProbabilisticTrigger(unittest.TestCase):
    """Test cases for ProbabilisticTrigger."""

    def test_probabilistic_trigger_distribution(self):
        """Test probabilistic trigger distribution."""
        probability = 0.3
        trigger = ProbabilisticTrigger(probability=probability)

        # Test with many trials
        triggers = 0
        trials = 10000

        for _ in range(trials):
            if trigger.should_trigger():
                triggers += 1

        # Should be approximately correct probability
        actual_probability = triggers / trials
        self.assertAlmostEqual(actual_probability, probability, delta=0.02)

    def test_probabilistic_trigger_edge_cases(self):
        """Test probabilistic trigger edge cases."""
        # Zero probability
        trigger = ProbabilisticTrigger(probability=0.0)
        for _ in range(100):
            self.assertFalse(trigger.should_trigger())

        # 100% probability
        trigger = ProbabilisticTrigger(probability=1.0)
        for _ in range(100):
            self.assertTrue(trigger.should_trigger())

    def test_invalid_probability(self):
        """Test invalid probability values."""
        with self.assertRaises(ValueError):
            ProbabilisticTrigger(probability=-0.1)

        with self.assertRaises(ValueError):
            ProbabilisticTrigger(probability=1.1)

    def test_reset(self):
        """Test reset method (no-op for probabilistic trigger)."""
        trigger = ProbabilisticTrigger(probability=0.5)
        trigger.reset()
        # Should still work normally
        results = [trigger.should_trigger() for _ in range(100)]
        self.assertTrue(any(results))  # Should have some True
        self.assertTrue(any(not r for r in results))  # Should have some False


class TestConditionalTrigger(unittest.TestCase):
    """Test cases for ConditionalTrigger."""

    def test_conditional_trigger_with_function(self):
        """Test conditional trigger with function."""
        condition_value = False

        def condition():
            return condition_value

        trigger = ConditionalTrigger(condition)

        # Should not trigger when condition is false
        self.assertFalse(trigger.should_trigger())

        # Change condition
        condition_value = True

        # Should trigger when condition is true
        self.assertTrue(trigger.should_trigger())

    def test_conditional_trigger_with_lambda(self):
        """Test conditional trigger with lambda."""
        counter = 0

        trigger = ConditionalTrigger(lambda: counter >= 3)

        # Should not trigger initially
        self.assertFalse(trigger.should_trigger())

        # Increment counter
        counter = 2
        self.assertFalse(trigger.should_trigger())

        # Reach threshold
        counter = 3
        self.assertTrue(trigger.should_trigger())

    def test_conditional_trigger_with_exception(self):
        """Test conditional trigger when condition raises exception."""

        def failing_condition():
            raise ValueError("Condition failed")

        trigger = ConditionalTrigger(failing_condition)

        # Should not trigger when condition fails
        self.assertFalse(trigger.should_trigger())

    def test_reset(self):
        """Test reset method (no-op for conditional trigger)."""
        trigger = ConditionalTrigger(lambda: True)
        trigger.reset()
        self.assertTrue(trigger.should_trigger())


class TestMetricThresholdTrigger(unittest.TestCase):
    """Test cases for MetricThresholdTrigger."""

    def test_metric_threshold_below_threshold(self):
        """Test metric threshold trigger below threshold."""

        def get_metric():
            return 50

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison="<")

        # Should trigger when metric is below threshold
        self.assertTrue(trigger.should_trigger())

    def test_metric_threshold_above_threshold(self):
        """Test metric threshold trigger above threshold."""

        def get_metric():
            return 150

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison=">")

        # Should trigger when metric is above threshold
        self.assertTrue(trigger.should_trigger())

    def test_metric_threshold_equal_threshold(self):
        """Test metric threshold trigger equal to threshold."""

        def get_metric():
            return 100

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison="==")

        # Should trigger when metric equals threshold
        self.assertTrue(trigger.should_trigger())

    def test_metric_threshold_not_equal(self):
        """Test metric threshold trigger not equal to threshold."""

        def get_metric():
            return 150

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison="!=")

        # Should trigger when metric is not equal to threshold
        self.assertTrue(trigger.should_trigger())

    def test_metric_threshold_dynamic_metric(self):
        """Test metric threshold with dynamically changing metric."""
        metric_value = 50

        def get_metric():
            return metric_value

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison=">")

        # Should not trigger initially
        self.assertFalse(trigger.should_trigger())

        # Increase metric
        metric_value = 150

        # Should trigger now
        self.assertTrue(trigger.should_trigger())

    def test_invalid_comparison(self):
        """Test invalid comparison operator."""

        def get_metric():
            return 100

        with self.assertRaises(ValueError):
            MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison="invalid")

    def test_reset(self):
        """Test reset method (no-op for metric threshold trigger)."""

        def get_metric():
            return 100

        trigger = MetricThresholdTrigger(metric_func=get_metric, threshold=100, comparison="==")
        trigger.reset()
        self.assertTrue(trigger.should_trigger())


class TestLogPatternTrigger(unittest.TestCase):
    """Test cases for LogPatternTrigger."""

    @patch("builtins.open")
    def test_log_pattern_trigger_found(self, mock_open):
        """Test log pattern trigger when pattern is found."""
        # Mock log file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            "2024-01-01 10:00:00 INFO Starting application\n",
            "2024-01-01 10:00:01 ERROR Connection failed\n",
            "2024-01-01 10:00:02 INFO Retrying connection\n",
        ]
        mock_open.return_value = mock_file

        trigger = LogPatternTrigger(log_file="test.log", pattern="ERROR.*Connection failed")

        # Should trigger when pattern is found
        self.assertTrue(trigger.should_trigger())

    @patch("builtins.open")
    def test_log_pattern_trigger_not_found(self, mock_open):
        """Test log pattern trigger when pattern is not found."""
        # Mock log file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            "2024-01-01 10:00:00 INFO Starting application\n",
            "2024-01-01 10:00:01 INFO All systems operational\n",
        ]
        mock_open.return_value = mock_file

        trigger = LogPatternTrigger(log_file="test.log", pattern="ERROR.*Connection failed")

        # Should not trigger when pattern is not found
        self.assertFalse(trigger.should_trigger())

    @patch("builtins.open")
    def test_log_pattern_trigger_multiple_occurrences(self, mock_open):
        """Test log pattern trigger with multiple occurrences."""
        # Mock log file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            "2024-01-01 10:00:01 ERROR Connection failed\n",
            "2024-01-01 10:00:02 INFO Retrying\n",
            "2024-01-01 10:00:03 ERROR Connection failed\n",
        ]
        mock_open.return_value = mock_file

        trigger = LogPatternTrigger(log_file="test.log", pattern="ERROR.*Connection failed", min_occurrences=2)

        # Should trigger when minimum occurrences are met
        self.assertTrue(trigger.should_trigger())

    def test_log_pattern_trigger_file_not_found(self):
        """Test log pattern trigger when file doesn't exist."""
        trigger = LogPatternTrigger(log_file="nonexistent.log", pattern="ERROR")

        # Should not trigger when file doesn't exist
        self.assertFalse(trigger.should_trigger())

    @patch("builtins.open")
    def test_log_pattern_trigger_with_check_interval(self, mock_open):
        """Test log pattern trigger with check interval."""
        # Mock log file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            "2024-01-01 10:00:00 INFO Starting\n",
            "2024-01-01 10:00:01 ERROR Failed\n",
        ]
        mock_open.return_value = mock_file

        trigger = LogPatternTrigger(
            log_file="test.log",
            pattern="ERROR",
            check_interval=0.1,  # 100ms interval
        )

        # First check should find pattern
        self.assertTrue(trigger.should_trigger())

        # Immediate second check should not (due to interval)
        self.assertFalse(trigger.should_trigger())

        # Wait for interval and check again
        time.sleep(0.1)
        self.assertTrue(trigger.should_trigger())

    def test_reset(self):
        """Test reset method."""
        trigger = LogPatternTrigger(log_file="test.log", pattern="ERROR")
        trigger.reset()
        # Should reset any internal state
        self.assertFalse(trigger.should_trigger())  # File doesn't exist


class TestTriggerManager(unittest.TestCase):
    """Test cases for TriggerManager."""

    def test_trigger_manager_single_trigger(self):
        """Test trigger manager with single trigger."""
        trigger = CountTrigger(max_count=2)
        manager = TriggerManager([trigger])

        # Should trigger for first 2 calls
        self.assertTrue(manager.should_trigger())
        self.assertTrue(manager.should_trigger())

        # Should not trigger after limit
        self.assertFalse(manager.should_trigger())

    def test_trigger_manager_multiple_triggers(self):
        """Test trigger manager with multiple triggers."""
        trigger1 = CountTrigger(max_count=2)
        trigger2 = ProbabilisticTrigger(probability=0.5)
        manager = TriggerManager([trigger1, trigger2])

        # First 2 calls should trigger (count trigger)
        self.assertTrue(manager.should_trigger())
        self.assertTrue(manager.should_trigger())

        # After count limit, should depend on probabilistic trigger
        results = [manager.should_trigger() for _ in range(100)]
        # Should have approximately 50% True (with tolerance)
        true_count = sum(results)
        self.assertGreater(true_count, 30)
        self.assertLess(true_count, 70)

    def test_trigger_manager_all_mode(self):
        """Test trigger manager in ALL mode."""
        # Create triggers where one will expire
        trigger1 = CountTrigger(max_count=1)
        trigger2 = ImmediateTrigger()
        manager = TriggerManager([trigger1, trigger2], mode="all")

        # First call should trigger (both triggers active)
        self.assertTrue(manager.should_trigger())

        # Second call should not trigger (count trigger expired)
        self.assertFalse(manager.should_trigger())

    def test_trigger_manager_reset(self):
        """Test trigger manager reset."""
        trigger1 = CountTrigger(max_count=1)
        trigger2 = CountTrigger(max_count=2)
        manager = TriggerManager([trigger1, trigger2])

        # Use up triggers
        self.assertTrue(manager.should_trigger())  # Both have 1 left
        self.assertFalse(manager.should_trigger())  # First trigger expired

        # Reset
        manager.reset()

        # Should trigger again
        self.assertTrue(manager.should_trigger())

    def test_trigger_manager_empty_list(self):
        """Test trigger manager with empty trigger list."""
        manager = TriggerManager([])

        # Should not trigger with no triggers
        self.assertFalse(manager.should_trigger())

    def test_trigger_manager_invalid_mode(self):
        """Test trigger manager with invalid mode."""
        trigger = ImmediateTrigger()

        with self.assertRaises(ValueError):
            TriggerManager([trigger], mode="invalid")


if __name__ == "__main__":
    unittest.main()
