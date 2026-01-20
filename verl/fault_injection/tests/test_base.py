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
Unit tests for base fault injection classes.
"""

import unittest

from verl.fault_injection.base import (
    BaseFaultInjector,
    FaultContext,
    FaultInjectorRegistry,
    FaultResult,
    FaultTargetSelector,
    FaultTrigger,
)


class TestFaultContext(unittest.TestCase):
    """Test cases for FaultContext."""

    def test_fault_context_creation(self):
        """Test FaultContext creation."""
        context = FaultContext(target="test_target", fault_type="test_fault", parameters={"param1": "value1"})

        self.assertEqual(context.target, "test_target")
        self.assertEqual(context.fault_type, "test_fault")
        self.assertEqual(context.parameters, {"param1": "value1"})

    def test_fault_context_with_timestamp(self):
        """Test FaultContext with timestamp."""
        import time

        current_time = time.time()

        context = FaultContext(target="test_target", fault_type="test_fault", parameters={}, timestamp=current_time)

        self.assertEqual(context.timestamp, current_time)

    def test_fault_context_immutable(self):
        """Test FaultContext immutability."""
        context = FaultContext(target="test_target", fault_type="test_fault", parameters={})

        # Should not be able to modify attributes
        with self.assertRaises(AttributeError):
            context.target = "new_target"


class TestFaultResult(unittest.TestCase):
    """Test cases for FaultResult."""

    def test_fault_result_creation(self):
        """Test FaultResult creation."""
        result = FaultResult(
            status="injected", fault_type="test_fault", target="test_target", metadata={"key": "value"}
        )

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.fault_type, "test_fault")
        self.assertEqual(result.target, "test_target")
        self.assertEqual(result.metadata, {"key": "value"})

    def test_fault_result_with_timestamp(self):
        """Test FaultResult with timestamp."""
        import time

        current_time = time.time()

        result = FaultResult(status="injected", fault_type="test_fault", target="test_target", timestamp=current_time)

        self.assertEqual(result.timestamp, current_time)

    def test_fault_result_immutable(self):
        """Test FaultResult immutability."""
        result = FaultResult(status="injected", fault_type="test_fault", target="test_target")

        # Should not be able to modify attributes
        with self.assertRaises(AttributeError):
            result.status = "failed"


class TestBaseFaultInjector(unittest.TestCase):
    """Test cases for BaseFaultInjector."""

    def test_base_injector_abstract(self):
        """Test that BaseFaultInjector is abstract."""
        with self.assertRaises(TypeError):
            BaseFaultInjector()

    def test_concrete_injector(self):
        """Test creating a concrete injector."""

        class TestInjector(BaseFaultInjector):
            def inject(self, context: FaultContext) -> FaultResult:
                return FaultResult(status="injected", fault_type=context.fault_type, target=context.target)

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                return FaultResult(status="recovered", fault_type=context.fault_type, target=context.target)

        injector = TestInjector()
        context = FaultContext(target="test_target", fault_type="test_fault", parameters={})

        # Test injection
        result = injector.inject(context)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.fault_type, "test_fault")
        self.assertEqual(result.target, "test_target")

        # Test recovery
        recover_result = injector.recover(context, result)
        self.assertEqual(recover_result.status, "recovered")

    def test_injector_with_validation(self):
        """Test injector with parameter validation."""

        class ValidatingInjector(BaseFaultInjector):
            def __init__(self, threshold: float):
                if threshold < 0 or threshold > 1:
                    raise ValueError("Threshold must be between 0 and 1")
                self.threshold = threshold

            def inject(self, context: FaultContext) -> FaultResult:
                return FaultResult(
                    status="injected",
                    fault_type=context.fault_type,
                    target=context.target,
                    metadata={"threshold": self.threshold},
                )

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                return FaultResult(status="recovered", fault_type=context.fault_type, target=context.target)

        # Valid threshold
        injector = ValidatingInjector(0.5)
        self.assertEqual(injector.threshold, 0.5)

        # Invalid threshold
        with self.assertRaises(ValueError):
            ValidatingInjector(1.5)


class TestFaultInjectorRegistry(unittest.TestCase):
    """Test cases for FaultInjectorRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = FaultInjectorRegistry()

    def test_register_injector(self):
        """Test registering an injector."""

        class TestInjector(BaseFaultInjector):
            def inject(self, context: FaultContext) -> FaultResult:
                return FaultResult(status="injected", fault_type=context.fault_type, target=context.target)

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                return FaultResult(status="recovered", fault_type=context.fault_type, target=context.target)

        # Register injector
        self.registry.register("test_fault", TestInjector())

        # Retrieve injector
        injector = self.registry.get("test_fault")
        self.assertIsInstance(injector, TestInjector)

    def test_register_duplicate_fault_type(self):
        """Test registering duplicate fault type."""

        class TestInjector1(BaseFaultInjector):
            def inject(self, context: FaultContext) -> FaultResult:
                pass

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                pass

        class TestInjector2(BaseFaultInjector):
            def inject(self, context: FaultContext) -> FaultResult:
                pass

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                pass

        self.registry.register("test_fault", TestInjector1())

        # Should overwrite with warning
        with self.assertWarns(UserWarning):
            self.registry.register("test_fault", TestInjector2())

        # Should get the second injector
        injector = self.registry.get("test_fault")
        self.assertIsInstance(injector, TestInjector2)

    def test_get_nonexistent_fault_type(self):
        """Test getting non-existent fault type."""
        injector = self.registry.get("nonexistent")
        self.assertIsNone(injector)

    def test_list_fault_types(self):
        """Test listing all registered fault types."""

        class TestInjector(BaseFaultInjector):
            def inject(self, context: FaultContext) -> FaultResult:
                pass

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                pass

        # Register multiple injectors
        self.registry.register("fault1", TestInjector())
        self.registry.register("fault2", TestInjector())
        self.registry.register("fault3", TestInjector())

        fault_types = self.registry.list_fault_types()
        self.assertEqual(set(fault_types), {"fault1", "fault2", "fault3"})

    def test_singleton_behavior(self):
        """Test that registry behaves as singleton."""
        registry1 = FaultInjectorRegistry()
        registry2 = FaultInjectorRegistry()

        # Should be the same instance
        self.assertIs(registry1, registry2)


class TestFaultTargetSelector(unittest.TestCase):
    """Test cases for FaultTargetSelector."""

    def test_random_selector(self):
        """Test random target selector."""
        selector = FaultTargetSelector.random_selector()

        targets = ["target1", "target2", "target3", "target4", "target5"]

        # Select multiple targets
        selected = selector.select(targets, count=3)

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(t in targets for t in selected))

        # Should be different selections (with high probability)
        selected2 = selector.select(targets, count=3)
        self.assertNotEqual(selected, selected2)

    def test_round_robin_selector(self):
        """Test round-robin target selector."""
        selector = FaultTargetSelector.round_robin_selector()

        targets = ["target1", "target2", "target3"]

        # Select in sequence
        selected1 = selector.select(targets, count=1)
        selected2 = selector.select(targets, count=1)
        selected3 = selector.select(targets, count=1)
        selected4 = selector.select(targets, count=1)  # Should wrap around

        self.assertEqual(selected1, ["target1"])
        self.assertEqual(selected2, ["target2"])
        self.assertEqual(selected3, ["target3"])
        self.assertEqual(selected4, ["target1"])  # Wrapped around

    def test_all_selector(self):
        """Test select-all target selector."""
        selector = FaultTargetSelector.all_selector()

        targets = ["target1", "target2", "target3"]

        selected = selector.select(targets, count=2)  # Count should be ignored

        self.assertEqual(selected, targets)

    def test_weighted_selector(self):
        """Test weighted target selector."""
        weights = {"target1": 0.5, "target2": 0.3, "target3": 0.2}
        selector = FaultTargetSelector.weighted_selector(weights)

        targets = ["target1", "target2", "target3"]

        # Select many times and check distribution
        selections = []
        for _ in range(1000):
            selected = selector.select(targets, count=1)
            selections.append(selected[0])

        # Count occurrences
        counts = {t: selections.count(t) for t in targets}

        # Check approximate weights (with some tolerance)
        total = len(selections)
        self.assertAlmostEqual(counts["target1"] / total, 0.5, delta=0.1)
        self.assertAlmostEqual(counts["target2"] / total, 0.3, delta=0.1)
        self.assertAlmostEqual(counts["target3"] / total, 0.2, delta=0.1)


class TestFaultTrigger(unittest.TestCase):
    """Test cases for FaultTrigger."""

    def test_immediate_trigger(self):
        """Test immediate trigger."""
        trigger = FaultTrigger.immediate_trigger()

        # Should always trigger immediately
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())

    def test_count_trigger(self):
        """Test count-based trigger."""
        trigger = FaultTrigger.count_trigger(max_count=3)

        # Should trigger for first 3 calls
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())

        # Should not trigger after max count
        self.assertFalse(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())

    def test_probabilistic_trigger(self):
        """Test probabilistic trigger."""
        trigger = FaultTrigger.probabilistic_trigger(probability=0.5)

        # Test with many trials
        triggers = 0
        trials = 10000

        for _ in range(trials):
            if trigger.should_trigger():
                triggers += 1

        # Should be approximately 50% (with some tolerance)
        actual_probability = triggers / trials
        self.assertAlmostEqual(actual_probability, 0.5, delta=0.05)

    def test_conditional_trigger(self):
        """Test conditional trigger."""
        condition_met = False

        def condition():
            return condition_met

        trigger = FaultTrigger.conditional_trigger(condition)

        # Should not trigger when condition is false
        self.assertFalse(trigger.should_trigger())
        self.assertFalse(trigger.should_trigger())

        # Should trigger when condition is true
        condition_met = True
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())

    def test_timed_trigger(self):
        """Test time-based trigger."""
        import time

        # Trigger after 0.1 seconds
        trigger = FaultTrigger.timed_trigger(delay_seconds=0.1)

        # Should not trigger immediately
        self.assertFalse(trigger.should_trigger())

        # Wait for delay
        time.sleep(0.1)

        # Should trigger after delay
        self.assertTrue(trigger.should_trigger())
        self.assertTrue(trigger.should_trigger())  # Should continue triggering


class TestIntegration(unittest.TestCase):
    """Integration tests for base classes."""

    def test_full_fault_injection_cycle(self):
        """Test a complete fault injection cycle."""

        # Create a test injector
        class TestInjector(BaseFaultInjector):
            def __init__(self, failure_rate: float = 0.1):
                self.failure_rate = failure_rate
                self.trigger = FaultTrigger.probabilistic_trigger(failure_rate)

            def inject(self, context: FaultContext) -> FaultResult:
                if not self.trigger.should_trigger():
                    return FaultResult(
                        status="skipped",
                        fault_type=context.fault_type,
                        target=context.target,
                        metadata={"reason": "trigger_not_met"},
                    )

                return FaultResult(
                    status="injected",
                    fault_type=context.fault_type,
                    target=context.target,
                    metadata={"failure_rate": self.failure_rate},
                )

            def recover(self, context: FaultContext, result: FaultResult) -> FaultResult:
                return FaultResult(
                    status="recovered",
                    fault_type=context.fault_type,
                    target=context.target,
                    metadata={"recovery_time": 0.1},
                )

        # Register the injector
        registry = FaultInjectorRegistry()
        registry.register("test_fault", TestInjector())

        # Get the injector
        injector = registry.get("test_fault")
        self.assertIsNotNone(injector)

        # Create context
        context = FaultContext(target="test_target", fault_type="test_fault", parameters={"failure_rate": 0.1})

        # Run multiple cycles
        injected_count = 0
        for _ in range(100):
            result = injector.inject(context)
            self.assertIn(result.status, ["injected", "skipped"])

            if result.status == "injected":
                injected_count += 1
                # Recover
                recover_result = injector.recover(context, result)
                self.assertEqual(recover_result.status, "recovered")

        # Should have some injections (with high probability)
        self.assertGreater(injected_count, 0)
        self.assertLess(injected_count, 100)


if __name__ == "__main__":
    unittest.main()
