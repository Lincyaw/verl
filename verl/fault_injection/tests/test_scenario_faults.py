"""Test scenario-based fault injection."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import yaml

from verl.fault_injection import (
    FaultScenarioConfig,
    FaultScenarioOrchestrator,
    FaultOrchestrator,
    ScenarioConfigLoader,
    TriggerManager,
    BaseTrigger,
    TimedTrigger,
    CountTrigger,
    ProbabilisticTrigger,
)


class TestScenarioFaultInjection(unittest.TestCase):
    """Test scenario-based fault injection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fault_orchestrator = FaultOrchestrator()
        self.scenario_orchestrator = FaultScenarioOrchestrator(self.fault_orchestrator)
        self.loader = ScenarioConfigLoader()

    def test_scenario_config_creation(self):
        """Test creating a fault scenario configuration."""
        scenario = FaultScenarioConfig(
            name="test_scenario",
            description="Test scenario",
            cascade_mode="linear",
            cascade_interval_seconds=5.0
        )

        self.assertEqual(scenario.name, "test_scenario")
        self.assertEqual(scenario.cascade_mode, "linear")
        self.assertEqual(scenario.cascade_interval_seconds, 5.0)

    def test_scenario_loader(self):
        """Test loading scenario from YAML."""
        # Create test YAML content
        yaml_content = """
test_scenario:
  description: "Test scenario"
  enabled: true
  cascade_mode: "linear"
  cascade_interval_seconds: 10
  faults:
    - name: "test_fault"
      type: "process_kill"
      layer: "orchestration"
      target:
        mode: "random"
        count: 1
      trigger:
        type: "immediate"
  dependencies: []
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            # Load scenario
            scenarios = self.loader.load_scenarios_from_yaml(f.name)
            scenario = scenarios["test_scenario"]

            self.assertEqual(scenario.name, "test_scenario")
            self.assertEqual(scenario.cascade_mode, "linear")
            self.assertEqual(len(scenario.faults), 1)
            self.assertEqual(scenario.faults[0].name, "test_scenario_test_fault")

        # Clean up
        Path(f.name).unlink()

    def test_trigger_types(self):
        """Test different trigger types."""
        # Test immediate trigger
        immediate = BaseTrigger.create_trigger({"type": "immediate"})
        self.assertIsInstance(immediate, BaseTrigger)

        # Test timed trigger
        timed = TimedTrigger({"type": "timed", "delay_seconds": 5})
        self.assertEqual(timed.config.delay_seconds, 5)

        # Test count trigger
        count = CountTrigger({"type": "count", "count_threshold": 10})
        for _ in range(10):
            count.increment()
        self.assertTrue(count.check_trigger({}))

        # Test probabilistic trigger
        prob = ProbabilisticTrigger({"type": "probabilistic", "probability": 1.0})
        self.assertTrue(prob.check_trigger({}))

    def test_template_creation(self):
        """Test creating scenarios from templates."""
        # Create scenario from template
        scenario = self.scenario_orchestrator.create_scenario_from_template(
            "system_failure_cascade",
            "test_system_failure"
        )

        self.assertEqual(scenario.name, "test_system_failure")
        self.assertEqual(scenario.cascade_mode, "tree")
        self.assertGreater(len(scenario.faults), 0)

    def test_dependency_graph(self):
        """Test building dependency graph."""
        # Create scenario with dependencies
        scenario = FaultScenarioConfig(
            name="dependency_test",
            faults=[
                {"name": "fault1", "type": "process_kill", "layer": "orchestration"},
                {"name": "fault2", "type": "network_delay", "layer": "orchestration"},
                {"name": "fault3", "type": "memory_oom", "layer": "worker"}
            ],
            dependencies=[
                {"fault_name": "fault1", "condition": "after"},
                {"fault_name": "fault2", "condition": "on_success"}
            ]
        )

        # Build execution graph
        graph = self.scenario_orchestrator.build_execution_graph(scenario)

        # Check dependencies
        self.assertIn("fault1", graph)
        self.assertIn("fault2", graph)
        self.assertIn("fault3", graph)

    def test_scenario_execution_context(self):
        """Test scenario execution context."""
        context = self.scenario_orchestrator.ScenarioExecutionContext(
            scenario_name="test"
        )

        self.assertEqual(context.scenario_name, "test")
        self.assertEqual(len(context.completed), 0)
        self.assertEqual(len(context.failed), 0)

    def test_trigger_manager(self):
        """Test trigger manager."""
        manager = TriggerManager()

        # Register triggers
        manager.register_trigger("fault1", BaseTrigger({"type": "immediate"}))
        manager.register_trigger("fault2", TimedTrigger({"type": "timed", "delay_seconds": 0}))

        # Check triggers
        triggered = manager.check_triggers({})
        self.assertIn("fault1", triggered)
        self.assertIn("fault2", triggered)

    def test_cascade_modes(self):
        """Test different cascade modes."""
        # Test burst mode
        burst_scenario = FaultScenarioConfig(
            name="burst_test",
            cascade_mode="burst"
        )
        self.assertEqual(burst_scenario.cascade_mode, "burst")

        # Test linear mode
        linear_scenario = FaultScenarioConfig(
            name="linear_test",
            cascade_mode="linear",
            cascade_interval_seconds=5
        )
        self.assertEqual(linear_scenario.cascade_mode, "linear")
        self.assertEqual(linear_scenario.cascade_interval_seconds, 5)

        # Test tree mode
        tree_scenario = FaultScenarioConfig(
            name="tree_test",
            cascade_mode="tree",
            max_cascade_depth=3
        )
        self.assertEqual(tree_scenario.cascade_mode, "tree")
        self.assertEqual(tree_scenario.max_cascade_depth, 3)

    def test_dependency_conditions(self):
        """Test dependency conditions."""
        deps = [
            {"fault_name": "f1", "condition": "after"},
            {"fault_name": "f2", "condition": "on_success"},
            {"fault_name": "f3", "condition": "on_failure"}
        ]

        for dep_data in deps:
            dep = FaultDependencyConfig(**dep_data)
            self.assertEqual(dep.fault_name, dep_data["fault_name"])
            self.assertEqual(dep.condition, dep_data["condition"])

    def test_scenario_status(self):
        """Test getting scenario status."""
        # Create a mock active scenario
        scenario = FaultScenarioConfig(name="status_test")
        self.scenario_orchestrator.active_scenarios["status_test"] = \
            self.scenario_orchestrator.ScenarioExecutionContext("status_test")

        status = self.scenario_orchestrator.get_scenario_status("status_test")
        self.assertIsNotNone(status)
        self.assertEqual(status["scenario_name"], "status_test")
        self.assertEqual(status["completed_faults"], 0)
        self.assertEqual(status["failed_faults"], 0)

    def test_save_and_load_scenario(self):
        """Test saving and loading scenario configuration."""
        # Create a scenario
        scenario = FaultScenarioConfig(
            name="save_test",
            description="Test save/load",
            cascade_mode="linear",
            faults=[
                {"name": "fault1", "type": "process_kill", "layer": "orchestration"}
            ]
        )

        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            self.loader.save_scenario_to_yaml(scenario, f.name)

            # Load back
            loaded = self.loader.load_scenario_from_yaml(f.name)

            self.assertEqual(loaded.name, scenario.name)
            self.assertEqual(loaded.cascade_mode, scenario.cascade_mode)

        # Clean up
        Path(f.name).unlink()


class TestAsyncScenarioExecution(unittest.TestCase):
    """Test async scenario execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.fault_orchestrator = FaultOrchestrator()
        self.scenario_orchestrator = FaultScenarioOrchestrator(self.fault_orchestrator)

    def test_async_scenario_execution(self):
        """Test executing a scenario asynchronously."""
        # Create a simple scenario
        scenario = FaultScenarioConfig(
            name="async_test",
            cascade_mode="burst",
            faults=[
                {"name": "fault1", "type": "process_kill", "layer": "orchestration"},
                {"name": "fault2", "type": "network_delay", "layer": "orchestration"}
            ]
        )

        # Execute scenario
        async def run_scenario():
            return await self.scenario_orchestrator.execute_scenario(scenario)

        result = asyncio.run(run_scenario())
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()