"""Unit tests for fault recovery system."""

import time
import unittest
from unittest.mock import MagicMock, patch

from verl.fault_injection.base import FaultContext, FaultResult, FaultStatus
from verl.fault_injection.config import FaultLayer
from verl.fault_injection.recovery.base import (
    RecoveryContext,
    RecoveryDecision,
    RecoveryMode,
    RecoveryPriority,
    RecoveryResult,
    RecoveryStatus,
)
from verl.fault_injection.recovery.strategies import (
    CheckpointRecoveryStrategy,
    DegradationRecoveryStrategy,
    IgnoreRecoveryStrategy,
    ManualRecoveryStrategy,
    ProcessRecoveryStrategy,
    RayRecoveryStrategy,
    RedeployRecoveryStrategy,
    ResourceRecoveryStrategy,
    RestartRecoveryStrategy,
    RetryRecoveryStrategy,
)


class TestRecoveryStrategies(unittest.TestCase):
    """Test recovery strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.fault_result = FaultResult(
            fault_id="test_fault_001",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "process_kill"},
        )

        self.fault_context = FaultContext(
            worker_id="worker_001",
            rank=0,
            host="localhost",
            process_type="actor",
            layer=FaultLayer.WORKER,
            metadata={"pid": 12345},
        )

        self.recovery_context = RecoveryContext(
            fault_result=self.fault_result,
            fault_context=self.fault_context,
            attempt_count=0,
            max_attempts=3,
        )

    def test_ignore_recovery_strategy(self):
        """Test ignore recovery strategy."""
        strategy = IgnoreRecoveryStrategy()

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "ignore")
        self.assertEqual(result.metadata["action"], "ignored")

    def test_retry_recovery_strategy(self):
        """Test retry recovery strategy."""
        config = {"retry_count": 2, "retry_delay": 0.1}
        strategy = RetryRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "retry")
        self.assertEqual(result.metadata["retry_count"], 2)
        self.assertEqual(result.metadata["retry_delay"], 0.1)

    def test_retry_recovery_strategy_max_attempts(self):
        """Test retry strategy respects max attempts."""
        self.recovery_context.attempt_count = 3
        strategy = RetryRecoveryStrategy()

        # Should not be able to recover when max attempts reached
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_restart_recovery_strategy(self):
        """Test restart recovery strategy."""
        config = {"restart_delay": 0.1, "cleanup_timeout": 5.0}
        strategy = RestartRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "restart")
        self.assertEqual(result.metadata["process_type"], "actor")
        self.assertEqual(result.metadata["worker_id"], "worker_001")

    def test_restart_recovery_strategy_no_process_type(self):
        """Test restart strategy without process type."""
        self.fault_context.process_type = None
        strategy = RestartRecoveryStrategy()

        # Should not be able to recover without process type
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_process_recovery_strategy(self):
        """Test process recovery strategy."""
        config = {"restart_command": "echo 'restart'"}
        strategy = ProcessRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "process_recovery")
        self.assertEqual(result.metadata["restart_command"], "echo 'restart'")

    def test_process_recovery_strategy_wrong_fault_type(self):
        """Test process strategy with wrong fault type."""
        self.fault_result.metadata["fault_type"] = "network_delay"
        strategy = ProcessRecoveryStrategy()

        # Should not be able to recover from network delay
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_resource_recovery_strategy(self):
        """Test resource recovery strategy."""
        self.fault_result.metadata["fault_type"] = "memory_oom"
        config = {"release_allocated_memory": True, "close_leaked_fds": True}
        strategy = ResourceRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "resource_recovery")
        self.assertIn("ran_garbage_collection", result.metadata["actions"])

    def test_checkpoint_recovery_strategy(self):
        """Test checkpoint recovery strategy."""
        with patch("os.path.exists", return_value=True), patch("os.path.getsize", return_value=1024):
            config = {"checkpoint_path": "/tmp/checkpoint.pt", "load_timeout": 1.0}
            strategy = CheckpointRecoveryStrategy(config)

            # Test can_recover
            self.assertTrue(strategy.can_recover(self.recovery_context))

            # Test execute
            result = strategy.execute(self.recovery_context)
            self.assertEqual(result.status, RecoveryStatus.SUCCESS)
            self.assertEqual(result.strategy_name, "checkpoint_recovery")
            self.assertEqual(result.metadata["checkpoint_path"], "/tmp/checkpoint.pt")

    def test_checkpoint_recovery_strategy_no_checkpoint(self):
        """Test checkpoint strategy without checkpoint file."""
        with patch("os.path.exists", return_value=False):
            strategy = CheckpointRecoveryStrategy({"checkpoint_path": "/tmp/missing.pt"})

            # Should not be able to recover without checkpoint
            self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_degradation_recovery_strategy(self):
        """Test degradation recovery strategy."""
        config = {
            "allow_degradation": True,
            "degradation_level": "partial",
            "reduced_batch_size": 1,
            "reduce_precision": True,
        }
        strategy = DegradationRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "degradation")
        self.assertEqual(result.metadata["degradation_level"], "partial")
        self.assertIn("reduced_batch_size_to_1", result.metadata["actions"])

    def test_degradation_recovery_strategy_disabled(self):
        """Test degradation strategy when disabled."""
        config = {"allow_degradation": False}
        strategy = DegradationRecoveryStrategy(config)

        # Should not be able to recover if degradation is disabled
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_manual_recovery_strategy(self):
        """Test manual recovery strategy."""
        config = {"instructions": "Call support", "simulated_wait_time": 0.1}
        strategy = ManualRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "manual")
        self.assertEqual(result.metadata["instructions"], "Call support")

    def test_ray_recovery_strategy(self):
        """Test Ray recovery strategy."""
        self.fault_context.layer = FaultLayer.ORCHESTRATION
        config = {
            "restart_actors": True,
            "reconnect_gcs": True,
            "recreate_placement_groups": True,
            "recovery_time": 0.1,
        }
        strategy = RayRecoveryStrategy(config)

        # Test can_recover
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "ray_recovery")
        self.assertIn("restarted_ray_actors", result.metadata["actions"])

    def test_ray_recovery_strategy_wrong_layer(self):
        """Test Ray strategy with wrong layer."""
        self.fault_context.layer = FaultLayer.WORKER
        strategy = RayRecoveryStrategy()

        # Should not be able to recover from non-orchestration layer
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_redeploy_recovery_strategy(self):
        """Test redeploy recovery strategy."""
        self.recovery_context.attempt_count = 2  # Last attempt
        config = {
            "stop_services": True,
            "cleanup_state": False,
            "redeploy_services": True,
            "redeploy_time": 0.1,
        }
        strategy = RedeployRecoveryStrategy(config)

        # Test can_recover (only on last attempt)
        self.assertTrue(strategy.can_recover(self.recovery_context))

        # Test execute
        result = strategy.execute(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "redeploy")
        self.assertIn("redeployed_services", result.metadata["actions"])
        self.assertTrue(result.metadata["last_resort"])

    def test_redeploy_recovery_strategy_not_last_attempt(self):
        """Test redeploy strategy when not last attempt."""
        self.recovery_context.attempt_count = 1
        strategy = RedeployRecoveryStrategy()

        # Should not be able to recover if not last attempt
        self.assertFalse(strategy.can_recover(self.recovery_context))

    def test_strategy_with_checks(self):
        """Test strategy with pre and post checks."""
        strategy = IgnoreRecoveryStrategy()

        # Test successful execution with checks
        result = strategy._execute_with_checks(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)

        # Test pre-check failure
        strategy.pre_check = lambda ctx: False
        result = strategy._execute_with_checks(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertEqual(result.metadata["reason"], "Pre-check failed")

        # Test post-check failure
        strategy.pre_check = lambda ctx: True
        strategy.post_check = lambda ctx: False
        result = strategy._execute_with_checks(self.recovery_context)
        self.assertEqual(result.status, RecoveryStatus.PARTIAL_SUCCESS)
        self.assertEqual(result.metadata["warning"], "Post-check failed")

    def test_strategy_exception_handling(self):
        """Test strategy exception handling."""

        class FailingStrategy(IgnoreRecoveryStrategy):
            def execute(self, context):
                raise RuntimeError("Simulated failure")

        strategy = FailingStrategy()
        result = strategy._execute_with_checks(self.recovery_context)

        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(str(result.error), "Simulated failure")


if __name__ == "__main__":
    unittest.main()