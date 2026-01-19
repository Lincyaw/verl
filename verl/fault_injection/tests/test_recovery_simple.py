#!/usr/bin/env python3
"""Simple test for recovery system without complex dependencies."""

import sys
import os
import time
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import base classes directly
from base import FaultContext, FaultResult, FaultStatus
from config import FaultLayer
from recovery.strategies import IgnoreRecoveryStrategy, RetryRecoveryStrategy
from recovery.base import RecoveryContext, RecoveryStatus


class SimpleRecoveryTest(unittest.TestCase):
    """Simple test for recovery strategies."""

    def test_ignore_recovery(self):
        """Test ignore recovery strategy."""
        # Create test data
        fault_result = FaultResult(
            fault_id="test_fault_001",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "test"},
        )

        fault_context = FaultContext(
            worker_id="worker_001",
            rank=0,
            host="localhost",
            process_type="actor",
            layer=FaultLayer.WORKER,
            metadata={"pid": 12345},
        )

        recovery_context = RecoveryContext(
            fault_result=fault_result,
            fault_context=fault_context,
            attempt_count=0,
            max_attempts=3,
        )

        # Test ignore strategy
        strategy = IgnoreRecoveryStrategy()

        # Should be able to recover
        self.assertTrue(strategy.can_recover(recovery_context))

        # Execute recovery
        result = strategy.execute(recovery_context)

        # Verify results
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "ignore")
        self.assertEqual(result.metadata["action"], "ignored")

    def test_retry_recovery(self):
        """Test retry recovery strategy."""
        # Create test data
        fault_result = FaultResult(
            fault_id="test_fault_002",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "test"},
        )

        fault_context = FaultContext(
            worker_id="worker_002",
            rank=1,
            host="localhost",
            process_type="critic",
            layer=FaultLayer.WORKER,
            metadata={"pid": 12346},
        )

        recovery_context = RecoveryContext(
            fault_result=fault_result,
            fault_context=fault_context,
            attempt_count=0,
            max_attempts=3,
        )

        # Test retry strategy
        config = {"retry_count": 2, "retry_delay": 0.01}
        strategy = RetryRecoveryStrategy(config)

        # Should be able to recover
        self.assertTrue(strategy.can_recover(recovery_context))

        # Execute recovery
        result = strategy.execute(recovery_context)

        # Verify results
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "retry")
        self.assertEqual(result.metadata["retry_count"], 2)
        self.assertEqual(result.metadata["retry_delay"], 0.01)

    def test_retry_max_attempts(self):
        """Test retry strategy respects max attempts."""
        # Create test data
        fault_result = FaultResult(
            fault_id="test_fault_003",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "test"},
        )

        fault_context = FaultContext()

        recovery_context = RecoveryContext(
            fault_result=fault_result,
            fault_context=fault_context,
            attempt_count=3,  # Max attempts reached
            max_attempts=3,
        )

        # Test retry strategy
        strategy = RetryRecoveryStrategy()

        # Should not be able to recover
        self.assertFalse(strategy.can_recover(recovery_context))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)