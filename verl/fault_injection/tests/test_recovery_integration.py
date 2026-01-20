# Copyright 2026 Aoyang Fang
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
# ==============================================================================

"""Unit tests for recovery integration."""

import time
import unittest
from unittest.mock import MagicMock, patch

from verl.fault_injection.base import FaultContext, FaultResult, FaultStatus
from verl.fault_injection.config import FaultInjectionConfig, RecoveryConfig, RecoveryStrategyConfig
from verl.fault_injection.orchestrator import FaultOrchestrator
from verl.fault_injection.recovery.base import RecoveryMode
from verl.fault_injection.recovery.integration import (
    RecoveryOrchestratorIntegration,
    create_recovery_integration,
)


class TestRecoveryIntegration(unittest.TestCase):
    """Test recovery integration with fault injection orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create recovery configuration
        recovery_config = RecoveryConfig(
            enabled=True,
            mode="automatic",
            max_recovery_attempts=3,
            recovery_timeout_seconds=30.0,
            strategies=[
                RecoveryStrategyConfig(name="retry", enabled=True, priority=1),
                RecoveryStrategyConfig(name="restart", enabled=True, priority=2),
                RecoveryStrategyConfig(name="ignore", enabled=True, priority=3),
            ],
        )

        # Create fault injection configuration
        self.config = FaultInjectionConfig(
            enabled=True,
            recovery=recovery_config,
        )

        # Create orchestrator
        self.orchestrator = FaultOrchestrator(self.config)

        # Create recovery integration
        self.integration = create_recovery_integration(self.orchestrator, self.config)

        # Create test fault result and context
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
            metadata={"pid": 12345},
        )

    def test_create_recovery_integration(self):
        """Test creating recovery integration."""
        self.assertIsInstance(self.integration, RecoveryOrchestratorIntegration)
        self.assertEqual(self.integration.config, self.config.recovery)
        self.assertTrue(self.integration._recovery_enabled)

    def test_on_fault_completed_with_recovery(self):
        """Test handling fault completion with recovery."""
        # Mock recovery manager
        mock_result = MagicMock()
        mock_result.status.value = "success"
        mock_result.strategy_name = "retry"
        mock_result.duration = 1.0
        mock_result.metadata = {"attempts": 1}

        self.integration.recovery_manager.handle_fault = MagicMock(return_value=mock_result)

        # Handle fault completion
        result = self.integration.on_fault_completed(self.fault_result, self.fault_context)

        # Verify recovery was triggered
        self.integration.recovery_manager.handle_fault.assert_called_once_with(
            self.fault_result, self.fault_context, RecoveryMode.AUTOMATIC
        )

        # Verify fault result was updated
        self.assertEqual(result.status, FaultStatus.RECOVERED)
        self.assertIn("recovery", result.metadata)
        self.assertEqual(result.metadata["recovery"]["strategy"], "retry")

    def test_on_fault_completed_recovery_failed(self):
        """Test handling fault completion when recovery fails."""
        # Mock failed recovery
        mock_result = MagicMock()
        mock_result.status.value = "failed"
        mock_result.strategy_name = "retry"
        mock_result.error = Exception("Recovery failed")
        mock_result.metadata = {"attempts": 3}

        self.integration.recovery_manager.handle_fault = MagicMock(return_value=mock_result)

        # Handle fault completion
        result = self.integration.on_fault_completed(self.fault_result, self.fault_context)

        # Verify fault result was updated with recovery failure
        self.assertEqual(result.status, FaultStatus.FAILED)
        self.assertIn("recovery_failure", result.metadata)
        self.assertEqual(result.metadata["recovery_failure"]["strategy"], "retry")

    def test_on_fault_completed_recovery_disabled(self):
        """Test handling fault completion when recovery is disabled."""
        # Disable recovery
        self.integration._recovery_enabled = False

        # Handle fault completion
        result = self.integration.on_fault_completed(self.fault_result, self.fault_context)

        # Verify no recovery was triggered
        self.assertIsNone(result)

    def test_on_fault_completed_non_failed_fault(self):
        """Test handling fault completion for non-failed fault."""
        # Set fault as already recovered
        self.fault_result.status = FaultStatus.RECOVERED

        # Handle fault completion
        result = self.integration.on_fault_completed(self.fault_result, self.fault_context)

        # Verify no recovery was triggered
        self.assertIsNone(result)

    def test_get_recovery_mode(self):
        """Test getting recovery mode."""
        # Test automatic mode
        self.integration.config.mode = "automatic"
        mode = self.integration._get_recovery_mode()
        self.assertEqual(mode, RecoveryMode.AUTOMATIC)

        # Test manual mode
        self.integration.config.mode = "manual"
        mode = self.integration._get_recovery_mode()
        self.assertEqual(mode, RecoveryMode.MANUAL)

        # Test semi-automatic mode
        self.integration.config.mode = "semi_automatic"
        mode = self.integration._get_recovery_mode()
        self.assertEqual(mode, RecoveryMode.SEMI_AUTOMATIC)

        # Test invalid mode (should default to automatic)
        self.integration.config.mode = "invalid"
        mode = self.integration._get_recovery_mode()
        self.assertEqual(mode, RecoveryMode.AUTOMATIC)

    def test_get_recovery_statistics(self):
        """Test getting recovery statistics."""
        # Mock recovery manager statistics
        mock_stats = {
            "total_recoveries": 10,
            "successful_recoveries": 8,
            "failed_recoveries": 2,
            "success_rate": 0.8,
            "average_recovery_time": 2.5,
        }
        self.integration.recovery_manager.get_recovery_statistics = MagicMock(return_value=mock_stats)

        # Get statistics
        stats = self.integration.get_recovery_statistics()

        # Verify statistics
        self.assertEqual(stats, mock_stats)

    def test_enable_preventive_recovery(self):
        """Test enabling preventive recovery."""
        # Enable for specific layer
        with patch.object(self.integration.recovery_manager, "enable_preventive_recovery_for_layer") as mock_method:
            self.integration.enable_preventive_recovery("worker")
            mock_method.assert_called_once()

        # Enable for all layers
        with patch.object(self.integration.recovery_manager, "enable_preventive_recovery_for_layer") as mock_method:
            self.integration.enable_preventive_recovery()
            # Should be called for each layer
            self.assertEqual(mock_method.call_count, 5)  # Number of fault layers

    def test_update_config(self):
        """Test updating recovery configuration."""
        # Create new configuration
        new_config = RecoveryConfig(
            enabled=False,
            mode="manual",
            strategies=[],
        )

        # Update configuration
        self.integration.update_config(new_config)

        # Verify configuration was updated
        self.assertEqual(self.integration.config, new_config)
        self.assertFalse(self.integration._recovery_enabled)

    def test_create_recovery_integration_factory(self):
        """Test factory function for creating recovery integration."""
        integration = create_recovery_integration(self.orchestrator, self.config)

        self.assertIsInstance(integration, RecoveryOrchestratorIntegration)
        self.assertEqual(integration.orchestrator, self.orchestrator)
        self.assertEqual(integration.config, self.config.recovery)

    def test_recovery_with_orchestrator_integration(self):
        """Test recovery integration with actual orchestrator."""
        # This is an integration test that verifies the recovery system works
        # with the fault injection orchestrator

        # Mock the recovery manager to avoid actual recovery execution
        self.integration.recovery_manager.handle_fault = MagicMock()

        # Create a simple fault result
        fault_result = FaultResult(
            fault_id="integration_test_fault",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "test"},
        )

        fault_context = FaultContext(
            worker_id="test_worker",
            rank=0,
            host="test_host",
            process_type="test",
            metadata={},
        )

        # Simulate fault injection completion
        self.integration.on_fault_completed(fault_result, fault_context)

        # Verify recovery was triggered
        self.integration.recovery_manager.handle_fault.assert_called_once()
        call_args = self.integration.recovery_manager.handle_fault.call_args[0]
        self.assertEqual(call_args[0], fault_result)
        self.assertEqual(call_args[1], fault_context)


if __name__ == "__main__":
    unittest.main()
