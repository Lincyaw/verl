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

"""
Unit tests for process fault injectors.
"""

import signal
import unittest
from unittest.mock import Mock, patch

from verl.fault_injection.base import FaultContext, FaultResult
from verl.fault_injection.injectors.process import (
    ProcessExitInjector,
    ProcessHangInjector,
    ProcessKillInjector,
)


class TestProcessKillInjector(unittest.TestCase):
    """Test cases for ProcessKillInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = ProcessKillInjector(signal_type="SIGTERM", graceful_timeout=5)
        self.context = FaultContext(target="test_process", fault_type="process_kill", parameters={"pid": 12345})

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.signal_type, "SIGTERM")
        self.assertEqual(self.injector.graceful_timeout, 5)

    @patch("os.kill")
    def test_inject_sigterm(self, mock_kill):
        """Test SIGTERM injection."""
        mock_kill.return_value = None

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["signal"], signal.SIGTERM)
        self.assertEqual(result.metadata["pid"], 12345)

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @patch("os.kill")
    def test_inject_sigkill(self, mock_kill):
        """Test SIGKILL injection."""
        injector = ProcessKillInjector(signal_type="SIGKILL")
        mock_kill.return_value = None

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["signal"], signal.SIGKILL)
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    @patch("os.kill")
    def test_inject_with_permission_error(self, mock_kill):
        """Test injection with permission error."""
        mock_kill.side_effect = PermissionError("Operation not permitted")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("PermissionError", result.metadata["error"])

    @patch("os.kill")
    def test_inject_with_process_not_found(self, mock_kill):
        """Test injection when process doesn't exist."""
        mock_kill.side_effect = ProcessLookupError("No such process")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("ProcessLookupError", result.metadata["error"])

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)
        self.assertIn("note", recover_result.metadata)

    def test_invalid_signal_type(self):
        """Test with invalid signal type."""
        with self.assertRaises(ValueError):
            ProcessKillInjector(signal_type="INVALID")


class TestProcessExitInjector(unittest.TestCase):
    """Test cases for ProcessExitInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = ProcessExitInjector(exit_code=1, graceful_shutdown=True)
        self.context = FaultContext(target="test_process", fault_type="process_exit", parameters={"pid": 12345})

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.exit_code, 1)
        self.assertTrue(self.injector.graceful_shutdown)

    @patch("os.kill")
    @patch("subprocess.run")
    def test_inject_graceful_exit(self, mock_run, mock_kill):
        """Test graceful process exit."""
        # Mock that process exists initially
        mock_kill.side_effect = [None, None]  # First call to check existence, second to send signal
        mock_run.return_value = Mock(returncode=0)

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["exit_code"], 1)
        self.assertEqual(result.metadata["graceful"], True)

    @patch("os.kill")
    def test_inject_force_exit(self, mock_kill):
        """Test forceful process exit."""
        injector = ProcessExitInjector(exit_code=9, graceful_shutdown=False)
        mock_kill.return_value = None

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["exit_code"], 9)
        self.assertEqual(result.metadata["graceful"], False)

        # Should send SIGKILL for force exit
        mock_kill.assert_called_with(12345, signal.SIGKILL)

    @patch("os.kill")
    def test_inject_with_process_not_found(self, mock_kill):
        """Test injection when process doesn't exist."""
        mock_kill.side_effect = ProcessLookupError("No such process")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("ProcessLookupError", result.metadata["error"])

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)
        self.assertIn("note", recover_result.metadata)

    def test_invalid_exit_code(self):
        """Test with invalid exit code."""
        with self.assertRaises(ValueError):
            ProcessExitInjector(exit_code=-1)

        with self.assertRaises(ValueError):
            ProcessExitInjector(exit_code=256)


class TestProcessHangInjector(unittest.TestCase):
    """Test cases for ProcessHangInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = ProcessHangInjector(hang_duration=2, hang_type="infinite_loop")
        self.context = FaultContext(target="test_process", fault_type="process_hang", parameters={"pid": 12345})

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.hang_duration, 2)
        self.assertEqual(self.injector.hang_type, "infinite_loop")

    @patch("os.kill")
    def test_inject_infinite_loop(self, mock_kill):
        """Test infinite loop hang injection."""
        mock_kill.return_value = None

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["hang_type"], "infinite_loop")
        self.assertEqual(result.metadata["duration"], 2)

        # Verify signal was sent
        mock_kill.assert_called()

    @patch("os.kill")
    def test_inject_deadlock(self, mock_kill):
        """Test deadlock hang injection."""
        injector = ProcessHangInjector(hang_duration=1, hang_type="deadlock")
        mock_kill.return_value = None

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["hang_type"], "deadlock")

    @patch("os.kill")
    def test_inject_resource_wait(self, mock_kill):
        """Test resource wait hang injection."""
        injector = ProcessHangInjector(hang_duration=1, hang_type="resource_wait")
        mock_kill.return_value = None

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["hang_type"], "resource_wait")

    @patch("os.kill")
    def test_inject_with_signal_error(self, mock_kill):
        """Test injection with signal error."""
        mock_kill.side_effect = OSError("Failed to send signal")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)

    @patch("os.kill")
    def test_recover(self, mock_kill):
        """Test recovery method."""
        # First inject
        mock_kill.return_value = None
        result = self.injector.inject(self.context)

        # Then recover
        mock_kill.reset_mock()
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)

        # Should send SIGCONT to resume process
        mock_kill.assert_called_with(12345, signal.SIGCONT)

    def test_invalid_hang_type(self):
        """Test with invalid hang type."""
        with self.assertRaises(ValueError):
            ProcessHangInjector(hang_type="invalid")

    def test_invalid_duration(self):
        """Test with invalid duration."""
        with self.assertRaises(ValueError):
            ProcessHangInjector(hang_duration=-1)


class TestProcessFaultsIntegration(unittest.TestCase):
    """Integration tests for process fault injectors."""

    @patch("os.kill")
    def test_sequential_process_faults(self, mock_kill):
        """Test applying multiple process faults sequentially."""
        kill_injector = ProcessKillInjector(signal_type="SIGTERM")
        exit_injector = ProcessExitInjector(exit_code=1)
        hang_injector = ProcessHangInjector(hang_duration=1)

        context = FaultContext(target="test_process", fault_type="process_combined", parameters={"pid": 12345})

        # Apply kill
        mock_kill.return_value = None
        kill_result = kill_injector.inject(context)
        self.assertEqual(kill_result.status, "injected")

        # Apply exit
        exit_result = exit_injector.inject(context)
        self.assertEqual(exit_result.status, "injected")

        # Apply hang
        hang_result = hang_injector.inject(context)
        self.assertEqual(hang_result.status, "injected")

        # Recover all
        kill_recover = kill_injector.recover(context, kill_result)
        exit_recover = exit_injector.recover(context, exit_result)
        hang_recover = hang_injector.recover(context, hang_result)

        self.assertEqual(kill_recover.status, "recovered")
        self.assertEqual(exit_recover.status, "recovered")
        self.assertEqual(hang_recover.status, "recovered")

    def test_process_fault_with_different_pids(self):
        """Test process faults with different PIDs."""
        injector = ProcessKillInjector()

        # Test with different PIDs
        for pid in [1000, 2000, 3000]:
            context = FaultContext(target=f"process_{pid}", fault_type="process_kill", parameters={"pid": pid})

            with patch("os.kill") as mock_kill:
                mock_kill.return_value = None
                result = injector.inject(context)
                self.assertEqual(result.metadata["pid"], pid)


if __name__ == "__main__":
    unittest.main()
