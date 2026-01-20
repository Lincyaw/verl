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


"""Unit tests for engine layer fault injectors."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from verl.fault_injection.base import FaultContext, FaultStatus
from verl.fault_injection.config import (
    EngineFaultConfig,
    FaultTarget,
    FaultTargetConfig,
    FaultTrigger,
    FaultTriggerConfig,
)
from verl.fault_injection.injectors.engine import (
    CheckpointCorruptionInjector,
    DeviceMapErrorInjector,
    DeviceMeshErrorInjector,
    EngineHangInjector,
    EngineInitFailureInjector,
    NCCLFailureInjector,
    PrecisionErrorInjector,
)


class TestEngineFaultInjectors(unittest.TestCase):
    """Test engine layer fault injectors."""

    def setUp(self):
        """Set up test fixtures."""
        self.context = FaultContext(worker_id="test_worker", rank=0, host="localhost", process_type="actor")

    def test_engine_init_failure_injector(self):
        """Test engine initialization failure injector."""
        config = EngineFaultConfig(
            name="test_engine_init_failure",
            layer="engine",
            type="engine_init_failure",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            engine_type="fsdp",
            init_error="Test initialization error",
        )

        injector = EngineInitFailureInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(RuntimeError) as cm:
            injector.inject(self.context)

        self.assertIn("FSDP engine initialization failed", str(cm.exception))
        self.assertIn("Test initialization error", str(cm.exception))

    def test_engine_hang_injector(self):
        """Test engine hang injector."""
        config = EngineFaultConfig(
            name="test_engine_hang",
            layer="engine",
            type="engine_hang",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            hang_point="forward",
            hang_duration_seconds=0.1,  # Short duration for testing
        )

        injector = EngineHangInjector(config)

        # Test injection
        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status, FaultStatus.COMPLETED)
        self.assertEqual(result.metadata["hang_point"], "forward")
        self.assertAlmostEqual(end_time - start_time, 0.1, places=1)

    def test_checkpoint_corruption_random_bytes(self):
        """Test checkpoint corruption with random bytes."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Original checkpoint data that should be corrupted")
            checkpoint_path = tmp.name

        config = EngineFaultConfig(
            name="test_checkpoint_corruption",
            layer="engine",
            type="checkpoint_corruption",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            corruption_type="random_bytes",
            checkpoint_path=checkpoint_path,
        )

        injector = CheckpointCorruptionInjector(config)

        # Read original content
        with open(checkpoint_path, "rb") as f:
            original_content = f.read()

        # Inject fault
        result = injector.inject(self.context)

        self.assertEqual(result.status, FaultStatus.COMPLETED)
        self.assertEqual(result.metadata["corruption_type"], "random_bytes")

        # Verify file was corrupted
        with open(checkpoint_path, "rb") as f:
            corrupted_content = f.read()

        self.assertNotEqual(original_content, corrupted_content)

        # Clean up
        os.unlink(checkpoint_path)

    def test_checkpoint_corruption_truncate(self):
        """Test checkpoint corruption by truncation."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"This is a longer checkpoint file that will be truncated")
            checkpoint_path = tmp.name

        config = EngineFaultConfig(
            name="test_checkpoint_truncation",
            layer="engine",
            type="checkpoint_corruption",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            corruption_type="truncate",
            checkpoint_path=checkpoint_path,
        )

        injector = CheckpointCorruptionInjector(config)

        # Get original size
        original_size = os.path.getsize(checkpoint_path)

        # Inject fault
        result = injector.inject(self.context)

        self.assertEqual(result.status, FaultStatus.COMPLETED)

        # Verify file was truncated
        new_size = os.path.getsize(checkpoint_path)
        self.assertLess(new_size, original_size)
        self.assertGreater(new_size, 0)

        # Clean up
        os.unlink(checkpoint_path)

    def test_checkpoint_corruption_delete(self):
        """Test checkpoint corruption by deletion."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Checkpoint data")
            checkpoint_path = tmp.name

        config = EngineFaultConfig(
            name="test_checkpoint_delete",
            layer="engine",
            type="checkpoint_corruption",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            corruption_type="delete",
            checkpoint_path=checkpoint_path,
        )

        injector = CheckpointCorruptionInjector(config)

        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))

        # Inject fault
        result = injector.inject(self.context)

        self.assertEqual(result.status, FaultStatus.COMPLETED)

        # Verify file was deleted
        self.assertFalse(os.path.exists(checkpoint_path))

    @patch("os._exit")
    def test_nccl_failure_abort(self, mock_exit):
        """Test NCCL failure with abort."""
        config = EngineFaultConfig(
            name="test_nccl_abort",
            layer="engine",
            type="nccl_failure",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            nccl_error_type="abort",
        )

        injector = NCCLFailureInjector(config)

        # Inject fault - should call os._exit
        injector.inject(self.context)

        # Verify exit was called
        mock_exit.assert_called_once_with(1)

    @patch("builtins.print")  # Suppress print output
    def test_nccl_failure_timeout(self, mock_print):
        """Test NCCL failure with timeout."""
        config = EngineFaultConfig(
            name="test_nccl_timeout",
            layer="engine",
            type="nccl_failure",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            nccl_error_type="timeout",
            nccl_timeout_ms=100,  # Short timeout for testing
        )

        injector = NCCLFailureInjector(config)

        # Inject fault
        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status, FaultStatus.COMPLETED)
        self.assertEqual(result.metadata["nccl_error_type"], "timeout")
        self.assertGreaterEqual(end_time - start_time, 0.1)  # Should sleep for at least 0.1s

    def test_device_mesh_invalid_shape(self):
        """Test device mesh error with invalid shape."""
        config = EngineFaultConfig(
            name="test_device_mesh_invalid",
            layer="engine",
            type="device_mesh_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            mesh_error_type="invalid_shape",
            mesh_shape=[2, 2, 2],
        )

        injector = DeviceMeshErrorInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(ValueError) as cm:
            injector.inject(self.context)

        self.assertIn("Invalid device mesh shape", str(cm.exception))

    def test_device_mesh_device_mismatch(self):
        """Test device mesh error with device mismatch."""
        config = EngineFaultConfig(
            name="test_device_mesh_mismatch",
            layer="engine",
            type="device_mesh_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            mesh_error_type="device_mismatch",
            mesh_shape=[100, 100, 100],  # Requires 1M devices
        )

        injector = DeviceMeshErrorInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(RuntimeError) as cm:
            injector.inject(self.context)

        self.assertIn("Not enough devices available", str(cm.exception))

    def test_precision_error_overflow(self):
        """Test precision error with overflow."""
        config = EngineFaultConfig(
            name="test_precision_overflow",
            layer="engine",
            type="precision_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            precision_type="fp16",
            precision_error_type="overflow",
        )

        injector = PrecisionErrorInjector(config)

        # Test injection
        result = injector.inject(self.context)

        self.assertEqual(result.status, FaultStatus.COMPLETED)
        self.assertEqual(result.metadata["precision_type"], "fp16")
        self.assertEqual(result.metadata["error_type"], "overflow")

    def test_precision_error_nan(self):
        """Test precision error with NaN."""
        config = EngineFaultConfig(
            name="test_precision_nan",
            layer="engine",
            type="precision_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            precision_type="fp16",
            precision_error_type="nan",
        )

        injector = PrecisionErrorInjector(config)

        # Test injection
        result = injector.inject(self.context)

        self.assertEqual(result.status, FaultStatus.COMPLETED)
        self.assertEqual(result.metadata["precision_type"], "fp16")
        self.assertEqual(result.metadata["error_type"], "nan")

    @patch("torch.cuda.device_count")
    def test_device_map_missing_device(self, mock_device_count):
        """Test device map error with missing device."""
        mock_device_count.return_value = 4  # Only 4 devices available

        config = EngineFaultConfig(
            name="test_device_missing",
            layer="engine",
            type="device_map_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            device_map_error="missing_device",
            target_device=999,
        )

        injector = DeviceMapErrorInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(RuntimeError) as cm:
            injector.inject(self.context)

        self.assertIn("invalid device ordinal", str(cm.exception))

    def test_device_map_device_busy(self):
        """Test device map error with device busy."""
        config = EngineFaultConfig(
            name="test_device_busy",
            layer="engine",
            type="device_map_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            device_map_error="device_busy",
            target_device=0,
        )

        injector = DeviceMapErrorInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(RuntimeError) as cm:
            injector.inject(self.context)

        self.assertIn("Device 0 is busy", str(cm.exception))

    def test_device_map_peer_access(self):
        """Test device map error with peer access."""
        config = EngineFaultConfig(
            name="test_peer_access",
            layer="engine",
            type="device_map_error",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
            device_map_error="peer_access",
            target_device=1,
        )

        injector = DeviceMapErrorInjector(config)

        # Test that injection raises an exception
        with self.assertRaises(RuntimeError) as cm:
            injector.inject(self.context)

        self.assertIn("Peer access is not supported", str(cm.exception))

    def test_recovery_methods(self):
        """Test that recovery methods can be called without errors."""
        config = EngineFaultConfig(
            name="test_recovery",
            layer="engine",
            type="engine_init_failure",
            trigger=FaultTriggerConfig(type=FaultTrigger.IMMEDIATE),
            target=FaultTargetConfig(mode=FaultTarget.RANDOM, count=1),
        )

        # Test all injector recovery methods
        injectors = [
            EngineInitFailureInjector(config),
            EngineHangInjector(config),
            CheckpointCorruptionInjector(config),
            NCCLFailureInjector(config),
            DeviceMeshErrorInjector(config),
            PrecisionErrorInjector(config),
            DeviceMapErrorInjector(config),
        ]

        for injector in injectors:
            # Recovery should not raise any exceptions
            injector.recover(self.context)


if __name__ == "__main__":
    unittest.main()
