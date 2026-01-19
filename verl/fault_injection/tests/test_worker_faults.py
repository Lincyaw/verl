"""Tests for worker layer fault injectors."""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from verl.fault_injection.config import FaultLayer, FaultTargetConfig, FaultTriggerConfig, WorkerFaultConfig
from verl.fault_injection.injectors.worker import (
    CudaMemoryFragmentationInjector,
    CudaOOMWorkerInjector,
    FSDPShardingErrorInjector,
    FSDPSyncFailureInjector,
    GradientSyncTimeoutInjector,
    MegatronPipelineErrorInjector,
    MegatronSyncFailureInjector,
    WorkerCrashInjector,
    WorkerHangInjector,
)


class TestWorkerFaultInjectors(unittest.TestCase):
    """Test cases for worker layer fault injectors."""

    def setUp(self):
        """Set up test fixtures."""
        self.context = MagicMock()
        self.context.worker_id = "test_worker"
        self.context.rank = 0
        self.context.host = "localhost"
        self.context.process_type = "actor"
        self.context.layer = FaultLayer.WORKER

    def test_fsdp_sync_failure_all_reduce(self):
        """Test FSDP all_reduce sync failure."""
        config = WorkerFaultConfig(
            name="test_fsdp_all_reduce",
            layer=FaultLayer.WORKER,
            type="fsdp_sync_failure",
            fsdp_sync_type="all_reduce",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = FSDPSyncFailureInjector(config)

        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_rank', return_value=0):

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["sync_type"], "all_reduce")
            self.assertEqual(result.metadata["action"], "skipped_all_reduce")

    def test_fsdp_sync_failure_broadcast(self):
        """Test FSDP broadcast sync failure."""
        config = WorkerFaultConfig(
            name="test_fsdp_broadcast",
            layer=FaultLayer.WORKER,
            type="fsdp_sync_failure",
            fsdp_sync_type="broadcast",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = FSDPSyncFailureInjector(config)

        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_rank', return_value=0):

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["sync_type"], "broadcast")
            self.assertEqual(result.metadata["action"], "broadcast_failed")

    def test_fsdp_sharding_error_forward(self):
        """Test FSDP forward sharding error."""
        config = WorkerFaultConfig(
            name="test_fsdp_forward",
            layer=FaultLayer.WORKER,
            type="fsdp_sharding_error",
            fsdp_sharding_stage="forward",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = FSDPShardingErrorInjector(config)

        # Mock torch.nn.Module
        with patch.object(torch.nn.Module, '_forward_unimpl', create=True) as mock_forward:
            mock_forward.return_value = None

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["stage"], "forward")
            self.assertEqual(result.metadata["action"], "corrupted_parameters")

    def test_megatron_sync_failure_all_reduce_overflow(self):
        """Test Megatron all_reduce overflow."""
        config = WorkerFaultConfig(
            name="test_megatron_all_reduce",
            layer=FaultLayer.WORKER,
            type="megatron_sync_failure",
            megatron_sync_op="all_reduce",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = MegatronSyncFailureInjector(config)

        # Mock distributed environment and tensor operations
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce') as mock_all_reduce:

            mock_all_reduce.side_effect = RuntimeError("Overflow in all_reduce")

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["sync_op"], "all_reduce")
            self.assertEqual(result.metadata["action"], "all_reduce_overflow")

    def test_megatron_pipeline_error_forward(self):
        """Test Megatron pipeline forward error."""
        config = WorkerFaultConfig(
            name="test_megatron_pipeline",
            layer=FaultLayer.WORKER,
            type="megatron_pipeline_error",
            megatron_pipeline_stage="forward",
            megatron_virtual_pipeline=1,
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = MegatronPipelineErrorInjector(config)

        # Mock torch.nn.Module
        with patch.object(torch.nn.Module, 'forward', create=True) as mock_forward:
            mock_forward.return_value = None

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["stage"], "forward")
            self.assertEqual(result.metadata["virtual_pipeline"], 1)
            self.assertEqual(result.metadata["action"], "corrupted_forward")

    def test_gradient_sync_timeout(self):
        """Test gradient synchronization timeout."""
        config = WorkerFaultConfig(
            name="test_gradient_timeout",
            layer=FaultLayer.WORKER,
            type="gradient_sync_timeout",
            gradient_sync_timeout_ms=100,
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = GradientSyncTimeoutInjector(config)

        # Mock distributed environment
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_timeout', return_value=30.0), \
             patch('torch.distributed.set_timeout') as mock_set_timeout, \
             patch('torch.distributed.all_reduce') as mock_all_reduce:

            mock_all_reduce.side_effect = RuntimeError("Timeout")

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["timeout_ms"], 100)
            self.assertEqual(result.metadata["action"], "gradient_sync_timeout")

            # Verify timeout was set
            mock_set_timeout.assert_called_with(0.1)

    @patch('torch.cuda.is_available')
    def test_cuda_oom_worker_success(self, mock_cuda_available):
        """Test CUDA OOM injection when CUDA is available."""
        mock_cuda_available.return_value = True

        config = WorkerFaultConfig(
            name="test_cuda_oom",
            layer=FaultLayer.WORKER,
            type="cuda_oom_worker",
            cuda_memory_mb=1024,
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = CudaOOMWorkerInjector(config)

        # Mock CUDA allocation to raise OOM
        with patch('torch.zeros', side_effect=torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 1.00 GiB")):

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["memory_mb"], 1024)
            self.assertEqual(result.metadata["action"], "cuda_oom")

    @patch('torch.cuda.is_available')
    def test_cuda_oom_worker_no_cuda(self, mock_cuda_available):
        """Test CUDA OOM injection when CUDA is not available."""
        mock_cuda_available.return_value = False

        config = WorkerFaultConfig(
            name="test_cuda_oom_no_cuda",
            layer=FaultLayer.WORKER,
            type="cuda_oom_worker",
            cuda_memory_mb=1024,
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = CudaOOMWorkerInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.error_message, "CUDA not available")

    @patch('torch.cuda.is_available')
    @patch('gc.collect')
    @patch('torch.cuda.empty_cache')
    def test_cuda_memory_fragmentation(self, mock_empty_cache, mock_gc, mock_cuda_available):
        """Test CUDA memory fragmentation."""
        mock_cuda_available.return_value = True

        config = WorkerFaultConfig(
            name="test_cuda_fragmentation",
            layer=FaultLayer.WORKER,
            type="cuda_memory_fragmentation",
            cuda_fragment_size_mb=64,
            cuda_fragment_count=100,
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = CudaMemoryFragmentationInjector(config)

        # Mock tensor allocation
        with patch('torch.zeros') as mock_zeros:
            mock_tensor = MagicMock()
            mock_zeros.return_value = mock_tensor

            result = injector.inject(self.context)

            self.assertEqual(result.status.value, "completed")
            self.assertEqual(result.metadata["fragment_size_mb"], 64)
            self.assertEqual(result.metadata["fragment_count"], 100)
            self.assertEqual(result.metadata["action"], "memory_fragmented")

            # Verify garbage collection was called
            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()

    def test_worker_crash_exception(self):
        """Test worker crash with exception."""
        config = WorkerFaultConfig(
            name="test_worker_crash",
            layer=FaultLayer.WORKER,
            type="worker_crash",
            crash_method="exception",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = WorkerCrashInjector(config)

        with self.assertRaises(RuntimeError) as context:
            injector.inject(self.context)

        self.assertIn("Simulated worker crash", str(context.exception))

    @patch('os._exit')
    def test_worker_crash_exit(self, mock_exit):
        """Test worker crash with exit."""
        config = WorkerFaultConfig(
            name="test_worker_exit",
            layer=FaultLayer.WORKER,
            type="worker_crash",
            crash_method="exit",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = WorkerCrashInjector(config)
        injector.inject(self.context)

        mock_exit.assert_called_once_with(1)

    @patch('os.kill')
    @patch('os.getpid')
    def test_worker_crash_signal(self, mock_getpid, mock_kill):
        """Test worker crash with signal."""
        mock_getpid.return_value = 12345

        config = WorkerFaultConfig(
            name="test_worker_signal",
            layer=FaultLayer.WORKER,
            type="worker_crash",
            crash_method="signal",
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = WorkerCrashInjector(config)
        injector.inject(self.context)

        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_worker_hang(self):
        """Test worker hang."""
        config = WorkerFaultConfig(
            name="test_worker_hang",
            layer=FaultLayer.WORKER,
            type="worker_hang",
            hang_location="forward",
            duration_seconds=1.0,  # Short duration for testing
            target=FaultTargetConfig(),
            trigger=FaultTriggerConfig(),
        )

        injector = WorkerHangInjector(config)

        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status.value, "completed")
        self.assertEqual(result.metadata["hang_location"], "forward")
        self.assertEqual(result.metadata["duration"], 1.0)
        self.assertEqual(result.metadata["action"], "worker_hung")

        # Verify it actually waited
        self.assertGreaterEqual(end_time - start_time, 1.0)


if __name__ == '__main__':
    unittest.main()