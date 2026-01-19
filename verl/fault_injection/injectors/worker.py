"""Worker layer fault injectors for verl fault injection system."""

import gc
import os
import signal
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from verl.fault_injection.base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from verl.fault_injection.config import FaultType, WorkerFaultConfig


@FaultInjectorRegistry.register(FaultType.FSDP_SYNC_FAILURE)
class FSDPSyncFailureInjector(BaseFaultInjector):
    """Injector for FSDP synchronization failures."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject FSDP synchronization failure."""
        sync_type = self.config.fsdp_sync_type

        if sync_type == "all_reduce":
            # Simulate all_reduce failure by not participating
            if dist.is_initialized() and dist.get_rank() == context.rank:
                # Skip the all_reduce operation
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"sync_type": sync_type, "action": "skipped_all_reduce"}
                )

        elif sync_type == "broadcast":
            # Simulate broadcast failure
            if dist.is_initialized() and dist.get_rank() == 0:
                # Root rank fails to broadcast
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"sync_type": sync_type, "action": "broadcast_failed"}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message=f"FSDP sync failure not applicable for {sync_type}",
            metadata={"sync_type": sync_type}
        )


@FaultInjectorRegistry.register(FaultType.FSDP_SHARDING_ERROR)
class FSDPShardingErrorInjector(BaseFaultInjector):
    """Injector for FSDP sharding errors."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject FSDP sharding error."""
        stage = self.config.fsdp_sharding_stage

        if stage == "forward":
            # Corrupt forward pass parameters
            if hasattr(torch.nn.Module, '_forward_unimpl'):
                # Store original forward
                original_forward = torch.nn.Module._forward_unimpl

                def corrupt_forward(self, *args, **kwargs):
                    # Corrupt parameters
                    for param in self.parameters():
                        if param is not None and hasattr(param, 'data'):
                            param.data.fill_(float('nan'))
                    return original_forward(*args, **kwargs)

                torch.nn.Module._forward_unimpl = corrupt_forward

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"stage": stage, "action": "corrupted_parameters"}
                )

        elif stage == "backward":
            # Corrupt gradient computation
            if hasattr(torch.Tensor, 'backward'):
                original_backward = torch.Tensor.backward

                def corrupt_backward(self, gradient=None, retain_graph=None, create_graph=False):
                    # Corrupt gradients
                    if gradient is not None:
                        gradient.fill_(float('nan'))
                    return original_backward(self, gradient, retain_graph, create_graph)

                torch.Tensor.backward = corrupt_backward

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"stage": stage, "action": "corrupted_gradients"}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message=f"FSDP sharding error not implemented for {stage}",
            metadata={"stage": stage}
        )


@FaultInjectorRegistry.register(FaultType.MEGATRON_SYNC_FAILURE)
class MegatronSyncFailureInjector(BaseFaultInjector):
    """Injector for Megatron synchronization failures."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject Megatron synchronization failure."""
        sync_op = self.config.megatron_sync_op

        if sync_op == "all_reduce":
            # Simulate all_reduce failure in Megatron
            if dist.is_initialized():
                # Create a tensor that will cause all_reduce to fail
                tensor = torch.tensor([float('inf')], device='cuda' if torch.cuda.is_available() else 'cpu')

                try:
                    # This should cause overflow in all_reduce
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                except Exception as e:
                    return FaultResult(
                        fault_id=self.config.name,
                        status=FaultStatus.COMPLETED,
                        start_time=time.time(),
                        error_message=str(e),
                        metadata={"sync_op": sync_op, "action": "all_reduce_overflow"}
                    )

        elif sync_op == "broadcast":
            # Simulate broadcast failure
            if dist.is_initialized() and dist.get_rank() == 0:
                # Root rank sends invalid data
                tensor = torch.tensor([float('nan')], device='cuda' if torch.cuda.is_available() else 'cpu')
                dist.broadcast(tensor, src=0)

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"sync_op": sync_op, "action": "broadcast_nan"}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message=f"Megatron sync failure not applicable for {sync_op}",
            metadata={"sync_op": sync_op}
        )


@FaultInjectorRegistry.register(FaultType.MEGATRON_PIPELINE_ERROR)
class MegatronPipelineErrorInjector(BaseFaultInjector):
    """Injector for Megatron pipeline errors."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject Megatron pipeline error."""
        stage = self.config.megatron_pipeline_stage
        virtual_pipeline = self.config.megatron_virtual_pipeline

        if stage == "forward":
            # Cause forward pass to fail in pipeline
            def corrupt_pipeline_forward(*args, **kwargs):
                raise RuntimeError(f"Pipeline forward error in stage {virtual_pipeline or 'unknown'}")

            # Store and replace forward function
            if hasattr(torch.nn.Module, 'forward'):
                original_forward = torch.nn.Module.forward
                torch.nn.Module.forward = corrupt_pipeline_forward

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"stage": stage, "virtual_pipeline": virtual_pipeline, "action": "corrupted_forward"}
                )

        elif stage == "backward":
            # Cause backward pass to fail
            def corrupt_pipeline_backward(self, grad_output):
                # Return None to break gradient flow
                return None

            if hasattr(torch.nn.Module, 'backward'):
                original_backward = torch.nn.Module.backward
                torch.nn.Module.backward = corrupt_pipeline_backward

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={"stage": stage, "virtual_pipeline": virtual_pipeline, "action": "corrupted_backward"}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message=f"Megatron pipeline error not implemented for {stage}",
            metadata={"stage": stage}
        )


@FaultInjectorRegistry.register(FaultType.GRADIENT_SYNC_TIMEOUT)
class GradientSyncTimeoutInjector(BaseFaultInjector):
    """Injector for gradient synchronization timeouts."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject gradient synchronization timeout."""
        timeout_ms = self.config.gradient_sync_timeout_ms

        if dist.is_initialized():
            # Set a very short timeout to cause timeout
            original_timeout = dist.get_timeout()
            dist.set_timeout(timeout_ms / 1000.0)  # Convert to seconds

            # Create a blocking operation that will timeout
            tensor = torch.zeros(1000000, device='cuda' if torch.cuda.is_available() else 'cpu')

            try:
                # This should timeout
                dist.all_reduce(tensor)
            except Exception as e:
                # Restore original timeout
                dist.set_timeout(original_timeout)

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    error_message=str(e),
                    metadata={"timeout_ms": timeout_ms, "action": "gradient_sync_timeout"}
                )

            # Restore original timeout
            dist.set_timeout(original_timeout)

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message="Gradient sync timeout not applicable",
            metadata={"timeout_ms": timeout_ms}
        )


@FaultInjectorRegistry.register(FaultType.CUDA_OOM_WORKER)
class CudaOOMWorkerInjector(BaseFaultInjector):
    """Injector for CUDA out of memory errors in workers."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject CUDA OOM error."""
        memory_mb = self.config.cuda_memory_mb

        if torch.cuda.is_available():
            try:
                # Try to allocate a large tensor that will cause OOM
                elements = (memory_mb * 1024 * 1024) // 4  # 4 bytes per float32
                tensor = torch.zeros(elements, dtype=torch.float32, device='cuda')

                # Force allocation
                tensor.fill_(1.0)

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error_message="CUDA OOM injection failed - memory was allocated successfully",
                    metadata={"memory_mb": memory_mb, "action": "allocation_succeeded"}
                )
            except torch.cuda.OutOfMemoryError as e:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    error_message=str(e),
                    metadata={"memory_mb": memory_mb, "action": "cuda_oom"}
                )
            except Exception as e:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error_message=str(e),
                    metadata={"memory_mb": memory_mb}
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message="CUDA not available",
            metadata={"memory_mb": memory_mb}
        )


@FaultInjectorRegistry.register(FaultType.CUDA_MEMORY_FRAGMENTATION)
class CudaMemoryFragmentationInjector(BaseFaultInjector):
    """Injector for CUDA memory fragmentation."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject CUDA memory fragmentation."""
        fragment_size_mb = self.config.cuda_fragment_size_mb
        fragment_count = self.config.cuda_fragment_count

        if torch.cuda.is_available():
            fragments = []

            try:
                # Allocate many small tensors to fragment memory
                elements = (fragment_size_mb * 1024 * 1024) // 4

                for i in range(fragment_count):
                    fragment = torch.zeros(elements, dtype=torch.float32, device='cuda')
                    fragments.append(fragment)

                    # Fill with pattern to ensure allocation
                    fragment.fill_(float(i % 100))

                # Delete every other fragment to create fragmentation
                for i in range(0, len(fragments), 2):
                    del fragments[i]

                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()

                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    metadata={
                        "fragment_size_mb": fragment_size_mb,
                        "fragment_count": fragment_count,
                        "action": "memory_fragmented"
                    }
                )
            except torch.cuda.OutOfMemoryError as e:
                return FaultResult(
                    fault_id=self.config.name,
                    status=FaultStatus.COMPLETED,
                    start_time=time.time(),
                    error_message=str(e),
                    metadata={
                        "fragment_size_mb": fragment_size_mb,
                        "fragment_count": fragment_count,
                        "action": "fragmentation_oom"
                    }
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message="CUDA not available",
            metadata={
                "fragment_size_mb": fragment_size_mb,
                "fragment_count": fragment_count
            }
        )


@FaultInjectorRegistry.register(FaultType.WORKER_CRASH)
class WorkerCrashInjector(BaseFaultInjector):
    """Injector for worker crashes."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject worker crash."""
        crash_method = self.config.crash_method

        if crash_method == "exception":
            # Raise an exception
            raise RuntimeError(f"Simulated worker crash: {self.config.name}")

        elif crash_method == "exit":
            # Exit the process
            os._exit(1)

        elif crash_method == "signal":
            # Send SIGKILL to self
            os.kill(os.getpid(), signal.SIGKILL)

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.FAILED,
            start_time=time.time(),
            error_message=f"Unknown crash method: {crash_method}",
            metadata={"crash_method": crash_method}
        )


@FaultInjectorRegistry.register(FaultType.WORKER_HANG)
class WorkerHangInjector(BaseFaultInjector):
    """Injector for worker hangs."""

    def __init__(self, config: WorkerFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject worker hang."""
        hang_location = self.config.hang_location
        duration = self.config.duration_seconds or 300.0  # Default 5 minutes

        start_time = time.time()

        # Hang in an infinite loop
        while time.time() - start_time < duration:
            # Busy wait to simulate hang
            pass

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={
                "hang_location": hang_location,
                "duration": duration,
                "action": "worker_hung"
            }
        )