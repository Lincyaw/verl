"""Engine layer fault injectors for verl fault injection system."""

import os
import random
import threading
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from verl.fault_injection.base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from verl.fault_injection.config import EngineFaultConfig, FaultType


@FaultInjectorRegistry.register(FaultType.ENGINE_INIT_FAILURE)
class EngineInitFailureInjector(BaseFaultInjector):
    """Injector for engine initialization failures."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject engine initialization failure."""
        engine_type = self.config.engine_type or "fsdp"
        error_msg = self.config.init_error or f"Failed to initialize {engine_type} engine"

        # Raise an exception to simulate initialization failure
        if engine_type == "fsdp":
            raise RuntimeError(f"FSDP engine initialization failed: {error_msg}")
        elif engine_type == "megatron":
            raise RuntimeError(f"Megatron engine initialization failed: {error_msg}")
        else:
            raise RuntimeError(f"Engine initialization failed for {engine_type}: {error_msg}")

    def recover(self, context: FaultContext) -> None:
        """Recovery is typically done by restarting the process."""
        pass


@FaultInjectorRegistry.register(FaultType.ENGINE_HANG)
class EngineHangInjector(BaseFaultInjector):
    """Injector for engine hangs during execution."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config
        self._hang_event = threading.Event()

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject engine hang at specified point."""
        hang_point = self.config.hang_point or "forward"
        hang_duration = getattr(self.config, 'hang_duration_seconds', 300.0)

        start_time = time.time()

        # Log the hang
        print(f"Simulating engine hang at {hang_point} for {hang_duration} seconds")

        # Hang for specified duration
        time.sleep(hang_duration)

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"hang_point": hang_point, "duration": hang_duration}
        )

    def recover(self, context: FaultContext) -> None:
        """Recover from hang by setting the event."""
        self._hang_event.set()


@FaultInjectorRegistry.register(FaultType.CHECKPOINT_CORRUPTION)
class CheckpointCorruptionInjector(BaseFaultInjector):
    """Injector for checkpoint file corruption."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Corrupt checkpoint files."""
        corruption_type = self.config.corruption_type or "random_bytes"
        checkpoint_path = self.config.checkpoint_path or "model.ckpt"

        start_time = time.time()

        if not os.path.exists(checkpoint_path):
            # Create a dummy checkpoint file if it doesn't exist
            with open(checkpoint_path, 'wb') as f:
                f.write(b"dummy checkpoint data")

        if corruption_type == "random_bytes":
            # Corrupt with random bytes
            with open(checkpoint_path, 'r+b') as f:
                # Corrupt 10% of the file with random data
                f.seek(0, 2)
                file_size = f.tell()
                corrupt_size = max(1, file_size // 10)

                # Choose random position to corrupt
                corrupt_pos = random.randint(0, max(0, file_size - corrupt_size))
                f.seek(corrupt_pos)

                # Write random bytes
                random_data = bytes([random.randint(0, 255) for _ in range(corrupt_size)])
                f.write(random_data)

        elif corruption_type == "truncate":
            # Truncate the file
            with open(checkpoint_path, 'r+b') as f:
                f.truncate(random.randint(1, 100))

        elif corruption_type == "delete":
            # Delete the file
            os.remove(checkpoint_path)

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"corruption_type": corruption_type, "checkpoint_path": checkpoint_path}
        )

    def recover(self, context: FaultContext) -> None:
        """Recovery would involve restoring from backup."""
        pass


@FaultInjectorRegistry.register(FaultType.NCCL_FAILURE)
class NCCLFailureInjector(BaseFaultInjector):
    """Injector for NCCL communication failures."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject NCCL communication failure."""
        nccl_error_type = self.config.nccl_error_type or "timeout"
        timeout_ms = self.config.nccl_timeout_ms or 60000

        start_time = time.time()

        if nccl_error_type == "timeout":
            # Simulate NCCL timeout by sleeping longer than timeout
            print(f"Simulating NCCL timeout for {timeout_ms}ms")
            time.sleep(timeout_ms / 1000.0 + 1.0)

        elif nccl_error_type == "abort":
            # Simulate NCCL abort by calling abort on process group
            if dist.is_initialized():
                # This would typically cause the process to exit
                print("Simulating NCCL abort")
                os._exit(1)

        elif nccl_error_type == "crash":
            # Simulate NCCL crash by segfaulting
            print("Simulating NCCL crash")
            # Access invalid memory to simulate crash
            import ctypes
            ctypes.string_at(0)  # This will cause a segfault

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"nccl_error_type": nccl_error_type, "timeout_ms": timeout_ms}
        )

    def recover(self, context: FaultContext) -> None:
        """NCCL failures typically require process restart."""
        pass


@FaultInjectorRegistry.register(FaultType.DEVICE_MESH_ERROR)
class DeviceMeshErrorInjector(BaseFaultInjector):
    """Injector for device mesh configuration errors."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject device mesh error."""
        mesh_error_type = self.config.mesh_error_type or "invalid_shape"
        mesh_shape = self.config.mesh_shape or [2, 2, 2]

        start_time = time.time()

        if mesh_error_type == "invalid_shape":
            # Create invalid mesh shape
            invalid_shape = [0, -1, 2]  # Invalid dimensions
            print(f"Creating device mesh with invalid shape: {invalid_shape}")

            # This would typically be passed to the engine initialization
            raise ValueError(f"Invalid device mesh shape: {invalid_shape}")

        elif mesh_error_type == "device_mismatch":
            # Simulate device mismatch
            available_devices = torch.cuda.device_count()
            required_devices = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]

            if available_devices < required_devices:
                raise RuntimeError(
                    f"Not enough devices available. Required: {required_devices}, "
                    f"Available: {available_devices}"
                )

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"mesh_error_type": mesh_error_type, "mesh_shape": mesh_shape}
        )

    def recover(self, context: FaultContext) -> None:
        """Recovery involves fixing mesh configuration."""
        pass


@FaultInjectorRegistry.register(FaultType.PRECISION_ERROR)
class PrecisionErrorInjector(BaseFaultInjector):
    """Injector for precision conversion errors."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject precision conversion error."""
        precision_type = self.config.precision_type or "fp16"
        error_type = self.config.precision_error_type or "overflow"

        start_time = time.time()

        if error_type == "overflow":
            # Create values that would overflow in fp16
            if precision_type == "fp16":
                large_value = torch.tensor([1e10], dtype=torch.float32)
                # This would overflow when converted to fp16
                print(f"Creating fp16 overflow with value: {large_value}")

        elif error_type == "underflow":
            # Create values that would underflow
            if precision_type == "fp16":
                small_value = torch.tensor([1e-10], dtype=torch.float32)
                # This would underflow when converted to fp16
                print(f"Creating fp16 underflow with value: {small_value}")

        elif error_type == "nan":
            # Create NaN values
            nan_tensor = torch.tensor([float('nan')], dtype=torch.float32)
            print(f"Creating NaN values in {precision_type}")

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"precision_type": precision_type, "error_type": error_type}
        )

    def recover(self, context: FaultContext) -> None:
        """Recovery involves fixing precision settings."""
        pass


@FaultInjectorRegistry.register(FaultType.DEVICE_MAP_ERROR)
class DeviceMapErrorInjector(BaseFaultInjector):
    """Injector for device mapping errors."""

    def __init__(self, config: EngineFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject device mapping error."""
        error_type = self.config.device_map_error or "missing_device"
        target_device = self.config.target_device or 999

        start_time = time.time()

        if error_type == "missing_device":
            # Try to use a device that doesn't exist
            print(f"Attempting to map to non-existent device: {target_device}")
            if target_device >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA error: invalid device ordinal. Requested device {target_device}, "
                                   f"but only {torch.cuda.device_count()} devices available")

        elif error_type == "device_busy":
            # Simulate device being busy
            print(f"Simulating device {target_device} being busy")
            # In real scenario, this would happen when trying to allocate memory
            raise RuntimeError(f"Device {target_device} is busy or memory is fragmented")

        elif error_type == "peer_access":
            # Simulate peer access error
            print(f"Simulating peer access error between devices")
            # This would happen in multi-GPU setups
            raise RuntimeError("Peer access is not supported between these devices")

        return FaultResult(
            fault_id=self.config.name,
            status=FaultStatus.COMPLETED,
            start_time=start_time,
            end_time=time.time(),
            metadata={"error_type": error_type, "target_device": target_device}
        )

    def recover(self, context: FaultContext) -> None:
        """Recovery involves fixing device mapping."""
        pass