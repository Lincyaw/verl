# Copyright 2026 Aoyang Fang Ltd. and/or its affiliates
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

#!/usr/bin/env python3
"""Comprehensive unit tests for all fault injectors."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.fault_injection import (
    EngineFaultConfig,
    FaultContext,
    FaultLayer,
    FaultType,
    InferenceFaultConfig,
    NetworkFaultConfig,
    OrchestrationFaultConfig,
    ProcessFaultConfig,
    UIFaultConfig,
    WorkerFaultConfig,
)
from verl.fault_injection.injectors import (
    ActorCrashInjector,
    CheckpointCorruptionInjector,
    CLIArgErrorInjector,
    ConnectionRefusedInjector,
    CudaOOMWorkerInjector,
    EngineInitFailureInjector,
    EnvVarErrorInjector,
    FSDPSyncFailureInjector,
    HydraConfigErrorInjector,
    # Inference Injectors
    InferenceOOMInjector,
    MemoryLeakInjector,
    NetworkLatencyInjector,
    ProcessCrashInjector,
    RayClusterFailureInjector,
    RayInitFailureInjector,
    ResourceExhaustionInjector,
    SchedulerDeadlockInjector,
    UICrashInjector,
    UIFreezeInjector,
)


class TestAllInjectors(unittest.TestCase):
    """Test all fault injectors comprehensively."""

    def setUp(self):
        """Set up test fixtures."""
        self.context = FaultContext()

    # UI Layer Tests
    def test_hydra_config_error_injector(self):
        """Test HydraConfigErrorInjector."""
        config = UIFaultConfig(
            name="test_hydra_error",
            layer=FaultLayer.UI,
            type=FaultType.HYDRA_CONFIG_ERROR,
            enabled=True,
            hydra_error_type="validate",
            description="Test Hydra config validation error",
        )

        injector = HydraConfigErrorInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("validation failed", result.error_message)
        self.assertEqual(result.fault_type, FaultType.HYDRA_CONFIG_ERROR)

    def test_ray_init_failure_injector(self):
        """Test RayInitFailureInjector."""
        config = UIFaultConfig(
            name="test_ray_init_failure",
            layer=FaultLayer.UI,
            type=FaultType.RAY_INIT_FAILURE,
            enabled=True,
            ray_init_error_type="timeout",
            ray_init_timeout=5.0,
            description="Test Ray initialization timeout",
        )

        injector = RayInitFailureInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Ray initialization failed", result.error_message)

    def test_cli_arg_error_injector(self):
        """Test CLIArgErrorInjector."""
        config = UIFaultConfig(
            name="test_cli_arg_error",
            layer=FaultLayer.UI,
            type=FaultType.CLI_ARG_ERROR,
            enabled=True,
            cli_arg_error_type="missing_required",
            missing_arg="--model-path",
            description="Test missing required argument",
        )

        injector = CLIArgErrorInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Missing required argument", result.error_message)

    def test_env_var_error_injector(self):
        """Test EnvVarErrorInjector."""
        config = UIFaultConfig(
            name="test_env_var_error",
            layer=FaultLayer.UI,
            type=FaultType.ENV_VAR_ERROR,
            enabled=True,
            env_var_error_type="not_set",
            env_var_name="CUDA_VISIBLE_DEVICES",
            description="Test environment variable not set",
        )

        injector = EnvVarErrorInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Environment variable error", result.error_message)

    def test_ui_freeze_injector(self):
        """Test UIFreezeInjector."""
        config = UIFaultConfig(
            name="test_ui_freeze",
            layer=FaultLayer.UI,
            type=FaultType.UI_FREEZE,
            enabled=True,
            freeze_duration=0.1,
            description="Test UI freeze",
        )

        injector = UIFreezeInjector(config)
        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status.value, "SUCCESS")
        self.assertGreaterEqual(end_time - start_time, 0.1)

    def test_ui_crash_injector(self):
        """Test UICrashInjector."""
        config = UIFaultConfig(
            name="test_ui_crash",
            layer=FaultLayer.UI,
            type=FaultType.UI_CRASH,
            enabled=True,
            crash_type="exception",
            description="Test UI crash",
        )

        injector = UICrashInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("UI crash", result.error_message)

    # Orchestration Layer Tests
    def test_ray_cluster_failure_injector(self):
        """Test RayClusterFailureInjector."""
        config = OrchestrationFaultConfig(
            name="test_ray_cluster_failure",
            layer=FaultLayer.ORCHESTRATION,
            type=FaultType.RAY_CLUSTER_FAILURE,
            enabled=True,
            cluster_failure_type="gcs_failure",
            duration=0.1,
            description="Test Ray cluster GCS failure",
        )

        injector = RayClusterFailureInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Ray cluster failure", result.error_message)

    def test_actor_crash_injector(self):
        """Test ActorCrashInjector."""
        config = OrchestrationFaultConfig(
            name="test_actor_crash",
            layer=FaultLayer.ORCHESTRATION,
            type=FaultType.ACTOR_CRASH,
            enabled=True,
            actor_crash_type="exception",
            actor_name_pattern="Worker*",
            description="Test actor crash",
        )

        injector = ActorCrashInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Actor crashed", result.error_message)

    def test_resource_exhaustion_injector(self):
        """Test ResourceExhaustionInjector."""
        config = OrchestrationFaultConfig(
            name="test_resource_exhaustion",
            layer=FaultLayer.ORCHESTRATION,
            type=FaultType.RESOURCE_EXHAUSTION,
            enabled=True,
            resource_type="memory",
            exhaustion_percentage=0.9,
            duration=0.1,
            description="Test memory exhaustion",
        )

        injector = ResourceExhaustionInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Resource exhaustion", result.error_message)

    # Worker Layer Tests
    def test_fsdp_sync_failure_injector(self):
        """Test FSDPSyncFailureInjector."""
        config = WorkerFaultConfig(
            name="test_fsdp_sync_failure",
            layer=FaultLayer.WORKER,
            type=FaultType.FSDP_SYNC_FAILURE,
            enabled=True,
            sync_operation="all_reduce",
            world_size=8,
            rank=0,
            description="Test FSDP all_reduce failure",
        )

        injector = FSDPSyncFailureInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("FSDP sync failure", result.error_message)

    def test_cuda_oom_worker_injector(self):
        """Test CudaOOMWorkerInjector."""
        config = WorkerFaultConfig(
            name="test_cuda_oom",
            layer=FaultLayer.WORKER,
            type=FaultType.CUDA_OOM,
            enabled=True,
            oom_size_mb=1024,
            description="Test CUDA OOM",
        )

        injector = CudaOOMWorkerInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("CUDA out of memory", result.error_message)

    # Engine Layer Tests
    def test_engine_init_failure_injector(self):
        """Test EngineInitFailureInjector."""
        config = EngineFaultConfig(
            name="test_engine_init_failure",
            layer=FaultLayer.ENGINE,
            type=FaultType.ENGINE_INIT_FAILURE,
            enabled=True,
            engine_type="fsdp",
            init_stage="model_load",
            description="Test engine initialization failure",
        )

        injector = EngineInitFailureInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Engine initialization failed", result.error_message)

    def test_checkpoint_corruption_injector(self):
        """Test CheckpointCorruptionInjector."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"original checkpoint data")
            checkpoint_path = tmp.name

        config = EngineFaultConfig(
            name="test_checkpoint_corruption",
            layer=FaultLayer.ENGINE,
            type=FaultType.CHECKPOINT_CORRUPTION,
            enabled=True,
            corruption_type="random_bytes",
            checkpoint_path=checkpoint_path,
            description="Test checkpoint corruption",
        )

        injector = CheckpointCorruptionInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Checkpoint corrupted", result.error_message)

        # Cleanup
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)

    # Inference Layer Tests
    def test_inference_oom_injector(self):
        """Test InferenceOOMInjector."""
        config = InferenceFaultConfig(
            name="test_inference_oom",
            layer=FaultLayer.INFERENCE,
            type=FaultType.INFERENCE_OOM,
            enabled=True,
            backend="vllm",
            oom_type="kv_cache",
            requested_memory_mb=1024,
            description="Test inference OOM",
        )

        injector = InferenceOOMInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Inference OOM", result.error_message)

    def test_scheduler_deadlock_injector(self):
        """Test SchedulerDeadlockInjector."""
        config = InferenceFaultConfig(
            name="test_scheduler_deadlock",
            layer=FaultLayer.INFERENCE,
            type=FaultType.SCHEDULER_DEADLOCK,
            enabled=True,
            deadlock_type="request_queue",
            max_queue_size=100,
            description="Test scheduler deadlock",
        )

        injector = SchedulerDeadlockInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Scheduler deadlock", result.error_message)

    # Process Layer Tests
    def test_process_crash_injector(self):
        """Test ProcessCrashInjector."""
        config = ProcessFaultConfig(
            name="test_process_crash",
            layer=FaultLayer.PROCESS,
            type=FaultType.PROCESS_CRASH,
            enabled=True,
            crash_type="sigkill",
            process_pattern="python",
            description="Test process crash",
        )

        injector = ProcessCrashInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Process crashed", result.error_message)

    def test_memory_leak_injector(self):
        """Test MemoryLeakInjector."""
        config = ProcessFaultConfig(
            name="test_memory_leak",
            layer=FaultLayer.PROCESS,
            type=FaultType.MEMORY_LEAK,
            enabled=True,
            leak_size_mb=10,
            leak_duration=0.1,
            description="Test memory leak",
        )

        injector = MemoryLeakInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "SUCCESS")
        self.assertIn("Memory leak", result.message)

    # Network Layer Tests
    def test_network_latency_injector(self):
        """Test NetworkLatencyInjector."""
        config = NetworkFaultConfig(
            name="test_network_latency",
            layer=FaultLayer.NETWORK,
            type=FaultType.NETWORK_LATENCY,
            enabled=True,
            latency_ms=100,
            jitter_ms=10,
            target_host="localhost",
            description="Test network latency",
        )

        injector = NetworkLatencyInjector(config)
        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status.value, "SUCCESS")
        self.assertGreaterEqual(end_time - start_time, 0.1)

    def test_connection_refused_injector(self):
        """Test ConnectionRefusedInjector."""
        config = NetworkFaultConfig(
            name="test_connection_refused",
            layer=FaultLayer.NETWORK,
            type=FaultType.CONNECTION_REFUSED,
            enabled=True,
            target_host="localhost",
            target_port=12345,
            description="Test connection refused",
        )

        injector = ConnectionRefusedInjector(config)
        result = injector.inject(self.context)

        self.assertEqual(result.status.value, "FAILED")
        self.assertIn("Connection refused", result.error_message)

    # Test Recovery Methods
    def test_injector_recovery(self):
        """Test that all injectors have working recovery methods."""
        configs = [
            UIFaultConfig(
                name="test_ui_recovery",
                layer=FaultLayer.UI,
                type=FaultType.HYDRA_CONFIG_ERROR,
                enabled=True,
                hydra_error_type="validate",
            ),
            OrchestrationFaultConfig(
                name="test_orchestration_recovery",
                layer=FaultLayer.ORCHESTRATION,
                type=FaultType.RAY_CLUSTER_FAILURE,
                enabled=True,
                cluster_failure_type="gcs_failure",
                duration=0.1,
            ),
            WorkerFaultConfig(
                name="test_worker_recovery",
                layer=FaultLayer.WORKER,
                type=FaultType.CUDA_OOM,
                enabled=True,
                oom_size_mb=100,
            ),
            EngineFaultConfig(
                name="test_engine_recovery",
                layer=FaultLayer.ENGINE,
                type=FaultType.ENGINE_INIT_FAILURE,
                enabled=True,
                engine_type="fsdp",
            ),
            InferenceFaultConfig(
                name="test_inference_recovery",
                layer=FaultLayer.INFERENCE,
                type=FaultType.INFERENCE_OOM,
                enabled=True,
                backend="vllm",
            ),
            ProcessFaultConfig(
                name="test_process_recovery",
                layer=FaultLayer.PROCESS,
                type=FaultType.PROCESS_CRASH,
                enabled=True,
                crash_type="exception",
            ),
            NetworkFaultConfig(
                name="test_network_recovery",
                layer=FaultLayer.NETWORK,
                type=FaultType.NETWORK_LATENCY,
                enabled=True,
                latency_ms=10,
            ),
        ]

        for config in configs:
            # Get the appropriate injector class based on config type
            if isinstance(config, UIFaultConfig):
                injector_class = HydraConfigErrorInjector
            elif isinstance(config, OrchestrationFaultConfig):
                injector_class = RayClusterFailureInjector
            elif isinstance(config, WorkerFaultConfig):
                injector_class = CudaOOMWorkerInjector
            elif isinstance(config, EngineFaultConfig):
                injector_class = EngineInitFailureInjector
            elif isinstance(config, InferenceFaultConfig):
                injector_class = InferenceOOMInjector
            elif isinstance(config, ProcessFaultConfig):
                injector_class = ProcessCrashInjector
            elif isinstance(config, NetworkFaultConfig):
                injector_class = NetworkLatencyInjector
            else:
                continue

            injector = injector_class(config)

            # Test recovery
            recovery_result = injector.recover(self.context)
            self.assertIsNotNone(recovery_result)
            self.assertIn(recovery_result.status.value, ["SUCCESS", "FAILED"])


if __name__ == "__main__":
    unittest.main()
