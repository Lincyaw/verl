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

"""Inference layer fault injectors for verl fault injection system."""

import gc
import random
import threading
import time

import torch

from verl.fault_injection.base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from verl.fault_injection.config import FaultType, InferenceFaultConfig


@FaultInjectorRegistry.register(FaultType.INFERENCE_OOM)
class InferenceOOMInjector(BaseFaultInjector):
    """Injector for inference out-of-memory errors in vLLM/SGLang."""

    def __init__(self, config: InferenceFaultConfig):
        self.config = config
        self._allocated_tensors = []

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject OOM by allocating large tensors in KV cache."""
        backend = self.config.backend or "vllm"
        kv_cache_size_mb = self.config.kv_cache_size_mb or 1024  # Default 1GB

        # Calculate number of elements for the desired memory size
        # Assuming float32 (4 bytes per element)
        num_elements = (kv_cache_size_mb * 1024 * 1024) // 4

        try:
            # Allocate large tensor to consume memory
            if backend == "vllm":
                # Simulate vLLM KV cache allocation
                tensor = torch.empty(num_elements, dtype=torch.float32, device="cuda")
                self._allocated_tensors.append(tensor)

                # Also allocate some smaller tensors to fragment memory
                for _ in range(10):
                    small_tensor = torch.empty(num_elements // 10, dtype=torch.float32, device="cuda")
                    self._allocated_tensors.append(small_tensor)

                raise RuntimeError(
                    f"vLLM KV cache allocation failed: Out of memory trying to allocate {kv_cache_size_mb}MB"
                )

            elif backend == "sglang":
                # Simulate SGLang memory allocation
                tensor = torch.empty(num_elements, dtype=torch.float32, device="cuda")
                self._allocated_tensors.append(tensor)

                # Simulate memory fragmentation
                for i in range(5):
                    frag_size = num_elements // (2 ** (i + 1))
                    frag_tensor = torch.empty(frag_size, dtype=torch.float32, device="cuda")
                    self._allocated_tensors.append(frag_tensor)

                raise RuntimeError(
                    f"SGLang memory allocation failed: Out of memory trying to allocate {kv_cache_size_mb}MB"
                )

            else:
                raise ValueError(f"Unknown backend: {backend}")

        except torch.cuda.OutOfMemoryError as e:
            # If we actually run out of memory, that's also a valid fault
            return FaultResult(
                status=FaultStatus.INJECTED,
                message=f"CUDA OOM error injected: {str(e)}",
                data={"backend": backend, "requested_mb": kv_cache_size_mb},
            )

    def recover(self, context: FaultContext) -> None:
        """Recover by freeing allocated tensors and clearing cache."""
        # Free all allocated tensors
        for tensor in self._allocated_tensors:
            del tensor
        self._allocated_tensors.clear()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()


@FaultInjectorRegistry.register(FaultType.SCHEDULER_DEADLOCK)
class SchedulerDeadlockInjector(BaseFaultInjector):
    """Injector for scheduler deadlocks in inference backends."""

    def __init__(self, config: InferenceFaultConfig):
        self.config = config
        self._deadlock_lock = threading.Lock()
        self._deadlock_thread = None
        self._should_deadlock = threading.Event()

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject scheduler deadlock by creating circular wait conditions."""
        backend = self.config.backend or "vllm"
        deadlock_type = self.config.deadlock_type or "request_queue"

        if deadlock_type == "request_queue":
            # Simulate request queue deadlock
            self._should_deadlock.set()

            # Create a thread that will try to acquire the same lock
            def deadlock_thread():
                with self._deadlock_lock:
                    # This thread holds the lock and waits for something
                    while self._should_deadlock.is_set():
                        time.sleep(0.1)

            self._deadlock_thread = threading.Thread(target=deadlock_thread)
            self._deadlock_thread.start()

            # Main thread also tries to acquire the lock (deadlock)
            with self._deadlock_lock:
                # This will never be reached if deadlock_thread holds the lock
                pass

            raise RuntimeError(f"{backend.upper()} scheduler deadlock: Request queue is blocked")

        elif deadlock_type == "kv_cache":
            # Simulate KV cache access deadlock
            cache_lock1 = threading.Lock()
            cache_lock2 = threading.Lock()

            def cache_deadlock_thread():
                with cache_lock1:
                    time.sleep(0.1)  # Ensure thread 1 gets lock1 first
                    with cache_lock2:
                        pass

            thread = threading.Thread(target=cache_deadlock_thread)
            thread.start()

            # Main thread gets locks in opposite order
            with cache_lock2:
                time.sleep(0.1)
                with cache_lock1:  # This will deadlock
                    pass

            raise RuntimeError(f"{backend.upper()} scheduler deadlock: KV cache access deadlock")

        elif deadlock_type == "batch_schedule":
            # Simulate batch scheduling deadlock
            batch_ready = threading.Event()
            batch_complete = threading.Event()

            def batch_thread():
                # Wait for batch to be ready
                batch_ready.wait()
                # Try to complete batch but need main thread
                while not batch_complete.is_set():
                    time.sleep(0.1)

            thread = threading.Thread(target=batch_thread)
            thread.start()

            # Main thread sets ready but waits for completion
            batch_ready.set()
            # This creates a circular dependency
            batch_complete.wait()  # Will wait forever

            raise RuntimeError(f"{backend.upper()} scheduler deadlock: Batch scheduling circular dependency")

        else:
            raise ValueError(f"Unknown deadlock type: {deadlock_type}")

    def recover(self, context: FaultContext) -> None:
        """Recover by releasing locks and stopping deadlock threads."""
        self._should_deadlock.clear()

        if self._deadlock_thread and self._deadlock_thread.is_alive():
            # We can't safely kill a thread with a lock, so we just clear the flag
            # In a real system, this would require process restart
            self._deadlock_thread.join(timeout=1.0)

        # Reset state
        self._deadlock_thread = None


@FaultInjectorRegistry.register(FaultType.COMPILATION_FAILURE)
class CompilationFailureInjector(BaseFaultInjector):
    """Injector for model compilation failures in inference backends."""

    def __init__(self, config: InferenceFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject compilation failure at various stages."""
        backend = self.config.backend or "vllm"
        compilation_stage = self.config.compilation_stage or "graph_capture"

        if backend == "vllm":
            if compilation_stage == "graph_capture":
                # Simulate CUDA graph capture failure
                raise RuntimeError(
                    "vLLM compilation failed: CUDA graph capture failed. "
                    "The model contains operations that cannot be captured in CUDA graphs. "
                    "This may be due to dynamic control flow or unsupported operations."
                )

            elif compilation_stage == "optimization":
                # Simulate graph optimization failure
                raise RuntimeError(
                    "vLLM compilation failed: Graph optimization failed. "
                    "Failed to apply tensor parallelism optimization. "
                    "Model architecture may not be compatible with current optimization passes."
                )

            elif compilation_stage == "memory_planning":
                # Simulate memory planning failure
                raise RuntimeError(
                    "vLLM compilation failed: Memory planning failed. "
                    "Cannot allocate sufficient memory for KV cache blocks. "
                    "Requested block size exceeds available GPU memory."
                )

            else:
                raise RuntimeError(f"vLLM compilation failed at stage: {compilation_stage}")

        elif backend == "sglang":
            if compilation_stage == "radix_attention":
                # Simulate RadixAttention compilation failure
                raise RuntimeError(
                    "SGLang compilation failed: RadixAttention kernel compilation failed. "
                    "The RadixAttention operation could not be compiled for the current GPU architecture. "
                    "This may be due to incompatible CUDA version or missing dependencies."
                )

            elif compilation_stage == "flashinfer":
                # Simulate FlashInfer compilation failure
                raise RuntimeError(
                    "SGLang compilation failed: FlashInfer kernel compilation failed. "
                    "FlashInfer operations require specific GPU compute capabilities. "
                    "Current GPU may not support required features."
                )

            elif compilation_stage == "torch_compile":
                # Simulate Torch compilation failure
                raise RuntimeError(
                    "SGLang compilation failed: torch.compile() failed. "
                    "Model graph contains dynamic shapes that cannot be compiled. "
                    "Consider disabling torch.compile or adjusting model configuration."
                )

            else:
                raise RuntimeError(f"SGLang compilation failed at stage: {compilation_stage}")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def recover(self, context: FaultContext) -> None:
        """Recovery typically requires recompilation or restart."""
        pass


class vLLMFaultInjector:
    """Helper class for vLLM-specific fault injection hooks."""

    @staticmethod
    def inject_kv_cache_fault(num_tokens: int, head_dim: int, num_heads: int, num_layers: int):
        """Inject fault during KV cache allocation in vLLM."""
        # Calculate required memory
        cache_size = num_tokens * head_dim * num_heads * num_layers * 2  # *2 for K and V

        # Check if we should inject fault
        if random.random() < 0.1:  # 10% chance
            raise RuntimeError(
                f"vLLM KV cache allocation failed: "
                f"Requested {cache_size * 4 / 1024 / 1024:.2f}MB for "
                f"{num_tokens} tokens with {num_heads} heads"
            )

    @staticmethod
    def inject_scheduler_fault(request_queue_size: int):
        """Inject fault in vLLM scheduler."""
        if request_queue_size > 100 and random.random() < 0.05:  # 5% chance for large queues
            raise RuntimeError(
                "vLLM scheduler fault: Request queue size exceeds maximum limit. "
                "Scheduler cannot handle more concurrent requests."
            )

    @staticmethod
    def inject_attention_fault(seq_len: int, max_seq_len: int):
        """Inject fault in attention computation."""
        if seq_len > max_seq_len * 0.9 and random.random() < 0.1:
            raise RuntimeError(
                f"vLLM attention fault: Sequence length {seq_len} exceeds maximum allowed length {max_seq_len}"
            )


class SGLangFaultInjector:
    """Helper class for SGLang-specific fault injection hooks."""

    @staticmethod
    def inject_radix_attention_fault(cache_hit_rate: float):
        """Inject fault in RadixAttention cache."""
        if cache_hit_rate < 0.5 and random.random() < 0.1:
            raise RuntimeError(
                f"SGLang RadixAttention fault: Cache hit rate {cache_hit_rate:.2f} is below minimum threshold 0.5"
            )

    @staticmethod
    def inject_memory_pool_fault(allocated_mb: int, total_mb: int):
        """Inject fault in memory pool allocation."""
        usage_ratio = allocated_mb / total_mb
        if usage_ratio > 0.95 and random.random() < 0.15:
            raise RuntimeError(f"SGLang memory pool fault: Memory usage {usage_ratio:.2f} exceeds safe threshold 0.95")

    @staticmethod
    def inject_tokenizer_fault(vocab_size: int, requested_id: int):
        """Inject fault in tokenizer."""
        if requested_id >= vocab_size and random.random() < 0.2:
            raise RuntimeError(f"SGLang tokenizer fault: Token ID {requested_id} exceeds vocabulary size {vocab_size}")
