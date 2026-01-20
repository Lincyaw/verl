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

"""Unit tests for inference layer fault injectors."""

import threading
import time
from unittest.mock import patch

import pytest
import torch

from verl.fault_injection.base import FaultContext, FaultStatus
from verl.fault_injection.config import InferenceFaultConfig
from verl.fault_injection.injectors.inference import (
    CompilationFailureInjector,
    InferenceOOMInjector,
    SchedulerDeadlockInjector,
    SGLangFaultInjector,
    vLLMFaultInjector,
)


class TestInferenceOOMInjector:
    """Test cases for InferenceOOMInjector."""

    def test_vllm_oom_injection(self):
        """Test vLLM OOM fault injection."""
        config = InferenceFaultConfig(
            backend="vllm",
            kv_cache_size_mb=100,  # Small size for testing
        )
        injector = InferenceOOMInjector(config)
        context = FaultContext()

        # Mock CUDA availability
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.empty") as mock_empty:
                # Test successful injection
                result = injector.inject(context)

                assert result.status == FaultStatus.INJECTED
                assert "CUDA OOM error injected" in result.message
                assert result.data["backend"] == "vllm"
                assert result.data["requested_mb"] == 100

                # Verify tensor allocation was attempted
                assert mock_empty.called

    def test_sglang_oom_injection(self):
        """Test SGLang OOM fault injection."""
        config = InferenceFaultConfig(backend="sglang", kv_cache_size_mb=200)
        injector = InferenceOOMInjector(config)
        context = FaultContext()

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.empty"):
                result = injector.inject(context)

                assert result.status == FaultStatus.INJECTED
                assert result.data["backend"] == "sglang"
                assert result.data["requested_mb"] == 200

    def test_oom_recovery(self):
        """Test OOM recovery."""
        config = InferenceFaultConfig(backend="vllm")
        injector = InferenceOOMInjector(config)

        # Allocate some dummy tensors
        injector._allocated_tensors = [torch.empty(100), torch.empty(200)]

        with patch("torch.cuda.empty_cache") as mock_cache_clear:
            with patch("gc.collect") as mock_gc:
                injector.recover(FaultContext())

                # Verify cleanup was performed
                assert len(injector._allocated_tensors) == 0
                mock_cache_clear.assert_called_once()
                mock_gc.assert_called_once()


class TestSchedulerDeadlockInjector:
    """Test cases for SchedulerDeadlockInjector."""

    def test_request_queue_deadlock(self):
        """Test request queue deadlock injection."""
        config = InferenceFaultConfig(backend="vllm", deadlock_type="request_queue")
        injector = SchedulerDeadlockInjector(config)
        context = FaultContext()

        # Test injection
        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "vLLM scheduler deadlock" in str(exc_info.value)
        assert "Request queue is blocked" in str(exc_info.value)

    def test_kv_cache_deadlock(self):
        """Test KV cache deadlock injection."""
        config = InferenceFaultConfig(backend="sglang", deadlock_type="kv_cache")
        injector = SchedulerDeadlockInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "SGLANG scheduler deadlock" in str(exc_info.value)
        assert "KV cache access deadlock" in str(exc_info.value)

    def test_batch_schedule_deadlock(self):
        """Test batch scheduling deadlock."""
        config = InferenceFaultConfig(backend="vllm", deadlock_type="batch_schedule")
        injector = SchedulerDeadlockInjector(config)
        context = FaultContext()

        # This will hang, so we need to test with timeout
        def inject_with_timeout():
            with pytest.raises(RuntimeError) as exc_info:
                injector.inject(context)
            return str(exc_info.value)

        # Run in thread with timeout
        result_thread = threading.Thread(target=inject_with_timeout)
        result_thread.start()
        result_thread.join(timeout=2.0)  # 2 second timeout

        # Thread should still be alive (deadlocked)
        assert result_thread.is_alive()

        # Clean up
        injector.recover(context)
        result_thread.join(timeout=1.0)

    def test_deadlock_recovery(self):
        """Test deadlock recovery."""
        config = InferenceFaultConfig()
        injector = SchedulerDeadlockInjector(config)

        # Set up deadlock state
        injector._should_deadlock.set()
        injector._deadlock_thread = threading.Thread(target=lambda: time.sleep(0.1))
        injector._deadlock_thread.start()

        # Recover
        injector.recover(FaultContext())

        assert not injector._should_deadlock.is_set()


class TestCompilationFailureInjector:
    """Test cases for CompilationFailureInjector."""

    def test_vllm_graph_capture_failure(self):
        """Test vLLM CUDA graph capture failure."""
        config = InferenceFaultConfig(backend="vllm", compilation_stage="graph_capture")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "vLLM compilation failed" in str(exc_info.value)
        assert "CUDA graph capture failed" in str(exc_info.value)

    def test_vllm_optimization_failure(self):
        """Test vLLM graph optimization failure."""
        config = InferenceFaultConfig(backend="vllm", compilation_stage="optimization")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "Graph optimization failed" in str(exc_info.value)
        assert "tensor parallelism" in str(exc_info.value)

    def test_vllm_memory_planning_failure(self):
        """Test vLLM memory planning failure."""
        config = InferenceFaultConfig(backend="vllm", compilation_stage="memory_planning")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "Memory planning failed" in str(exc_info.value)
        assert "KV cache blocks" in str(exc_info.value)

    def test_sglang_radix_attention_failure(self):
        """Test SGLang RadixAttention compilation failure."""
        config = InferenceFaultConfig(backend="sglang", compilation_stage="radix_attention")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "SGLang compilation failed" in str(exc_info.value)
        assert "RadixAttention kernel compilation failed" in str(exc_info.value)

    def test_sglang_flashinfer_failure(self):
        """Test SGLang FlashInfer compilation failure."""
        config = InferenceFaultConfig(backend="sglang", compilation_stage="flashinfer")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "FlashInfer kernel compilation failed" in str(exc_info.value)
        assert "compute capabilities" in str(exc_info.value)

    def test_sglang_torch_compile_failure(self):
        """Test SGLang torch.compile failure."""
        config = InferenceFaultConfig(backend="sglang", compilation_stage="torch_compile")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "torch.compile() failed" in str(exc_info.value)
        assert "dynamic shapes" in str(exc_info.value)

    def test_unknown_backend(self):
        """Test unknown backend."""
        config = InferenceFaultConfig(backend="unknown")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(ValueError) as exc_info:
            injector.inject(context)

        assert "Unknown backend" in str(exc_info.value)

    def test_unknown_compilation_stage(self):
        """Test unknown compilation stage."""
        config = InferenceFaultConfig(backend="vllm", compilation_stage="unknown_stage")
        injector = CompilationFailureInjector(config)
        context = FaultContext()

        with pytest.raises(RuntimeError) as exc_info:
            injector.inject(context)

        assert "compilation failed at stage: unknown_stage" in str(exc_info.value)


class TestvLLMFaultInjector:
    """Test cases for vLLMFaultInjector helper class."""

    def test_kv_cache_fault_injection(self):
        """Test KV cache fault injection."""
        # Test with low probability (should not inject)
        with patch("random.random", return_value=0.5):  # > 0.1
            result = vLLMFaultInjector.inject_kv_cache_fault(num_tokens=1000, head_dim=128, num_heads=32, num_layers=40)
            assert result is None

        # Test with high probability (should inject)
        with patch("random.random", return_value=0.05):  # < 0.1
            with pytest.raises(RuntimeError) as exc_info:
                vLLMFaultInjector.inject_kv_cache_fault(num_tokens=1000, head_dim=128, num_heads=32, num_layers=40)

            assert "vLLM KV cache allocation failed" in str(exc_info.value)

    def test_scheduler_fault_injection(self):
        """Test scheduler fault injection."""
        # Test with small queue (should not inject)
        result = vLLMFaultInjector.inject_scheduler_fault(request_queue_size=50)
        assert result is None

        # Test with large queue and low probability
        with patch("random.random", return_value=0.5):  # > 0.05
            result = vLLMFaultInjector.inject_scheduler_fault(request_queue_size=150)
            assert result is None

        # Test with large queue and high probability
        with patch("random.random", return_value=0.02):  # < 0.05
            with pytest.raises(RuntimeError) as exc_info:
                vLLMFaultInjector.inject_scheduler_fault(request_queue_size=150)

            assert "vLLM scheduler fault" in str(exc_info.value)

    def test_attention_fault_injection(self):
        """Test attention fault injection."""
        # Test with short sequence (should not inject)
        result = vLLMFaultInjector.inject_attention_fault(seq_len=100, max_seq_len=1000)
        assert result is None

        # Test with long sequence and low probability
        with patch("random.random", return_value=0.5):  # > 0.1
            result = vLLMFaultInjector.inject_attention_fault(seq_len=950, max_seq_len=1000)
            assert result is None

        # Test with long sequence and high probability
        with patch("random.random", return_value=0.05):  # < 0.1
            with pytest.raises(RuntimeError) as exc_info:
                vLLMFaultInjector.inject_attention_fault(seq_len=950, max_seq_len=1000)

            assert "vLLM attention fault" in str(exc_info.value)


class TestSGLangFaultInjector:
    """Test cases for SGLangFaultInjector helper class."""

    def test_radix_attention_fault(self):
        """Test RadixAttention fault injection."""
        # Test with high cache hit rate (should not inject)
        result = SGLangFaultInjector.inject_radix_attention_fault(cache_hit_rate=0.8)
        assert result is None

        # Test with low cache hit rate and low probability
        with patch("random.random", return_value=0.5):  # > 0.1
            result = SGLangFaultInjector.inject_radix_attention_fault(cache_hit_rate=0.3)
            assert result is None

        # Test with low cache hit rate and high probability
        with patch("random.random", return_value=0.05):  # < 0.1
            with pytest.raises(RuntimeError) as exc_info:
                SGLangFaultInjector.inject_radix_attention_fault(cache_hit_rate=0.3)

            assert "SGLang RadixAttention fault" in str(exc_info.value)

    def test_memory_pool_fault(self):
        """Test memory pool fault injection."""
        # Test with low usage (should not inject)
        result = SGLangFaultInjector.inject_memory_pool_fault(allocated_mb=500, total_mb=1000)
        assert result is None

        # Test with high usage and low probability
        with patch("random.random", return_value=0.5):  # > 0.15
            result = SGLangFaultInjector.inject_memory_pool_fault(allocated_mb=970, total_mb=1000)
            assert result is None

        # Test with high usage and high probability
        with patch("random.random", return_value=0.1):  # < 0.15
            with pytest.raises(RuntimeError) as exc_info:
                SGLangFaultInjector.inject_memory_pool_fault(allocated_mb=970, total_mb=1000)

            assert "SGLang memory pool fault" in str(exc_info.value)

    def test_tokenizer_fault(self):
        """Test tokenizer fault injection."""
        # Test with valid token ID (should not inject)
        result = SGLangFaultInjector.inject_tokenizer_fault(vocab_size=50000, requested_id=1000)
        assert result is None

        # Test with invalid token ID and low probability
        with patch("random.random", return_value=0.5):  # > 0.2
            result = SGLangFaultInjector.inject_tokenizer_fault(vocab_size=50000, requested_id=50000)
            assert result is None

        # Test with invalid token ID and high probability
        with patch("random.random", return_value=0.1):  # < 0.2
            with pytest.raises(RuntimeError) as exc_info:
                SGLangFaultInjector.inject_tokenizer_fault(vocab_size=50000, requested_id=50000)

            assert "SGLang tokenizer fault" in str(exc_info.value)
