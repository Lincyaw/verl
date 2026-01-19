#!/usr/bin/env python3
"""
Fault injector that runs inside the child process.
Reads fault configuration from environment variables and injects faults at runtime.
"""

import json
import os
import random
import sys
import threading
import time

_injected = False
_stop_event = threading.Event()


def init():
    """Initialize fault injection from environment variables."""
    global _injected
    if _injected:
        return
    _injected = True

    config_json = os.environ.get("FAULT_INJECTION_CONFIG")
    if not config_json:
        return

    try:
        config = json.loads(config_json)
        faults = config.get("faults", [])

        # Start background threads for each fault
        for fault in faults:
            t = threading.Thread(target=_inject_fault, args=(fault,), daemon=True)
            t.start()
    except Exception as e:
        print(f"[fault-injector] Failed to initialize: {e}", file=sys.stderr)


def _inject_fault(fault_spec):
    """Execute fault injection based on trigger conditions."""
    try:
        # Wait for trigger
        trigger = fault_spec.get("trigger", {})
        trigger_type = trigger.get("type", "timed")

        if trigger_type == "timed":
            after_seconds = trigger.get("after_seconds", 0)
            if _stop_event.wait(after_seconds):
                return
        elif trigger_type == "probability":
            after_seconds = trigger.get("after_seconds", 0)
            if _stop_event.wait(after_seconds):
                return
            probability = trigger.get("probability", 1.0)
            if random.random() > probability:
                return

        # Check rank matching
        target_rank = fault_spec.get("target_rank")
        if target_rank is not None:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
            if rank != target_rank:
                return

        # Execute fault
        fault_type = fault_spec["type"]
        params = fault_spec.get("params", {})

        print(f"[fault-injector] Injecting fault: {fault_type}", file=sys.stderr)

        if fault_type == "nccl_timeout":
            _inject_nccl_timeout(params)
        elif fault_type == "nccl_hang":
            _inject_nccl_hang(params)
        elif fault_type == "cuda_oom":
            _inject_cuda_oom(params)
        elif fault_type == "cuda_error":
            _inject_cuda_error(params)
        elif fault_type == "gradient_nan":
            _inject_gradient_nan(params)
        elif fault_type == "gradient_explode":
            _inject_gradient_explode(params)
        elif fault_type == "code_sleep":
            _inject_code_sleep(params)
        elif fault_type == "code_oom":
            _inject_code_oom(params)
        elif fault_type == "ray_actor_crash":
            _inject_ray_actor_crash(params)
        elif fault_type == "ray_resource_exhausted":
            _inject_ray_resource_exhausted(params)
        elif fault_type == "vllm_init_fail":
            _inject_vllm_init_fail(params)
        elif fault_type == "vllm_kv_cache_oom":
            _inject_vllm_kv_cache_oom(params)
        elif fault_type == "rank_desync":
            _inject_rank_desync(params)
        elif fault_type == "barrier_timeout":
            _inject_barrier_timeout(params)
        elif fault_type == "checkpoint_corrupt":
            _inject_checkpoint_corrupt(params)
        elif fault_type == "checkpoint_missing":
            _inject_checkpoint_missing(params)
        elif fault_type == "batch_size_mismatch":
            _inject_batch_size_mismatch(params)
        elif fault_type == "env_modify":
            _inject_env_modify(params)

    except Exception as e:
        print(f"[fault-injector] Error injecting fault: {e}", file=sys.stderr)


def _inject_nccl_timeout(params):
    """Inject NCCL timeout by sleeping before collective operations."""
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return

        duration_sec = params.get("duration_sec", 120)
        original_all_reduce = dist.all_reduce

        def faulty_all_reduce(tensor, *args, **kwargs):
            print(f"[fault-injector] Sleeping {duration_sec}s to trigger NCCL timeout", file=sys.stderr)
            time.sleep(duration_sec)
            return original_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = faulty_all_reduce
    except ImportError:
        pass


def _inject_nccl_hang(params):
    """Inject NCCL hang by making barrier hang indefinitely."""
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return

        original_barrier = dist.barrier

        def hanging_barrier(*args, **kwargs):
            print("[fault-injector] Hanging barrier indefinitely", file=sys.stderr)
            while True:
                time.sleep(1)

        dist.barrier = hanging_barrier
    except ImportError:
        pass


def _inject_cuda_oom(params):
    """Trigger CUDA OOM by allocating large GPU memory."""
    try:
        import torch

        size_gb = params.get("size_gb", 100)
        print(f"[fault-injector] Allocating {size_gb}GB GPU memory", file=sys.stderr)
        _ = torch.zeros(size_gb * 1024 * 1024 * 256, device="cuda")
    except Exception as e:
        print(f"[fault-injector] CUDA OOM triggered: {e}", file=sys.stderr)


def _inject_cuda_error(params):
    """Trigger CUDA error by setting invalid device."""
    try:
        import torch

        print("[fault-injector] Triggering CUDA error", file=sys.stderr)
        torch.cuda.set_device(999)
    except Exception as e:
        print(f"[fault-injector] CUDA error triggered: {e}", file=sys.stderr)


def _inject_gradient_nan(params):
    """Inject NaN into gradients during backward pass."""
    try:
        import torch

        original_backward = torch.Tensor.backward

        def faulty_backward(self, *args, **kwargs):
            result = original_backward(self, *args, **kwargs)
            if self.grad is not None:
                print("[fault-injector] Injecting NaN into gradients", file=sys.stderr)
                self.grad.data.fill_(float("nan"))
            return result

        torch.Tensor.backward = faulty_backward
    except ImportError:
        pass


def _inject_gradient_explode(params):
    """Inject gradient explosion by multiplying gradients."""
    try:
        import torch

        multiplier = params.get("multiplier", 1e10)
        original_backward = torch.Tensor.backward

        def faulty_backward(self, *args, **kwargs):
            result = original_backward(self, *args, **kwargs)
            if self.grad is not None:
                print(f"[fault-injector] Multiplying gradients by {multiplier}", file=sys.stderr)
                self.grad.data.mul_(multiplier)
            return result

        torch.Tensor.backward = faulty_backward
    except ImportError:
        pass


def _inject_code_sleep(params):
    """Sleep to trigger timeouts."""
    duration_sec = params.get("duration_sec", 60)
    print(f"[fault-injector] Sleeping {duration_sec}s", file=sys.stderr)
    time.sleep(duration_sec)


def _inject_code_oom(params):
    """Allocate large memory to trigger OOM."""
    size_gb = params.get("size_gb", 100)
    print(f"[fault-injector] Allocating {size_gb}GB memory", file=sys.stderr)
    try:
        _ = bytearray(size_gb * 1024 * 1024 * 1024)
    except MemoryError:
        print("[fault-injector] OOM triggered", file=sys.stderr)


def _inject_ray_actor_crash(params):
    """Crash Ray actor by calling os._exit."""
    try:
        import ray

        if ray.is_initialized():
            print("[fault-injector] Crashing Ray actor", file=sys.stderr)
            os._exit(1)
    except ImportError:
        pass


def _inject_ray_resource_exhausted(params):
    """Simulate Ray resource exhaustion."""
    try:
        import ray

        original_get = ray.get

        def faulty_get(*args, **kwargs):
            print("[fault-injector] Simulating Ray resource exhaustion", file=sys.stderr)
            raise ray.exceptions.RayTaskError("Resource exhausted")

        ray.get = faulty_get
    except ImportError:
        pass


def _inject_vllm_init_fail(params):
    """Cause vLLM initialization failure."""
    print("[fault-injector] Setting invalid vLLM config", file=sys.stderr)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "invalid"


def _inject_vllm_kv_cache_oom(params):
    """Trigger vLLM KV cache OOM."""
    print("[fault-injector] Setting vLLM memory utilization > 1.0", file=sys.stderr)
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "1.5"


def _inject_rank_desync(params):
    """Desynchronize ranks by sleeping."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
    duration_sec = params.get("duration_sec", 60)
    print(f"[fault-injector] Rank {rank} sleeping {duration_sec}s to desync", file=sys.stderr)
    time.sleep(duration_sec)


def _inject_barrier_timeout(params):
    """Sleep before barrier to trigger timeout."""
    duration_sec = params.get("duration_sec", 60)
    print(f"[fault-injector] Sleeping {duration_sec}s before barrier", file=sys.stderr)
    time.sleep(duration_sec)


def _inject_checkpoint_corrupt(params):
    """Corrupt checkpoint file."""
    ckpt_path = params.get("checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[fault-injector] Corrupting checkpoint {ckpt_path}", file=sys.stderr)
        with open(ckpt_path, "wb") as f:
            f.write(b"corrupted")


def _inject_checkpoint_missing(params):
    """Delete checkpoint file."""
    ckpt_path = params.get("checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[fault-injector] Deleting checkpoint {ckpt_path}", file=sys.stderr)
        os.unlink(ckpt_path)


def _inject_batch_size_mismatch(params):
    """Set mismatched batch size."""
    batch_size = params.get("batch_size", 999)
    print(f"[fault-injector] Setting batch size to {batch_size}", file=sys.stderr)
    os.environ["BATCH_SIZE_OVERRIDE"] = str(batch_size)


def _inject_env_modify(params):
    """Modify environment variables."""
    for k, v in params.items():
        print(f"[fault-injector] Setting env {k}={v}", file=sys.stderr)
        os.environ[k] = str(v)
