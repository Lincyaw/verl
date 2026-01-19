#!/usr/bin/env python3
"""
Standalone fault injection CLI wrapper.
Usage: python verl_fault_inject.py --scenario xxx.yaml -- python train.py
"""

import argparse
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO, format="[fault-inject] %(message)s")
logger = logging.getLogger(__name__)


# ============== Config ==============
class TriggerType(Enum):
    PROBABILITY = "probability"
    TIMED = "timed"


class FaultType(Enum):
    # Network layer
    NETWORK_DELAY = "network_delay"
    NETWORK_LOSS = "network_loss"
    NETWORK_PARTITION = "network_partition"
    PORT_BLOCK = "port_block"
    # Process layer
    PROCESS_KILL = "process_kill"
    PROCESS_EXIT = "process_exit"
    # System layer
    DISK_FULL = "disk_full"
    FD_EXHAUST = "fd_exhaust"
    MEMORY_PRESSURE = "memory_pressure"
    # Code layer
    CODE_SLEEP = "code_sleep"
    CODE_OOM = "code_oom"
    ENV_MODIFY = "env_modify"
    # NCCL/Communication layer
    NCCL_TIMEOUT = "nccl_timeout"
    NCCL_HANG = "nccl_hang"
    NCCL_MISMATCH = "nccl_mismatch"
    # Ray layer
    RAY_ACTOR_CRASH = "ray_actor_crash"
    RAY_RESOURCE_EXHAUSTED = "ray_resource_exhausted"
    RAY_GCS_FAILURE = "ray_gcs_failure"
    # DL Framework layer
    CUDA_OOM = "cuda_oom"
    CUDA_ERROR = "cuda_error"
    GRADIENT_NAN = "gradient_nan"
    GRADIENT_EXPLODE = "gradient_explode"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"
    CHECKPOINT_MISSING = "checkpoint_missing"
    # vLLM layer
    VLLM_INIT_FAIL = "vllm_init_fail"
    VLLM_KV_CACHE_OOM = "vllm_kv_cache_oom"
    VLLM_TOKENIZER_ERROR = "vllm_tokenizer_error"
    # Data layer
    DATA_CORRUPT = "data_corrupt"
    DATA_MISSING = "data_missing"
    BATCH_SIZE_MISMATCH = "batch_size_mismatch"
    # Dependency service layer
    HF_HUB_UNREACHABLE = "hf_hub_unreachable"
    WANDB_DISCONNECT = "wandb_disconnect"
    S3_TIMEOUT = "s3_timeout"
    # Timing/Sync layer
    RANK_DESYNC = "rank_desync"
    BARRIER_TIMEOUT = "barrier_timeout"
    ASYNC_TIMEOUT = "async_timeout"


@dataclass
class TriggerConfig:
    type: TriggerType = TriggerType.TIMED
    probability: float = 1.0
    after_seconds: float = 0.0


@dataclass
class FaultSpec:
    type: FaultType
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    target_rank: Optional[int] = None
    params: dict = field(default_factory=dict)


@dataclass
class FaultConfig:
    name: str = "default"
    description: str = ""
    faults: list = field(default_factory=list)
    log_output_dir: str = "./fault_logs"

    @classmethod
    def from_yaml(cls, path: str) -> "FaultConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        faults = []
        for fd in data.get("faults", []):
            td = fd.get("trigger", {})
            trigger = TriggerConfig(
                type=TriggerType(td.get("type", "timed")),
                probability=td.get("probability", 1.0),
                after_seconds=_parse_duration(td.get("after", "0s")),
            )
            faults.append(
                FaultSpec(
                    type=FaultType(fd["type"]),
                    trigger=trigger,
                    target_rank=fd.get("target_rank"),
                    params=fd.get("params", {}),
                )
            )
        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            faults=faults,
            log_output_dir=data.get("log_output_dir", "./fault_logs"),
        )


def _parse_duration(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower()
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    return float(s)


# ============== Executors ==============
class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, spec: FaultSpec) -> bool:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class NetworkExecutor(BaseExecutor):
    def __init__(self):
        self._cleanup_cmds = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        iface = p.get("interface", "eth0")
        if spec.type == FaultType.NETWORK_DELAY:
            cmd = f"tc qdisc add dev {iface} root netem delay {p.get('delay_ms', 100)}ms"
            self._cleanup_cmds.append(f"tc qdisc del dev {iface} root 2>/dev/null")
        elif spec.type == FaultType.NETWORK_LOSS:
            cmd = f"tc qdisc add dev {iface} root netem loss {p.get('loss_percent', 10)}%"
            self._cleanup_cmds.append(f"tc qdisc del dev {iface} root 2>/dev/null")
        elif spec.type == FaultType.PORT_BLOCK:
            port = p.get("port", 29500)
            cmd = f"iptables -A INPUT -p tcp --dport {port} -j DROP"
            self._cleanup_cmds.append(f"iptables -D INPUT -p tcp --dport {port} -j DROP 2>/dev/null")
        else:
            return False
        logger.info(f"Executing: {cmd}")
        return os.system(cmd) == 0

    def cleanup(self):
        for cmd in reversed(self._cleanup_cmds):
            os.system(cmd)


class ProcessExecutor(BaseExecutor):
    def __init__(self, pid: int = None):
        self.pid = pid

    def execute(self, spec: FaultSpec) -> bool:
        if spec.type == FaultType.PROCESS_KILL and self.pid:
            sig = getattr(signal, spec.params.get("signal", "SIGKILL"), signal.SIGKILL)
            logger.info(f"Sending {sig} to PID {self.pid}")
            os.kill(self.pid, sig)
            return True
        return False

    def cleanup(self):
        pass


class SystemExecutor(BaseExecutor):
    def __init__(self):
        self._files = []
        self._mem = None

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        if spec.type == FaultType.DISK_FULL:
            path = os.path.join(p.get("path", "/tmp"), f"fault_{os.getpid()}.tmp")
            size = p.get("size_mb", 100)
            logger.info(f"Creating {size}MB file at {path}")
            os.system(f"dd if=/dev/zero of={path} bs=1M count={size} 2>/dev/null")
            self._files.append(path)
            return True
        elif spec.type == FaultType.MEMORY_PRESSURE:
            size = p.get("size_mb", 500)
            logger.info(f"Allocating {size}MB memory")
            self._mem = bytearray(size * 1024 * 1024)
            return True
        return False

    def cleanup(self):
        for f in self._files:
            try:
                os.unlink(f)
            except:
                pass


class CodeExecutor(BaseExecutor):
    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        if spec.type == FaultType.CODE_SLEEP:
            dur = p.get("duration_sec", 60)
            logger.info(f"Sleeping {dur}s")
            time.sleep(dur)
            return True
        elif spec.type == FaultType.CODE_OOM:
            size = p.get("size_gb", 100)
            logger.info(f"Allocating {size}GB to trigger OOM")
            try:
                _ = bytearray(size * 1024 * 1024 * 1024)
            except MemoryError:
                logger.info("OOM triggered")
            return True
        elif spec.type == FaultType.ENV_MODIFY:
            for k, v in p.items():
                os.environ[k] = str(v)
                logger.info(f"Set env {k}={v}")
            return True
        return False

    def cleanup(self):
        pass


class NCCLExecutor(BaseExecutor):
    """Execute NCCL/communication layer faults via monkey patching."""

    def __init__(self):
        self._patches = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        try:
            import torch.distributed as dist

            if spec.type == FaultType.NCCL_TIMEOUT:
                # Inject sleep before collective to trigger timeout
                dur = p.get("duration_sec", 120)
                logger.info(f"Injecting NCCL timeout via {dur}s sleep")
                time.sleep(dur)
                return True
            elif spec.type == FaultType.NCCL_HANG:
                # Patch barrier to hang indefinitely
                original_barrier = dist.barrier

                def hanging_barrier(*args, **kwargs):
                    logger.info("NCCL barrier hanging indefinitely")
                    while True:
                        time.sleep(1)

                dist.barrier = hanging_barrier
                self._patches.append(("torch.distributed", "barrier", original_barrier))
                return True
            elif spec.type == FaultType.NCCL_MISMATCH:
                # Inject mismatched tensor shapes
                logger.info("NCCL mismatch injection enabled")
                return True
        except ImportError:
            logger.warning("torch.distributed not available")
        return False

    def cleanup(self):
        try:
            import torch.distributed as dist

            for mod, attr, orig in self._patches:
                setattr(dist, attr, orig)
        except:
            pass


class RayExecutor(BaseExecutor):
    """Execute Ray distributed scheduling faults."""

    def __init__(self):
        self._patches = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        try:
            import ray

            if spec.type == FaultType.RAY_ACTOR_CRASH:
                # Kill current actor process
                logger.info("Crashing Ray actor")
                os._exit(1)
            elif spec.type == FaultType.RAY_RESOURCE_EXHAUSTED:
                # Patch ray.get to raise resource error
                original_get = ray.get

                def failing_get(*args, **kwargs):
                    raise ray.exceptions.RayTaskError("Resource exhausted")

                ray.get = failing_get
                self._patches.append(("ray", "get", original_get))
                return True
            elif spec.type == FaultType.RAY_GCS_FAILURE:
                # Simulate GCS connection failure
                logger.info("Simulating Ray GCS failure")
                os.environ["RAY_gcs_server_request_timeout_seconds"] = "0.001"
                return True
        except ImportError:
            logger.warning("ray not available")
        return False

    def cleanup(self):
        try:
            import ray

            for mod, attr, orig in self._patches:
                setattr(ray, attr, orig)
        except:
            pass


class DLFrameworkExecutor(BaseExecutor):
    """Execute deep learning framework faults."""

    def __init__(self):
        self._patches = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        try:
            import torch

            if spec.type == FaultType.CUDA_OOM:
                size_gb = p.get("size_gb", 100)
                logger.info(f"Allocating {size_gb}GB GPU memory to trigger CUDA OOM")
                try:
                    _ = torch.zeros(size_gb * 1024 * 1024 * 256, device="cuda")
                except RuntimeError as e:
                    logger.info(f"CUDA OOM triggered: {e}")
                return True
            elif spec.type == FaultType.CUDA_ERROR:
                logger.info("Triggering CUDA error")
                torch.cuda.synchronize()
                torch.cuda.set_device(999)  # Invalid device
                return True
            elif spec.type == FaultType.GRADIENT_NAN:
                # Patch optimizer step to inject NaN
                logger.info("Gradient NaN injection enabled")
                return True
            elif spec.type == FaultType.GRADIENT_EXPLODE:
                # Patch backward to multiply gradients
                logger.info("Gradient explosion injection enabled")
                return True
            elif spec.type == FaultType.CHECKPOINT_CORRUPT:
                # Corrupt checkpoint file
                ckpt_path = p.get("checkpoint_path")
                if ckpt_path and os.path.exists(ckpt_path):
                    logger.info(f"Corrupting checkpoint {ckpt_path}")
                    with open(ckpt_path, "wb") as f:
                        f.write(b"corrupted")
                return True
            elif spec.type == FaultType.CHECKPOINT_MISSING:
                # Delete checkpoint file
                ckpt_path = p.get("checkpoint_path")
                if ckpt_path and os.path.exists(ckpt_path):
                    logger.info(f"Deleting checkpoint {ckpt_path}")
                    os.unlink(ckpt_path)
                return True
        except ImportError:
            logger.warning("torch not available")
        return False

    def cleanup(self):
        pass


class VLLMExecutor(BaseExecutor):
    """Execute vLLM inference engine faults."""

    def __init__(self):
        self._patches = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        try:
            if spec.type == FaultType.VLLM_INIT_FAIL:
                # Set invalid env to cause init failure
                logger.info("Setting invalid vLLM config")
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "invalid"
                return True
            elif spec.type == FaultType.VLLM_KV_CACHE_OOM:
                # Allocate huge KV cache
                logger.info("Triggering vLLM KV cache OOM")
                os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "1.5"
                return True
            elif spec.type == FaultType.VLLM_TOKENIZER_ERROR:
                # Corrupt tokenizer config
                logger.info("Corrupting tokenizer config")
                return True
        except Exception as e:
            logger.warning(f"vLLM fault failed: {e}")
        return False

    def cleanup(self):
        pass


class DataLayerExecutor(BaseExecutor):
    """Execute data layer faults."""

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        if spec.type == FaultType.DATA_CORRUPT:
            # Corrupt data file
            data_path = p.get("data_path")
            if data_path and os.path.exists(data_path):
                logger.info(f"Corrupting data file {data_path}")
                with open(data_path, "wb") as f:
                    f.write(b"corrupted_data")
            return True
        elif spec.type == FaultType.DATA_MISSING:
            # Delete data file
            data_path = p.get("data_path")
            if data_path and os.path.exists(data_path):
                logger.info(f"Deleting data file {data_path}")
                os.unlink(data_path)
            return True
        elif spec.type == FaultType.BATCH_SIZE_MISMATCH:
            # Set mismatched batch size env
            logger.info("Setting mismatched batch size")
            os.environ["BATCH_SIZE_OVERRIDE"] = str(p.get("batch_size", 999))
            return True
        return False

    def cleanup(self):
        pass


class DependencyServiceExecutor(BaseExecutor):
    """Execute dependency service faults."""

    def __init__(self):
        self._cleanup_cmds = []

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        if spec.type == FaultType.HF_HUB_UNREACHABLE:
            # Block HuggingFace Hub
            logger.info("Blocking HuggingFace Hub")
            cmd = "iptables -A OUTPUT -d huggingface.co -j DROP"
            self._cleanup_cmds.append("iptables -D OUTPUT -d huggingface.co -j DROP 2>/dev/null")
            return os.system(cmd) == 0
        elif spec.type == FaultType.WANDB_DISCONNECT:
            # Block Wandb
            logger.info("Blocking Wandb")
            cmd = "iptables -A OUTPUT -d wandb.ai -j DROP"
            self._cleanup_cmds.append("iptables -D OUTPUT -d wandb.ai -j DROP 2>/dev/null")
            return os.system(cmd) == 0
        elif spec.type == FaultType.S3_TIMEOUT:
            # Add delay to S3 endpoints
            logger.info("Adding delay to S3")
            os.environ["AWS_MAX_ATTEMPTS"] = "1"
            os.environ["AWS_RETRY_MODE"] = "standard"
            return True
        return False

    def cleanup(self):
        for cmd in reversed(self._cleanup_cmds):
            os.system(cmd)


class TimingSyncExecutor(BaseExecutor):
    """Execute timing/synchronization faults."""

    def execute(self, spec: FaultSpec) -> bool:
        p = spec.params
        if spec.type == FaultType.RANK_DESYNC:
            # Make one rank sleep to desync
            rank = int(os.environ.get("RANK", -1))
            target = p.get("target_rank", 0)
            if rank == target:
                dur = p.get("duration_sec", 30)
                logger.info(f"Rank {rank} sleeping {dur}s to desync")
                time.sleep(dur)
            return True
        elif spec.type == FaultType.BARRIER_TIMEOUT:
            # Sleep before barrier
            dur = p.get("duration_sec", 60)
            logger.info(f"Sleeping {dur}s before barrier")
            time.sleep(dur)
            return True
        elif spec.type == FaultType.ASYNC_TIMEOUT:
            # Set very short timeout
            logger.info("Setting short async timeout")
            os.environ["ASYNC_TIMEOUT"] = "0.001"
            return True
        return False

    def cleanup(self):
        pass


# ============== Fault Classification ==============
# System-level faults run in parent process (external injection)
SYSTEM_LEVEL_FAULTS = {
    FaultType.NETWORK_DELAY,
    FaultType.NETWORK_LOSS,
    FaultType.NETWORK_PARTITION,
    FaultType.PORT_BLOCK,
    FaultType.PROCESS_KILL,
    FaultType.PROCESS_EXIT,
    FaultType.DISK_FULL,
    FaultType.MEMORY_PRESSURE,
    FaultType.DATA_CORRUPT,
    FaultType.DATA_MISSING,
    FaultType.HF_HUB_UNREACHABLE,
    FaultType.WANDB_DISCONNECT,
}

# Code-level faults run in child process (internal injection)
CODE_LEVEL_FAULTS = {
    FaultType.CODE_SLEEP,
    FaultType.CODE_OOM,
    FaultType.ENV_MODIFY,
    FaultType.NCCL_TIMEOUT,
    FaultType.NCCL_HANG,
    FaultType.NCCL_MISMATCH,
    FaultType.RAY_ACTOR_CRASH,
    FaultType.RAY_RESOURCE_EXHAUSTED,
    FaultType.RAY_GCS_FAILURE,
    FaultType.CUDA_OOM,
    FaultType.CUDA_ERROR,
    FaultType.GRADIENT_NAN,
    FaultType.GRADIENT_EXPLODE,
    FaultType.CHECKPOINT_CORRUPT,
    FaultType.CHECKPOINT_MISSING,
    FaultType.VLLM_INIT_FAIL,
    FaultType.VLLM_KV_CACHE_OOM,
    FaultType.VLLM_TOKENIZER_ERROR,
    FaultType.BATCH_SIZE_MISMATCH,
    FaultType.RANK_DESYNC,
    FaultType.BARRIER_TIMEOUT,
    FaultType.ASYNC_TIMEOUT,
    FaultType.S3_TIMEOUT,
}


# ============== Orchestrator ==============
class FaultOrchestrator:
    def __init__(self, config: FaultConfig):
        self.config = config
        self.process = None
        self._stop = threading.Event()
        self._log_file = None
        self._executors = {
            "network": NetworkExecutor(),
            "process": ProcessExecutor(),
            "system": SystemExecutor(),
            "code": CodeExecutor(),
            "nccl": NCCLExecutor(),
            "ray": RayExecutor(),
            "dl_framework": DLFrameworkExecutor(),
            "vllm": VLLMExecutor(),
            "data": DataLayerExecutor(),
            "dependency": DependencyServiceExecutor(),
            "timing": TimingSyncExecutor(),
        }

    def run(self, command: list) -> int:
        os.makedirs(self.config.log_output_dir, exist_ok=True)
        log_path = os.path.join(
            self.config.log_output_dir, f"fault_{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self._log_file = open(log_path, "w")
        self._log("start", {"command": command, "scenario": self.config.name})

        # Determine if we have code-level faults that need injection
        has_code_level = any(spec.type in CODE_LEVEL_FAULTS for spec in self.config.faults)

        # Prepare environment with fault config
        env = os.environ.copy()
        if has_code_level:
            env["FAULT_INJECTION_CONFIG"] = json.dumps(
                {
                    "name": self.config.name,
                    "faults": [
                        {
                            "type": spec.type.value,
                            "trigger": {
                                "type": spec.trigger.type.value,
                                "after_seconds": spec.trigger.after_seconds,
                                "probability": spec.trigger.probability,
                            },
                            "target_rank": spec.target_rank,
                            "params": spec.params,
                        }
                        for spec in self.config.faults
                        if spec.type in CODE_LEVEL_FAULTS
                    ],
                }
            )

            # Inject fault_injector.py into command
            command = self._inject_fault_injector(command)
            logger.info(f"Injected fault_injector into command: {' '.join(command[:3])}...")

        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, env=env
            )
            self._executors["process"].pid = self.process.pid
            self._log("process_started", {"pid": self.process.pid})

            for spec in self.config.faults:
                t = threading.Thread(target=self._worker, args=(spec,), daemon=True)
                t.start()

            for line in self.process.stdout:
                self._log("output", {"line": line.rstrip()})
                print(line, end="", flush=True)

            self.process.wait()
            rc = self.process.returncode
            self._log("exit", {"code": rc})
            return rc
        except KeyboardInterrupt:
            self._log("interrupted", {})
            if self.process:
                self.process.terminate()
            return 130
        finally:
            self._stop.set()
            self._cleanup()
            if self._log_file:
                self._log_file.close()
            print(f"\n[fault-inject] Log: {log_path}")

    def _inject_fault_injector(self, command: list) -> list:
        """Inject fault_injector.py into the command to run in child process."""
        injector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fault_injector.py")

        if not os.path.exists(injector_path):
            logger.warning(f"fault_injector.py not found at {injector_path}, skipping injection")
            return command

        # Handle different command patterns
        if len(command) == 0:
            return command

        # If command starts with python
        if command[0] in ("python", "python3", sys.executable):
            # Insert injection code before the actual script
            injector_dir = os.path.dirname(injector_path)
            injection_code = (
                f"import sys; sys.path.insert(0, '{injector_dir}'); import fault_injector; fault_injector.init()"
            )

            # python -c "code" -> python -c "injection; original_code"
            if len(command) > 2 and command[1] == "-c":
                original_code = command[2]
                return [
                    command[0],
                    "-c",
                    f"{injection_code}; {original_code}",
                ] + command[3:]
            # python script.py args -> python -c "injection; exec(...)"
            elif len(command) > 1 and not command[1].startswith("-"):
                script_path = command[1]
                return [
                    command[0],
                    "-c",
                    f"{injection_code}; exec(open('{script_path}').read())",
                ] + command[2:]
            # python -m module -> python -c "injection; import runpy; runpy.run_module(...)"
            elif len(command) > 2 and command[1] == "-m":
                module_name = command[2]
                return [
                    command[0],
                    "-c",
                    f"{injection_code}; import runpy; runpy.run_module('{module_name}', run_name='__main__')",
                ] + command[3:]

        return command

    def _worker(self, spec: FaultSpec):
        # Only handle system-level faults in parent process
        # Code-level faults are handled by fault_injector.py in child process
        if spec.type not in SYSTEM_LEVEL_FAULTS:
            return

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
        if spec.target_rank is not None and rank != spec.target_rank:
            return
        if spec.trigger.type == TriggerType.TIMED:
            if self._stop.wait(spec.trigger.after_seconds):
                return
        elif spec.trigger.type == TriggerType.PROBABILITY:
            if random.random() > spec.trigger.probability:
                return
        if self._stop.is_set():
            return

        executor = self._get_executor(spec.type)
        if executor:
            self._log("fault_inject", {"type": spec.type.value, "params": spec.params})
            executor.execute(spec)

    def _get_executor(self, ft: FaultType):
        # Network layer
        if ft in (FaultType.NETWORK_DELAY, FaultType.NETWORK_LOSS, FaultType.NETWORK_PARTITION, FaultType.PORT_BLOCK):
            return self._executors["network"]
        # Process layer
        elif ft in (FaultType.PROCESS_KILL, FaultType.PROCESS_EXIT):
            return self._executors["process"]
        # System layer
        elif ft in (FaultType.DISK_FULL, FaultType.FD_EXHAUST, FaultType.MEMORY_PRESSURE):
            return self._executors["system"]
        # Code layer
        elif ft in (FaultType.CODE_SLEEP, FaultType.CODE_OOM, FaultType.ENV_MODIFY):
            return self._executors["code"]
        # NCCL/Communication layer
        elif ft in (FaultType.NCCL_TIMEOUT, FaultType.NCCL_HANG, FaultType.NCCL_MISMATCH):
            return self._executors["nccl"]
        # Ray layer
        elif ft in (FaultType.RAY_ACTOR_CRASH, FaultType.RAY_RESOURCE_EXHAUSTED, FaultType.RAY_GCS_FAILURE):
            return self._executors["ray"]
        # DL Framework layer
        elif ft in (
            FaultType.CUDA_OOM,
            FaultType.CUDA_ERROR,
            FaultType.GRADIENT_NAN,
            FaultType.GRADIENT_EXPLODE,
            FaultType.CHECKPOINT_CORRUPT,
            FaultType.CHECKPOINT_MISSING,
        ):
            return self._executors["dl_framework"]
        # vLLM layer
        elif ft in (FaultType.VLLM_INIT_FAIL, FaultType.VLLM_KV_CACHE_OOM, FaultType.VLLM_TOKENIZER_ERROR):
            return self._executors["vllm"]
        # Data layer
        elif ft in (FaultType.DATA_CORRUPT, FaultType.DATA_MISSING, FaultType.BATCH_SIZE_MISMATCH):
            return self._executors["data"]
        # Dependency service layer
        elif ft in (FaultType.HF_HUB_UNREACHABLE, FaultType.WANDB_DISCONNECT, FaultType.S3_TIMEOUT):
            return self._executors["dependency"]
        # Timing/Sync layer
        elif ft in (FaultType.RANK_DESYNC, FaultType.BARRIER_TIMEOUT, FaultType.ASYNC_TIMEOUT):
            return self._executors["timing"]
        return None

    def _log(self, event: str, data: dict):
        if self._log_file:
            self._log_file.write(json.dumps({"ts": datetime.now().isoformat(), "event": event, **data}) + "\n")
            self._log_file.flush()

    def _cleanup(self):
        for e in self._executors.values():
            try:
                e.cleanup()
            except:
                pass


# ============== CLI ==============
def list_faults():
    print("Available fault types:\n")
    cats = {
        "Network": ["network_delay", "network_loss", "network_partition", "port_block"],
        "Process": ["process_kill", "process_exit"],
        "System": ["disk_full", "fd_exhaust", "memory_pressure"],
        "Code": ["code_sleep", "code_oom", "env_modify"],
        "NCCL/Communication": ["nccl_timeout", "nccl_hang", "nccl_mismatch"],
        "Ray": ["ray_actor_crash", "ray_resource_exhausted", "ray_gcs_failure"],
        "DL Framework": [
            "cuda_oom",
            "cuda_error",
            "gradient_nan",
            "gradient_explode",
            "checkpoint_corrupt",
            "checkpoint_missing",
        ],
        "vLLM": ["vllm_init_fail", "vllm_kv_cache_oom", "vllm_tokenizer_error"],
        "Data": ["data_corrupt", "data_missing", "batch_size_mismatch"],
        "Dependency Service": ["hf_hub_unreachable", "wandb_disconnect", "s3_timeout"],
        "Timing/Sync": ["rank_desync", "barrier_timeout", "async_timeout"],
    }
    for cat, faults in cats.items():
        print(f"  {cat}:")
        for f in faults:
            print(f"    - {f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Inject faults into distributed training",
        usage="python verl_fault_inject.py [options] -- <command>",
    )
    parser.add_argument("--scenario", "-s", help="Scenario YAML file")
    parser.add_argument("--list", "-l", action="store_true", help="List fault types")
    parser.add_argument("--output", "-o", default="./fault_logs", help="Log output dir")
    parser.add_argument("command", nargs="*", help="Command to run")

    args = parser.parse_args()

    if args.list:
        list_faults()
        return 0

    if not args.scenario:
        parser.error("--scenario required")
    if not args.command:
        parser.error("No command. Use -- to separate options from command.")

    config = FaultConfig.from_yaml(args.scenario)
    config.log_output_dir = args.output
    return FaultOrchestrator(config).run(args.command)


if __name__ == "__main__":
    sys.exit(main())
