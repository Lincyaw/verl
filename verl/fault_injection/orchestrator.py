"""Fault injection orchestrator - manages subprocess and fault execution."""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from .config import FaultConfig, FaultSpec, FaultType, TriggerType
from .executors import CodeExecutor, NetworkExecutor, ProcessExecutor, SystemExecutor

logger = logging.getLogger(__name__)


class FaultOrchestrator:
    """Orchestrates fault injection for a subprocess."""

    def __init__(self, config: FaultConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.executors = {
            "system": SystemExecutor(),
            "network": NetworkExecutor(),
            "process": ProcessExecutor(),
            "code": CodeExecutor(),
        }
        self._fault_threads = []
        self._stop_event = threading.Event()
        self._log_file = None

    def run(self, command: list[str]) -> int:
        """Run command with fault injection."""
        os.makedirs(self.config.log_output_dir, exist_ok=True)
        log_path = os.path.join(
            self.config.log_output_dir, f"fault_{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self._log_file = open(log_path, "w")
        self._log_event("start", {"command": command, "config": self.config.name})

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            self.executors["process"].target_pid = self.process.pid
            self._log_event("process_started", {"pid": self.process.pid})

            # Start fault injection threads
            for spec in self.config.faults:
                t = threading.Thread(target=self._fault_worker, args=(spec,), daemon=True)
                t.start()
                self._fault_threads.append(t)

            # Stream output
            for line in self.process.stdout:
                self._log_output(line.rstrip())
                print(line, end="", flush=True)

            self.process.wait()
            return_code = self.process.returncode
            self._log_event("process_exit", {"return_code": return_code})
            return return_code

        except KeyboardInterrupt:
            self._log_event("interrupted", {})
            if self.process:
                self.process.terminate()
            return 130
        finally:
            self._stop_event.set()
            self._cleanup()
            if self._log_file:
                self._log_file.close()
            print(f"\n[fault-inject] Log saved to: {log_path}")

    def _fault_worker(self, spec: FaultSpec):
        """Worker thread for a single fault."""
        # Check rank filter
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
        if spec.target_rank is not None and rank != spec.target_rank:
            return

        # Wait for trigger
        if spec.trigger.type == TriggerType.TIMED:
            if self._stop_event.wait(spec.trigger.after_seconds):
                return
        elif spec.trigger.type == TriggerType.PROBABILITY:
            import random

            if random.random() > spec.trigger.probability:
                return

        if self._stop_event.is_set():
            return

        # Execute fault
        executor = self._get_executor(spec.type)
        if executor:
            self._log_event(
                "fault_inject",
                {
                    "type": spec.type.value,
                    "params": spec.params,
                    "rank": rank,
                },
            )
            logger.info(f"Injecting fault: {spec.type.value}")
            executor.execute(spec)

    def _get_executor(self, fault_type: FaultType):
        if fault_type in (
            FaultType.NETWORK_DELAY,
            FaultType.NETWORK_LOSS,
            FaultType.PORT_BLOCK,
            FaultType.NETWORK_PARTITION,
        ):
            return self.executors["network"]
        elif fault_type in (FaultType.PROCESS_KILL, FaultType.PROCESS_EXIT):
            return self.executors["process"]
        elif fault_type in (FaultType.DISK_FULL, FaultType.FD_EXHAUST, FaultType.MEMORY_PRESSURE):
            return self.executors["system"]
        elif fault_type in (FaultType.CODE_SLEEP, FaultType.CODE_OOM, FaultType.CODE_NAN, FaultType.ENV_MODIFY):
            return self.executors["code"]
        return None

    def _log_event(self, event_type: str, data: dict):
        if self._log_file:
            entry = {"ts": datetime.now().isoformat(), "event": event_type, **data}
            self._log_file.write(json.dumps(entry) + "\n")
            self._log_file.flush()

    def _log_output(self, line: str):
        if self._log_file:
            entry = {"ts": datetime.now().isoformat(), "event": "output", "line": line}
            self._log_file.write(json.dumps(entry) + "\n")

    def _cleanup(self):
        for executor in self.executors.values():
            try:
                executor.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
