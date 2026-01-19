"""Process fault executor."""

import logging
import os
import signal

from ..config import FaultSpec, FaultType
from .base import BaseExecutor

logger = logging.getLogger(__name__)


class ProcessExecutor(BaseExecutor):
    """Execute process-level faults."""

    def __init__(self, target_pid: int = None):
        self.target_pid = target_pid

    def execute(self, spec: FaultSpec) -> bool:
        pid = self.target_pid
        if not pid:
            logger.error("No target PID set")
            return False

        try:
            if spec.type == FaultType.PROCESS_KILL:
                sig = spec.params.get("signal", "SIGKILL")
                return self._kill(pid, sig)
            elif spec.type == FaultType.PROCESS_EXIT:
                # This will be called from within the target process
                code = spec.params.get("exit_code", 1)
                os._exit(code)
        except Exception as e:
            logger.error(f"Process fault failed: {e}")
            return False
        return False

    def _kill(self, pid: int, sig_name: str) -> bool:
        sig = getattr(signal, sig_name, signal.SIGKILL)
        logger.info(f"Sending {sig_name} to PID {pid}")
        os.kill(pid, sig)
        return True

    def cleanup(self) -> None:
        pass
