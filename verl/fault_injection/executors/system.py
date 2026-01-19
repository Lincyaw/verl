"""System fault executor."""

import logging
import os
import subprocess
import tempfile

from ..config import FaultSpec, FaultType
from .base import BaseExecutor

logger = logging.getLogger(__name__)


class SystemExecutor(BaseExecutor):
    """Execute system-level faults (disk, fd, memory)."""

    def __init__(self):
        self._temp_files = []
        self._cleanup_commands = []

    def execute(self, spec: FaultSpec) -> bool:
        try:
            if spec.type == FaultType.DISK_FULL:
                return self._fill_disk(spec.params.get("path", "/tmp"), spec.params.get("size_mb", 100))
            elif spec.type == FaultType.FD_EXHAUST:
                return self._exhaust_fd(spec.params.get("count", 1000))
            elif spec.type == FaultType.MEMORY_PRESSURE:
                return self._memory_pressure(spec.params.get("size_mb", 500))
        except Exception as e:
            logger.error(f"System fault failed: {e}")
            return False
        return False

    def _fill_disk(self, path: str, size_mb: int) -> bool:
        fpath = os.path.join(path, f"fault_fill_{os.getpid()}.tmp")
        cmd = f"dd if=/dev/zero of={fpath} bs=1M count={size_mb} 2>/dev/null"
        self._temp_files.append(fpath)
        logger.info(f"Filling disk at {fpath} with {size_mb}MB")
        return subprocess.run(cmd, shell=True).returncode == 0

    def _exhaust_fd(self, count: int) -> bool:
        # Open many file descriptors
        logger.info(f"Exhausting {count} file descriptors")
        for _ in range(count):
            try:
                f = open("/dev/null", "r")
                self._temp_files.append(f)
            except OSError:
                break
        return True

    def _memory_pressure(self, size_mb: int) -> bool:
        logger.info(f"Allocating {size_mb}MB memory")
        # Allocate memory that won't be garbage collected
        self._memory_block = bytearray(size_mb * 1024 * 1024)
        return True

    def cleanup(self) -> None:
        for f in self._temp_files:
            try:
                if isinstance(f, str):
                    os.unlink(f)
                else:
                    f.close()
            except:
                pass
        self._temp_files.clear()
        if hasattr(self, "_memory_block"):
            del self._memory_block
