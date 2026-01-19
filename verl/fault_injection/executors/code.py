"""Code-level fault executor (sleep, OOM, NaN injection)."""

import logging
import os
import time

from ..config import FaultSpec, FaultType
from .base import BaseExecutor

logger = logging.getLogger(__name__)


class CodeExecutor(BaseExecutor):
    """Execute code-level faults via monkey patching."""

    def __init__(self):
        self._patches = []

    def execute(self, spec: FaultSpec) -> bool:
        try:
            if spec.type == FaultType.CODE_SLEEP:
                return self._inject_sleep(spec.params.get("duration_sec", 60))
            elif spec.type == FaultType.CODE_OOM:
                return self._trigger_oom(spec.params.get("size_gb", 100))
            elif spec.type == FaultType.CODE_NAN:
                return self._inject_nan(spec.params.get("target_module"))
            elif spec.type == FaultType.ENV_MODIFY:
                return self._modify_env(spec.params)
        except Exception as e:
            logger.error(f"Code fault failed: {e}")
            return False
        return False

    def _inject_sleep(self, duration: float) -> bool:
        """Block execution to trigger timeout."""
        logger.info(f"Sleeping for {duration}s to trigger timeout")
        time.sleep(duration)
        return True

    def _trigger_oom(self, size_gb: int) -> bool:
        """Allocate huge memory to trigger OOM."""
        logger.info(f"Allocating {size_gb}GB to trigger OOM")
        try:
            _ = bytearray(size_gb * 1024 * 1024 * 1024)
        except MemoryError:
            logger.info("OOM triggered successfully")
        return True

    def _inject_nan(self, target_module: str) -> bool:
        """Patch torch operations to inject NaN."""
        if not target_module:
            return False
        try:
            import torch

            original_matmul = torch.matmul

            def nan_matmul(*args, **kwargs):
                result = original_matmul(*args, **kwargs)
                result[0, 0] = float("nan")
                return result

            torch.matmul = nan_matmul
            self._patches.append(("torch", "matmul", original_matmul))
            logger.info("NaN injection patch applied to torch.matmul")
            return True
        except ImportError:
            logger.warning("torch not available for NaN injection")
            return False

    def _modify_env(self, params: dict) -> bool:
        """Modify environment variables."""
        for key, value in params.items():
            if key not in ("type", "trigger"):
                logger.info(f"Setting env {key}={value}")
                os.environ[key] = str(value)
        return True

    def cleanup(self) -> None:
        for module_name, attr, original in self._patches:
            try:
                import importlib

                module = importlib.import_module(module_name)
                setattr(module, attr, original)
            except:
                pass
        self._patches.clear()
