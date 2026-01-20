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

"""Process-level fault injectors."""

import os
import threading
import time
from typing import Optional

from ..base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from ..config import FaultType, ProcessFaultConfig


@FaultInjectorRegistry.register(FaultType.PROCESS_KILL)
class ProcessKillInjector(BaseFaultInjector):
    """Injector for killing processes."""

    def __init__(self, config: ProcessFaultConfig):
        super().__init__(config)
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Kill the target process."""
        pid = os.getpid()

        logger = self._get_logger()
        logger.info(f"Killing process {pid} with signal {self.config.signal}")

        try:
            # Give logger time to flush
            time.sleep(0.1)
            os.kill(pid, self.config.signal)

            # If we get here, the signal didn't kill us
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                error=Exception(f"Failed to kill process with signal {self.config.signal}"),
                metadata={"pid": pid, "signal": self.config.signal},
            )
        except Exception as e:
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                error=e,
                metadata={"pid": pid, "signal": self.config.signal},
            )

    def recover(self, context: FaultContext) -> None:
        """Recovery is not possible for process kill."""
        pass


@FaultInjectorRegistry.register(FaultType.PROCESS_EXIT)
class ProcessExitInjector(BaseFaultInjector):
    """Injector for process exit."""

    def __init__(self, config: ProcessFaultConfig):
        super().__init__(config)
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Exit the process with specified code."""
        logger = self._get_logger()
        logger.info(f"Exiting process with code {self.config.exit_code}")

        try:
            # Give logger time to flush
            time.sleep(0.1)
            os._exit(self.config.exit_code)
        except Exception as e:
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                error=e,
                metadata={"exit_code": self.config.exit_code},
            )

    def recover(self, context: FaultContext) -> None:
        """Recovery is not possible for process exit."""
        pass


@FaultInjectorRegistry.register(FaultType.PROCESS_HANG)
class ProcessHangInjector(BaseFaultInjector):
    """Injector for process hangs."""

    def __init__(self, config: ProcessFaultConfig):
        super().__init__(config)
        self.config = config
        self._hang_thread: Optional[threading.Thread] = None
        self._stop_hang = threading.Event()

    def inject(self, context: FaultContext) -> FaultResult:
        """Hang the process."""
        logger = self._get_logger()
        duration = self.config.hang_duration or 3600  # Default 1 hour

        logger.info(f"Hanging process for {duration} seconds")

        def hang_loop():
            start_time = time.time()
            while time.time() - start_time < duration and not self._stop_hang.is_set():
                time.sleep(1)

        self._hang_thread = threading.Thread(target=hang_loop, daemon=True)
        self._hang_thread.start()

        # Wait for the hang to complete or be interrupted
        self._hang_thread.join()

        return FaultResult(
            fault_id=self.fault_id,
            status=FaultStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"hang_duration": duration},
        )

    def recover(self, context: FaultContext) -> None:
        """Stop the hang."""
        self._stop_hang.set()
        if self._hang_thread and self._hang_thread.is_alive():
            self._hang_thread.join(timeout=1)

    def _get_logger(self):
        """Get logger instance."""
        import logging

        return logging.getLogger(f"{__name__}.{self.__class__.__name__}")
