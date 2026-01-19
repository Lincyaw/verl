"""Base executor interface."""

from abc import ABC, abstractmethod

from ..config import FaultSpec


class BaseExecutor(ABC):
    """Base class for fault executors."""

    @abstractmethod
    def execute(self, spec: FaultSpec) -> bool:
        """Execute the fault. Returns True if successful."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup any resources."""
        pass
