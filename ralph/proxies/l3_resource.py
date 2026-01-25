"""
L3 Resource Layer proxies for Ralph fault injection framework.

Contains injectors for GPU memory, CPU, and other resource-related fault injection.
"""

from typing import TYPE_CHECKING, Any, Optional

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.base import BaseProxy

if TYPE_CHECKING:
    from ralph.collectors.dual_stream import DualStreamCollector

# Try to import torch for GPU operations
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class GPUMemoryInjector(BaseProxy):
    """
    GPU Memory fault injector for simulating memory-related issues.

    This injector can simulate:
    - Memory pressure: Fill GPU to a specified ratio
    - OOM (Out Of Memory): Fill GPU completely to cause OOM errors
    - Memory fragmentation: Allocate/deallocate many small tensors
    - Memory leak: Gradually consume memory over time

    Unlike typical proxies that wrap functions, this injector actively
    manipulates GPU memory to create fault conditions.

    Note: This injector requires CUDA to be available. Operations will
    be no-ops if CUDA is not available.

    Example:
        injector = GPUMemoryInjector(lambda: None, collector)
        injector.allocate_pressure(0.9)  # Fill to 90% capacity
        # ... run tests ...
        injector.release()  # Free allocated memory
    """

    SUPPORTED_STRATEGIES: set[StrategyType] = {
        StrategyType.MEMORY_PRESSURE,
        StrategyType.OOM_SIMULATION,
        StrategyType.MEMORY_FRAGMENTATION,
        StrategyType.MEMORY_LEAK,
    }

    def __init__(
        self,
        original_fn: Any = None,
        collector: Optional["DualStreamCollector"] = None,
        device: Optional[int] = None,
    ):
        """
        Initialize the GPU Memory injector.

        Args:
            original_fn: Not used for this injector (kept for API compatibility)
            collector: Optional DualStreamCollector for recording fault injections
            device: CUDA device index to target (default: current device)
        """
        # Use a no-op function as original since this isn't wrapping a function
        super().__init__(original_fn or (lambda: None), collector)

        # Track allocated tensors for cleanup
        self._allocated_tensors: list[Any] = []
        self._leak_tensors: list[Any] = []
        self._fragmentation_tensors: list[Any] = []

        # Memory leak state
        self._leak_rate_mb: float = 0.0
        self._leak_active: bool = False

        # Device configuration
        self._device = device

    def _get_layer(self) -> str:
        """Return the layer this injector belongs to."""
        return "L3"

    def _get_device(self) -> Optional[int]:
        """Get the target CUDA device index."""
        if self._device is not None:
            return self._device
        if HAS_TORCH and torch.cuda.is_available():
            return torch.cuda.current_device()
        return None

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU operations."""
        return HAS_TORCH and torch.cuda.is_available()

    def _get_free_memory(self) -> int:
        """
        Get the amount of free GPU memory in bytes.

        Returns:
            Free memory in bytes, or 0 if CUDA not available
        """
        if not self._is_cuda_available():
            return 0

        device = self._get_device()
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        return free

    def _get_total_memory(self) -> int:
        """
        Get the total GPU memory in bytes.

        Returns:
            Total memory in bytes, or 0 if CUDA not available
        """
        if not self._is_cuda_available():
            return 0

        device = self._get_device()
        free, total = torch.cuda.mem_get_info(device)
        return total

    def _get_allocated_memory(self) -> int:
        """
        Get the currently allocated GPU memory in bytes.

        Returns:
            Allocated memory in bytes, or 0 if CUDA not available
        """
        if not self._is_cuda_available():
            return 0

        device = self._get_device()
        return torch.cuda.memory_allocated(device)

    def allocate_pressure(self, ratio: float) -> bool:
        """
        Allocate GPU memory to reach the specified utilization ratio.

        Args:
            ratio: Target memory utilization ratio (0.0 to 1.0)
                   e.g., 0.9 means fill to 90% of total GPU memory

        Returns:
            True if allocation succeeded, False otherwise

        Raises:
            ValueError: If ratio is not between 0 and 1
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")

        if not self._is_cuda_available():
            return False

        device = self._get_device()
        total_memory = self._get_total_memory()
        target_allocated = int(total_memory * ratio)
        current_allocated = self._get_allocated_memory()

        # Calculate how much more we need to allocate
        bytes_to_allocate = target_allocated - current_allocated

        if bytes_to_allocate <= 0:
            # Already at or above target
            return True

        try:
            # Allocate in large chunks for efficiency
            # Use float32 (4 bytes per element)
            elements = bytes_to_allocate // 4
            if elements > 0:
                tensor = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
                self._allocated_tensors.append(tensor)

            return True
        except RuntimeError:
            # OOM during pressure allocation
            return False

    def trigger_oom(self, leave_headroom_mb: float = 0) -> None:
        """
        Fill GPU memory completely to cause an OOM error on next allocation.

        This method allocates memory until the GPU is nearly full, leaving
        only a small headroom. The next allocation attempt will fail with
        a CUDA out of memory error.

        Args:
            leave_headroom_mb: Amount of memory (in MB) to leave free
                              (default: 0, try to fill completely)

        Raises:
            RuntimeError: CUDA out of memory (expected behavior)
        """
        if not self._is_cuda_available():
            raise RuntimeError("CUDA not available - cannot trigger OOM")

        device = self._get_device()
        headroom_bytes = int(leave_headroom_mb * 1024 * 1024)

        # First, try to fill most of the memory
        while True:
            free_memory = self._get_free_memory()

            if free_memory <= headroom_bytes:
                break

            # Allocate in increasingly smaller chunks
            try_size = min(free_memory - headroom_bytes, 256 * 1024 * 1024)  # Max 256MB chunks
            if try_size < 1024:  # Less than 1KB remaining
                break

            try:
                elements = try_size // 4
                tensor = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
                self._allocated_tensors.append(tensor)
            except RuntimeError:
                # OOM, try smaller size
                try_size = try_size // 2
                if try_size < 1024:
                    break
                continue

        # Now try to allocate one more tensor to trigger OOM
        if leave_headroom_mb == 0:
            # Force OOM by trying to allocate more than available
            remaining = self._get_free_memory()
            if remaining > 0:
                try:
                    # Try to allocate more than available
                    tensor = torch.empty(remaining + 1024 * 1024, dtype=torch.float32, device=f"cuda:{device}")
                    self._allocated_tensors.append(tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise RuntimeError(f"CUDA out of memory (simulated OOM): {e}") from e
                    raise

    def cause_fragmentation(self, count: int) -> int:
        """
        Create memory fragmentation by allocating and freeing many small tensors.

        This simulates memory fragmentation where total free memory may be
        sufficient but no contiguous block is large enough for allocation.

        Args:
            count: Number of allocation/deallocation cycles

        Returns:
            Number of fragmentation operations performed
        """
        if not self._is_cuda_available():
            return 0

        if count < 0:
            count = 0

        device = self._get_device()
        ops_performed = 0

        # Allocate tensors of varying sizes
        temp_tensors = []
        sizes = [1024, 4096, 16384, 65536, 262144]  # Varying sizes in elements

        try:
            for i in range(count):
                size = sizes[i % len(sizes)]
                tensor = torch.empty(size, dtype=torch.float32, device=f"cuda:{device}")
                temp_tensors.append(tensor)
                ops_performed += 1

            # Free every other tensor to create fragmentation
            for i in range(0, len(temp_tensors), 2):
                temp_tensors[i] = None

            # Force garbage collection
            if HAS_TORCH:
                torch.cuda.empty_cache()

            # Keep remaining tensors to maintain fragmentation
            self._fragmentation_tensors.extend([t for t in temp_tensors if t is not None])

        except RuntimeError:
            # OOM during fragmentation - stop and clean up
            pass

        return ops_performed

    def start_leak(self, rate_mb_per_step: float) -> None:
        """
        Start a gradual memory leak that allocates memory each step.

        Memory will be allocated when step_leak() is called, at the specified
        rate per step. Use stop_leak() to stop the leak without releasing
        already leaked memory.

        Args:
            rate_mb_per_step: Amount of memory (in MB) to leak per step
        """
        if rate_mb_per_step < 0:
            rate_mb_per_step = 0

        self._leak_rate_mb = rate_mb_per_step
        self._leak_active = True

    def step_leak(self) -> int:
        """
        Execute one step of memory leak if active.

        Call this method once per training step to simulate gradual memory leak.

        Returns:
            Amount of memory leaked in bytes (0 if leak not active or failed)
        """
        if not self._leak_active or self._leak_rate_mb <= 0:
            return 0

        if not self._is_cuda_available():
            return 0

        device = self._get_device()
        bytes_to_leak = int(self._leak_rate_mb * 1024 * 1024)

        try:
            elements = bytes_to_leak // 4
            if elements > 0:
                tensor = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
                self._leak_tensors.append(tensor)
                return bytes_to_leak
        except RuntimeError:
            # OOM during leak
            pass

        return 0

    def stop_leak(self) -> None:
        """Stop the memory leak without releasing leaked memory."""
        self._leak_active = False

    def get_leak_total_mb(self) -> float:
        """
        Get the total amount of memory leaked.

        Returns:
            Total leaked memory in MB
        """
        total_elements = sum(t.numel() for t in self._leak_tensors if t is not None)
        return (total_elements * 4) / (1024 * 1024)  # Convert bytes to MB

    def release(self) -> None:
        """
        Release all allocated tensors and clean up memory.

        This releases:
        - Pressure allocation tensors
        - Leaked tensors
        - Fragmentation tensors
        """
        # Clear all tensor lists
        self._allocated_tensors.clear()
        self._leak_tensors.clear()
        self._fragmentation_tensors.clear()

        # Stop any active leak
        self._leak_active = False
        self._leak_rate_mb = 0.0

        # Force garbage collection and clear CUDA cache
        if self._is_cuda_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self._get_device())

    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats (all values in bytes):
            - total: Total GPU memory
            - allocated: Currently allocated memory
            - free: Free memory
            - pressure_allocated: Memory allocated by pressure operations
            - leak_allocated: Memory allocated by leak operations
            - fragmentation_allocated: Memory held for fragmentation
        """
        if not self._is_cuda_available():
            return {
                "total": 0,
                "allocated": 0,
                "free": 0,
                "pressure_allocated": 0,
                "leak_allocated": 0,
                "fragmentation_allocated": 0,
            }

        pressure_elements = sum(t.numel() for t in self._allocated_tensors if t is not None)
        leak_elements = sum(t.numel() for t in self._leak_tensors if t is not None)
        frag_elements = sum(t.numel() for t in self._fragmentation_tensors if t is not None)

        return {
            "total": self._get_total_memory(),
            "allocated": self._get_allocated_memory(),
            "free": self._get_free_memory(),
            "pressure_allocated": pressure_elements * 4,
            "leak_allocated": leak_elements * 4,
            "fragmentation_allocated": frag_elements * 4,
        }

    # Strategy methods for BaseProxy compatibility
    def _strategy_memory_pressure(self, *args, **kwargs) -> Any:
        """
        Execute memory pressure strategy.

        Reads 'pressure_ratio' from config.parameters (default 0.9).
        """
        ratio = self._config.parameters.get("pressure_ratio", 0.9) if self._config else 0.9
        self.allocate_pressure(ratio)
        return self._original(*args, **kwargs)

    def _strategy_oom_simulation(self, *args, **kwargs) -> Any:
        """
        Execute OOM simulation strategy.

        Reads 'leave_headroom_mb' from config.parameters (default 0).
        """
        headroom = self._config.parameters.get("leave_headroom_mb", 0) if self._config else 0
        self.trigger_oom(headroom)
        return self._original(*args, **kwargs)

    def _strategy_memory_fragmentation(self, *args, **kwargs) -> Any:
        """
        Execute memory fragmentation strategy.

        Reads 'fragment_count' from config.parameters (default 1000).
        """
        count = self._config.parameters.get("fragment_count", 1000) if self._config else 1000
        self.cause_fragmentation(count)
        return self._original(*args, **kwargs)

    def _strategy_memory_leak(self, *args, **kwargs) -> Any:
        """
        Execute memory leak strategy.

        Reads 'leak_rate_mb' from config.parameters (default 100).
        Starts the leak and executes one step.
        """
        rate = self._config.parameters.get("leak_rate_mb", 100) if self._config else 100
        self.start_leak(rate)
        self.step_leak()
        return self._original(*args, **kwargs)

    def __del__(self):
        """Clean up allocated tensors on deletion."""
        try:
            self.release()
        except Exception:
            pass  # Ignore errors during cleanup


# Register the injector with the proxy registry
ProxyRegistry.register("gpu_memory")(GPUMemoryInjector)
