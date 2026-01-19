"""Async fault injection support for performance optimization."""

import asyncio
import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

from .base import BaseFaultInjector, FaultContext, FaultResult

logger = logging.getLogger(__name__)


class AsyncBaseFaultInjector(BaseFaultInjector):
    """Base class for async fault injectors."""

    async def execute_async(self, target_context: Optional[FaultContext] = None) -> FaultResult:
        """Execute fault injection asynchronously."""
        # Check if should inject
        if not self.should_inject(target_context):
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultResult.Status.SKIPPED,
                message="Fault injection skipped by trigger",
            )

        # Execute async fault injection
        try:
            start_time = asyncio.get_event_loop().time()
            result = await self._inject_async(target_context)
            end_time = asyncio.get_event_loop().time()

            # Update result with timing
            result.duration = end_time - start_time
            result.timestamp = end_time

            logger.info(f"Async fault injection completed: {self.fault_id} - Status: {result.status.value}")
            return result

        except Exception as e:
            logger.error(f"Async fault injection failed: {self.fault_id} - Error: {e}")
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultResult.Status.FAILED,
                error=str(e),
                timestamp=asyncio.get_event_loop().time(),
            )

    @abstractmethod
    async def _inject_async(self, target_context: Optional[FaultContext] = None) -> FaultResult:
        """Async implementation of fault injection."""
        pass

    async def should_inject_async(self, target_context: Optional[FaultContext] = None) -> bool:
        """Async version of should_inject check."""
        # Run trigger check in thread pool if needed
        if self._trigger:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._trigger.should_trigger, target_context)
        return True

    async def cleanup_async(self) -> None:
        """Async cleanup of fault injection effects."""
        try:
            await self._cleanup_async()
        except Exception as e:
            logger.error(f"Async cleanup failed for {self.fault_id}: {e}")

    @abstractmethod
    async def _cleanup_async(self) -> None:
        """Async implementation of cleanup."""
        pass


class AsyncBatchFaultInjector(AsyncBaseFaultInjector):
    """Base class for batch fault injection."""

    def __init__(self, config: Any, batch_size: int = 10):
        super().__init__(config)
        self.batch_size = batch_size
        self._pending_faults: list[Dict[str, Any]] = []

    async def execute_batch_async(self, contexts: list[Optional[FaultContext]]) -> list[FaultResult]:
        """Execute multiple fault injections in batch."""
        results = []

        # Process in batches
        for i in range(0, len(contexts), self.batch_size):
            batch = contexts[i:i + self.batch_size]
            batch_results = await self._inject_batch_async(batch)
            results.extend(batch_results)

        return results

    @abstractmethod
    async def _inject_batch_async(self, contexts: list[Optional[FaultContext]]) -> list[FaultResult]:
        """Async batch implementation of fault injection."""
        pass


class AsyncCachedFaultInjector(AsyncBaseFaultInjector):
    """Base class for cached fault injection."""

    def __init__(self, config: Any, cache_ttl: int = 300):
        super().__init__(config)
        self.cache_ttl = cache_ttl
        self._result_cache: Dict[str, tuple[FaultResult, float]] = {}

    async def execute_cached_async(self, target_context: Optional[FaultContext] = None) -> FaultResult:
        """Execute with result caching."""
        cache_key = self._get_cache_key(target_context)

        # Check cache
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            if asyncio.get_event_loop().time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached result for {self.fault_id}")
                return result

        # Execute and cache
        result = await self.execute_async(target_context)
        self._result_cache[cache_key] = (result, asyncio.get_event_loop().time())
        return result

    def _get_cache_key(self, target_context: Optional[FaultContext]) -> str:
        """Generate cache key for the fault and context."""
        key_parts = [self.fault_id]
        if target_context:
            key_parts.extend([
                str(target_context.worker_id),
                str(target_context.rank),
                str(target_context.layer),
            ])
        return ":".join(key_parts)

    async def clear_cache_async(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()


class AsyncNetworkFaultInjector(AsyncBaseFaultInjector):
    """Async network fault injector."""

    async def _inject_async(self, target_context: Optional[FaultContext] = None) -> FaultResult:
        """Async network fault injection."""
        # Use async subprocess for network operations
        proc = await asyncio.create_subprocess_shell(
            self._get_network_command(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultResult.Status.COMPLETED,
                message="Network fault injected successfully",
                details={"stdout": stdout.decode(), "command": self._get_network_command()},
            )
        else:
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultResult.Status.FAILED,
                error=stderr.decode(),
                details={"stdout": stdout.decode(), "command": self._get_network_command()},
            )

    def _get_network_command(self) -> str:
        """Get the network fault command."""
        # Override in subclasses
        return "echo 'Network fault simulation'"

    async def _cleanup_async(self) -> None:
        """Async cleanup of network faults."""
        # Restore network settings asynchronously
        restore_cmd = self._get_restore_command()
        if restore_cmd:
            proc = await asyncio.create_subprocess_shell(
                restore_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

    def _get_restore_command(self) -> str:
        """Get the network restore command."""
        # Override in subclasses
        return ""


class AsyncResourceFaultInjector(AsyncBatchFaultInjector):
    """Async resource exhaustion fault injector."""

    async def _inject_batch_async(self, contexts: list[Optional[FaultContext]]) -> list[FaultResult]:
        """Async batch resource exhaustion."""
        results = []

        # Create async tasks for resource allocation
        tasks = []
        for context in contexts:
            task = asyncio.create_task(self._allocate_resource_async(context))
            tasks.append(task)

        # Wait for all allocations
        allocations = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, allocation in enumerate(allocations):
            if isinstance(allocation, Exception):
                results.append(FaultResult(
                    fault_id=self.fault_id,
                    status=FaultResult.Status.FAILED,
                    error=str(allocation),
                    target_context=contexts[i],
                ))
            else:
                results.append(FaultResult(
                    fault_id=self.fault_id,
                    status=FaultResult.Status.COMPLETED,
                    message="Resource allocated for fault injection",
                    details={"allocation": allocation},
                    target_context=contexts[i],
                ))

        return results

    async def _allocate_resource_async(self, target_context: Optional[FaultContext]) -> Any:
        """Async resource allocation."""
        # Simulate async resource allocation
        await asyncio.sleep(0.1)  # Simulate async operation
        return {"type": "memory", "size": 1024 * 1024}  # 1MB


class AsyncDelayedFaultInjector(AsyncCachedFaultInjector):
    """Async fault injector with delayed execution."""

    def __init__(self, config: Any, delay_seconds: float = 1.0):
        super().__init__(config)
        self.delay_seconds = delay_seconds

    async def _inject_async(self, target_context: Optional[FaultContext] = None) -> FaultResult:
        """Inject fault after a delay."""
        # Wait for the delay
        await asyncio.sleep(self.delay_seconds)

        # Execute the actual fault
        return FaultResult(
            fault_id=self.fault_id,
            status=FaultResult.Status.COMPLETED,
            message=f"Fault injected after {self.delay_seconds}s delay",
            details={"delay": self.delay_seconds},
        )


# Helper functions for async fault injection
async def run_async_fault_injection(
    injector: AsyncBaseFaultInjector,
    target_context: Optional[FaultContext] = None,
    timeout: float = 30.0,
) -> FaultResult:
    """Run async fault injection with timeout."""
    try:
        return await asyncio.wait_for(
            injector.execute_async(target_context),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Async fault injection timed out: {injector.fault_id}")
        return FaultResult(
            fault_id=injector.fault_id,
            status=FaultResult.Status.FAILED,
            error="Fault injection timed out",
        )


async def run_batch_async_fault_injection(
    injectors: list[AsyncBaseFaultInjector],
    contexts: list[Optional[FaultContext]],
    max_concurrent: int = 10,
) -> list[FaultResult]:
    """Run multiple async fault injections with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(injector: AsyncBaseFaultInjector, context: Optional[FaultContext]) -> FaultResult:
        async with semaphore:
            return await injector.execute_async(context)

    # Create tasks
    tasks = [
        asyncio.create_task(run_with_semaphore(injector, context))
        for injector, context in zip(injectors, contexts)
    ]

    # Wait for all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(FaultResult(
                fault_id=injectors[i].fault_id,
                status=FaultResult.Status.FAILED,
                error=str(result),
            ))
        else:
            processed_results.append(result)

    return processed_results