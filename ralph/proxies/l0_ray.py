"""
L0 Ray proxies for Ralph fault injection framework.

Contains proxy classes for Ray-level operations like ray.get, ray.put, etc.
"""

import random
from typing import Any, List, Optional, Set, Union

from ralph.core.config import StrategyType
from ralph.core.registry import ProxyRegistry
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.proxies.base import BaseProxy


# Attempt to import ray exceptions, falling back to custom exceptions if not available
try:
    import ray
    import ray.exceptions

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class ObjectLostError(Exception):
    """
    Fallback exception for when Ray is not installed.

    Mimics ray.exceptions.ObjectLostError for testing purposes.
    """

    def __init__(
        self,
        object_ref: Any = None,
        owner_address: str = "",
        call_site: str = "",
    ):
        self.object_ref = object_ref
        self.owner_address = owner_address
        self.call_site = call_site
        super().__init__(
            f"Object lost: ref={object_ref}, owner={owner_address}, call_site={call_site}"
        )


@ProxyRegistry.register("ray.get")
class RayGetProxy(BaseProxy, DelayMixin, ExceptionMixin, TensorCorruptionMixin):
    """
    Proxy for ray.get() operations.

    Supports fault injection at the Ray object store level (L0).

    Supported strategies:
    - DELAY: Adds delay before calling original ray.get
    - RAISE_EXCEPTION: Raises a configurable exception instead of calling ray.get
    - CORRUPT_TENSOR: Corrupts tensor data in ray.get results with Gaussian noise
    - OBJECT_LOST: Raises ObjectLostError simulating lost objects
    - PARTIAL_FAILURE: Sets a ratio of results to None simulating partial failures

    Config parameters:
    - DELAY: delay_seconds (float, default 10.0) - seconds to delay
    - RAISE_EXCEPTION: exc_type (str, required), message (str, optional)
    - CORRUPT_TENSOR: noise_scale (float, default 0.1)
    - OBJECT_LOST: No specific parameters
    - PARTIAL_FAILURE: fail_ratio (float, default 0.5) - ratio of results to set to None
    """

    SUPPORTED_STRATEGIES: Set[StrategyType] = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.OBJECT_LOST,
        StrategyType.PARTIAL_FAILURE,
    }

    def _get_layer(self) -> str:
        """Return the layer this proxy belongs to."""
        return "L0"

    def _strategy_object_lost(
        self,
        object_refs: Union[Any, List[Any]],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate ray ObjectLostError.

        Raises ObjectLostError as if the object was lost from the object store.
        The original function is never called.

        Args:
            object_refs: Single object reference or list of references
            *args: Additional positional arguments (ignored)
            timeout: Timeout parameter (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Raises:
            ray.exceptions.ObjectLostError: If Ray is installed
            ObjectLostError: If Ray is not installed (fallback)
        """
        # Get the first reference to report in the error
        ref = object_refs[0] if isinstance(object_refs, list) else object_refs

        if HAS_RAY:
            raise ray.exceptions.ObjectLostError(
                object_ref=ref,
                owner_address="ralph_simulated",
                call_site="ralph_fault_injection",
            )
        else:
            raise ObjectLostError(
                object_ref=ref,
                owner_address="ralph_simulated",
                call_site="ralph_fault_injection",
            )

    def _strategy_partial_failure(
        self,
        object_refs: Union[Any, List[Any]],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Simulate partial failure where some results are None.

        Calls the original ray.get, then sets a ratio of the results to None
        to simulate partial failures in the object store.

        Args:
            object_refs: Single object reference or list of references
            *args: Additional positional arguments to pass to original
            timeout: Timeout parameter to pass to original
            **kwargs: Additional keyword arguments to pass to original

        Returns:
            Results with fail_ratio of them set to None.
            For single reference, may return None or the actual result.
            For list of references, returns list with some elements as None.

        Config parameters:
            fail_ratio (float): Ratio of results to set to None (0.0 to 1.0, default 0.5)
        """
        # Build kwargs for original call
        call_kwargs = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout

        # Call original to get results
        results = self._original(object_refs, *args, **call_kwargs)

        # Get fail ratio from config
        fail_ratio = self._config.parameters.get("fail_ratio", 0.5)

        # Handle single reference case
        if not isinstance(object_refs, list):
            if random.random() < fail_ratio:
                return None
            return results

        # Handle list of references - results should be a list
        if not isinstance(results, list):
            # If results is not a list, wrap and unwrap
            results = [results]
            for i in range(len(results)):
                if random.random() < fail_ratio:
                    results[i] = None
            return results[0]

        # Process list results
        for i in range(len(results)):
            if random.random() < fail_ratio:
                results[i] = None

        return results
