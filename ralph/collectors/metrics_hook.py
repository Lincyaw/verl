"""
Metrics hook for capturing verl training metrics during fault injection.

Integrates with verl's tracking backend (wandb/tensorboard) to capture
training metrics such as throughput (TPS), loss, and gradient norms.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import torch
except ImportError:
    torch = None


@dataclass
class MetricSample:
    """A single metric sample with timestamp and step info."""

    timestamp: str
    step: int
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
        }


@dataclass
class MetricsSummary:
    """Summary statistics for captured metrics."""

    metric_name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    last_value: float
    first_step: int
    last_step: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "last_value": self.last_value,
            "first_step": self.first_step,
            "last_step": self.last_step,
        }


class VerlMetricsHook:
    """
    Hook for capturing verl training metrics during fault injection experiments.

    Integrates with verl's Tracking class to intercept and record metrics
    from training loops. Supports multiple tracking backends including
    wandb, tensorboard, mlflow, etc.

    Usage:
        hook = VerlMetricsHook()
        hook.start_capture()
        # ... training loop runs ...
        hook.stop_capture()
        metrics = hook.get_metrics()

    The hook can intercept metrics by wrapping a Tracking instance's log method,
    or can be used standalone to record metrics directly.
    """

    # Standard metric names for verl training
    METRIC_NAMES = {
        # Throughput metrics
        "throughput": ["throughput", "tps", "tokens_per_second", "samples_per_second"],
        # Loss metrics
        "loss": ["loss", "train_loss", "policy_loss", "value_loss", "critic_loss", "actor_loss"],
        # Gradient metrics
        "gradient_norm": ["grad_norm", "gradient_norm", "global_grad_norm", "actor_grad_norm", "critic_grad_norm"],
        # KL divergence
        "kl": ["kl", "kl_divergence", "kl_penalty", "approx_kl"],
        # Reward metrics
        "reward": ["reward", "mean_reward", "returns", "advantages"],
        # Learning rate
        "lr": ["lr", "learning_rate", "actor_lr", "critic_lr"],
    }

    def __init__(
        self,
        metric_filter: Optional[List[str]] = None,
        step_interval: int = 1,
        buffer_size: int = 10000,
    ):
        """
        Initialize the metrics hook.

        Args:
            metric_filter: List of metric name patterns to capture. If None, captures all.
            step_interval: Only capture metrics every N steps. Default is 1 (all steps).
            buffer_size: Maximum number of samples to buffer before oldest are dropped.
        """
        self._metric_filter = metric_filter
        self._step_interval = step_interval
        self._buffer_size = buffer_size

        # Capture state
        self._capturing = False
        self._capture_start_time: Optional[float] = None
        self._capture_start_step: Optional[int] = None

        # Metrics storage
        self._samples: List[MetricSample] = []
        self._metrics_by_name: Dict[str, List[MetricSample]] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Wrapped tracker reference
        self._wrapped_tracker: Optional[Any] = None
        self._original_log_method: Optional[Callable] = None

        # Step tracking
        self._current_step: int = 0
        self._last_capture_step: int = -1

    def start_capture(self) -> None:
        """
        Start capturing metrics.

        Call this before the training loop begins or at the point where
        you want to start recording metrics.
        """
        with self._lock:
            self._capturing = True
            self._capture_start_time = time.time()
            self._capture_start_step = self._current_step

    def stop_capture(self) -> None:
        """
        Stop capturing metrics.

        Call this after the training loop ends or when you want to stop
        recording. Captured metrics remain available via get_metrics().
        """
        with self._lock:
            self._capturing = False

    def is_capturing(self) -> bool:
        """Check if metrics capture is currently active."""
        with self._lock:
            return self._capturing

    def set_step(self, step: int) -> None:
        """
        Update the current training step.

        Args:
            step: Current training step number.
        """
        with self._lock:
            self._current_step = step

    def get_step(self) -> int:
        """Get the current training step."""
        with self._lock:
            return self._current_step

    def record_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Record a single metric value.

        Args:
            name: Name of the metric (e.g., "loss", "grad_norm").
            value: Numeric value of the metric.
            step: Training step. If None, uses current step.
            tags: Optional tags for categorization.

        Returns:
            True if metric was recorded, False if capture is not active or filtered.
        """
        with self._lock:
            if not self._capturing:
                return False

            actual_step = step if step is not None else self._current_step

            # Check step interval
            if self._step_interval > 1:
                if actual_step % self._step_interval != 0:
                    return False

            # Check metric filter
            if self._metric_filter and not self._matches_filter(name):
                return False

            # Create sample
            sample = MetricSample(
                timestamp=datetime.now(timezone.utc).isoformat(),
                step=actual_step,
                name=name,
                value=float(value),
                tags=tags or {},
            )

            # Add to storage
            self._samples.append(sample)
            if name not in self._metrics_by_name:
                self._metrics_by_name[name] = []
            self._metrics_by_name[name].append(sample)

            # Enforce buffer size
            if len(self._samples) > self._buffer_size:
                removed = self._samples.pop(0)
                if removed.name in self._metrics_by_name:
                    name_samples = self._metrics_by_name[removed.name]
                    if name_samples and name_samples[0] == removed:
                        name_samples.pop(0)

            self._last_capture_step = actual_step
            return True

    def record_metrics(
        self,
        data: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Record multiple metrics at once.

        Args:
            data: Dictionary mapping metric names to values.
            step: Training step. If None, uses current step.
            tags: Optional tags for all metrics.

        Returns:
            Number of metrics actually recorded.
        """
        recorded = 0
        for name, value in data.items():
            if isinstance(value, (int, float)):
                if self.record_metric(name, value, step, tags):
                    recorded += 1
            elif torch is not None and isinstance(value, torch.Tensor):
                # Handle scalar tensors
                if value.numel() == 1:
                    if self.record_metric(name, value.item(), step, tags):
                        recorded += 1
        return recorded

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all captured metrics.

        Returns:
            Dictionary containing:
            - samples: List of all metric samples
            - by_name: Dictionary grouping samples by metric name
            - summary: Summary statistics for each metric
            - capture_info: Information about the capture session
        """
        with self._lock:
            samples = [s.to_dict() for s in self._samples]

            by_name = {}
            for name, name_samples in self._metrics_by_name.items():
                by_name[name] = [s.to_dict() for s in name_samples]

            summaries = {}
            for name, name_samples in self._metrics_by_name.items():
                if name_samples:
                    summaries[name] = self._compute_summary(name, name_samples).to_dict()

            capture_info = {
                "capturing": self._capturing,
                "start_time": self._capture_start_time,
                "start_step": self._capture_start_step,
                "current_step": self._current_step,
                "last_capture_step": self._last_capture_step,
                "total_samples": len(self._samples),
                "unique_metrics": len(self._metrics_by_name),
            }

            return {
                "samples": samples,
                "by_name": by_name,
                "summary": summaries,
                "capture_info": capture_info,
            }

    def get_metric_values(self, name: str) -> List[float]:
        """
        Get all values for a specific metric.

        Args:
            name: Metric name.

        Returns:
            List of values, empty if metric not found.
        """
        with self._lock:
            if name in self._metrics_by_name:
                return [s.value for s in self._metrics_by_name[name]]
            return []

    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a specific metric.

        Args:
            name: Metric name.

        Returns:
            Summary dict or None if metric not found.
        """
        with self._lock:
            if name in self._metrics_by_name and self._metrics_by_name[name]:
                return self._compute_summary(name, self._metrics_by_name[name]).to_dict()
            return None

    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get the most recent value for each metric.

        Returns:
            Dictionary mapping metric names to their latest values.
        """
        with self._lock:
            latest = {}
            for name, samples in self._metrics_by_name.items():
                if samples:
                    latest[name] = samples[-1].value
            return latest

    def clear(self) -> None:
        """Clear all captured metrics."""
        with self._lock:
            self._samples.clear()
            self._metrics_by_name.clear()
            self._last_capture_step = -1

    def wrap_tracker(self, tracker: Any) -> None:
        """
        Wrap a verl Tracking instance to intercept logged metrics.

        Args:
            tracker: A verl.utils.tracking.Tracking instance.
        """
        if hasattr(tracker, "log"):
            self._wrapped_tracker = tracker
            self._original_log_method = tracker.log

            def wrapped_log(data, step, backend=None):
                # Record metrics through our hook
                self.record_metrics(data, step)
                # Call original log method
                return self._original_log_method(data, step, backend)

            tracker.log = wrapped_log

    def unwrap_tracker(self) -> None:
        """Restore the original log method on the wrapped tracker."""
        if self._wrapped_tracker is not None and self._original_log_method is not None:
            self._wrapped_tracker.log = self._original_log_method
            self._wrapped_tracker = None
            self._original_log_method = None

    def _matches_filter(self, name: str) -> bool:
        """Check if metric name matches the filter patterns."""
        if not self._metric_filter:
            return True

        name_lower = name.lower()
        for pattern in self._metric_filter:
            pattern_lower = pattern.lower()
            # Support simple wildcard matching
            if pattern_lower == "*":
                return True
            if pattern_lower in name_lower or name_lower in pattern_lower:
                return True
            # Check against standard metric name groups
            for group_name, aliases in self.METRIC_NAMES.items():
                if pattern_lower == group_name:
                    for alias in aliases:
                        if alias in name_lower:
                            return True
        return False

    def _compute_summary(self, name: str, samples: List[MetricSample]) -> MetricsSummary:
        """Compute summary statistics for a list of samples."""
        values = [s.value for s in samples]
        steps = [s.step for s in samples]

        return MetricsSummary(
            metric_name=name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=sum(values) / len(values),
            last_value=values[-1],
            first_step=min(steps),
            last_step=max(steps),
        )

    def __enter__(self) -> "VerlMetricsHook":
        """Context manager entry - starts capture."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops capture."""
        self.stop_capture()

    def __repr__(self) -> str:
        """String representation of the hook."""
        with self._lock:
            return (
                f"VerlMetricsHook("
                f"capturing={self._capturing}, "
                f"samples={len(self._samples)}, "
                f"metrics={len(self._metrics_by_name)}, "
                f"step={self._current_step})"
            )


class ThroughputTracker:
    """
    Utility class for tracking throughput (tokens/samples per second).

    Usage:
        tracker = ThroughputTracker()
        tracker.start()
        # ... process batch ...
        tracker.record_batch(num_tokens=1024)
        tps = tracker.get_throughput()
    """

    def __init__(self):
        """Initialize the throughput tracker."""
        self._start_time: Optional[float] = None
        self._total_tokens: int = 0
        self._total_samples: int = 0
        self._batch_times: List[float] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start or restart throughput tracking."""
        with self._lock:
            self._start_time = time.time()
            self._total_tokens = 0
            self._total_samples = 0
            self._batch_times.clear()

    def record_batch(
        self,
        num_tokens: int = 0,
        num_samples: int = 0,
        batch_time: Optional[float] = None,
    ) -> None:
        """
        Record a processed batch.

        Args:
            num_tokens: Number of tokens in the batch.
            num_samples: Number of samples in the batch.
            batch_time: Time taken for this batch (if tracking per-batch).
        """
        with self._lock:
            self._total_tokens += num_tokens
            self._total_samples += num_samples
            if batch_time is not None:
                self._batch_times.append(batch_time)

    def get_throughput(self) -> Dict[str, float]:
        """
        Get current throughput metrics.

        Returns:
            Dictionary with tokens_per_second and samples_per_second.
        """
        with self._lock:
            if self._start_time is None:
                return {"tokens_per_second": 0.0, "samples_per_second": 0.0}

            elapsed = time.time() - self._start_time
            if elapsed <= 0:
                return {"tokens_per_second": 0.0, "samples_per_second": 0.0}

            return {
                "tokens_per_second": self._total_tokens / elapsed,
                "samples_per_second": self._total_samples / elapsed,
            }

    def get_total_tokens(self) -> int:
        """Get total tokens processed."""
        with self._lock:
            return self._total_tokens

    def get_total_samples(self) -> int:
        """Get total samples processed."""
        with self._lock:
            return self._total_samples

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time


class GradientNormTracker:
    """
    Utility class for tracking gradient norms.

    Usage:
        tracker = GradientNormTracker()
        for step in training_loop:
            norm = tracker.compute_grad_norm(model.parameters())
            tracker.record(norm, step)
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the gradient norm tracker.

        Args:
            max_history: Maximum number of gradient norms to store.
        """
        self._history: List[tuple] = []  # (step, norm)
        self._max_history = max_history
        self._lock = threading.Lock()

    def compute_grad_norm(
        self,
        parameters,
        norm_type: float = 2.0,
    ) -> float:
        """
        Compute the gradient norm for a set of parameters.

        Args:
            parameters: Iterable of parameters (or parameter groups).
            norm_type: Type of norm (default: L2).

        Returns:
            Total gradient norm as a float.
        """
        if torch is None:
            return 0.0

        # Handle parameter groups
        params = []
        for p in parameters:
            if isinstance(p, dict):
                # Parameter group
                params.extend(p.get("params", []))
            else:
                params.append(p)

        # Filter parameters with gradients
        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        if norm_type == float("inf"):
            total_norm = max(g.abs().max().item() for g in grads)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(g, norm_type) for g in grads]),
                norm_type,
            ).item()

        return total_norm

    def record(self, grad_norm: float, step: int) -> None:
        """
        Record a gradient norm value.

        Args:
            grad_norm: Computed gradient norm.
            step: Training step.
        """
        with self._lock:
            self._history.append((step, grad_norm))
            if len(self._history) > self._max_history:
                self._history.pop(0)

    def get_history(self) -> List[tuple]:
        """Get the gradient norm history as list of (step, norm) tuples."""
        with self._lock:
            return list(self._history)

    def get_latest(self) -> Optional[float]:
        """Get the most recent gradient norm."""
        with self._lock:
            if self._history:
                return self._history[-1][1]
            return None

    def get_mean(self) -> float:
        """Get the mean gradient norm over history."""
        with self._lock:
            if not self._history:
                return 0.0
            return sum(norm for _, norm in self._history) / len(self._history)

    def clear(self) -> None:
        """Clear the gradient norm history."""
        with self._lock:
            self._history.clear()
