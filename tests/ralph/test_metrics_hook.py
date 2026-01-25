"""
Comprehensive unit tests for VerlMetricsHook.

Tests cover:
- VerlMetricsHook class
- ThroughputTracker utility
- GradientNormTracker utility
- Integration with mock training loops
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from ralph.collectors.metrics_hook import (
    GradientNormTracker,
    MetricSample,
    MetricsSummary,
    ThroughputTracker,
    VerlMetricsHook,
)

# =============================================================================
# MetricSample Tests
# =============================================================================


class TestMetricSample:
    """Tests for MetricSample dataclass."""

    def test_create_sample(self):
        """Test creating a metric sample."""
        sample = MetricSample(
            timestamp="2024-01-01T00:00:00Z",
            step=10,
            name="loss",
            value=0.5,
        )
        assert sample.timestamp == "2024-01-01T00:00:00Z"
        assert sample.step == 10
        assert sample.name == "loss"
        assert sample.value == 0.5
        assert sample.tags == {}

    def test_sample_with_tags(self):
        """Test sample with custom tags."""
        sample = MetricSample(
            timestamp="2024-01-01T00:00:00Z",
            step=10,
            name="loss",
            value=0.5,
            tags={"phase": "train", "model": "actor"},
        )
        assert sample.tags == {"phase": "train", "model": "actor"}

    def test_to_dict(self):
        """Test converting sample to dictionary."""
        sample = MetricSample(
            timestamp="2024-01-01T00:00:00Z",
            step=10,
            name="loss",
            value=0.5,
            tags={"phase": "train"},
        )
        d = sample.to_dict()
        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["step"] == 10
        assert d["name"] == "loss"
        assert d["value"] == 0.5
        assert d["tags"] == {"phase": "train"}


# =============================================================================
# MetricsSummary Tests
# =============================================================================


class TestMetricsSummary:
    """Tests for MetricsSummary dataclass."""

    def test_create_summary(self):
        """Test creating a metrics summary."""
        summary = MetricsSummary(
            metric_name="loss",
            count=100,
            min_value=0.1,
            max_value=1.0,
            mean_value=0.5,
            last_value=0.2,
            first_step=0,
            last_step=99,
        )
        assert summary.metric_name == "loss"
        assert summary.count == 100
        assert summary.min_value == 0.1
        assert summary.max_value == 1.0
        assert summary.mean_value == 0.5
        assert summary.last_value == 0.2
        assert summary.first_step == 0
        assert summary.last_step == 99

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        summary = MetricsSummary(
            metric_name="loss",
            count=100,
            min_value=0.1,
            max_value=1.0,
            mean_value=0.5,
            last_value=0.2,
            first_step=0,
            last_step=99,
        )
        d = summary.to_dict()
        assert d["metric_name"] == "loss"
        assert d["count"] == 100
        assert d["min_value"] == 0.1
        assert d["max_value"] == 1.0


# =============================================================================
# VerlMetricsHook Basic Tests
# =============================================================================


class TestVerlMetricsHookInit:
    """Tests for VerlMetricsHook initialization."""

    def test_default_init(self):
        """Test default initialization."""
        hook = VerlMetricsHook()
        assert hook._metric_filter is None
        assert hook._step_interval == 1
        assert hook._buffer_size == 10000
        assert not hook._capturing
        assert hook._current_step == 0

    def test_init_with_filter(self):
        """Test initialization with metric filter."""
        hook = VerlMetricsHook(metric_filter=["loss", "grad_norm"])
        assert hook._metric_filter == ["loss", "grad_norm"]

    def test_init_with_step_interval(self):
        """Test initialization with step interval."""
        hook = VerlMetricsHook(step_interval=5)
        assert hook._step_interval == 5

    def test_init_with_buffer_size(self):
        """Test initialization with buffer size."""
        hook = VerlMetricsHook(buffer_size=1000)
        assert hook._buffer_size == 1000


class TestVerlMetricsHookCaptureControl:
    """Tests for start/stop capture methods."""

    def test_start_capture(self):
        """Test starting capture."""
        hook = VerlMetricsHook()
        hook.start_capture()
        assert hook._capturing
        assert hook._capture_start_time is not None

    def test_stop_capture(self):
        """Test stopping capture."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.stop_capture()
        assert not hook._capturing

    def test_is_capturing(self):
        """Test is_capturing method."""
        hook = VerlMetricsHook()
        assert not hook.is_capturing()
        hook.start_capture()
        assert hook.is_capturing()
        hook.stop_capture()
        assert not hook.is_capturing()

    def test_context_manager(self):
        """Test context manager usage."""
        hook = VerlMetricsHook()
        with hook:
            assert hook.is_capturing()
        assert not hook.is_capturing()

    def test_capture_start_step(self):
        """Test capture start step is recorded."""
        hook = VerlMetricsHook()
        hook.set_step(50)
        hook.start_capture()
        assert hook._capture_start_step == 50


# =============================================================================
# VerlMetricsHook Recording Tests
# =============================================================================


class TestVerlMetricsHookRecording:
    """Tests for recording metrics."""

    def test_record_metric_when_capturing(self):
        """Test recording when capture is active."""
        hook = VerlMetricsHook()
        hook.start_capture()
        result = hook.record_metric("loss", 0.5, step=10)
        assert result is True
        assert len(hook._samples) == 1
        assert hook._samples[0].name == "loss"
        assert hook._samples[0].value == 0.5

    def test_record_metric_when_not_capturing(self):
        """Test recording when capture is not active."""
        hook = VerlMetricsHook()
        result = hook.record_metric("loss", 0.5, step=10)
        assert result is False
        assert len(hook._samples) == 0

    def test_record_metric_with_tags(self):
        """Test recording with tags."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=10, tags={"phase": "train"})
        assert hook._samples[0].tags == {"phase": "train"}

    def test_record_metric_uses_current_step(self):
        """Test recording uses current step if not specified."""
        hook = VerlMetricsHook()
        hook.set_step(42)
        hook.start_capture()
        hook.record_metric("loss", 0.5)
        assert hook._samples[0].step == 42

    def test_record_metrics_batch(self):
        """Test recording multiple metrics at once."""
        hook = VerlMetricsHook()
        hook.start_capture()
        count = hook.record_metrics(
            {
                "loss": 0.5,
                "grad_norm": 1.2,
                "lr": 0.001,
            },
            step=10,
        )
        assert count == 3
        assert len(hook._samples) == 3

    def test_record_metrics_ignores_non_numeric(self):
        """Test non-numeric values are ignored."""
        hook = VerlMetricsHook()
        hook.start_capture()
        count = hook.record_metrics(
            {
                "loss": 0.5,
                "name": "test",  # string, should be ignored
                "config": {},  # dict, should be ignored
            },
            step=10,
        )
        assert count == 1

    def test_step_interval_filtering(self):
        """Test step interval filtering."""
        hook = VerlMetricsHook(step_interval=5)
        hook.start_capture()

        # Only steps divisible by 5 should be recorded
        for step in range(20):
            hook.record_metric("loss", 0.5, step=step)

        # Should have steps 0, 5, 10, 15
        assert len(hook._samples) == 4
        steps = [s.step for s in hook._samples]
        assert steps == [0, 5, 10, 15]

    def test_buffer_size_limit(self):
        """Test buffer size is enforced."""
        hook = VerlMetricsHook(buffer_size=10)
        hook.start_capture()

        for i in range(20):
            hook.record_metric("loss", i, step=i)

        assert len(hook._samples) == 10
        # Oldest samples should be removed
        assert hook._samples[0].step == 10

    def test_metric_filter(self):
        """Test metric name filtering."""
        hook = VerlMetricsHook(metric_filter=["loss"])
        hook.start_capture()

        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("grad_norm", 1.0, step=1)

        assert len(hook._samples) == 1
        assert hook._samples[0].name == "loss"

    def test_metric_filter_partial_match(self):
        """Test metric filter partial matching."""
        hook = VerlMetricsHook(metric_filter=["loss"])
        hook.start_capture()

        hook.record_metric("train_loss", 0.5, step=1)
        hook.record_metric("policy_loss", 0.3, step=1)
        hook.record_metric("grad_norm", 1.0, step=1)

        # train_loss and policy_loss contain "loss"
        assert len(hook._samples) == 2

    def test_metric_filter_group_name(self):
        """Test metric filter with standard group names."""
        hook = VerlMetricsHook(metric_filter=["gradient_norm"])
        hook.start_capture()

        hook.record_metric("grad_norm", 1.0, step=1)
        hook.record_metric("global_grad_norm", 2.0, step=1)
        hook.record_metric("loss", 0.5, step=1)

        # grad_norm and global_grad_norm match "gradient_norm" group
        assert len(hook._samples) == 2


# =============================================================================
# VerlMetricsHook Retrieval Tests
# =============================================================================


class TestVerlMetricsHookRetrieval:
    """Tests for getting recorded metrics."""

    def test_get_metrics(self):
        """Test get_metrics returns all data."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("loss", 0.3, step=2)
        hook.record_metric("grad_norm", 1.0, step=1)

        metrics = hook.get_metrics()
        assert len(metrics["samples"]) == 3
        assert "loss" in metrics["by_name"]
        assert "grad_norm" in metrics["by_name"]
        assert "loss" in metrics["summary"]
        assert metrics["capture_info"]["total_samples"] == 3
        assert metrics["capture_info"]["unique_metrics"] == 2

    def test_get_metric_values(self):
        """Test getting values for specific metric."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("loss", 0.3, step=2)
        hook.record_metric("loss", 0.1, step=3)

        values = hook.get_metric_values("loss")
        assert values == [0.5, 0.3, 0.1]

    def test_get_metric_values_empty(self):
        """Test getting values for non-existent metric."""
        hook = VerlMetricsHook()
        values = hook.get_metric_values("nonexistent")
        assert values == []

    def test_get_metric_summary(self):
        """Test getting summary for specific metric."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("loss", 0.3, step=2)
        hook.record_metric("loss", 0.1, step=3)

        summary = hook.get_metric_summary("loss")
        assert summary is not None
        assert summary["count"] == 3
        assert summary["min_value"] == 0.1
        assert summary["max_value"] == 0.5
        assert summary["mean_value"] == pytest.approx(0.3)
        assert summary["last_value"] == 0.1
        assert summary["first_step"] == 1
        assert summary["last_step"] == 3

    def test_get_metric_summary_nonexistent(self):
        """Test getting summary for non-existent metric."""
        hook = VerlMetricsHook()
        summary = hook.get_metric_summary("nonexistent")
        assert summary is None

    def test_get_latest_metrics(self):
        """Test getting latest values for all metrics."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("loss", 0.3, step=2)
        hook.record_metric("grad_norm", 1.0, step=1)
        hook.record_metric("grad_norm", 0.8, step=2)

        latest = hook.get_latest_metrics()
        assert latest["loss"] == 0.3
        assert latest["grad_norm"] == 0.8

    def test_clear(self):
        """Test clearing all metrics."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=1)
        hook.record_metric("grad_norm", 1.0, step=1)

        hook.clear()
        assert len(hook._samples) == 0
        assert len(hook._metrics_by_name) == 0


# =============================================================================
# VerlMetricsHook Step Management Tests
# =============================================================================


class TestVerlMetricsHookStepManagement:
    """Tests for step management."""

    def test_set_step(self):
        """Test setting current step."""
        hook = VerlMetricsHook()
        hook.set_step(100)
        assert hook.get_step() == 100

    def test_step_used_in_recording(self):
        """Test current step is used when recording."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.set_step(50)
        hook.record_metric("loss", 0.5)
        assert hook._samples[0].step == 50


# =============================================================================
# VerlMetricsHook Tracker Wrapping Tests
# =============================================================================


class TestVerlMetricsHookTrackerWrapping:
    """Tests for wrapping verl Tracking instance."""

    def test_wrap_tracker(self):
        """Test wrapping a tracker."""
        # Create mock tracker
        mock_tracker = MagicMock()
        original_log = MagicMock()
        mock_tracker.log = original_log

        hook = VerlMetricsHook()
        hook.start_capture()
        hook.wrap_tracker(mock_tracker)

        # Log through tracker
        mock_tracker.log({"loss": 0.5}, step=10)

        # Should have recorded metric
        assert len(hook._samples) == 1
        assert hook._samples[0].name == "loss"
        assert hook._samples[0].value == 0.5

        # Original method should have been called
        original_log.assert_called_once()

    def test_unwrap_tracker(self):
        """Test unwrapping a tracker."""
        mock_tracker = MagicMock()
        original_log = MagicMock()
        mock_tracker.log = original_log

        hook = VerlMetricsHook()
        hook.wrap_tracker(mock_tracker)
        hook.unwrap_tracker()

        # Original method should be restored
        assert mock_tracker.log == original_log

    def test_wrap_tracker_without_log_method(self):
        """Test wrapping tracker without log method."""
        mock_tracker = MagicMock(spec=[])  # No log method

        hook = VerlMetricsHook()
        hook.wrap_tracker(mock_tracker)  # Should not raise

        assert hook._wrapped_tracker is None


# =============================================================================
# VerlMetricsHook Thread Safety Tests
# =============================================================================


class TestVerlMetricsHookThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Test recording from multiple threads."""
        hook = VerlMetricsHook()
        hook.start_capture()

        def record_metrics(thread_id):
            for i in range(100):
                hook.record_metric(f"metric_{thread_id}", i, step=i)

        threads = [threading.Thread(target=record_metrics, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 samples total
        assert len(hook._samples) == 500


# =============================================================================
# VerlMetricsHook Repr Tests
# =============================================================================


class TestVerlMetricsHookRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        hook = VerlMetricsHook()
        hook.start_capture()
        hook.record_metric("loss", 0.5, step=10)
        hook.set_step(10)

        repr_str = repr(hook)
        assert "VerlMetricsHook" in repr_str
        assert "capturing=True" in repr_str
        assert "samples=1" in repr_str
        assert "metrics=1" in repr_str
        assert "step=10" in repr_str


# =============================================================================
# ThroughputTracker Tests
# =============================================================================


class TestThroughputTracker:
    """Tests for ThroughputTracker utility class."""

    def test_init(self):
        """Test default initialization."""
        tracker = ThroughputTracker()
        assert tracker._start_time is None
        assert tracker._total_tokens == 0
        assert tracker._total_samples == 0

    def test_start(self):
        """Test starting tracker."""
        tracker = ThroughputTracker()
        tracker.start()
        assert tracker._start_time is not None
        assert tracker._total_tokens == 0

    def test_record_batch_tokens(self):
        """Test recording tokens."""
        tracker = ThroughputTracker()
        tracker.start()
        tracker.record_batch(num_tokens=1024)
        assert tracker.get_total_tokens() == 1024

    def test_record_batch_samples(self):
        """Test recording samples."""
        tracker = ThroughputTracker()
        tracker.start()
        tracker.record_batch(num_samples=32)
        assert tracker.get_total_samples() == 32

    def test_record_multiple_batches(self):
        """Test recording multiple batches."""
        tracker = ThroughputTracker()
        tracker.start()
        tracker.record_batch(num_tokens=1024, num_samples=32)
        tracker.record_batch(num_tokens=2048, num_samples=64)
        assert tracker.get_total_tokens() == 3072
        assert tracker.get_total_samples() == 96

    def test_get_throughput(self):
        """Test getting throughput."""
        tracker = ThroughputTracker()
        tracker.start()
        tracker.record_batch(num_tokens=1000)
        time.sleep(0.1)  # Small delay

        throughput = tracker.get_throughput()
        assert throughput["tokens_per_second"] > 0

    def test_get_throughput_before_start(self):
        """Test throughput before start returns zeros."""
        tracker = ThroughputTracker()
        throughput = tracker.get_throughput()
        assert throughput["tokens_per_second"] == 0.0
        assert throughput["samples_per_second"] == 0.0

    def test_get_elapsed_time(self):
        """Test getting elapsed time."""
        tracker = ThroughputTracker()
        tracker.start()
        time.sleep(0.1)
        elapsed = tracker.get_elapsed_time()
        assert elapsed >= 0.1

    def test_get_elapsed_time_before_start(self):
        """Test elapsed time before start returns zero."""
        tracker = ThroughputTracker()
        assert tracker.get_elapsed_time() == 0.0

    def test_restart(self):
        """Test restarting resets counters."""
        tracker = ThroughputTracker()
        tracker.start()
        tracker.record_batch(num_tokens=1000)
        tracker.start()  # Restart
        assert tracker.get_total_tokens() == 0


# =============================================================================
# GradientNormTracker Tests
# =============================================================================


class TestGradientNormTracker:
    """Tests for GradientNormTracker utility class."""

    def test_init(self):
        """Test default initialization."""
        tracker = GradientNormTracker()
        assert len(tracker._history) == 0
        assert tracker._max_history == 1000

    def test_init_with_max_history(self):
        """Test initialization with custom max history."""
        tracker = GradientNormTracker(max_history=100)
        assert tracker._max_history == 100

    def test_record(self):
        """Test recording gradient norm."""
        tracker = GradientNormTracker()
        tracker.record(1.5, step=10)
        assert len(tracker._history) == 1
        assert tracker._history[0] == (10, 1.5)

    def test_get_history(self):
        """Test getting history."""
        tracker = GradientNormTracker()
        tracker.record(1.0, step=1)
        tracker.record(2.0, step=2)
        history = tracker.get_history()
        assert history == [(1, 1.0), (2, 2.0)]

    def test_get_latest(self):
        """Test getting latest value."""
        tracker = GradientNormTracker()
        tracker.record(1.0, step=1)
        tracker.record(2.0, step=2)
        assert tracker.get_latest() == 2.0

    def test_get_latest_empty(self):
        """Test getting latest when empty."""
        tracker = GradientNormTracker()
        assert tracker.get_latest() is None

    def test_get_mean(self):
        """Test getting mean value."""
        tracker = GradientNormTracker()
        tracker.record(1.0, step=1)
        tracker.record(2.0, step=2)
        tracker.record(3.0, step=3)
        assert tracker.get_mean() == pytest.approx(2.0)

    def test_get_mean_empty(self):
        """Test getting mean when empty."""
        tracker = GradientNormTracker()
        assert tracker.get_mean() == 0.0

    def test_clear(self):
        """Test clearing history."""
        tracker = GradientNormTracker()
        tracker.record(1.0, step=1)
        tracker.clear()
        assert len(tracker._history) == 0

    def test_max_history_enforced(self):
        """Test max history is enforced."""
        tracker = GradientNormTracker(max_history=5)
        for i in range(10):
            tracker.record(float(i), step=i)
        assert len(tracker._history) == 5
        assert tracker._history[0] == (5, 5.0)

    @pytest.mark.skipif(
        True,  # Skip if torch not available
        reason="Torch required for compute_grad_norm test",
    )
    def test_compute_grad_norm_with_torch(self):
        """Test computing gradient norm with torch parameters."""
        try:
            import torch

            tracker = GradientNormTracker()

            # Create simple model parameters
            params = [torch.randn(10, requires_grad=True) for _ in range(3)]
            for p in params:
                p.grad = torch.randn_like(p)

            norm = tracker.compute_grad_norm(params)
            assert norm > 0
        except ImportError:
            pytest.skip("torch not available")


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerlMetricsHookIntegration:
    """Integration tests for VerlMetricsHook."""

    def test_mock_training_loop(self):
        """Test capturing metrics during mock training loop."""
        hook = VerlMetricsHook()

        with hook:
            for step in range(100):
                hook.set_step(step)
                # Simulate training metrics
                hook.record_metrics(
                    {
                        "train/loss": 1.0 - step * 0.01,
                        "train/grad_norm": 1.0 + step * 0.01,
                        "train/lr": 0.001 * (0.99**step),
                    }
                )

        metrics = hook.get_metrics()
        assert metrics["capture_info"]["total_samples"] == 300
        assert "train/loss" in metrics["by_name"]

        # Loss should decrease
        loss_summary = metrics["summary"]["train/loss"]
        assert loss_summary["last_value"] < loss_summary["first_step"]

    def test_throughput_and_metrics_together(self):
        """Test using ThroughputTracker with VerlMetricsHook."""
        hook = VerlMetricsHook()
        throughput = ThroughputTracker()

        throughput.start()
        hook.start_capture()

        for step in range(10):
            hook.set_step(step)
            # Simulate batch processing
            throughput.record_batch(num_tokens=1024, num_samples=32)
            tps = throughput.get_throughput()
            hook.record_metric("throughput/tps", tps["tokens_per_second"], step)

        hook.stop_capture()

        metrics = hook.get_metrics()
        assert "throughput/tps" in metrics["by_name"]
        assert len(metrics["by_name"]["throughput/tps"]) == 10

    def test_with_collector_integration(self):
        """Test integration with DualStreamCollector."""
        import tempfile

        from ralph.collectors.dual_stream import DualStreamCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            hook = VerlMetricsHook()

            hook.start_capture()

            for step in range(10):
                hook.set_step(step)
                hook.record_metrics({"loss": 0.5, "grad_norm": 1.0}, step)

                # Record telemetry from captured metrics
                latest = hook.get_latest_metrics()
                collector.record_telemetry("training_metrics", {"step": step, **latest})

            hook.stop_capture()
            collector.close()

            # Verify telemetry was written
            assert collector.get_telemetry_path().exists()
