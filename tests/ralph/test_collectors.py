"""
Unit tests for DualStreamCollector.

Tests the fault injection and telemetry data collection functionality
from ralph.collectors.dual_stream.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Generator

import pytest

from ralph.collectors.dual_stream import (
    DualStreamCollector,
    FaultInjectionRecord,
    FaultOutcomeRecord,
    TelemetryRecord,
)


@pytest.fixture
def temp_output_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def collector(temp_output_dir: str) -> Generator[DualStreamCollector, None, None]:
    """Create a collector for testing."""
    c = DualStreamCollector(output_dir=temp_output_dir, auto_flush=False)
    yield c
    c.close()


@pytest.fixture
def auto_flush_collector(
    temp_output_dir: str,
) -> Generator[DualStreamCollector, None, None]:
    """Create a collector with auto-flush enabled."""
    c = DualStreamCollector(output_dir=temp_output_dir, buffer_size=2, auto_flush=True)
    yield c
    c.close()


# =============================================================================
# Data Record Tests
# =============================================================================


class TestDataRecords:
    """Tests for data record dataclasses."""

    def test_fault_injection_record(self):
        """Test FaultInjectionRecord creation."""
        record = FaultInjectionRecord(
            fault_id="test-123",
            timestamp="2024-01-01T00:00:00Z",
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={"delay_seconds": 5.0},
            expected_behavior="Delay by 5 seconds",
        )

        assert record.fault_id == "test-123"
        assert record.fault_type == "delay"
        assert record.parameters["delay_seconds"] == 5.0

    def test_fault_outcome_record(self):
        """Test FaultOutcomeRecord creation."""
        record = FaultOutcomeRecord(
            fault_id="test-123",
            timestamp="2024-01-01T00:00:01Z",
            outcome="success",
            duration_ms=100.5,
        )

        assert record.fault_id == "test-123"
        assert record.outcome == "success"
        assert record.duration_ms == 100.5

    def test_telemetry_record(self):
        """Test TelemetryRecord creation."""
        record = TelemetryRecord(
            event_id="event-456",
            timestamp="2024-01-01T00:00:00Z",
            event_type="ray_log",
            data={"message": "Test log"},
        )

        assert record.event_id == "event-456"
        assert record.event_type == "ray_log"
        assert record.data["message"] == "Test log"


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDualStreamCollectorInit:
    """Tests for DualStreamCollector initialization."""

    def test_creates_output_directory(self, temp_output_dir: str):
        """Test collector creates output directory."""
        subdir = os.path.join(temp_output_dir, "nested", "subdir")
        collector = DualStreamCollector(output_dir=subdir)

        assert os.path.exists(subdir)
        collector.close()

    def test_default_filenames(self, temp_output_dir: str):
        """Test default filenames are set."""
        collector = DualStreamCollector(output_dir=temp_output_dir)

        assert collector.get_telemetry_path().name == "telemetry.jsonl"
        assert collector.get_labels_path().name == "labels.jsonl"
        collector.close()

    def test_custom_filenames(self, temp_output_dir: str):
        """Test custom filenames are used."""
        collector = DualStreamCollector(
            output_dir=temp_output_dir,
            telemetry_file="custom_tel.jsonl",
            labels_file="custom_lbl.jsonl",
        )

        assert collector.get_telemetry_path().name == "custom_tel.jsonl"
        assert collector.get_labels_path().name == "custom_lbl.jsonl"
        collector.close()


# =============================================================================
# Fault Injection Recording Tests
# =============================================================================


class TestRecordFaultInjection:
    """Tests for record_fault_injection method."""

    def test_returns_fault_id(self, collector: DualStreamCollector):
        """Test record_fault_injection returns unique fault_id."""
        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={"delay_seconds": 5.0},
            expected_behavior="Delay by 5 seconds",
        )

        assert fault_id is not None
        assert len(fault_id) > 0

    def test_returns_unique_ids(self, collector: DualStreamCollector):
        """Test each call returns unique fault_id."""
        ids = set()
        for _ in range(100):
            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="medium",
                parameters={},
                expected_behavior="test",
            )
            ids.add(fault_id)

        assert len(ids) == 100

    def test_increments_active_fault_count(self, collector: DualStreamCollector):
        """Test recording increments active fault count."""
        assert collector.get_active_fault_count() == 0

        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        assert collector.get_active_fault_count() == 1

        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        assert collector.get_active_fault_count() == 2

    def test_handles_none_parameters(self, collector: DualStreamCollector):
        """Test handling None parameters."""
        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters=None,
            expected_behavior="test",
        )

        assert fault_id is not None

    def test_returns_empty_when_closed(self, collector: DualStreamCollector):
        """Test returns empty string when collector is closed."""
        collector.close()

        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        assert fault_id == ""


# =============================================================================
# Fault Outcome Recording Tests
# =============================================================================


class TestRecordFaultOutcome:
    """Tests for record_fault_outcome method."""

    def test_decrements_active_fault_count(self, collector: DualStreamCollector):
        """Test recording outcome decrements active fault count."""
        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        assert collector.get_active_fault_count() == 1

        collector.record_fault_outcome(
            fault_id=fault_id,
            outcome="success",
            duration_ms=100.0,
        )

        assert collector.get_active_fault_count() == 0

    def test_handles_empty_fault_id(self, collector: DualStreamCollector):
        """Test handling empty fault_id gracefully."""
        # Should not raise
        collector.record_fault_outcome(
            fault_id="",
            outcome="success",
            duration_ms=100.0,
        )

    def test_handles_unknown_fault_id(self, collector: DualStreamCollector):
        """Test handling unknown fault_id gracefully."""
        # Should not raise
        collector.record_fault_outcome(
            fault_id="unknown-id",
            outcome="success",
            duration_ms=100.0,
        )


# =============================================================================
# Telemetry Recording Tests
# =============================================================================


class TestRecordTelemetry:
    """Tests for record_telemetry method."""

    def test_returns_event_id(self, collector: DualStreamCollector):
        """Test record_telemetry returns event_id."""
        event_id = collector.record_telemetry(
            event_type="ray_log",
            data={"message": "Test log"},
        )

        assert event_id is not None
        assert len(event_id) > 0

    def test_handles_none_data(self, collector: DualStreamCollector):
        """Test handling None data."""
        event_id = collector.record_telemetry(
            event_type="test_event",
            data=None,
        )

        assert event_id is not None

    def test_returns_empty_when_closed(self, collector: DualStreamCollector):
        """Test returns empty string when collector is closed."""
        collector.close()

        event_id = collector.record_telemetry(
            event_type="test_event",
            data={"key": "value"},
        )

        assert event_id == ""


# =============================================================================
# Buffer and Flush Tests
# =============================================================================


class TestBufferAndFlush:
    """Tests for buffer management and flushing."""

    def test_get_buffer_sizes(self, collector: DualStreamCollector):
        """Test get_buffer_sizes returns correct counts."""
        assert collector.get_buffer_sizes() == {"telemetry": 0, "labels": 0}

        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        sizes = collector.get_buffer_sizes()
        assert sizes["labels"] == 1

        collector.record_telemetry("test", {"key": "value"})
        sizes = collector.get_buffer_sizes()
        assert sizes["telemetry"] == 1

    def test_flush_writes_files(self, collector: DualStreamCollector):
        """Test flush writes data to files."""
        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )
        collector.record_telemetry("test", {"key": "value"})

        collector.flush()

        # Check files exist and have content
        assert collector.get_labels_path().exists()
        assert collector.get_telemetry_path().exists()

        with open(collector.get_labels_path()) as f:
            content = f.read()
            assert len(content) > 0

        with open(collector.get_telemetry_path()) as f:
            content = f.read()
            assert len(content) > 0

    def test_flush_clears_buffers(self, collector: DualStreamCollector):
        """Test flush clears buffers."""
        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        assert collector.get_buffer_sizes()["labels"] == 1

        collector.flush()

        assert collector.get_buffer_sizes()["labels"] == 0

    def test_auto_flush_on_buffer_full(self, auto_flush_collector: DualStreamCollector):
        """Test auto-flush when buffer reaches limit."""
        # Buffer size is 2
        auto_flush_collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test1",
        )

        assert auto_flush_collector.get_buffer_sizes()["labels"] == 1

        # This should trigger auto-flush
        auto_flush_collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test2",
        )

        # Buffer should be cleared after auto-flush
        assert auto_flush_collector.get_buffer_sizes()["labels"] == 0


# =============================================================================
# JSONL Output Tests
# =============================================================================


class TestJSONLOutput:
    """Tests for JSONL file output format."""

    def test_labels_jsonl_format(self, collector: DualStreamCollector):
        """Test labels output is valid JSONL."""
        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={"delay_seconds": 5.0},
            expected_behavior="test",
        )
        collector.record_fault_outcome(fault_id, "success", 100.0)
        collector.flush()

        with open(collector.get_labels_path()) as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "record_type" in record
            assert "fault_id" in record

    def test_telemetry_jsonl_format(self, collector: DualStreamCollector):
        """Test telemetry output is valid JSONL."""
        collector.record_telemetry("test_event", {"key": "value", "number": 42})
        collector.flush()

        with open(collector.get_telemetry_path()) as f:
            lines = f.readlines()

        assert len(lines) == 1

        record = json.loads(lines[0])
        assert "event_id" in record
        assert "event_type" in record
        assert record["event_type"] == "test_event"
        assert record["data"]["key"] == "value"

    def test_labels_contains_fault_info_in_outcome(self, collector: DualStreamCollector):
        """Test fault outcome includes original fault info."""
        fault_id = collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )
        collector.record_fault_outcome(fault_id, "success", 100.0)
        collector.flush()

        with open(collector.get_labels_path()) as f:
            lines = f.readlines()

        end_record = json.loads(lines[1])
        assert end_record["record_type"] == "fault_injection_end"
        assert end_record["fault_type"] == "delay"
        assert end_record["target_layer"] == "L0"
        assert end_record["target_function"] == "ray.get"


# =============================================================================
# Close and Context Manager Tests
# =============================================================================


class TestCloseAndContextManager:
    """Tests for close method and context manager."""

    def test_close_flushes_data(self, temp_output_dir: str):
        """Test close flushes buffered data."""
        collector = DualStreamCollector(output_dir=temp_output_dir, auto_flush=False)
        collector.record_fault_injection(
            fault_type="delay",
            target_layer="L0",
            target_function="ray.get",
            severity="medium",
            parameters={},
            expected_behavior="test",
        )

        collector.close()

        # Data should be flushed to file
        assert collector.get_labels_path().exists()

    def test_is_closed(self, collector: DualStreamCollector):
        """Test is_closed returns correct state."""
        assert not collector.is_closed()

        collector.close()

        assert collector.is_closed()

    def test_context_manager(self, temp_output_dir: str):
        """Test context manager usage."""
        with DualStreamCollector(output_dir=temp_output_dir, auto_flush=False) as collector:
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="medium",
                parameters={},
                expected_behavior="test",
            )

        # After context manager exits, data should be flushed and collector closed
        assert collector.is_closed()
        assert collector.get_labels_path().exists()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_fault_injection(self, collector: DualStreamCollector):
        """Test concurrent fault injection recording."""
        results = []

        def record_injection(n):
            for i in range(n):
                fault_id = collector.record_fault_injection(
                    fault_type="delay",
                    target_layer="L0",
                    target_function="ray.get",
                    severity="medium",
                    parameters={},
                    expected_behavior=f"test-{i}",
                )
                results.append(fault_id)

        threads = [threading.Thread(target=record_injection, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All IDs should be unique
        assert len(set(results)) == 50

    def test_concurrent_flush(self, temp_output_dir: str):
        """Test concurrent flush operations."""
        collector = DualStreamCollector(output_dir=temp_output_dir, auto_flush=False)

        # Add some data
        for i in range(20):
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="medium",
                parameters={},
                expected_behavior=f"test-{i}",
            )

        # Concurrent flushes
        threads = [threading.Thread(target=collector.flush) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        collector.close()

        # Should not raise and file should exist
        assert collector.get_labels_path().exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDualStreamCollectorIntegration:
    """Integration tests for DualStreamCollector."""

    def test_full_workflow(self, temp_output_dir: str):
        """Test complete fault injection workflow."""
        with DualStreamCollector(output_dir=temp_output_dir) as collector:
            # Record some telemetry
            collector.record_telemetry("system_start", {"version": "1.0"})

            # Record fault injections
            fault_id1 = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="medium",
                parameters={"delay_seconds": 5.0},
                expected_behavior="Delay by 5 seconds",
            )

            fault_id2 = collector.record_fault_injection(
                fault_type="corrupt_tensor",
                target_layer="L1",
                target_function="all_reduce",
                severity="high",
                parameters={"noise_scale": 0.1},
                expected_behavior="Add Gaussian noise",
            )

            # More telemetry
            collector.record_telemetry("training_step", {"step": 100})

            # Record outcomes
            collector.record_fault_outcome(fault_id1, "success", 5001.5)
            collector.record_fault_outcome(fault_id2, "success", 50.0)

            # Final telemetry
            collector.record_telemetry("system_end", {"total_steps": 100})

        # Verify output files
        labels_path = Path(temp_output_dir) / "labels.jsonl"
        telemetry_path = Path(temp_output_dir) / "telemetry.jsonl"

        assert labels_path.exists()
        assert telemetry_path.exists()

        # Verify labels content
        with open(labels_path) as f:
            labels = [json.loads(line) for line in f]

        assert len(labels) == 4  # 2 starts + 2 ends

        starts = [label for label in labels if label["record_type"] == "fault_injection_start"]
        ends = [label for label in labels if label["record_type"] == "fault_injection_end"]

        assert len(starts) == 2
        assert len(ends) == 2

        # Verify telemetry content
        with open(telemetry_path) as f:
            telemetry = [json.loads(line) for line in f]

        assert len(telemetry) == 3

        event_types = [t["event_type"] for t in telemetry]
        assert "system_start" in event_types
        assert "training_step" in event_types
        assert "system_end" in event_types
