"""
Tests for the DualStreamCollector class.
"""

import json
import os
import tempfile
import threading
import time

from ralph.collectors.dual_stream import (
    DualStreamCollector,
)


class TestDualStreamCollectorInit:
    """Tests for DualStreamCollector initialization."""

    def test_init_creates_output_directory(self):
        """Test that initialization creates the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_dir")
            assert not os.path.exists(output_dir)

            collector = DualStreamCollector(output_dir)
            assert os.path.exists(output_dir)
            collector.close()

    def test_init_with_default_filenames(self):
        """Test default telemetry and labels file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            assert collector.get_telemetry_path().name == "telemetry.jsonl"
            assert collector.get_labels_path().name == "labels.jsonl"
            collector.close()

    def test_init_with_custom_filenames(self):
        """Test custom telemetry and labels file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(
                tmpdir,
                telemetry_file="custom_telemetry.jsonl",
                labels_file="custom_labels.jsonl",
            )
            assert collector.get_telemetry_path().name == "custom_telemetry.jsonl"
            assert collector.get_labels_path().name == "custom_labels.jsonl"
            collector.close()

    def test_init_sets_buffer_size(self):
        """Test buffer size configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, buffer_size=50)
            # Buffer size is internal, so we can only test indirectly
            assert not collector.is_closed()
            collector.close()


class TestRecordFaultInjection:
    """Tests for record_fault_injection method."""

    def test_record_fault_injection_returns_uuid(self):
        """Test that fault injection returns a valid UUID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={"delay_seconds": 1.0},
                expected_behavior="Slow response",
            )

            assert fault_id is not None
            assert len(fault_id) == 36  # UUID format
            collector.close()

    def test_record_fault_injection_tracks_active_faults(self):
        """Test that fault injection is tracked as active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            assert collector.get_active_fault_count() == 1
            collector.close()

    def test_record_fault_injection_increments_buffer(self):
        """Test that fault injection adds to buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            sizes = collector.get_buffer_sizes()
            assert sizes["labels"] == 1
            assert sizes["telemetry"] == 0
            collector.close()

    def test_record_fault_injection_with_none_parameters(self):
        """Test handling of None parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters=None,
                expected_behavior=None,
            )

            assert fault_id is not None
            collector.close()

    def test_record_fault_injection_after_close_returns_empty(self):
        """Test that recording after close returns empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.close()

            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            assert fault_id == ""


class TestRecordFaultOutcome:
    """Tests for record_fault_outcome method."""

    def test_record_fault_outcome_removes_active_fault(self):
        """Test that outcome removes fault from active tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            assert collector.get_active_fault_count() == 1
            collector.record_fault_outcome(fault_id, "success", 100.0)
            assert collector.get_active_fault_count() == 0
            collector.close()

    def test_record_fault_outcome_with_exception(self):
        """Test recording exception outcome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            fault_id = collector.record_fault_injection(
                fault_type="raise_exception",
                target_layer="L0",
                target_function="ray.get",
                severity="high",
                parameters={},
                expected_behavior="",
            )

            collector.record_fault_outcome(fault_id, "exception: RuntimeError: test error", 50.0)

            # Should have 2 records: start and end
            sizes = collector.get_buffer_sizes()
            assert sizes["labels"] == 2
            collector.close()

    def test_record_fault_outcome_with_empty_fault_id(self):
        """Test that empty fault_id is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            collector.record_fault_outcome("", "success", 0.0)

            sizes = collector.get_buffer_sizes()
            assert sizes["labels"] == 0  # Nothing recorded
            collector.close()

    def test_record_fault_outcome_after_close(self):
        """Test that recording after close does nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )
            collector.close()

            # This should not raise
            collector.record_fault_outcome(fault_id, "success", 0.0)


class TestRecordTelemetry:
    """Tests for record_telemetry method."""

    def test_record_telemetry_returns_event_id(self):
        """Test that telemetry returns a valid event ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            event_id = collector.record_telemetry(
                event_type="ray_log",
                data={"message": "Worker started"},
            )

            assert event_id is not None
            assert len(event_id) == 36  # UUID format
            collector.close()

    def test_record_telemetry_increments_buffer(self):
        """Test that telemetry adds to buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            collector.record_telemetry(
                event_type="metrics",
                data={"throughput": 100.5},
            )

            sizes = collector.get_buffer_sizes()
            assert sizes["telemetry"] == 1
            assert sizes["labels"] == 0
            collector.close()

    def test_record_telemetry_with_none_data(self):
        """Test handling of None data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            event_id = collector.record_telemetry(
                event_type="event",
                data=None,
            )

            assert event_id is not None
            collector.close()

    def test_record_telemetry_after_close(self):
        """Test that recording after close returns empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.close()

            event_id = collector.record_telemetry(
                event_type="event",
                data={},
            )

            assert event_id == ""


class TestFlush:
    """Tests for flush method."""

    def test_flush_writes_labels_to_file(self):
        """Test that flush writes labels buffer to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)

            fault_id = collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={"delay_seconds": 1.0},
                expected_behavior="Slow response",
            )
            collector.record_fault_outcome(fault_id, "success", 100.0)

            # Buffer should have 2 records
            assert collector.get_buffer_sizes()["labels"] == 2

            # Flush
            collector.flush()

            # Buffer should be empty
            assert collector.get_buffer_sizes()["labels"] == 0

            # File should exist and have content
            with open(collector.get_labels_path()) as f:
                lines = f.readlines()
                assert len(lines) == 2

            collector.close()

    def test_flush_writes_telemetry_to_file(self):
        """Test that flush writes telemetry buffer to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)

            collector.record_telemetry("event1", {"key": "value1"})
            collector.record_telemetry("event2", {"key": "value2"})

            collector.flush()

            with open(collector.get_telemetry_path()) as f:
                lines = f.readlines()
                assert len(lines) == 2

            collector.close()

    def test_flush_on_closed_collector(self):
        """Test that flush on closed collector does nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.close()

            # Should not raise
            collector.flush()


class TestClose:
    """Tests for close method."""

    def test_close_flushes_data(self):
        """Test that close flushes all buffered data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)

            collector.record_telemetry("event", {"key": "value"})
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            collector.close()

            # Files should have content
            assert collector.get_telemetry_path().exists()
            assert collector.get_labels_path().exists()

    def test_close_sets_closed_flag(self):
        """Test that close sets the closed flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            assert not collector.is_closed()
            collector.close()
            assert collector.is_closed()

    def test_close_clears_active_faults(self):
        """Test that close clears active faults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.record_fault_injection(
                fault_type="delay",
                target_layer="L0",
                target_function="ray.get",
                severity="low",
                parameters={},
                expected_behavior="",
            )

            assert collector.get_active_fault_count() == 1
            collector.close()
            assert collector.get_active_fault_count() == 0

    def test_double_close(self):
        """Test that double close is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir)
            collector.close()
            collector.close()  # Should not raise


class TestContextManager:
    """Tests for context manager interface."""

    def test_context_manager_closes_on_exit(self):
        """Test that exiting context manager closes collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                collector.record_telemetry("event", {"key": "value"})
                assert not collector.is_closed()

            assert collector.is_closed()

    def test_context_manager_flushes_on_exit(self):
        """Test that exiting context manager flushes data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir, auto_flush=False) as collector:
                collector.record_telemetry("event", {"key": "value"})
                telemetry_path = collector.get_telemetry_path()

            # File should exist with content after context exit
            with open(telemetry_path) as f:
                lines = f.readlines()
                assert len(lines) == 1


class TestAutoFlush:
    """Tests for auto-flush functionality."""

    def test_auto_flush_on_buffer_full(self):
        """Test that auto-flush triggers when buffer is full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, buffer_size=3, auto_flush=True)

            # Add 2 records (below threshold)
            collector.record_telemetry("event1", {})
            collector.record_telemetry("event2", {})

            # File should not exist yet or be empty
            if collector.get_telemetry_path().exists():
                with open(collector.get_telemetry_path()) as f:
                    assert len(f.readlines()) == 0

            # Add 3rd record (triggers flush)
            collector.record_telemetry("event3", {})

            # Buffer should be empty after auto-flush
            assert collector.get_buffer_sizes()["telemetry"] == 0

            # File should have content
            with open(collector.get_telemetry_path()) as f:
                lines = f.readlines()
                assert len(lines) == 3

            collector.close()

    def test_no_auto_flush_when_disabled(self):
        """Test that auto-flush doesn't trigger when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, buffer_size=2, auto_flush=False)

            # Add 3 records (would exceed threshold)
            collector.record_telemetry("event1", {})
            collector.record_telemetry("event2", {})
            collector.record_telemetry("event3", {})

            # Buffer should still have all 3
            assert collector.get_buffer_sizes()["telemetry"] == 3

            collector.close()


class TestJSONLFormat:
    """Tests for JSONL output format."""

    def test_telemetry_jsonl_valid_format(self):
        """Test that telemetry output is valid JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                collector.record_telemetry("event", {"key": "value"})
                telemetry_path = collector.get_telemetry_path()

            with open(telemetry_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    assert "event_id" in record
                    assert "timestamp" in record
                    assert "event_type" in record
                    assert "data" in record

    def test_labels_jsonl_valid_format(self):
        """Test that labels output is valid JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                fault_id = collector.record_fault_injection(
                    fault_type="delay",
                    target_layer="L0",
                    target_function="ray.get",
                    severity="low",
                    parameters={"delay_seconds": 1.0},
                    expected_behavior="Slow response",
                )
                collector.record_fault_outcome(fault_id, "success", 100.0)
                labels_path = collector.get_labels_path()

            with open(labels_path) as f:
                lines = f.readlines()
                assert len(lines) == 2

                # First line is fault_injection_start
                start_record = json.loads(lines[0])
                assert start_record["record_type"] == "fault_injection_start"
                assert start_record["fault_id"] == fault_id
                assert start_record["fault_type"] == "delay"
                assert start_record["target_layer"] == "L0"

                # Second line is fault_injection_end
                end_record = json.loads(lines[1])
                assert end_record["record_type"] == "fault_injection_end"
                assert end_record["fault_id"] == fault_id
                assert end_record["outcome"] == "success"

    def test_unicode_in_data(self):
        """Test that unicode characters are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                collector.record_telemetry(
                    "event",
                    {"message": "测试中文字符", "emoji": "🚀"},
                )
                telemetry_path = collector.get_telemetry_path()

            with open(telemetry_path, encoding="utf-8") as f:
                record = json.loads(f.readline())
                assert record["data"]["message"] == "测试中文字符"
                assert record["data"]["emoji"] == "🚀"


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_fault_injection(self):
        """Test concurrent fault injection recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            fault_ids = []
            errors = []

            def record_fault(i):
                try:
                    fault_id = collector.record_fault_injection(
                        fault_type="delay",
                        target_layer="L0",
                        target_function="ray.get",
                        severity="low",
                        parameters={"index": i},
                        expected_behavior="",
                    )
                    fault_ids.append(fault_id)
                except Exception as e:
                    errors.append(e)

            # Launch multiple threads
            threads = [threading.Thread(target=record_fault, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(fault_ids) == 10
            assert len(set(fault_ids)) == 10  # All unique
            collector.close()

    def test_concurrent_telemetry_recording(self):
        """Test concurrent telemetry recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DualStreamCollector(tmpdir, auto_flush=False)
            event_ids = []
            errors = []

            def record_event(i):
                try:
                    event_id = collector.record_telemetry(
                        event_type="test",
                        data={"index": i},
                    )
                    event_ids.append(event_id)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=record_event, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(event_ids) == 10
            collector.close()


class TestIntegrationWithBaseProxy:
    """Integration tests simulating BaseProxy usage pattern."""

    def test_full_fault_injection_workflow(self):
        """Test the complete workflow as used by BaseProxy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                # Simulate multiple fault injections
                for i in range(3):
                    # Start injection
                    fault_id = collector.record_fault_injection(
                        fault_type="delay",
                        target_layer="L0",
                        target_function="ray.get",
                        severity="low",
                        parameters={"delay_seconds": 0.1 * (i + 1)},
                        expected_behavior="Slow response",
                    )

                    # Simulate some work
                    time.sleep(0.01)

                    # Record outcome
                    collector.record_fault_outcome(fault_id, "success", 10.0 * (i + 1))

                labels_path = collector.get_labels_path()

            # Verify output
            with open(labels_path) as f:
                lines = f.readlines()
                assert len(lines) == 6  # 3 starts + 3 ends

                # Parse all records
                records = [json.loads(line) for line in lines]
                starts = [r for r in records if r["record_type"] == "fault_injection_start"]
                ends = [r for r in records if r["record_type"] == "fault_injection_end"]

                assert len(starts) == 3
                assert len(ends) == 3

                # All ends should reference valid starts
                start_ids = {r["fault_id"] for r in starts}
                end_ids = {r["fault_id"] for r in ends}
                assert start_ids == end_ids

    def test_fault_with_exception_outcome(self):
        """Test fault injection that results in exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DualStreamCollector(tmpdir) as collector:
                fault_id = collector.record_fault_injection(
                    fault_type="raise_exception",
                    target_layer="L0",
                    target_function="ray.get",
                    severity="critical",
                    parameters={"exc_type": "RuntimeError", "message": "Test error"},
                    expected_behavior="RuntimeError raised",
                )

                collector.record_fault_outcome(
                    fault_id,
                    "exception: RuntimeError: Test error",
                    5.0,
                )

                labels_path = collector.get_labels_path()

            with open(labels_path) as f:
                records = [json.loads(line) for line in f]

            end_record = [r for r in records if r["record_type"] == "fault_injection_end"][0]
            assert "exception" in end_record["outcome"]
            assert "RuntimeError" in end_record["outcome"]
