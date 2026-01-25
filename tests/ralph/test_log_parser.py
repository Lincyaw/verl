"""
Unit tests for ralph/collectors/log_parser.py.

Tests RayLogParser and NCCLLogParser classes.
"""

import os
import tempfile
from pathlib import Path

from ralph.collectors.log_parser import (
    LogEvent,
    NCCLLogParser,
    RayLogParser,
    parse_nccl_logs,
    parse_ray_logs,
)

# =============================================================================
# LogEvent Tests
# =============================================================================


class TestLogEvent:
    """Tests for LogEvent dataclass."""

    def test_create_basic_event(self):
        """Test creating a basic LogEvent."""
        event = LogEvent(
            timestamp="2024-01-15T10:30:00Z",
            event_type="object_lost",
            message="ObjectLostError: Object lost",
        )
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.event_type == "object_lost"
        assert event.message == "ObjectLostError: Object lost"
        assert event.source_file == ""
        assert event.line_number == 0
        assert event.level == "INFO"
        assert event.metadata == {}

    def test_create_full_event(self):
        """Test creating a LogEvent with all fields."""
        event = LogEvent(
            timestamp="2024-01-15T10:30:00Z",
            event_type="worker_death",
            message="Worker died with exit code 1",
            source_file="/tmp/ray/logs/worker.log",
            line_number=42,
            level="ERROR",
            metadata={"exit_code": 1},
        )
        assert event.source_file == "/tmp/ray/logs/worker.log"
        assert event.line_number == 42
        assert event.level == "ERROR"
        assert event.metadata["exit_code"] == 1

    def test_to_dict(self):
        """Test converting LogEvent to dictionary."""
        event = LogEvent(
            timestamp="2024-01-15T10:30:00Z",
            event_type="object_lost",
            message="Test message",
            source_file="test.log",
            line_number=10,
            level="ERROR",
            metadata={"key": "value"},
        )
        result = event.to_dict()
        assert result["timestamp"] == "2024-01-15T10:30:00Z"
        assert result["event_type"] == "object_lost"
        assert result["message"] == "Test message"
        assert result["source_file"] == "test.log"
        assert result["line_number"] == 10
        assert result["level"] == "ERROR"
        assert result["metadata"]["key"] == "value"


# =============================================================================
# RayLogParser Tests
# =============================================================================


class TestRayLogParserInit:
    """Tests for RayLogParser initialization."""

    def test_init_default(self):
        """Test default initialization."""
        parser = RayLogParser()
        assert parser._log_level_filter is None

    def test_init_with_filter(self):
        """Test initialization with log level filter."""
        parser = RayLogParser(log_level_filter="WARNING")
        assert parser._log_level_filter == "WARNING"


class TestRayLogParserExtraction:
    """Tests for extraction methods."""

    def test_extract_timestamp_iso(self):
        """Test extracting ISO format timestamp."""
        parser = RayLogParser()
        line = "2024-01-15T10:30:00.123Z ERROR: Something happened"
        result = parser._extract_timestamp(line)
        assert result == "2024-01-15T10:30:00.123Z"

    def test_extract_timestamp_space_format(self):
        """Test extracting space-separated timestamp."""
        parser = RayLogParser()
        line = "2024-01-15 10:30:00 ERROR: Something happened"
        result = parser._extract_timestamp(line)
        assert result == "2024-01-15 10:30:00"

    def test_extract_timestamp_unix(self):
        """Test extracting Unix timestamp."""
        parser = RayLogParser()
        line = "[1705315800.123] ERROR: Something happened"
        result = parser._extract_timestamp(line)
        assert result == "[1705315800.123]"

    def test_extract_timestamp_none(self):
        """Test no timestamp found."""
        parser = RayLogParser()
        line = "ERROR: Something happened"
        result = parser._extract_timestamp(line)
        assert result == ""

    def test_extract_log_level_error(self):
        """Test extracting ERROR level."""
        parser = RayLogParser()
        line = "2024-01-15 10:30:00 ERROR: Something happened"
        result = parser._extract_log_level(line)
        assert result == "ERROR"

    def test_extract_log_level_warn_normalized(self):
        """Test that WARN is normalized to WARNING."""
        parser = RayLogParser()
        line = "2024-01-15 10:30:00 WARN: Something happened"
        result = parser._extract_log_level(line)
        assert result == "WARNING"

    def test_extract_log_level_default(self):
        """Test default level when not found."""
        parser = RayLogParser()
        line = "Something happened without level"
        result = parser._extract_log_level(line)
        assert result == "INFO"


class TestRayLogParserClassification:
    """Tests for event classification."""

    def test_classify_object_lost(self):
        """Test classifying ObjectLostError."""
        parser = RayLogParser()
        line = "ObjectLostError object_ref = abc123def456"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "object_lost"
        assert "object_ref" in metadata

    def test_classify_object_lost_hex_ref(self):
        """Test classifying ObjectLostError with hex ref."""
        parser = RayLogParser()
        line = "ObjectLostError: 12345ffffffff67890"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "object_lost"

    def test_classify_worker_death(self):
        """Test classifying worker death."""
        parser = RayLogParser()
        line = "Worker died with exit code 1"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "worker_death"
        assert metadata.get("exit_code") == 1

    def test_classify_worker_crashed(self):
        """Test classifying worker crash."""
        parser = RayLogParser()
        line = "Worker crashed unexpectedly"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "worker_death"

    def test_classify_ray_worker_died(self):
        """Test classifying RayWorkerDied."""
        parser = RayLogParser()
        line = "RayWorkerDied: Worker process terminated"
        event_type, _ = parser._classify_event(line)
        assert event_type == "worker_death"

    def test_classify_task_failure(self):
        """Test classifying task failure."""
        parser = RayLogParser()
        line = "RayTaskError: Task failed with exception"
        event_type, _ = parser._classify_event(line)
        assert event_type == "task_failure"

    def test_classify_task_cancelled(self):
        """Test classifying TaskCancelledError."""
        parser = RayLogParser()
        line = "TaskCancelledError: Task was cancelled"
        event_type, _ = parser._classify_event(line)
        assert event_type == "task_failure"

    def test_classify_actor_failure(self):
        """Test classifying actor failure."""
        parser = RayLogParser()
        line = "RayActorError: Actor failed"
        event_type, _ = parser._classify_event(line)
        assert event_type == "actor_failure"

    def test_classify_actor_died(self):
        """Test classifying ActorDiedError."""
        parser = RayLogParser()
        line = "ActorDiedError: Actor process died"
        event_type, _ = parser._classify_event(line)
        assert event_type == "actor_failure"

    def test_classify_scheduling_timeout(self):
        """Test classifying scheduling timeout."""
        parser = RayLogParser()
        line = "Scheduling timeout while waiting for resources"
        event_type, _ = parser._classify_event(line)
        assert event_type == "scheduling_timeout"

    def test_classify_object_store_full(self):
        """Test classifying ObjectStoreFullError."""
        parser = RayLogParser()
        line = "ObjectStoreFullError: Object store is full"
        event_type, _ = parser._classify_event(line)
        assert event_type == "object_store_full"

    def test_classify_exception(self):
        """Test classifying general exception."""
        parser = RayLogParser()
        line = "Exception: Something went wrong"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "exception"
        assert "exception_message" in metadata

    def test_classify_unknown(self):
        """Test classifying unknown event."""
        parser = RayLogParser()
        line = "This is just a regular log message"
        event_type, _ = parser._classify_event(line)
        assert event_type == "unknown"


class TestRayLogParserParseLine:
    """Tests for parse_line method."""

    def test_parse_line_basic(self):
        """Test parsing a basic log line."""
        parser = RayLogParser()
        line = "2024-01-15T10:30:00Z ERROR: ObjectLostError object_ref = abc123"
        event = parser.parse_line(line)
        assert event is not None
        assert event.event_type == "object_lost"
        assert event.level == "ERROR"
        assert "abc123" in event.message

    def test_parse_line_with_context(self):
        """Test parsing with source file context."""
        parser = RayLogParser()
        line = "2024-01-15T10:30:00Z ERROR: Worker died"
        event = parser.parse_line(line, source_file="test.log", line_number=42)
        assert event is not None
        assert event.source_file == "test.log"
        assert event.line_number == 42

    def test_parse_line_empty(self):
        """Test parsing empty line."""
        parser = RayLogParser()
        event = parser.parse_line("")
        assert event is None

    def test_parse_line_whitespace_only(self):
        """Test parsing whitespace-only line."""
        parser = RayLogParser()
        event = parser.parse_line("   \t  \n  ")
        assert event is None

    def test_parse_line_unknown_info_filtered(self):
        """Test that unknown INFO events are filtered."""
        parser = RayLogParser()
        line = "2024-01-15T10:30:00Z INFO: Regular log message"
        event = parser.parse_line(line)
        assert event is None

    def test_parse_line_unknown_error_included(self):
        """Test that unknown ERROR events are included."""
        parser = RayLogParser()
        line = "2024-01-15T10:30:00Z ERROR: Unknown error occurred"
        event = parser.parse_line(line)
        assert event is not None
        assert event.level == "ERROR"

    def test_parse_line_level_filter(self):
        """Test log level filtering."""
        parser = RayLogParser(log_level_filter="ERROR")
        # INFO line should be filtered
        event = parser.parse_line("2024-01-15 INFO: ObjectLostError abc")
        assert event is None


class TestRayLogParserParseFile:
    """Tests for parse_file method."""

    def test_parse_file_basic(self):
        """Test parsing a log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-15T10:30:00Z ERROR: ObjectLostError object_ref = abc123\n")
            f.write("2024-01-15T10:30:01Z INFO: Regular message\n")
            f.write("2024-01-15T10:30:02Z ERROR: Worker died exit code 1\n")

        try:
            parser = RayLogParser()
            events = parser.parse_file(f.name)
            assert len(events) == 2
            assert events[0].event_type == "object_lost"
            assert events[1].event_type == "worker_death"
        finally:
            os.unlink(f.name)

    def test_parse_file_nonexistent(self):
        """Test parsing non-existent file."""
        parser = RayLogParser()
        events = parser.parse_file("/nonexistent/path/file.log")
        assert events == []

    def test_parse_file_empty(self):
        """Test parsing empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            fname = f.name

        try:
            parser = RayLogParser()
            events = parser.parse_file(fname)
            assert events == []
        finally:
            os.unlink(fname)


class TestRayLogParserParseRayLogs:
    """Tests for parse_ray_logs method."""

    def test_parse_ray_logs_basic(self):
        """Test parsing Ray logs from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock log file
            log_file = Path(tmpdir) / "raylet.log"
            log_file.write_text(
                "2024-01-15T10:30:00Z ERROR: ObjectLostError object_ref = abc123\n"
                "2024-01-15T10:30:01Z ERROR: Worker died\n"
            )

            parser = RayLogParser()
            events = parser.parse_ray_logs(tmpdir)
            assert len(events) == 2
            assert events[0]["event_type"] == "object_lost"
            assert events[1]["event_type"] == "worker_death"

    def test_parse_ray_logs_multiple_files(self):
        """Test parsing multiple log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple log files
            (Path(tmpdir) / "raylet.log").write_text("2024-01-15T10:30:00Z ERROR: ObjectLostError\n")
            (Path(tmpdir) / "worker.out").write_text("2024-01-15T10:30:01Z ERROR: Task failed\n")

            parser = RayLogParser()
            events = parser.parse_ray_logs(tmpdir)
            assert len(events) == 2

    def test_parse_ray_logs_nonexistent(self):
        """Test parsing non-existent directory."""
        parser = RayLogParser()
        events = parser.parse_ray_logs("/nonexistent/path")
        assert events == []

    def test_parse_ray_logs_empty_dir(self):
        """Test parsing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = RayLogParser()
            events = parser.parse_ray_logs(tmpdir)
            assert events == []

    def test_parse_ray_logs_nested(self):
        """Test parsing nested log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            subdir = Path(tmpdir) / "logs" / "session"
            subdir.mkdir(parents=True)
            (subdir / "raylet.log").write_text("2024-01-15T10:30:00Z ERROR: ObjectLostError\n")

            parser = RayLogParser()
            events = parser.parse_ray_logs(tmpdir)
            assert len(events) == 1


# =============================================================================
# NCCLLogParser Tests
# =============================================================================


class TestNCCLLogParserInit:
    """Tests for NCCLLogParser initialization."""

    def test_init_default(self):
        """Test default initialization."""
        parser = NCCLLogParser()
        assert parser._log_level_filter is None

    def test_init_with_filter(self):
        """Test initialization with log level filter."""
        parser = NCCLLogParser(log_level_filter="ERROR")
        assert parser._log_level_filter == "ERROR"


class TestNCCLLogParserExtraction:
    """Tests for extraction methods."""

    def test_extract_rank_equals(self):
        """Test extracting rank with = format."""
        parser = NCCLLogParser()
        line = "rank=3: NCCL timeout"
        rank = parser._extract_rank(line)
        assert rank == 3

    def test_extract_rank_colon(self):
        """Test extracting rank with : format."""
        parser = NCCLLogParser()
        line = "rank: 5 encountered error"
        rank = parser._extract_rank(line)
        assert rank == 5

    def test_extract_rank_brackets(self):
        """Test extracting rank in brackets."""
        parser = NCCLLogParser()
        line = "[2] NCCL error occurred"
        rank = parser._extract_rank(line)
        assert rank == 2

    def test_extract_rank_worker(self):
        """Test extracting worker number as rank."""
        parser = NCCLLogParser()
        line = "worker 4 timeout"
        rank = parser._extract_rank(line)
        assert rank == 4

    def test_extract_rank_none(self):
        """Test no rank found."""
        parser = NCCLLogParser()
        line = "NCCL error without rank"
        rank = parser._extract_rank(line)
        assert rank is None


class TestNCCLLogParserClassification:
    """Tests for event classification."""

    def test_classify_watchdog_timeout(self):
        """Test classifying Watchdog timeout."""
        parser = NCCLLogParser()
        line = "Watchdog timeout detected on rank 0"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "watchdog_timeout"
        assert metadata.get("rank") == 0

    def test_classify_nccl_timeout(self):
        """Test classifying NCCL timeout."""
        parser = NCCLLogParser()
        line = "NCCL timeout after 1800 seconds"
        event_type, _ = parser._classify_event(line)
        assert event_type == "watchdog_timeout"

    def test_classify_nccl_error(self):
        """Test classifying NCCL error."""
        parser = NCCLLogParser()
        line = "ncclInternalError: Operation failed"
        event_type, _ = parser._classify_event(line)
        assert event_type == "nccl_error"

    def test_classify_nccl_system_error(self):
        """Test classifying ncclSystemError."""
        parser = NCCLLogParser()
        line = "ncclSystemError occurred"
        event_type, _ = parser._classify_event(line)
        assert event_type == "nccl_error"

    def test_classify_comm_failure_allreduce(self):
        """Test classifying all_reduce failure."""
        parser = NCCLLogParser()
        line = "all_reduce failed on rank 3"
        event_type, metadata = parser._classify_event(line)
        assert event_type == "comm_failure"
        assert metadata.get("rank") == 3

    def test_classify_comm_failure_barrier(self):
        """Test classifying barrier timeout."""
        parser = NCCLLogParser()
        line = "barrier timeout after 300s"
        event_type, _ = parser._classify_event(line)
        assert event_type == "comm_failure"

    def test_classify_process_group_failure(self):
        """Test classifying ProcessGroup failure."""
        parser = NCCLLogParser()
        line = "ProcessGroup destroyed unexpectedly"
        event_type, _ = parser._classify_event(line)
        assert event_type == "process_group_failure"

    def test_classify_c10d_error(self):
        """Test classifying c10d error."""
        parser = NCCLLogParser()
        line = "c10d error: Backend failed"
        event_type, _ = parser._classify_event(line)
        assert event_type == "process_group_failure"

    def test_classify_deadlock(self):
        """Test classifying deadlock."""
        parser = NCCLLogParser()
        line = "Deadlock detected in collective operation"
        event_type, _ = parser._classify_event(line)
        assert event_type == "deadlock"

    def test_classify_hung_process(self):
        """Test classifying hung process."""
        parser = NCCLLogParser()
        line = "Process stuck waiting for collective"
        event_type, _ = parser._classify_event(line)
        assert event_type == "deadlock"

    def test_classify_rank_desync(self):
        """Test classifying rank desync."""
        parser = NCCLLogParser()
        line = "Rank mismatch detected in collective"
        event_type, _ = parser._classify_event(line)
        assert event_type == "rank_desync"

    def test_classify_cuda_nccl_error(self):
        """Test classifying CUDA NCCL error."""
        parser = NCCLLogParser()
        line = "CUDA error in NCCL operation"
        event_type, _ = parser._classify_event(line)
        assert event_type == "cuda_nccl_error"

    def test_classify_connection_error(self):
        """Test classifying connection error."""
        parser = NCCLLogParser()
        line = "Socket connection refused"
        event_type, _ = parser._classify_event(line)
        assert event_type == "connection_error"

    def test_classify_econnreset(self):
        """Test classifying ECONNRESET."""
        parser = NCCLLogParser()
        line = "ECONNRESET: Connection reset by peer"
        event_type, _ = parser._classify_event(line)
        assert event_type == "connection_error"

    def test_classify_unknown(self):
        """Test classifying unknown event."""
        parser = NCCLLogParser()
        line = "Regular log message without errors"
        event_type, _ = parser._classify_event(line)
        assert event_type == "unknown"


class TestNCCLLogParserParseLine:
    """Tests for parse_line method."""

    def test_parse_line_basic(self):
        """Test parsing a basic log line."""
        parser = NCCLLogParser()
        line = "2024-01-15T10:30:00Z ERROR: Watchdog timeout rank=3"
        event = parser.parse_line(line)
        assert event is not None
        assert event.event_type == "watchdog_timeout"
        assert event.level == "ERROR"
        assert event.metadata.get("rank") == 3

    def test_parse_line_empty(self):
        """Test parsing empty line."""
        parser = NCCLLogParser()
        event = parser.parse_line("")
        assert event is None

    def test_parse_line_unknown_info_filtered(self):
        """Test that unknown INFO events are filtered."""
        parser = NCCLLogParser()
        line = "INFO: Regular NCCL initialization message"
        event = parser.parse_line(line)
        assert event is None


class TestNCCLLogParserParseNCCLLogs:
    """Tests for parse_nccl_logs method."""

    def test_parse_nccl_logs_basic(self):
        """Test parsing NCCL log content."""
        parser = NCCLLogParser()
        content = """2024-01-15T10:30:00Z ERROR: Watchdog timeout rank=0
2024-01-15T10:30:01Z INFO: Regular message
2024-01-15T10:30:02Z ERROR: NCCL error on rank 1"""

        events = parser.parse_nccl_logs(content)
        assert len(events) == 2
        assert events[0]["event_type"] == "watchdog_timeout"
        assert events[1]["event_type"] == "nccl_error"

    def test_parse_nccl_logs_empty(self):
        """Test parsing empty content."""
        parser = NCCLLogParser()
        events = parser.parse_nccl_logs("")
        assert events == []

    def test_parse_nccl_logs_sorted(self):
        """Test events are sorted by timestamp."""
        parser = NCCLLogParser()
        content = """2024-01-15T10:30:02Z ERROR: NCCL error
2024-01-15T10:30:00Z ERROR: Watchdog timeout
2024-01-15T10:30:01Z ERROR: Deadlock detected"""

        events = parser.parse_nccl_logs(content)
        assert len(events) == 3
        # Should be sorted by timestamp
        assert events[0]["timestamp"] == "2024-01-15T10:30:00Z"
        assert events[1]["timestamp"] == "2024-01-15T10:30:01Z"
        assert events[2]["timestamp"] == "2024-01-15T10:30:02Z"

    def test_parse_nccl_logs_with_ranks(self):
        """Test extracting ranks from logs."""
        parser = NCCLLogParser()
        content = """2024-01-15T10:30:00Z ERROR: [0] Watchdog timeout
2024-01-15T10:30:01Z ERROR: rank=2 NCCL error"""

        events = parser.parse_nccl_logs(content)
        assert len(events) == 2
        assert events[0]["metadata"]["rank"] == 0
        assert events[1]["metadata"]["rank"] == 2


class TestNCCLLogParserParseFile:
    """Tests for parse_file method."""

    def test_parse_file_basic(self):
        """Test parsing NCCL log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-15T10:30:00Z ERROR: Watchdog timeout\n")
            f.write("2024-01-15T10:30:01Z ERROR: NCCL error\n")
            fname = f.name

        try:
            parser = NCCLLogParser()
            events = parser.parse_file(fname)
            assert len(events) == 2
        finally:
            os.unlink(fname)

    def test_parse_file_nonexistent(self):
        """Test parsing non-existent file."""
        parser = NCCLLogParser()
        events = parser.parse_file("/nonexistent/file.log")
        assert events == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_parse_ray_logs_function(self):
        """Test parse_ray_logs convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "raylet.log").write_text("2024-01-15T10:30:00Z ERROR: ObjectLostError\n")

            events = parse_ray_logs(tmpdir)
            assert len(events) == 1
            assert events[0]["event_type"] == "object_lost"

    def test_parse_nccl_logs_function(self):
        """Test parse_nccl_logs convenience function."""
        content = "2024-01-15T10:30:00Z ERROR: Watchdog timeout"
        events = parse_nccl_logs(content)
        assert len(events) == 1
        assert events[0]["event_type"] == "watchdog_timeout"


# =============================================================================
# Integration Tests
# =============================================================================


class TestLogParserIntegration:
    """Integration tests for log parsers."""

    def test_parse_realistic_ray_logs(self):
        """Test parsing realistic Ray log content."""
        content = """2024-01-15 10:30:00,123 INFO worker.py:1234 -- Starting worker process
2024-01-15 10:30:01,456 ERROR worker.py:1235 -- RayTaskError: Task 1234 failed
2024-01-15 10:30:02,789 ERROR driver.py:100 -- ObjectLostError: Object lost object_ref=abc123
2024-01-15 10:30:03,012 WARNING scheduler.py:50 -- Scheduling timeout detected
2024-01-15 10:30:04,345 ERROR worker.py:200 -- Worker died with exit status=1"""

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "worker.log").write_text(content)

            parser = RayLogParser()
            events = parser.parse_ray_logs(tmpdir)

            # Should have task_failure, object_lost, scheduling_timeout, worker_death
            assert len(events) >= 4
            event_types = {e["event_type"] for e in events}
            assert "task_failure" in event_types
            assert "object_lost" in event_types
            assert "scheduling_timeout" in event_types
            assert "worker_death" in event_types

    def test_parse_realistic_nccl_logs(self):
        """Test parsing realistic NCCL log content."""
        content = """[2024-01-15 10:30:00] INFO: NCCL initialized with 8 GPUs
[2024-01-15 10:30:01] ERROR: [0] Watchdog timeout after 1800s
[2024-01-15 10:30:02] ERROR: [1] ncclInternalError: Collective operation failed
[2024-01-15 10:30:03] ERROR: rank=2 ProcessGroup destroyed
[2024-01-15 10:30:04] ERROR: [3] all_reduce failed with timeout"""

        events = parse_nccl_logs(content)

        # Should have watchdog_timeout, nccl_error, process_group_failure, comm_failure
        assert len(events) >= 4
        event_types = {e["event_type"] for e in events}
        assert "watchdog_timeout" in event_types
        assert "nccl_error" in event_types
        assert "process_group_failure" in event_types
        assert "comm_failure" in event_types

    def test_parse_mixed_ray_nccl_logs(self):
        """Test parsing logs with both Ray and NCCL errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Ray logs
            (Path(tmpdir) / "raylet.log").write_text(
                "2024-01-15T10:30:00Z ERROR: ObjectLostError\n2024-01-15T10:30:01Z ERROR: Worker crashed\n"
            )
            # NCCL logs (might be in stderr)
            (Path(tmpdir) / "worker.err").write_text(
                "2024-01-15T10:30:02Z ERROR: NCCL timeout\n2024-01-15T10:30:03Z ERROR: Watchdog timeout\n"
            )

            # Parse Ray logs
            ray_parser = RayLogParser()
            ray_events = ray_parser.parse_ray_logs(tmpdir)

            # Parse NCCL logs from file content
            nccl_content = (Path(tmpdir) / "worker.err").read_text()
            nccl_events = parse_nccl_logs(nccl_content)

            assert len(ray_events) >= 2
            assert len(nccl_events) >= 2
