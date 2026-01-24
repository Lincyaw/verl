"""
Log parsers for Ray and NCCL logs.

Extracts structured events from system logs for correlation with
fault injection labels.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


@dataclass
class LogEvent:
    """Structured representation of a log event."""

    timestamp: str
    event_type: str
    message: str
    source_file: str = ""
    line_number: int = 0
    level: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "message": self.message,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "level": self.level,
            "metadata": self.metadata,
        }


class RayLogParser:
    """
    Parser for Ray runtime logs.

    Extracts structured events from Ray log files, including:
    - ObjectLostError events
    - Worker failures
    - Task exceptions
    - Scheduling events
    - Actor creation/destruction
    """

    # Common Ray log patterns
    PATTERNS = {
        # ObjectLostError pattern
        "object_lost": re.compile(
            r"ObjectLostError.*?object_ref\s*=\s*([a-f0-9]+)"
            r"|ObjectLostError.*?(\S+ffffffff\S+)",
            re.IGNORECASE,
        ),
        # Worker death/crash pattern
        "worker_death": re.compile(
            r"Worker\s+(?:died|crashed|killed|terminated)"
            r"|RayWorkerDied"
            r"|Worker.*(?:exit|died).*(?:code|status)\s*[=:]\s*(\d+)",
            re.IGNORECASE,
        ),
        # Task failure pattern
        "task_failure": re.compile(
            r"RayTaskError"
            r"|TaskCancelledError"
            r"|Task.*(?:failed|error|exception)",
            re.IGNORECASE,
        ),
        # Actor failure pattern
        "actor_failure": re.compile(
            r"RayActorError"
            r"|ActorDiedError"
            r"|Actor.*(?:died|crashed|failed)",
            re.IGNORECASE,
        ),
        # Scheduling timeout
        "scheduling_timeout": re.compile(
            r"(?:scheduling|placement).*?timeout"
            r"|resource.*?unavailable.*?timeout",
            re.IGNORECASE,
        ),
        # Object store full
        "object_store_full": re.compile(
            r"ObjectStoreFullError"
            r"|object\s+store.*(?:full|out\s+of\s+memory|OOM)",
            re.IGNORECASE,
        ),
        # General exception
        "exception": re.compile(
            r"(?:Exception|Error|Traceback).*?:\s*(.+)",
            re.IGNORECASE,
        ),
        # Timestamp extraction (multiple formats)
        "timestamp": re.compile(
            r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
            r"|(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"
            r"|(\[\d+\.\d+\])",  # Unix timestamp
        ),
        # Log level
        "log_level": re.compile(
            r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b",
            re.IGNORECASE,
        ),
    }

    # Ray-specific log file patterns
    RAY_LOG_PATTERNS = [
        "raylet.*.log",
        "raylet.out",
        "raylet.err",
        "gcs_server.*.log",
        "gcs_server.out",
        "gcs_server.err",
        "monitor.log",
        "dashboard.log",
        "python-core-worker-*.log",
        "python-core-driver-*.log",
        "worker-*.out",
        "worker-*.err",
    ]

    def __init__(self, log_level_filter: Optional[str] = None):
        """
        Initialize the Ray log parser.

        Args:
            log_level_filter: Minimum log level to include (e.g., "WARNING").
                              None means include all levels.
        """
        self._log_level_filter = log_level_filter
        self._level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "WARN": 2,
            "ERROR": 3,
            "CRITICAL": 4,
            "FATAL": 4,
        }

    def _should_include_level(self, level: str) -> bool:
        """Check if a log level should be included based on filter."""
        if self._log_level_filter is None:
            return True
        filter_priority = self._level_priority.get(
            self._log_level_filter.upper(), 0
        )
        level_priority = self._level_priority.get(level.upper(), 0)
        return level_priority >= filter_priority

    def _extract_timestamp(self, line: str) -> str:
        """Extract timestamp from log line."""
        match = self.PATTERNS["timestamp"].search(line)
        if match:
            # Return the first non-None group
            for group in match.groups():
                if group:
                    return group
        return ""

    def _extract_log_level(self, line: str) -> str:
        """Extract log level from log line."""
        match = self.PATTERNS["log_level"].search(line)
        if match:
            level = match.group(1).upper()
            return "WARNING" if level == "WARN" else level
        return "INFO"

    def _classify_event(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify a log line into an event type.

        Returns:
            Tuple of (event_type, metadata_dict)
        """
        metadata: Dict[str, Any] = {}

        # Check specific patterns in priority order
        if self.PATTERNS["object_lost"].search(line):
            match = self.PATTERNS["object_lost"].search(line)
            if match:
                object_ref = match.group(1) or match.group(2)
                if object_ref:
                    metadata["object_ref"] = object_ref
            return "object_lost", metadata

        if self.PATTERNS["worker_death"].search(line):
            match = self.PATTERNS["worker_death"].search(line)
            if match and match.group(1):
                metadata["exit_code"] = int(match.group(1))
            return "worker_death", metadata

        if self.PATTERNS["task_failure"].search(line):
            return "task_failure", metadata

        if self.PATTERNS["actor_failure"].search(line):
            return "actor_failure", metadata

        if self.PATTERNS["scheduling_timeout"].search(line):
            return "scheduling_timeout", metadata

        if self.PATTERNS["object_store_full"].search(line):
            return "object_store_full", metadata

        if self.PATTERNS["exception"].search(line):
            match = self.PATTERNS["exception"].search(line)
            if match:
                metadata["exception_message"] = match.group(1).strip()
            return "exception", metadata

        return "unknown", metadata

    def parse_line(
        self, line: str, source_file: str = "", line_number: int = 0
    ) -> Optional[LogEvent]:
        """
        Parse a single log line into a LogEvent.

        Args:
            line: The log line to parse
            source_file: Source file name (for context)
            line_number: Line number in source file

        Returns:
            LogEvent if the line contains a relevant event, None otherwise
        """
        line = line.strip()
        if not line:
            return None

        timestamp = self._extract_timestamp(line)
        level = self._extract_log_level(line)

        if not self._should_include_level(level):
            return None

        event_type, metadata = self._classify_event(line)

        # Only return events that are classified (not "unknown")
        # unless the log level is ERROR or higher
        if event_type == "unknown" and level not in ("ERROR", "CRITICAL", "FATAL"):
            return None

        return LogEvent(
            timestamp=timestamp,
            event_type=event_type,
            message=line,
            source_file=source_file,
            line_number=line_number,
            level=level,
            metadata=metadata,
        )

    def parse_file(self, file_path: Union[str, Path]) -> List[LogEvent]:
        """
        Parse a log file and extract structured events.

        Args:
            file_path: Path to the log file

        Returns:
            List of LogEvent objects
        """
        file_path = Path(file_path)
        events: List[LogEvent] = []

        if not file_path.exists():
            return events

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    event = self.parse_line(
                        line,
                        source_file=str(file_path),
                        line_number=line_num,
                    )
                    if event:
                        events.append(event)
        except (IOError, OSError):
            pass

        return events

    def parse_ray_logs(self, log_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parse all Ray logs in a directory and extract structured events.

        This method searches for Ray log files in the given directory
        and its subdirectories, parsing them for relevant events.

        Args:
            log_dir: Path to Ray logs directory (e.g., /tmp/ray/session_latest/logs)

        Returns:
            List of event dictionaries with timestamp, type, and message
        """
        log_dir = Path(log_dir)
        all_events: List[LogEvent] = []

        if not log_dir.exists():
            return []

        # Find Ray log files
        log_files: List[Path] = []

        # Check for log files matching Ray patterns
        for pattern in self.RAY_LOG_PATTERNS:
            log_files.extend(log_dir.glob(f"**/{pattern}"))

        # Also check .log and .out/.err files directly
        log_files.extend(log_dir.glob("**/*.log"))
        log_files.extend(log_dir.glob("**/*.out"))
        log_files.extend(log_dir.glob("**/*.err"))

        # Remove duplicates while preserving order
        seen: set = set()
        unique_files: List[Path] = []
        for f in log_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        # Parse each file
        for log_file in unique_files:
            events = self.parse_file(log_file)
            all_events.extend(events)

        # Sort by timestamp if available
        all_events.sort(key=lambda e: e.timestamp or "")

        # Convert to dict format
        return [event.to_dict() for event in all_events]


class NCCLLogParser:
    """
    Parser for NCCL (NVIDIA Collective Communications Library) logs.

    Extracts structured events from NCCL logs, including:
    - Watchdog timeouts
    - Communication failures
    - Process group errors
    - CUDA errors related to NCCL
    """

    # NCCL-specific patterns
    PATTERNS = {
        # Watchdog timeout
        "watchdog_timeout": re.compile(
            r"Watchdog.*?timeout"
            r"|NCCL.*?timeout"
            r"|Timed\s+out.*?NCCL"
            r"|WATCHDOG.*?timeout"
            r"|ncclCommWatchdog.*?timeout",
            re.IGNORECASE,
        ),
        # NCCL error
        "nccl_error": re.compile(
            r"NCCL\s+error"
            r"|ncclInternalError"
            r"|ncclSystemError"
            r"|ncclInvalidArgument"
            r"|ncclInvalidUsage"
            r"|NCCL.*?failed",
            re.IGNORECASE,
        ),
        # Communication failure
        "comm_failure": re.compile(
            r"(?:all_?reduce|all_?gather|broadcast|reduce_?scatter|barrier).*?(?:fail|error|timeout)"
            r"|communication.*?fail"
            r"|collective.*?(?:fail|error)",
            re.IGNORECASE,
        ),
        # Process group failure
        "process_group_failure": re.compile(
            r"ProcessGroup.*?(?:fail|error|destroy|abort)"
            r"|pg.*?(?:fail|destroy)"
            r"|c10d.*?(?:fail|error)",
            re.IGNORECASE,
        ),
        # Deadlock detection
        "deadlock": re.compile(
            r"deadlock"
            r"|hung.*?(?:process|operation)"
            r"|(?:process|rank).*?(?:stuck|hanging)",
            re.IGNORECASE,
        ),
        # Rank mismatch/desync
        "rank_desync": re.compile(
            r"rank.*?(?:mismatch|desync|out\s+of\s+sync)"
            r"|(?:mismatch|inconsistent).*?rank"
            r"|collective.*?(?:mismatch|inconsistent)",
            re.IGNORECASE,
        ),
        # CUDA/GPU error related to NCCL
        "cuda_nccl_error": re.compile(
            r"CUDA.*?NCCL"
            r"|NCCL.*?CUDA"
            r"|cudaErrorLaunchFailure.*?NCCL"
            r"|GPU.*?(?:NCCL|collective)",
            re.IGNORECASE,
        ),
        # Connection/socket errors
        "connection_error": re.compile(
            r"(?:socket|connection|network).*?(?:fail|error|refused|reset)"
            r"|ECONNREFUSED"
            r"|ECONNRESET"
            r"|ETIMEDOUT",
            re.IGNORECASE,
        ),
        # Timestamp extraction
        "timestamp": re.compile(
            r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
            r"|(\[\d+\.\d+\])"
            r"|(\d{2}:\d{2}:\d{2}\.\d+)",
        ),
        # Log level
        "log_level": re.compile(
            r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b",
            re.IGNORECASE,
        ),
        # Rank extraction
        "rank": re.compile(
            r"rank\s*[=:]\s*(\d+)"
            r"|\[(\d+)\]"
            r"|worker\s*(\d+)",
            re.IGNORECASE,
        ),
    }

    def __init__(self, log_level_filter: Optional[str] = None):
        """
        Initialize the NCCL log parser.

        Args:
            log_level_filter: Minimum log level to include (e.g., "WARNING").
                              None means include all levels.
        """
        self._log_level_filter = log_level_filter
        self._level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "WARN": 2,
            "ERROR": 3,
            "CRITICAL": 4,
            "FATAL": 4,
        }

    def _should_include_level(self, level: str) -> bool:
        """Check if a log level should be included based on filter."""
        if self._log_level_filter is None:
            return True
        filter_priority = self._level_priority.get(
            self._log_level_filter.upper(), 0
        )
        level_priority = self._level_priority.get(level.upper(), 0)
        return level_priority >= filter_priority

    def _extract_timestamp(self, line: str) -> str:
        """Extract timestamp from log line."""
        match = self.PATTERNS["timestamp"].search(line)
        if match:
            for group in match.groups():
                if group:
                    return group
        return ""

    def _extract_log_level(self, line: str) -> str:
        """Extract log level from log line."""
        match = self.PATTERNS["log_level"].search(line)
        if match:
            level = match.group(1).upper()
            return "WARNING" if level == "WARN" else level
        return "INFO"

    def _extract_rank(self, line: str) -> Optional[int]:
        """Extract rank number from log line."""
        match = self.PATTERNS["rank"].search(line)
        if match:
            for group in match.groups():
                if group:
                    return int(group)
        return None

    def _classify_event(self, line: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify a log line into an event type.

        Returns:
            Tuple of (event_type, metadata_dict)
        """
        metadata: Dict[str, Any] = {}

        # Extract rank if present
        rank = self._extract_rank(line)
        if rank is not None:
            metadata["rank"] = rank

        # Check specific patterns in priority order
        if self.PATTERNS["watchdog_timeout"].search(line):
            return "watchdog_timeout", metadata

        if self.PATTERNS["deadlock"].search(line):
            return "deadlock", metadata

        if self.PATTERNS["nccl_error"].search(line):
            return "nccl_error", metadata

        if self.PATTERNS["process_group_failure"].search(line):
            return "process_group_failure", metadata

        if self.PATTERNS["comm_failure"].search(line):
            return "comm_failure", metadata

        if self.PATTERNS["rank_desync"].search(line):
            return "rank_desync", metadata

        if self.PATTERNS["cuda_nccl_error"].search(line):
            return "cuda_nccl_error", metadata

        if self.PATTERNS["connection_error"].search(line):
            return "connection_error", metadata

        return "unknown", metadata

    def parse_line(
        self, line: str, source_file: str = "", line_number: int = 0
    ) -> Optional[LogEvent]:
        """
        Parse a single log line into a LogEvent.

        Args:
            line: The log line to parse
            source_file: Source file name (for context)
            line_number: Line number in source file

        Returns:
            LogEvent if the line contains a relevant event, None otherwise
        """
        line = line.strip()
        if not line:
            return None

        timestamp = self._extract_timestamp(line)
        level = self._extract_log_level(line)

        if not self._should_include_level(level):
            return None

        event_type, metadata = self._classify_event(line)

        # Only return events that are classified (not "unknown")
        # unless the log level is ERROR or higher
        if event_type == "unknown" and level not in ("ERROR", "CRITICAL", "FATAL"):
            return None

        return LogEvent(
            timestamp=timestamp,
            event_type=event_type,
            message=line,
            source_file=source_file,
            line_number=line_number,
            level=level,
            metadata=metadata,
        )

    def parse_nccl_logs(self, log_content: str) -> List[Dict[str, Any]]:
        """
        Parse NCCL log content and extract structured events.

        This method parses the content of NCCL logs (typically from
        NCCL_DEBUG output or torch.distributed logs) and extracts
        timeout/error events.

        Args:
            log_content: String content of NCCL logs

        Returns:
            List of event dictionaries with timestamp, type, and message
        """
        events: List[LogEvent] = []

        for line_num, line in enumerate(log_content.splitlines(), start=1):
            event = self.parse_line(line, line_number=line_num)
            if event:
                events.append(event)

        # Sort by timestamp if available
        events.sort(key=lambda e: e.timestamp or "")

        return [event.to_dict() for event in events]

    def parse_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parse an NCCL log file and extract structured events.

        Args:
            file_path: Path to the log file

        Returns:
            List of event dictionaries
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self.parse_nccl_logs(content)
        except (IOError, OSError):
            return []


def parse_ray_logs(log_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Convenience function to parse Ray logs from a directory.

    Args:
        log_dir: Path to Ray logs directory

    Returns:
        List of event dictionaries with timestamp, type, message
    """
    parser = RayLogParser()
    return parser.parse_ray_logs(log_dir)


def parse_nccl_logs(log_content: str) -> List[Dict[str, Any]]:
    """
    Convenience function to parse NCCL log content.

    Args:
        log_content: String content of NCCL logs

    Returns:
        List of event dictionaries with timestamp, type, message
    """
    parser = NCCLLogParser()
    return parser.parse_nccl_logs(log_content)
