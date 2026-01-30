"""
Dual stream data collector for Ralph fault injection framework.

Collects two streams of data:
- Stream A (telemetry): System telemetry events (logs, metrics, etc.)
- Stream B (labels): Ground truth fault injection labels

Both streams are written to JSONL format for easy processing.
"""

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FaultInjectionRecord:
    """Record of a fault injection event (start)."""

    fault_id: str
    timestamp: str
    fault_type: str
    target_layer: str
    target_function: str
    severity: str
    parameters: dict[str, Any]
    expected_behavior: str


@dataclass
class FaultOutcomeRecord:
    """Record of a fault injection outcome (end)."""

    fault_id: str
    timestamp: str
    outcome: str
    duration_ms: float


@dataclass
class TelemetryRecord:
    """Record of a telemetry event."""

    event_id: str
    timestamp: str
    event_type: str
    data: dict[str, Any]


class DualStreamCollector:
    """
    Collector for dual-stream fault injection data.

    Manages two output streams:
    - Telemetry stream: System events and metrics
    - Labels stream: Fault injection ground truth labels

    Data is buffered and periodically flushed to JSONL files.
    """

    def __init__(
        self,
        output_dir: str,
        telemetry_file: str = "telemetry.jsonl",
        labels_file: str = "labels.jsonl",
        buffer_size: int = 100,
        auto_flush: bool = True,
    ):
        """
        Initialize the dual stream collector.

        Args:
            output_dir: Directory to write output files
            telemetry_file: Filename for telemetry stream (default: telemetry.jsonl)
            labels_file: Filename for labels stream (default: labels.jsonl)
            buffer_size: Number of records to buffer before auto-flushing
            auto_flush: Whether to auto-flush when buffer is full
        """
        self._output_dir = Path(output_dir)
        self._telemetry_path = self._output_dir / telemetry_file
        self._labels_path = self._output_dir / labels_file
        self._buffer_size = buffer_size
        self._auto_flush = auto_flush

        # Buffers for pending records
        self._telemetry_buffer: list[TelemetryRecord] = []
        self._labels_buffer: list[dict[str, Any]] = []

        # Track active fault injections (fault_id -> FaultInjectionRecord)
        self._active_faults: dict[str, FaultInjectionRecord] = {}

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Track if collector has been closed
        self._closed = False

        # File handles (lazy initialization)
        self._telemetry_file_handle: Optional[Any] = None
        self._labels_file_handle: Optional[Any] = None

        # Ensure output directory exists
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format with UTC timezone."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_id(self) -> str:
        """Generate a unique ID for events."""
        return str(uuid.uuid4())

    def record_fault_injection(
        self,
        fault_type: str,
        target_layer: str,
        target_function: str,
        severity: str,
        parameters: dict[str, Any],
        expected_behavior: str,
    ) -> str:
        """
        Record the start of a fault injection.

        Args:
            fault_type: Type of fault being injected (e.g., "delay", "corrupt_tensor")
            target_layer: Layer where fault is injected (e.g., "L0", "L1", "L2")
            target_function: Name of the function being proxied
            severity: Severity level of the fault
            parameters: Configuration parameters for the fault
            expected_behavior: Expected system behavior during this fault

        Returns:
            Unique fault_id for tracking this injection
        """
        if self._closed:
            return ""

        fault_id = self._generate_id()
        timestamp = self._get_timestamp()

        record = FaultInjectionRecord(
            fault_id=fault_id,
            timestamp=timestamp,
            fault_type=fault_type,
            target_layer=target_layer,
            target_function=target_function,
            severity=severity,
            parameters=parameters or {},
            expected_behavior=expected_behavior or "",
        )

        with self._lock:
            # Store in active faults for later outcome matching
            self._active_faults[fault_id] = record

            # Add to labels buffer with record_type marker
            label_entry = {
                "record_type": "fault_injection_start",
                **asdict(record),
            }
            self._labels_buffer.append(label_entry)

            if self._auto_flush and len(self._labels_buffer) >= self._buffer_size:
                self._flush_labels_buffer()

        return fault_id

    def record_fault_outcome(
        self,
        fault_id: str,
        outcome: str,
        duration_ms: float = 0.0,
    ) -> None:
        """
        Record the outcome of a fault injection.

        Args:
            fault_id: The unique ID returned by record_fault_injection()
            outcome: Description of the outcome (e.g., "success", "exception: RuntimeError")
            duration_ms: Duration of the injection in milliseconds
        """
        if self._closed or not fault_id:
            return

        timestamp = self._get_timestamp()

        outcome_record = FaultOutcomeRecord(
            fault_id=fault_id,
            timestamp=timestamp,
            outcome=outcome,
            duration_ms=duration_ms,
        )

        with self._lock:
            # Remove from active faults
            start_record = self._active_faults.pop(fault_id, None)

            # Add to labels buffer with record_type marker
            label_entry = {
                "record_type": "fault_injection_end",
                **asdict(outcome_record),
            }

            # Include original fault info for context if available
            if start_record:
                label_entry["fault_type"] = start_record.fault_type
                label_entry["target_layer"] = start_record.target_layer
                label_entry["target_function"] = start_record.target_function

            self._labels_buffer.append(label_entry)

            if self._auto_flush and len(self._labels_buffer) >= self._buffer_size:
                self._flush_labels_buffer()

    def record_telemetry(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> str:
        """
        Record a telemetry event.

        Args:
            event_type: Type of telemetry event (e.g., "ray_log", "metrics", "gpu_status")
            data: Event data dictionary

        Returns:
            Unique event_id for this telemetry record
        """
        if self._closed:
            return ""

        event_id = self._generate_id()
        timestamp = self._get_timestamp()

        record = TelemetryRecord(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            data=data or {},
        )

        with self._lock:
            self._telemetry_buffer.append(record)

            if self._auto_flush and len(self._telemetry_buffer) >= self._buffer_size:
                self._flush_telemetry_buffer()

        return event_id

    def _flush_telemetry_buffer(self) -> None:
        """Flush telemetry buffer to file (internal, assumes lock is held).

        Only clears the buffer after successful write to prevent data loss.

        Raises:
            IOError, OSError: If file write fails
            TypeError, ValueError: If JSON serialization fails
        """
        if not self._telemetry_buffer:
            return

        try:
            with open(self._telemetry_path, "a", encoding="utf-8") as f:
                for record in self._telemetry_buffer:
                    json_line = json.dumps(asdict(record), ensure_ascii=False, default=self._json_default)
                    f.write(json_line + "\n")
            # Only clear buffer after successful write
            self._telemetry_buffer.clear()
        except (IOError, OSError) as e:
            logger.error(f"Failed to flush telemetry buffer to {self._telemetry_path}: {e}")
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error in telemetry buffer: {e}")
            # Try to identify and keep valid records
            valid_records = []
            for record in self._telemetry_buffer:
                try:
                    json.dumps(asdict(record), ensure_ascii=False, default=self._json_default)
                    valid_records.append(record)
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-serializable telemetry record: {record.event_id}")
            self._telemetry_buffer = valid_records
            raise

    def _flush_labels_buffer(self) -> None:
        """Flush labels buffer to file (internal, assumes lock is held).

        Only clears the buffer after successful write to prevent data loss.

        Raises:
            IOError, OSError: If file write fails
            TypeError, ValueError: If JSON serialization fails
        """
        if not self._labels_buffer:
            return

        try:
            with open(self._labels_path, "a", encoding="utf-8") as f:
                for record in self._labels_buffer:
                    json_line = json.dumps(record, ensure_ascii=False, default=self._json_default)
                    f.write(json_line + "\n")
            # Only clear buffer after successful write
            self._labels_buffer.clear()
        except (IOError, OSError) as e:
            logger.error(f"Failed to flush labels buffer to {self._labels_path}: {e}")
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error in labels buffer: {e}")
            raise

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    def flush(self) -> None:
        """
        Flush all buffered data to files.

        This method is thread-safe and can be called at any time.
        """
        if self._closed:
            return

        with self._lock:
            self._flush_telemetry_buffer()
            self._flush_labels_buffer()

    def close(self) -> None:
        """
        Flush all data and close the collector.

        After calling close(), no more data can be recorded.
        """
        if self._closed:
            return

        # Flush remaining data
        self.flush()

        with self._lock:
            self._closed = True
            self._active_faults.clear()

    def get_telemetry_path(self) -> Path:
        """Get the path to the telemetry file."""
        return self._telemetry_path

    def get_labels_path(self) -> Path:
        """Get the path to the labels file."""
        return self._labels_path

    def get_active_fault_count(self) -> int:
        """Get the number of active (unfinished) fault injections."""
        with self._lock:
            return len(self._active_faults)

    def get_buffer_sizes(self) -> dict[str, int]:
        """Get current buffer sizes."""
        with self._lock:
            return {
                "telemetry": len(self._telemetry_buffer),
                "labels": len(self._labels_buffer),
            }

    def is_closed(self) -> bool:
        """Check if the collector has been closed."""
        return self._closed

    def __enter__(self) -> "DualStreamCollector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures flush and close."""
        self.close()
