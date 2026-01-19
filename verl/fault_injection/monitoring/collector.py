"""Real-time metrics collector for fault injection system."""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Deque

from ..base import FaultLayer, FaultStatus, FaultResult

logger = logging.getLogger(__name__)


@dataclass
class FaultMetrics:
    """Metrics for fault injection events."""

    fault_id: str
    fault_type: str
    layer: FaultLayer
    status: FaultStatus
    timestamp: datetime
    duration_ms: float
    target_info: Dict[str, Any]
    error_message: Optional[str] = None
    recovery_time_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System-level metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_utilization: Optional[Dict[int, float]] = None
    gpu_memory_used_mb: Optional[Dict[int, float]] = None
    gpu_memory_total_mb: Optional[Dict[int, float]] = None
    disk_usage_percent: float = 0.0
    network_io_mb: Optional[Dict[str, float]] = None
    ray_cluster_nodes: int = 0
    ray_active_actors: int = 0
    ray_pending_tasks: int = 0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over time windows."""

    window_start: datetime
    window_end: datetime
    window_seconds: int

    # Fault statistics
    total_faults: int = 0
    successful_faults: int = 0
    failed_faults: int = 0
    recovered_faults: int = 0
    active_faults: int = 0

    # Layer breakdown
    faults_by_layer: Dict[FaultLayer, int] = field(default_factory=lambda: defaultdict(int))

    # Type breakdown
    faults_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Timing statistics
    avg_injection_duration_ms: float = 0.0
    max_injection_duration_ms: float = 0.0
    min_injection_duration_ms: float = 0.0

    # Recovery statistics
    avg_recovery_time_ms: float = 0.0
    recovery_success_rate: float = 0.0

    # System health
    avg_cpu_percent: float = 0.0
    avg_memory_percent: float = 0.0
    avg_gpu_utilization: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics for the fault injection system."""

    def __init__(
        self,
        collection_interval: float = 5.0,
        aggregation_window: int = 60,
        max_history_size: int = 10000,
    ):
        self.collection_interval = collection_interval
        self.aggregation_window = aggregation_window
        self.max_history_size = max_history_size

        # Metrics storage
        self._fault_metrics: Deque[FaultMetrics] = deque(maxlen=max_history_size)
        self._system_metrics: Deque[SystemMetrics] = deque(maxlen=max_history_size)
        self._aggregated_metrics: Deque[AggregatedMetrics] = deque(maxlen=1440)  # 24 hours

        # Threading
        self._collection_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._lock = threading.Lock()

        # Callbacks
        self._metric_callbacks: List[callable] = []

        # GPU monitoring
        self._gpu_monitoring_enabled = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._gpu_monitoring_enabled = True
            logger.info(f"GPU monitoring enabled for {self._gpu_count} GPUs")
        except ImportError:
            logger.info("pynvml not available, GPU monitoring disabled")
        except pynvml.NVMLError:
            logger.info("No NVIDIA GPUs found, GPU monitoring disabled")

        # Ray monitoring
        self._ray_monitoring_enabled = False
        try:
            import ray
            if ray.is_initialized():
                self._ray_monitoring_enabled = True
                logger.info("Ray monitoring enabled")
        except ImportError:
            logger.info("Ray not available, Ray monitoring disabled")

    def start(self) -> None:
        """Start metrics collection."""
        if self._collection_thread is not None:
            return

        self._shutdown = False
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info(f"Metrics collector started (interval: {self.collection_interval}s)")

    def stop(self) -> None:
        """Stop metrics collection."""
        self._shutdown = True
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
            self._collection_thread = None
        logger.info("Metrics collector stopped")

    def record_fault(self, result: FaultResult) -> None:
        """Record a fault injection event."""
        metrics = FaultMetrics(
            fault_id=result.fault_id,
            fault_type=result.fault_type,
            layer=result.layer,
            status=result.status,
            timestamp=datetime.fromtimestamp(result.end_time or time.time()),
            duration_ms=(result.duration or 0) * 1000,
            target_info=result.target_info or {},
            error_message=result.error_message,
            recovery_time_ms=result.recovery_time_ms,
        )

        with self._lock:
            self._fault_metrics.append(metrics)

        # Notify callbacks
        for callback in self._metric_callbacks:
            try:
                callback("fault", metrics)
            except Exception as e:
                logger.error(f"Fault metric callback failed: {e}")

    def register_callback(self, callback: callable) -> None:
        """Register a callback for metric updates."""
        self._metric_callbacks.append(callback)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            # Calculate current statistics
            fault_count = len(self._fault_metrics)
            if fault_count == 0:
                return {
                    "faults_total": 0,
                    "faults_active": 0,
                    "faults_success_rate": 0.0,
                    "system_health": "unknown",
                }

            # Recent fault statistics (last 5 minutes)
            recent_cutoff = time.time() - 300
            recent_faults = [
                f for f in self._fault_metrics
                if f.timestamp.timestamp() > recent_cutoff
            ]

            successful = sum(1 for f in recent_faults if f.status == FaultStatus.COMPLETED)
            total_recent = len(recent_faults)

            # System health
            system_health = self._calculate_system_health()

            return {
                "faults_total": fault_count,
                "faults_active": sum(1 for f in self._fault_metrics if f.status not in [FaultStatus.COMPLETED, FaultStatus.FAILED]),
                "faults_success_rate": successful / total_recent if total_recent > 0 else 0.0,
                "system_health": system_health,
                "recent_faults": total_recent,
                "by_layer": self._get_layer_breakdown(),
                "by_type": self._get_type_breakdown(),
            }

    def get_aggregated_metrics(self, window_seconds: int = 300) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics for a time window."""
        with self._lock:
            cutoff_time = datetime.now().timestamp() - window_seconds

            # Filter faults in window
            window_faults = [
                f for f in self._fault_metrics
                if f.timestamp.timestamp() >= cutoff_time
            ]

            if not window_faults:
                return None

            # Calculate aggregated metrics
            return self._calculate_aggregation(window_faults, window_seconds)

    def _collection_loop(self) -> None:
        """Main collection loop."""
        logger.info("Starting metrics collection loop")

        while not self._shutdown:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                with self._lock:
                    self._system_metrics.append(system_metrics)

                # Calculate aggregations every aggregation_window seconds
                if int(time.time()) % self.aggregation_window == 0:
                    self._calculate_periodic_aggregation()

                # Sleep until next collection
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        import psutil

        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disk usage
        disk_usage = psutil.disk_usage('/')

        # GPU metrics
        gpu_utilization = None
        gpu_memory_used = None
        gpu_memory_total = None

        if self._gpu_monitoring_enabled:
            try:
                import pynvml
                gpu_utilization = {}
                gpu_memory_used = {}
                gpu_memory_total = {}

                for i in range(self._gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization[i] = utilization.gpu

                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used[i] = mem_info.used / 1024 / 1024  # MB
                    gpu_memory_total[i] = mem_info.total / 1024 / 1024  # MB

            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")

        # Network I/O
        network_io = None
        try:
            net_io = psutil.net_io_counters()
            network_io = {
                "bytes_sent_mb": net_io.bytes_sent / 1024 / 1024,
                "bytes_recv_mb": net_io.bytes_recv / 1024 / 1024,
            }
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")

        # Ray cluster metrics
        ray_nodes = 0
        ray_actors = 0
        ray_tasks = 0

        if self._ray_monitoring_enabled:
            try:
                import ray
                ray_nodes = len(ray.nodes())
                ray_actors = len(ray.actors())
                # This is a simplified metric - in practice you'd use Ray metrics
                ray_tasks = 0
            except Exception as e:
                logger.error(f"Error collecting Ray metrics: {e}")

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / 1024 / 1024,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            disk_usage_percent=disk_usage.percent,
            network_io_mb=network_io,
            ray_cluster_nodes=ray_nodes,
            ray_active_actors=ray_actors,
            ray_pending_tasks=ray_tasks,
        )

    def _calculate_system_health(self) -> str:
        """Calculate overall system health."""
        if not self._system_metrics:
            return "unknown"

        # Get recent system metrics
        recent = list(self._system_metrics)[-10:]  # Last 10 measurements
        if not recent:
            return "unknown"

        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_percent for m in recent) / len(recent)

        # Determine health based on thresholds
        if avg_cpu > 90 or avg_memory > 90:
            return "critical"
        elif avg_cpu > 70 or avg_memory > 70:
            return "warning"
        else:
            return "healthy"

    def _get_layer_breakdown(self) -> Dict[str, int]:
        """Get fault count breakdown by layer."""
        breakdown = defaultdict(int)
        for fault in self._fault_metrics:
            breakdown[fault.layer.value] += 1
        return dict(breakdown)

    def _get_type_breakdown(self) -> Dict[str, int]:
        """Get fault count breakdown by type."""
        breakdown = defaultdict(int)
        for fault in self._fault_metrics:
            breakdown[fault.fault_type] += 1
        return dict(breakdown)

    def _calculate_aggregation(self, faults: List[FaultMetrics], window_seconds: int) -> AggregatedMetrics:
        """Calculate aggregated metrics for a list of faults."""
        if not faults:
            return AggregatedMetrics(
                window_start=datetime.now(),
                window_end=datetime.now(),
                window_seconds=window_seconds,
            )

        # Basic statistics
        total = len(faults)
        successful = sum(1 for f in faults if f.status == FaultStatus.COMPLETED)
        failed = sum(1 for f in faults if f.status == FaultStatus.FAILED)
        recovered = sum(1 for f in faults if f.recovery_time_ms is not None)

        # Timing statistics
        durations = [f.duration_ms for f in faults if f.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0

        # Recovery statistics
        recovery_times = [f.recovery_time_ms for f in faults if f.recovery_time_ms is not None]
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        recovery_success_rate = recovered / total if total > 0 else 0

        # Layer breakdown
        layer_stats = defaultdict(int)
        for fault in faults:
            layer_stats[fault.layer] += 1

        # Type breakdown
        type_stats = defaultdict(int)
        for fault in faults:
            type_stats[fault.fault_type] += 1

        return AggregatedMetrics(
            window_start=faults[0].timestamp,
            window_end=faults[-1].timestamp,
            window_seconds=window_seconds,
            total_faults=total,
            successful_faults=successful,
            failed_faults=failed,
            recovered_faults=recovered,
            faults_by_layer=dict(layer_stats),
            faults_by_type=dict(type_stats),
            avg_injection_duration_ms=avg_duration,
            max_injection_duration_ms=max_duration,
            min_injection_duration_ms=min_duration,
            avg_recovery_time_ms=avg_recovery,
            recovery_success_rate=recovery_success_rate,
        )

    def _calculate_periodic_aggregation(self) -> None:
        """Calculate periodic aggregation."""
        with self._lock:
            if not self._fault_metrics:
                return

            # Create aggregation for the last window
            aggregation = self._calculate_aggregation(
                list(self._fault_metrics),
                self.aggregation_window,
            )

            self._aggregated_metrics.append(aggregation)

            logger.debug(f"Calculated periodic aggregation: {aggregation.total_faults} faults")