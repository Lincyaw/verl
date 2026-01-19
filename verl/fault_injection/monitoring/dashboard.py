"""Web-based dashboard for fault injection monitoring."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from .collector import MetricsCollector, AggregatedMetrics

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard server."""

    host: str = "0.0.0.0"
    port: int = 8080
    update_interval: float = 2.0  # Seconds between updates
    enable_cors: bool = True
    authentication: Optional[Dict[str, str]] = None  # username: password
    max_datapoints: int = 1000  # Maximum datapoints to show


class DashboardServer:
    """Web-based dashboard for monitoring fault injection."""

    def __init__(self, metrics_collector: MetricsCollector, config: DashboardConfig):
        self.metrics_collector = metrics_collector
        self.config = config
        self.app = Flask(__name__)
        if config.enable_cors:
            CORS(self.app)
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_TEMPLATE)

        @self.app.route("/api/metrics/current")
        def get_current_metrics():
            """Get current metrics."""
            metrics = self.metrics_collector.get_current_metrics()
            return jsonify(metrics)

        @self.app.route("/api/metrics/aggregated")
        def get_aggregated_metrics():
            """Get aggregated metrics."""
            window = request.args.get("window", 300, type=int)
            aggregation = self.metrics_collector.get_aggregated_metrics(window)
            if aggregation:
                return jsonify({
                    "window_start": aggregation.window_start.isoformat(),
                    "window_end": aggregation.window_end.isoformat(),
                    "total_faults": aggregation.total_faults,
                    "successful_faults": aggregation.successful_faults,
                    "failed_faults": aggregation.failed_faults,
                    "recovered_faults": aggregation.recovered_faults,
                    "avg_injection_duration_ms": aggregation.avg_injection_duration_ms,
                    "recovery_success_rate": aggregation.recovery_success_rate,
                    "faults_by_layer": aggregation.faults_by_layer,
                    "faults_by_type": aggregation.faults_by_type,
                })
            return jsonify({"error": "No data available"}), 404

        @self.app.route("/api/faults/history")
        def get_fault_history():
            """Get fault history."""
            limit = request.args.get("limit", 100, type=int)
            history = self.metrics_collector._fault_history

            # Return recent faults
            recent_faults = list(history)[-limit:] if history else []

            return jsonify([
                {
                    "fault_id": f.fault_id,
                    "fault_type": f.fault_type,
                    "layer": f.layer.value,
                    "status": f.status.value,
                    "timestamp": f.timestamp.isoformat(),
                    "duration_ms": f.duration_ms,
                    "recovery_time_ms": f.recovery_time_ms,
                }
                for f in recent_faults
            ])

        @self.app.route("/api/system/health")
        def get_system_health():
            """Get system health status."""
            current = self.metrics_collector.get_current_metrics()

            # Calculate health score (0-100)
            health_score = 100
            if current["system_health"] == "critical":
                health_score = 25
            elif current["system_health"] == "warning":
                health_score = 50

            # Adjust based on fault rate
            if current["recent_faults"] > 10:
                health_score -= 20
            elif current["recent_faults"] > 5:
                health_score -= 10

            # Adjust based on success rate
            success_rate = current.get("faults_success_rate", 1.0)
            if success_rate < 0.5:
                health_score -= 30
            elif success_rate < 0.8:
                health_score -= 15

            return jsonify({
                "health_score": max(0, health_score),
                "status": current["system_health"],
                "active_faults": current["faults_active"],
                "recent_faults": current["recent_faults"],
                "success_rate": current["faults_success_rate"],
            })

        @self.app.route("/api/faults/by-layer")
        def get_faults_by_layer():
            """Get fault distribution by layer."""
            breakdown = self.metrics_collector._get_layer_breakdown()
            return jsonify(breakdown)

        @self.app.route("/api/faults/by-type")
        def get_faults_by_type():
            """Get fault distribution by type."""
            breakdown = self.metrics_collector._get_type_breakdown()
            return jsonify(breakdown)

    def start(self) -> None:
        """Start the dashboard server."""
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()
        logger.info(f"Dashboard server started on http://{self.config.host}:{self.config.port}")

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
        if self._server_thread:
            self._server_thread.join(timeout=5)
        logger.info("Dashboard server stopped")

    def _run_server(self) -> None:
        """Run the Flask server."""
        try:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=False,
                use_reloader=False,
            )
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")


# Simple HTML dashboard template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fault Injection Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status-healthy { color: #22c55e; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        .fault-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .fault-table th, .fault-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .fault-table th {
            background-color: #f9fafb;
            font-weight: 600;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.875em;
            font-weight: 500;
        }
        .status-completed { background-color: #d1fae5; color: #065f46; }
        .status-failed { background-color: #fee2e2; color: #991b1b; }
        .status-active { background-color: #dbeafe; color: #1e40af; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Fault Injection Monitor</h1>
            <p>Real-time monitoring of fault injection system</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="health-score">--</div>
                <div class="metric-label">System Health Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="active-faults">--</div>
                <div class="metric-label">Active Faults</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-faults">--</div>
                <div class="metric-label">Total Faults</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="success-rate">--</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Fault Distribution by Layer</h2>
            <canvas id="layer-chart"></canvas>
        </div>

        <div class="chart-container">
            <h2>Fault Distribution by Type</h2>
            <canvas id="type-chart"></canvas>
        </div>

        <div class="chart-container">
            <h2>Recent Faults</h2>
            <table class="fault-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Fault ID</th>
                        <th>Type</th>
                        <th>Layer</th>
                        <th>Status</th>
                        <th>Duration (ms)</th>
                    </tr>
                </thead>
                <tbody id="fault-history">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Initialize charts
        const layerCtx = document.getElementById('layer-chart').getContext('2d');
        const typeCtx = document.getElementById('type-chart').getContext('2d');

        const layerChart = new Chart(layerCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });

        const typeChart = new Chart(typeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fault Count',
                    data: [],
                    backgroundColor: '#3b82f6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Update function
        async function updateMetrics() {
            try {
                // Get current metrics
                const metricsRes = await fetch('/api/metrics/current');
                const metrics = await metricsRes.json();

                // Update metric cards
                document.getElementById('health-score').textContent =
                    Math.round(metrics.faults_success_rate * 100) + '%';
                document.getElementById('active-faults').textContent = metrics.faults_active;
                document.getElementById('total-faults').textContent = metrics.faults_total;
                document.getElementById('success-rate').textContent =
                    (metrics.faults_success_rate * 100).toFixed(1) + '%';

                // Get layer distribution
                const layerRes = await fetch('/api/faults/by-layer');
                const layerData = await layerRes.json();
                layerChart.data.labels = Object.keys(layerData);
                layerChart.data.datasets[0].data = Object.values(layerData);
                layerChart.update();

                // Get type distribution
                const typeRes = await fetch('/api/faults/by-type');
                const typeData = await typeRes.json();
                typeChart.data.labels = Object.keys(typeData);
                typeChart.data.datasets[0].data = Object.values(typeData);
                typeChart.update();

                // Get fault history
                const historyRes = await fetch('/api/faults/history?limit=10');
                const history = await historyRes.json();

                const historyHtml = history.map(fault => `
                    <tr>
                        <td>${new Date(fault.timestamp).toLocaleTimeString()}</td>
                        <td>${fault.fault_id}</td>
                        <td>${fault.fault_type}</td>
                        <td>${fault.layer}</td>
                        <td><span class="status-badge status-${fault.status}">${fault.status}</span></td>
                        <td>${fault.duration_ms.toFixed(0)}</td>
                    </tr>
                `).join('');

                document.getElementById('fault-history').innerHTML = historyHtml;

            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }

        // Start updates
        updateMetrics();
        setInterval(updateMetrics, 2000);
    </script>
</body>
</html>
"""