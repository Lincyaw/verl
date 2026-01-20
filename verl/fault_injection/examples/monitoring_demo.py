# Copyright 2026 Individual Contributor: Aoyang Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
"""Demonstration of the fault injection monitoring system."""

import logging
import time

from verl.fault_injection import FaultInjectionConfig, FaultOrchestrator
from verl.fault_injection.config import (
    FaultLayer,
    FaultTrigger,
    FaultType,
    MonitoringConfig,
    UIFaultConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_config():
    """Create a demo configuration with monitoring enabled."""
    config = FaultInjectionConfig(
        enabled=True,
        log_level="INFO",
        monitoring=MonitoringConfig(
            enabled=True,
            metrics={
                "enabled": True,
                "collection_interval": 2.0,
                "aggregation_window": 10,
            },
            dashboard={
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8080,
                "update_interval": 2.0,
            },
            alerts={
                "enabled": True,
                "rules": [],  # Use default rules
                "notification_channels": [{"type": "console", "enabled": True, "min_severity": "info"}],
            },
            impact_analysis={
                "enabled": True,
                "historical_window_hours": 1,
            },
        ),
    )

    # Add some demo faults
    config.faults = [
        UIFaultConfig(
            name="demo_hydra_error",
            fault_type=FaultType.HYDRA_CONFIG_ERROR,
            layer=FaultLayer.UI,
            trigger=FaultTrigger.IMMEDIATE,
            enabled=True,
            hydra_error_type="parse",
            description="Demo: Hydra configuration parsing error",
        ),
        UIFaultConfig(
            name="demo_ray_init_failure",
            fault_type=FaultType.RAY_INIT_FAILURE,
            layer=FaultLayer.UI,
            trigger=FaultTrigger.TIMED,
            enabled=True,
            trigger_delay=5.0,
            ray_init_error="Failed to initialize Ray cluster",
            description="Demo: Ray initialization failure",
        ),
    ]

    return config


def main():
    """Main demo function."""
    logger.info("Starting fault injection monitoring demo")

    # Create configuration
    config = create_demo_config()

    # Create orchestrator
    orchestrator = FaultOrchestrator(config)

    # Enable monitoring
    orchestrator.enable_monitoring()

    # Get dashboard URL
    dashboard_url = orchestrator.get_dashboard_url()
    if dashboard_url:
        logger.info(f"Dashboard available at: {dashboard_url}")

    # Start monitoring
    orchestrator.start_monitoring()

    try:
        logger.info("Injecting demo faults...")

        # Inject faults
        results = orchestrator.inject_all_enabled()
        logger.info(f"Injected {len(results)} faults")

        # Wait and show monitoring summary
        time.sleep(10)

        # Get monitoring summary
        summary = orchestrator.get_monitoring_summary()
        if summary:
            logger.info("Monitoring Summary:")
            logger.info(f"  Components: {summary['components_status']}")
            logger.info(f"  Current Metrics: {summary['current_metrics']}")
            logger.info(f"  Active Alerts: {summary['active_alerts']}")

        # Keep running to show dashboard
        logger.info("Demo running. Press Ctrl+C to stop.")
        logger.info("View dashboard at: http://localhost:8080")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping demo...")

    finally:
        # Cleanup
        orchestrator.shutdown()
        logger.info("Demo completed")


if __name__ == "__main__":
    main()
