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


"""
Unit tests for network fault injectors.
"""

import asyncio
import time
import unittest
from unittest.mock import Mock, patch

import pytest

from verl.fault_injection.base import FaultContext, FaultResult
from verl.fault_injection.injectors.network import (
    NetworkDelayInjector,
    NetworkLossInjector,
    NetworkPartitionInjector,
)


class TestNetworkDelayInjector(unittest.TestCase):
    """Test cases for NetworkDelayInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = NetworkDelayInjector(delay_ms=100, jitter_ms=20, correlation=0.5, distribution="normal")
        self.context = FaultContext(target="test_target", fault_type="network_delay", parameters={"delay_ms": 100})

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.delay_ms, 100)
        self.assertEqual(self.injector.jitter_ms, 20)
        self.assertEqual(self.injector.correlation, 0.5)
        self.assertEqual(self.injector.distribution, "normal")

    def test_inject_delay(self):
        """Test delay injection."""
        start_time = time.time()
        result = self.injector.inject(self.context)
        end_time = time.time()

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertIn("delay_ms", result.metadata)

        # Verify delay was applied
        actual_delay = (end_time - start_time) * 1000
        self.assertGreaterEqual(actual_delay, 80)  # Allow some tolerance

    def test_inject_with_zero_delay(self):
        """Test injection with zero delay."""
        injector = NetworkDelayInjector(delay_ms=0)
        start_time = time.time()
        result = injector.inject(self.context)
        end_time = time.time()

        self.assertEqual(result.status, "injected")
        actual_delay = (end_time - start_time) * 1000
        self.assertLess(actual_delay, 50)  # Should be very fast

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)

    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        with self.assertRaises(ValueError):
            NetworkDelayInjector(delay_ms=-10)

        with self.assertRaises(ValueError):
            NetworkDelayInjector(jitter_ms=-5)


class TestNetworkLossInjector(unittest.TestCase):
    """Test cases for NetworkLossInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = NetworkLossInjector(loss_percent=10, burst_length=3, correlation=0.2)
        self.context = FaultContext(target="test_target", fault_type="network_loss", parameters={"loss_percent": 10})

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.loss_percent, 10)
        self.assertEqual(self.injector.burst_length, 3)
        self.assertEqual(self.injector.correlation, 0.2)

    def test_inject_packet_loss(self):
        """Test packet loss injection."""
        # Test multiple injections to verify statistical behavior
        loss_count = 0
        total_packets = 1000

        for i in range(total_packets):
            self.context.parameters["packet_id"] = i
            result = self.injector.inject(self.context)

            if result.metadata.get("dropped", False):
                loss_count += 1

        # Verify loss rate is approximately correct (within 3% tolerance)
        actual_loss_percent = (loss_count / total_packets) * 100
        self.assertGreaterEqual(actual_loss_percent, 7)
        self.assertLessEqual(actual_loss_percent, 13)

    def test_inject_burst_loss(self):
        """Test burst loss injection."""
        # Configure for burst loss
        injector = NetworkLossInjector(loss_percent=20, burst_length=5, correlation=0.8)

        # Track consecutive drops
        # in_burst = False  # Not used
        burst_lengths = []
        current_burst = 0

        for i in range(200):
            self.context.parameters["packet_id"] = i
            result = injector.inject(self.context)

            if result.metadata.get("dropped", False):
                current_burst += 1
            else:
                if current_burst > 0:
                    burst_lengths.append(current_burst)
                    current_burst = 0

        # Should have some bursts
        self.assertGreater(len(burst_lengths), 0)
        # Burst lengths should be reasonable
        for length in burst_lengths:
            self.assertLessEqual(length, 10)  # Allow some variance

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)

    def test_invalid_loss_percent(self):
        """Test with invalid loss percentage."""
        with self.assertRaises(ValueError):
            NetworkLossInjector(loss_percent=-5)

        with self.assertRaises(ValueError):
            NetworkLossInjector(loss_percent=150)


class TestNetworkPartitionInjector(unittest.TestCase):
    """Test cases for NetworkPartitionInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = NetworkPartitionInjector(
            partition_type="partial", isolated_nodes=["node1", "node2"], duration_seconds=5
        )
        self.context = FaultContext(
            target="test_cluster", fault_type="network_partition", parameters={"partition_type": "partial"}
        )

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.partition_type, "partial")
        self.assertEqual(self.injector.isolated_nodes, ["node1", "node2"])
        self.assertEqual(self.injector.duration_seconds, 5)

    @patch("subprocess.run")
    def test_inject_partial_partition(self, mock_run):
        """Test partial network partition injection."""
        mock_run.return_value = Mock(returncode=0)

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertIn("partition_rules", result.metadata)
        self.assertIn("start_time", result.metadata)

        # Verify iptables rules were created
        self.assertGreater(mock_run.call_count, 0)

    @patch("subprocess.run")
    def test_inject_complete_partition(self, mock_run):
        """Test complete network partition injection."""
        injector = NetworkPartitionInjector(partition_type="complete", isolated_nodes=["node1"], duration_seconds=2)
        mock_run.return_value = Mock(returncode=0)

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertIn("partition_type", result.metadata)
        self.assertEqual(result.metadata["partition_type"], "complete")

    @patch("subprocess.run")
    def test_inject_with_iptables_failure(self, mock_run):
        """Test injection when iptables command fails."""
        mock_run.return_value = Mock(returncode=1, stderr="iptables: Permission denied")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)

    @patch("subprocess.run")
    def test_recover(self, mock_run):
        """Test recovery method."""
        # First inject
        mock_run.return_value = Mock(returncode=0)
        result = self.injector.inject(self.context)

        # Then recover
        mock_run.reset_mock()
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)

        # Verify iptables rules were removed
        self.assertGreater(mock_run.call_count, 0)

    def test_invalid_partition_type(self):
        """Test with invalid partition type."""
        with self.assertRaises(ValueError):
            NetworkPartitionInjector(partition_type="invalid", isolated_nodes=["node1"])


class TestNetworkFaultsIntegration(unittest.TestCase):
    """Integration tests for network fault injectors."""

    def test_sequential_faults(self):
        """Test applying multiple network faults sequentially."""
        delay_injector = NetworkDelayInjector(delay_ms=50)
        loss_injector = NetworkLossInjector(loss_percent=5)

        context = FaultContext(target="test_target", fault_type="network_combined", parameters={})

        # Apply delay
        delay_result = delay_injector.inject(context)
        self.assertEqual(delay_result.status, "injected")

        # Apply loss
        context.parameters["packet_id"] = 1
        loss_result = loss_injector.inject(context)
        self.assertEqual(loss_result.status, "injected")

        # Recover both
        delay_recover = delay_injector.recover(context, delay_result)
        loss_recover = loss_injector.recover(context, loss_result)

        self.assertEqual(delay_recover.status, "recovered")
        self.assertEqual(loss_recover.status, "recovered")

    @pytest.mark.asyncio
    async def test_async_network_delay(self):
        """Test async network delay injection."""
        injector = NetworkDelayInjector(delay_ms=100)
        context = FaultContext(target="async_target", fault_type="async_network_delay", parameters={})

        start_time = time.time()
        result = await asyncio.to_thread(injector.inject, context)
        end_time = time.time()

        self.assertEqual(result.status, "injected")
        actual_delay = (end_time - start_time) * 1000
        self.assertGreaterEqual(actual_delay, 80)


if __name__ == "__main__":
    unittest.main()
