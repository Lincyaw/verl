# Copyright 2026 Aoyang Fang
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
# ==============================================================================

"""
Unit tests for orchestration fault injectors.
"""

import time
import unittest
from unittest.mock import Mock, patch

from verl.fault_injection.base import FaultContext, FaultResult
from verl.fault_injection.injectors.orchestration import (
    PlacementGroupFaultInjector,
    RayClusterFailureInjector,
    ResourceDeadlockInjector,
)


class TestRayClusterFailureInjector(unittest.TestCase):
    """Test cases for RayClusterFailureInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = RayClusterFailureInjector(
            failure_type="gcs_disconnect", duration_seconds=10, affected_nodes=["node1", "node2"]
        )
        self.context = FaultContext(
            target="ray_cluster", fault_type="ray_cluster_failure", parameters={"failure_type": "gcs_disconnect"}
        )

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.failure_type, "gcs_disconnect")
        self.assertEqual(self.injector.duration_seconds, 10)
        self.assertEqual(self.injector.affected_nodes, ["node1", "node2"])

    @patch("ray._private.services.find_redis_address_or_die")
    @patch("redis.Redis")
    def test_inject_gcs_disconnect(self, mock_redis, mock_find_redis):
        """Test GCS disconnect injection."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_find_redis.return_value = "localhost:6379"

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["failure_type"], "gcs_disconnect")
        self.assertIn("redis_client", result.metadata)

        # Verify Redis operations were called
        mock_redis_client.execute_command.assert_called()

    @patch("ray._private.services.find_redis_address_or_die")
    @patch("redis.Redis")
    def test_inject_node_failure(self, mock_redis, mock_find_redis):
        """Test node failure injection."""
        injector = RayClusterFailureInjector(failure_type="node_failure", duration_seconds=5, affected_nodes=["node1"])
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_find_redis.return_value = "localhost:6379"

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["failure_type"], "node_failure")
        self.assertIn("affected_nodes", result.metadata)

    @patch("ray._private.services.find_redis_address_or_die")
    def test_inject_with_no_redis(self, mock_find_redis):
        """Test injection when Redis is not available."""
        mock_find_redis.side_effect = RuntimeError("Redis not found")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("Redis not found", result.metadata["error"])

    @patch("ray._private.services.find_redis_address_or_die")
    @patch("redis.Redis")
    def test_recover(self, mock_redis, mock_find_redis):
        """Test recovery method."""
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_find_redis.return_value = "localhost:6379"

        # First inject
        result = self.injector.inject(self.context)

        # Then recover
        mock_redis_client.reset_mock()
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)

        # Verify recovery operations were called
        mock_redis_client.execute_command.assert_called()

    def test_invalid_failure_type(self):
        """Test with invalid failure type."""
        with self.assertRaises(ValueError):
            RayClusterFailureInjector(failure_type="invalid")


class TestPlacementGroupFaultInjector(unittest.TestCase):
    """Test cases for PlacementGroupFaultInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = PlacementGroupFaultInjector(
            placement_group_name="test_pg", fault_type="creation_failure", retry_count=3
        )
        self.context = FaultContext(
            target="placement_group", fault_type="placement_group_fault", parameters={"name": "test_pg"}
        )

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.placement_group_name, "test_pg")
        self.assertEqual(self.injector.fault_type, "creation_failure")
        self.assertEqual(self.injector.retry_count, 3)

    @patch("ray.util.placement_group.placement_group")
    @patch("ray.get")
    def test_inject_creation_failure(self, mock_ray_get, mock_placement_group):
        """Test placement group creation failure."""
        mock_pg = Mock()
        mock_placement_group.return_value = mock_pg
        mock_ray_get.side_effect = RuntimeError("Placement group creation failed")

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["fault_type"], "creation_failure")
        self.assertEqual(result.metadata["placement_group"], "test_pg")

        # Verify placement group was created
        mock_placement_group.assert_called_once()

    @patch("ray.util.placement_group.placement_group")
    @patch("ray.get")
    def test_inject_removal_failure(self, mock_ray_get, mock_placement_group):
        """Test placement group removal failure."""
        injector = PlacementGroupFaultInjector(placement_group_name="test_pg", fault_type="removal_failure")
        mock_pg = Mock()
        mock_placement_group.return_value = mock_pg
        mock_pg.remove.side_effect = RuntimeError("Removal failed")

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["fault_type"], "removal_failure")

    @patch("ray.util.placement_group.placement_group")
    def test_inject_with_ray_not_initialized(self, mock_placement_group):
        """Test injection when Ray is not initialized."""
        mock_placement_group.side_effect = RuntimeError("Ray is not initialized")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("Ray is not initialized", result.metadata["error"])

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)
        self.assertIn("placement_group_restored", recover_result.metadata)

    def test_invalid_fault_type(self):
        """Test with invalid fault type."""
        with self.assertRaises(ValueError):
            PlacementGroupFaultInjector(placement_group_name="test", fault_type="invalid")


class TestResourceDeadlockInjector(unittest.TestCase):
    """Test cases for ResourceDeadlockInjector."""

    def setUp(self):
        """Set up test fixtures."""
        self.injector = ResourceDeadlockInjector(
            resource_type="memory", deadlock_duration=30, involved_actors=["actor1", "actor2"]
        )
        self.context = FaultContext(
            target="resource_manager", fault_type="resource_deadlock", parameters={"resource": "memory"}
        )

    def test_init(self):
        """Test injector initialization."""
        self.assertEqual(self.injector.resource_type, "memory")
        self.assertEqual(self.injector.deadlock_duration, 30)
        self.assertEqual(self.injector.involved_actors, ["actor1", "actor2"])

    @patch("ray.get")
    @patch("ray.put")
    def test_inject_memory_deadlock(self, mock_ray_put, mock_ray_get):
        """Test memory resource deadlock injection."""
        # Simulate deadlock by making ray.get block
        mock_ray_get.side_effect = lambda ref: time.sleep(1000)  # Simulate infinite wait

        result = self.injector.inject(self.context)

        self.assertIsInstance(result, FaultResult)
        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["resource_type"], "memory")
        self.assertEqual(result.metadata["deadlock_duration"], 30)
        self.assertIn("involved_actors", result.metadata)

    @patch("ray.get")
    @patch("ray.put")
    def test_inject_cpu_deadlock(self, mock_ray_put, mock_ray_get):
        """Test CPU resource deadlock injection."""
        injector = ResourceDeadlockInjector(resource_type="cpu", deadlock_duration=15, involved_actors=["actor1"])
        mock_ray_get.return_value = {"cpu_available": 0}  # Simulate no CPU available

        result = injector.inject(self.context)

        self.assertEqual(result.status, "injected")
        self.assertEqual(result.metadata["resource_type"], "cpu")

    @patch("ray.get")
    def test_inject_with_ray_timeout(self, mock_ray_get):
        """Test injection with Ray timeout."""
        from ray.exceptions import RayTimeoutError

        mock_ray_get.side_effect = RayTimeoutError("Get timed out")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "injected")  # Timeout is treated as deadlock
        self.assertIn("timeout_occurred", result.metadata)

    @patch("ray.get")
    def test_inject_with_ray_not_initialized(self, mock_ray_get):
        """Test injection when Ray is not initialized."""
        mock_ray_get.side_effect = RuntimeError("Ray is not initialized")

        result = self.injector.inject(self.context)

        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.metadata)
        self.assertIn("Ray is not initialized", result.metadata["error"])

    def test_recover(self):
        """Test recovery method."""
        result = self.injector.inject(self.context)
        recover_result = self.injector.recover(self.context, result)

        self.assertEqual(recover_result.status, "recovered")
        self.assertIn("recovery_time", recover_result.metadata)
        self.assertIn("resource_deadlock_resolved", recover_result.metadata)

    def test_invalid_resource_type(self):
        """Test with invalid resource type."""
        with self.assertRaises(ValueError):
            ResourceDeadlockInjector(resource_type="invalid", deadlock_duration=10)

    def test_invalid_duration(self):
        """Test with invalid duration."""
        with self.assertRaises(ValueError):
            ResourceDeadlockInjector(resource_type="memory", deadlock_duration=-10)


class TestOrchestrationFaultsIntegration(unittest.TestCase):
    """Integration tests for orchestration fault injectors."""

    @patch("ray._private.services.find_redis_address_or_die")
    @patch("redis.Redis")
    @patch("ray.util.placement_group.placement_group")
    @patch("ray.get")
    def test_combined_orchestration_faults(self, mock_ray_get, mock_placement_group, mock_redis, mock_find_redis):
        """Test applying multiple orchestration faults."""
        # Set up mocks
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        mock_find_redis.return_value = "localhost:6379"
        mock_placement_group.return_value = Mock()
        mock_ray_get.side_effect = [
            RuntimeError("GCS error"),  # Ray cluster failure
            RuntimeError("PG creation failed"),  # Placement group failure
            {"memory_available": 0},  # Resource deadlock
        ]

        # Create injectors
        cluster_injector = RayClusterFailureInjector(failure_type="gcs_disconnect", duration_seconds=5)
        pg_injector = PlacementGroupFaultInjector(placement_group_name="test_pg", fault_type="creation_failure")
        deadlock_injector = ResourceDeadlockInjector(resource_type="memory", deadlock_duration=10)

        context = FaultContext(target="orchestration_system", fault_type="combined_orchestration", parameters={})

        # Apply cluster failure
        cluster_result = cluster_injector.inject(context)
        self.assertEqual(cluster_result.status, "injected")

        # Apply placement group fault
        pg_result = pg_injector.inject(context)
        self.assertEqual(pg_result.status, "injected")

        # Apply resource deadlock
        deadlock_result = deadlock_injector.inject(context)
        self.assertEqual(deadlock_result.status, "injected")

        # Recover all
        cluster_recover = cluster_injector.recover(context, cluster_result)
        pg_recover = pg_injector.recover(context, pg_result)
        deadlock_recover = deadlock_injector.recover(context, deadlock_result)

        self.assertEqual(cluster_recover.status, "recovered")
        self.assertEqual(pg_recover.status, "recovered")
        self.assertEqual(deadlock_recover.status, "recovered")


if __name__ == "__main__":
    unittest.main()
