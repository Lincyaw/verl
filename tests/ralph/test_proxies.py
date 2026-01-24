"""
Consolidated unit tests for all Ralph proxy classes.

Tests all proxy implementations including L0 (Ray), L1 (Distributed),
L2 (verl), L3 (Resource), Algorithm, Inference, and Worker proxies.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.base import BaseProxy


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_collector():
    """Create a mock collector for testing."""
    collector = MagicMock()
    collector.record_fault_injection.return_value = "fault_123"
    return collector


@pytest.fixture
def one_shot_trigger():
    """Create a ONE_SHOT trigger at step 10."""
    return TriggerConfig(type=TriggerType.ONE_SHOT, at_step=10)


@pytest.fixture
def periodic_trigger():
    """Create a PERIODIC trigger every 5 steps."""
    return TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5)


# =============================================================================
# BaseProxy Tests
# =============================================================================


class TestBaseProxyAbstract:
    """Test that BaseProxy is abstract."""

    def test_cannot_instantiate_directly(self):
        """Test BaseProxy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseProxy(lambda x: x, None)


class TestBaseProxyConcreteImplementation:
    """Test BaseProxy with a concrete implementation."""

    @pytest.fixture
    def concrete_proxy_class(self):
        """Create a concrete proxy class for testing."""

        class ConcreteProxy(BaseProxy):
            SUPPORTED_STRATEGIES = {StrategyType.DELAY, StrategyType.SKIP}

            def _get_layer(self):
                return "TestLayer"

            def _strategy_delay(self, *args, **kwargs):
                return self._original(*args, **kwargs)

            def _strategy_skip(self, *args, **kwargs):
                return None

        return ConcreteProxy

    def test_init_stores_original(self, concrete_proxy_class, mock_collector):
        """Test __init__ stores original function."""
        original = MagicMock(return_value="result")
        proxy = concrete_proxy_class(original, mock_collector)

        assert proxy.get_original() == original

    def test_call_without_config(self, concrete_proxy_class):
        """Test __call__ without config calls original."""
        original = MagicMock(return_value="result")
        proxy = concrete_proxy_class(original, None)

        result = proxy("arg1", kwarg1="value")

        assert result == "result"
        original.assert_called_once_with("arg1", kwarg1="value")

    def test_call_with_disabled_config(self, concrete_proxy_class, one_shot_trigger):
        """Test __call__ with disabled config calls original."""
        original = MagicMock(return_value="result")
        proxy = concrete_proxy_class(original, None)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=one_shot_trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(10)

        result = proxy()

        assert result == "result"
        original.assert_called_once()

    def test_set_config_validates_strategy(self, concrete_proxy_class):
        """Test set_config validates strategy is supported."""
        proxy = concrete_proxy_class(MagicMock(), None)

        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(
                FaultConfig(
                    id="test",
                    strategy=StrategyType.CORRUPT_TENSOR,  # Not supported
                    trigger=TriggerConfig(type=TriggerType.ONE_SHOT, at_step=1),
                )
            )

    def test_set_step(self, concrete_proxy_class):
        """Test set_step updates current step."""
        proxy = concrete_proxy_class(MagicMock(), None)

        proxy.set_step(100)

        assert proxy.get_step() == 100

    def test_get_layer(self, concrete_proxy_class):
        """Test _get_layer returns layer name."""
        proxy = concrete_proxy_class(MagicMock(), None)

        assert proxy._get_layer() == "TestLayer"


# =============================================================================
# L0 Ray Proxy Tests
# =============================================================================


class TestRayGetProxy:
    """Tests for RayGetProxy."""

    @pytest.fixture
    def ray_get_proxy_class(self):
        """Get the RayGetProxy class."""
        from ralph.proxies.l0_ray import RayGetProxy

        return RayGetProxy

    def test_registration(self, ray_get_proxy_class):
        """Test RayGetProxy is registered."""
        assert ProxyRegistry.is_registered("ray.get")
        assert ProxyRegistry.get_proxy("ray.get") == ray_get_proxy_class

    def test_supported_strategies(self, ray_get_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.RAISE_EXCEPTION,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.OBJECT_LOST,
            StrategyType.PARTIAL_FAILURE,
        }
        assert ray_get_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, ray_get_proxy_class, mock_collector):
        """Test _get_layer returns 'L0'."""
        proxy = ray_get_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "L0"


class TestRayPutProxy:
    """Tests for RayPutProxy."""

    @pytest.fixture
    def ray_put_proxy_class(self):
        """Get the RayPutProxy class."""
        from ralph.proxies.l0_ray import RayPutProxy

        return RayPutProxy

    def test_registration(self, ray_put_proxy_class):
        """Test RayPutProxy is registered."""
        assert ProxyRegistry.is_registered("ray.put")
        assert ProxyRegistry.get_proxy("ray.put") == ray_put_proxy_class

    def test_supported_strategies(self, ray_put_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.STORE_FULL,
            StrategyType.SILENT_DROP,
        }
        assert ray_put_proxy_class.SUPPORTED_STRATEGIES == expected


class TestExecuteAllProxy:
    """Tests for ExecuteAllProxy."""

    @pytest.fixture
    def execute_all_proxy_class(self):
        """Get the ExecuteAllProxy class."""
        from ralph.proxies.l0_ray import ExecuteAllProxy

        return ExecuteAllProxy

    def test_registration(self, execute_all_proxy_class):
        """Test ExecuteAllProxy is registered."""
        assert ProxyRegistry.is_registered("execute_all_sync")
        assert ProxyRegistry.get_proxy("execute_all_sync") == execute_all_proxy_class

    def test_supported_strategies(self, execute_all_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.WORKER_DEATH,
            StrategyType.STRAGGLER,
            StrategyType.SKIP_WORKER,
            StrategyType.DUPLICATE_CALL,
        }
        assert execute_all_proxy_class.SUPPORTED_STRATEGIES == expected


# =============================================================================
# L1 Distributed Proxy Tests
# =============================================================================


class TestAllReduceProxy:
    """Tests for AllReduceProxy."""

    @pytest.fixture
    def all_reduce_proxy_class(self):
        """Get the AllReduceProxy class."""
        from ralph.proxies.l1_distributed import AllReduceProxy

        return AllReduceProxy

    def test_registration(self, all_reduce_proxy_class):
        """Test AllReduceProxy is registered."""
        assert ProxyRegistry.is_registered("torch.distributed.all_reduce")
        assert (
            ProxyRegistry.get_proxy("torch.distributed.all_reduce")
            == all_reduce_proxy_class
        )

    def test_supported_strategies(self, all_reduce_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.INJECT_NAN,
            StrategyType.INJECT_INF,
            StrategyType.DEADLOCK,
        }
        assert all_reduce_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, all_reduce_proxy_class, mock_collector):
        """Test _get_layer returns 'L1'."""
        proxy = all_reduce_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "L1"


class TestAllGatherProxy:
    """Tests for AllGatherProxy."""

    @pytest.fixture
    def all_gather_proxy_class(self):
        """Get the AllGatherProxy class."""
        from ralph.proxies.l1_distributed import AllGatherProxy

        return AllGatherProxy

    def test_registration(self, all_gather_proxy_class):
        """Test AllGatherProxy is registered."""
        assert ProxyRegistry.is_registered("torch.distributed.all_gather")
        assert (
            ProxyRegistry.get_proxy("torch.distributed.all_gather")
            == all_gather_proxy_class
        )

    def test_supported_strategies(self, all_gather_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_GATHERED,
            StrategyType.MISSING_RANK,
            StrategyType.SHAPE_MISMATCH,
        }
        assert all_gather_proxy_class.SUPPORTED_STRATEGIES == expected


class TestBarrierProxy:
    """Tests for BarrierProxy."""

    @pytest.fixture
    def barrier_proxy_class(self):
        """Get the BarrierProxy class."""
        from ralph.proxies.l1_distributed import BarrierProxy

        return BarrierProxy

    def test_registration(self, barrier_proxy_class):
        """Test BarrierProxy is registered."""
        assert ProxyRegistry.is_registered("torch.distributed.barrier")
        assert (
            ProxyRegistry.get_proxy("torch.distributed.barrier") == barrier_proxy_class
        )

    def test_supported_strategies(self, barrier_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.BARRIER_TIMEOUT,
            StrategyType.BARRIER_SKIP,
            StrategyType.ASYNC_DESYNC,
        }
        assert barrier_proxy_class.SUPPORTED_STRATEGIES == expected


# =============================================================================
# L2 verl Proxy Tests
# =============================================================================


class TestRewardManagerProxy:
    """Tests for RewardManagerProxy."""

    @pytest.fixture
    def reward_manager_proxy_class(self):
        """Get the RewardManagerProxy class."""
        from ralph.proxies.l2_verl import RewardManagerProxy

        return RewardManagerProxy

    def test_registration(self, reward_manager_proxy_class):
        """Test RewardManagerProxy is registered."""
        assert ProxyRegistry.is_registered("RewardManager.__call__")
        assert (
            ProxyRegistry.get_proxy("RewardManager.__call__")
            == reward_manager_proxy_class
        )

    def test_supported_strategies(self, reward_manager_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.CORRUPT_TENSOR,
            StrategyType.REWARD_FLIP,
            StrategyType.CONSTANT_REWARD,
        }
        assert reward_manager_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, reward_manager_proxy_class, mock_collector):
        """Test _get_layer returns 'L2'."""
        proxy = reward_manager_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "L2"


class TestCheckpointSaveProxy:
    """Tests for CheckpointSaveProxy."""

    @pytest.fixture
    def checkpoint_save_proxy_class(self):
        """Get the CheckpointSaveProxy class."""
        from ralph.proxies.l2_verl import CheckpointSaveProxy

        return CheckpointSaveProxy

    def test_registration(self, checkpoint_save_proxy_class):
        """Test CheckpointSaveProxy is registered."""
        assert ProxyRegistry.is_registered("FSDPCheckpointManager.save_checkpoint")
        assert (
            ProxyRegistry.get_proxy("FSDPCheckpointManager.save_checkpoint")
            == checkpoint_save_proxy_class
        )

    def test_supported_strategies(self, checkpoint_save_proxy_class):
        """Test supported strategies."""
        expected = {StrategyType.DELAY, StrategyType.RAISE_EXCEPTION}
        assert checkpoint_save_proxy_class.SUPPORTED_STRATEGIES == expected


class TestCheckpointLoadProxy:
    """Tests for CheckpointLoadProxy."""

    @pytest.fixture
    def checkpoint_load_proxy_class(self):
        """Get the CheckpointLoadProxy class."""
        from ralph.proxies.l2_verl import CheckpointLoadProxy

        return CheckpointLoadProxy

    def test_registration(self, checkpoint_load_proxy_class):
        """Test CheckpointLoadProxy is registered."""
        assert ProxyRegistry.is_registered("FSDPCheckpointManager.load_checkpoint")
        assert (
            ProxyRegistry.get_proxy("FSDPCheckpointManager.load_checkpoint")
            == checkpoint_load_proxy_class
        )

    def test_supported_strategies(self, checkpoint_load_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.RAISE_EXCEPTION,
            StrategyType.FILE_NOT_FOUND,
            StrategyType.CORRUPT_STATE_DICT,
            StrategyType.PARTIAL_LOAD,
        }
        assert checkpoint_load_proxy_class.SUPPORTED_STRATEGIES == expected


class TestUpdateActorProxy:
    """Tests for UpdateActorProxy."""

    @pytest.fixture
    def update_actor_proxy_class(self):
        """Get the UpdateActorProxy class."""
        from ralph.proxies.l2_verl import UpdateActorProxy

        return UpdateActorProxy

    def test_registration(self, update_actor_proxy_class):
        """Test UpdateActorProxy is registered."""
        assert ProxyRegistry.is_registered("RayPPOTrainer._update_actor")
        assert (
            ProxyRegistry.get_proxy("RayPPOTrainer._update_actor")
            == update_actor_proxy_class
        )

    def test_supported_strategies(self, update_actor_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.NAN_INPUT,
            StrategyType.EXPLODING_GRADIENTS,
            StrategyType.VANISHING_GRADIENTS,
            StrategyType.SKIP_UPDATE,
            StrategyType.DOUBLE_UPDATE,
        }
        assert update_actor_proxy_class.SUPPORTED_STRATEGIES == expected


# =============================================================================
# L3 Resource Proxy Tests
# =============================================================================


class TestGPUMemoryInjector:
    """Tests for GPUMemoryInjector."""

    @pytest.fixture
    def gpu_memory_injector_class(self):
        """Get the GPUMemoryInjector class."""
        from ralph.proxies.l3_resource import GPUMemoryInjector

        return GPUMemoryInjector

    def test_registration(self, gpu_memory_injector_class):
        """Test GPUMemoryInjector is registered."""
        assert ProxyRegistry.is_registered("gpu_memory")
        assert ProxyRegistry.get_proxy("gpu_memory") == gpu_memory_injector_class

    def test_supported_strategies(self, gpu_memory_injector_class):
        """Test supported strategies."""
        expected = {
            StrategyType.MEMORY_PRESSURE,
            StrategyType.OOM_SIMULATION,
            StrategyType.MEMORY_FRAGMENTATION,
            StrategyType.MEMORY_LEAK,
        }
        assert gpu_memory_injector_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, gpu_memory_injector_class, mock_collector):
        """Test _get_layer returns 'L3'."""
        proxy = gpu_memory_injector_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "L3"


# =============================================================================
# Algorithm Proxy Tests
# =============================================================================


class TestGAEProxy:
    """Tests for GAEProxy."""

    @pytest.fixture
    def gae_proxy_class(self):
        """Get the GAEProxy class."""
        from ralph.proxies.algorithm import GAEProxy

        return GAEProxy

    def test_registration(self, gae_proxy_class):
        """Test GAEProxy is registered."""
        assert ProxyRegistry.is_registered("compute_gae_advantage_return")
        assert ProxyRegistry.get_proxy("compute_gae_advantage_return") == gae_proxy_class

    def test_supported_strategies(self, gae_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.WRONG_ADVANTAGE,
            StrategyType.ZERO_ADVANTAGE,
            StrategyType.INVERTED_ADVANTAGE,
            StrategyType.SCALED_ADVANTAGE,
            StrategyType.DELAYED_ADVANTAGE,
        }
        assert gae_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, gae_proxy_class, mock_collector):
        """Test _get_layer returns 'Algorithm'."""
        proxy = gae_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "Algorithm"


class TestKLPenaltyProxy:
    """Tests for KLPenaltyProxy."""

    @pytest.fixture
    def kl_penalty_proxy_class(self):
        """Get the KLPenaltyProxy class."""
        from ralph.proxies.algorithm import KLPenaltyProxy

        return KLPenaltyProxy

    def test_registration(self, kl_penalty_proxy_class):
        """Test KLPenaltyProxy is registered."""
        assert ProxyRegistry.is_registered("apply_kl_penalty")
        assert ProxyRegistry.get_proxy("apply_kl_penalty") == kl_penalty_proxy_class

    def test_supported_strategies(self, kl_penalty_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.WRONG_KL,
            StrategyType.ZERO_KL,
            StrategyType.EXTREME_KL,
            StrategyType.NEGATIVE_KL,
        }
        assert kl_penalty_proxy_class.SUPPORTED_STRATEGIES == expected


# =============================================================================
# Inference Proxy Tests
# =============================================================================


class TestGenerateProxy:
    """Tests for GenerateProxy."""

    @pytest.fixture
    def generate_proxy_class(self):
        """Get the GenerateProxy class."""
        from ralph.proxies.inference import GenerateProxy

        return GenerateProxy

    def test_registration(self, generate_proxy_class):
        """Test GenerateProxy is registered."""
        assert ProxyRegistry.is_registered("vllm.generate")
        assert ProxyRegistry.get_proxy("vllm.generate") == generate_proxy_class

    def test_supported_strategies(self, generate_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.GENERATION_TIMEOUT,
            StrategyType.EMPTY_RESPONSE,
            StrategyType.TRUNCATED_OUTPUT,
            StrategyType.GARBAGE_OUTPUT,
        }
        assert generate_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, generate_proxy_class, mock_collector):
        """Test _get_layer returns 'Inference'."""
        proxy = generate_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "Inference"


# =============================================================================
# Worker Proxy Tests
# =============================================================================


class TestComputeValuesProxy:
    """Tests for ComputeValuesProxy."""

    @pytest.fixture
    def compute_values_proxy_class(self):
        """Get the ComputeValuesProxy class."""
        from ralph.proxies.worker import ComputeValuesProxy

        return ComputeValuesProxy

    def test_registration(self, compute_values_proxy_class):
        """Test ComputeValuesProxy is registered."""
        assert ProxyRegistry.is_registered("CriticWorker.compute_values")
        assert (
            ProxyRegistry.get_proxy("CriticWorker.compute_values")
            == compute_values_proxy_class
        )

    def test_supported_strategies(self, compute_values_proxy_class):
        """Test supported strategies."""
        expected = {
            StrategyType.DELAY,
            StrategyType.WRONG_VALUES,
            StrategyType.CONSTANT_VALUES,
            StrategyType.NAN_VALUES,
            StrategyType.INVERTED_VALUES,
        }
        assert compute_values_proxy_class.SUPPORTED_STRATEGIES == expected

    def test_get_layer(self, compute_values_proxy_class, mock_collector):
        """Test _get_layer returns 'Worker'."""
        proxy = compute_values_proxy_class(MagicMock(), mock_collector)
        assert proxy._get_layer() == "Worker"


# =============================================================================
# Proxy Strategy Execution Tests
# =============================================================================


class TestProxyStrategyExecution:
    """Test proxy strategy execution patterns."""

    def test_delay_strategy_calls_original(self, mock_collector, one_shot_trigger):
        """Test DELAY strategy calls original function."""
        from ralph.proxies.l0_ray import RayGetProxy

        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=one_shot_trigger,
            parameters={"delay_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        with patch("time.sleep"):
            result = proxy()

        assert result == "result"
        original.assert_called_once()

    def test_corrupt_tensor_modifies_result(self, mock_collector, one_shot_trigger):
        """Test CORRUPT_TENSOR strategy modifies tensor result."""
        from ralph.proxies.l0_ray import RayGetProxy

        tensor = torch.zeros(10)
        original = MagicMock(return_value=tensor)
        proxy = RayGetProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_TENSOR,
            trigger=one_shot_trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        result = proxy()

        # Result should be different from original
        assert not torch.allclose(result, tensor)

    def test_raise_exception_strategy(self, mock_collector, one_shot_trigger):
        """Test RAISE_EXCEPTION strategy raises exception."""
        from ralph.proxies.l2_verl import CheckpointSaveProxy

        original = MagicMock()
        proxy = CheckpointSaveProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=one_shot_trigger,
            parameters={"exc_type": "IOError", "message": "Test error"},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        with pytest.raises(IOError, match="Test error"):
            proxy()


# =============================================================================
# Proxy Collector Integration Tests
# =============================================================================


class TestProxyCollectorIntegration:
    """Test proxy integration with collectors."""

    def test_records_fault_injection(self, mock_collector, one_shot_trigger):
        """Test proxy records fault injection to collector."""
        from ralph.proxies.l0_ray import RayGetProxy

        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=one_shot_trigger,
            parameters={"delay_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        with patch("time.sleep"):
            proxy()

        mock_collector.record_fault_injection.assert_called_once()

    def test_records_fault_outcome(self, mock_collector, one_shot_trigger):
        """Test proxy records fault outcome to collector."""
        from ralph.proxies.l0_ray import RayGetProxy

        original = MagicMock(return_value="result")
        proxy = RayGetProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=one_shot_trigger,
            parameters={"delay_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        with patch("time.sleep"):
            proxy()

        mock_collector.record_fault_outcome.assert_called_once()

    def test_records_failure_on_exception(self, mock_collector, one_shot_trigger):
        """Test proxy records failure outcome when exception occurs."""
        from ralph.proxies.l2_verl import CheckpointSaveProxy

        original = MagicMock()
        proxy = CheckpointSaveProxy(original, mock_collector)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.RAISE_EXCEPTION,
            trigger=one_shot_trigger,
            parameters={"exc_type": "RuntimeError"},
        )
        proxy.set_config(config)
        proxy.set_step(10)

        with pytest.raises(RuntimeError):
            proxy()

        # Verify fault_outcome was recorded with failure
        mock_collector.record_fault_outcome.assert_called_once()
        call_args = mock_collector.record_fault_outcome.call_args
        assert "failure" in call_args[0] or call_args[1].get("outcome") == "failure"
