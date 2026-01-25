"""
Unit tests for L3 Resource Layer proxies.

Tests cover GPUMemoryInjector with all 4 supported strategies:
- MEMORY_PRESSURE: Fill GPU to specified utilization ratio
- OOM_SIMULATION: Fill GPU completely to cause OOM
- MEMORY_FRAGMENTATION: Create memory fragmentation
- MEMORY_LEAK: Gradual memory consumption over time
"""

from unittest.mock import MagicMock, patch

import pytest

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.l3_resource import GPUMemoryInjector

# Check if torch is available
try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


class TestGPUMemoryInjectorRegistration:
    """Tests for injector registration."""

    def test_registered_with_gpu_memory_target(self):
        """GPUMemoryInjector is registered for 'gpu_memory' target."""
        assert ProxyRegistry.is_registered("gpu_memory")
        assert ProxyRegistry.get_proxy("gpu_memory") is GPUMemoryInjector

    def test_supported_strategies(self):
        """GPUMemoryInjector declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("gpu_memory")
        expected = {
            StrategyType.MEMORY_PRESSURE,
            StrategyType.OOM_SIMULATION,
            StrategyType.MEMORY_FRAGMENTATION,
            StrategyType.MEMORY_LEAK,
        }
        assert strategies == expected


class TestGPUMemoryInjectorBasics:
    """Tests for basic injector functionality."""

    def test_get_layer_returns_l3(self):
        """_get_layer returns 'L3'."""
        injector = GPUMemoryInjector()
        assert injector._get_layer() == "L3"

    def test_init_with_no_args(self):
        """Injector can be initialized with no arguments."""
        injector = GPUMemoryInjector()
        assert injector._allocated_tensors == []
        assert injector._leak_tensors == []
        assert injector._fragmentation_tensors == []
        assert injector._leak_active is False
        assert injector._leak_rate_mb == 0.0

    def test_init_with_collector(self):
        """Injector can be initialized with a collector."""
        collector = MagicMock()
        injector = GPUMemoryInjector(collector=collector)
        assert injector._collector is collector

    def test_init_with_device(self):
        """Injector can be initialized with a specific device."""
        injector = GPUMemoryInjector(device=0)
        assert injector._device == 0

    def test_call_without_config_calls_original(self):
        """Injector calls original when no config is set."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        result = injector()
        original.assert_called_once()
        assert result == "result"

    def test_call_with_disabled_config_calls_original(self):
        """Injector calls original when config is disabled."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            enabled=False,
        )
        injector.set_config(config)
        injector.set_step(0)
        result = injector()
        original.assert_called_once()
        assert result == "result"

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        injector = GPUMemoryInjector()
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,  # Not supported by GPUMemoryInjector
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            injector.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestAllocatePressure:
    """Tests for allocate_pressure method."""

    def test_invalid_ratio_raises_error(self):
        """allocate_pressure raises ValueError for invalid ratio."""
        injector = GPUMemoryInjector()
        with pytest.raises(ValueError) as exc_info:
            injector.allocate_pressure(1.5)
        assert "between 0 and 1" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            injector.allocate_pressure(-0.1)
        assert "between 0 and 1" in str(exc_info.value)

    def test_returns_false_without_cuda(self):
        """allocate_pressure returns False when CUDA not available."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            result = injector.allocate_pressure(0.5)
            assert result is False

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_allocates_memory(self):
        """allocate_pressure allocates GPU memory."""
        injector = GPUMemoryInjector()
        try:
            injector._get_allocated_memory()
            result = injector.allocate_pressure(0.1)
            injector._get_allocated_memory()

            # Should have allocated some memory
            assert result is True
            # Note: might already be above 10% so we just check it doesn't fail
        finally:
            injector.release()


class TestTriggerOOM:
    """Tests for trigger_oom method."""

    def test_raises_runtime_error_without_cuda(self):
        """trigger_oom raises RuntimeError when CUDA not available."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                injector.trigger_oom()
            assert "CUDA not available" in str(exc_info.value)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_fills_memory(self):
        """trigger_oom fills GPU memory."""
        injector = GPUMemoryInjector()
        try:
            # Use a large headroom to avoid actual OOM during test
            injector.trigger_oom(leave_headroom_mb=1000)
            stats = injector.get_memory_stats()
            # Should have allocated significant memory
            assert stats["pressure_allocated"] > 0
        finally:
            injector.release()


class TestCauseFragmentation:
    """Tests for cause_fragmentation method."""

    def test_returns_zero_without_cuda(self):
        """cause_fragmentation returns 0 when CUDA not available."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            result = injector.cause_fragmentation(100)
            assert result == 0

    def test_negative_count_handled(self):
        """cause_fragmentation handles negative count gracefully."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            result = injector.cause_fragmentation(-5)
            assert result == 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_creates_fragmentation(self):
        """cause_fragmentation performs allocation cycles."""
        injector = GPUMemoryInjector()
        try:
            result = injector.cause_fragmentation(10)
            # Should have performed some operations
            assert result > 0
            # Should have some fragmentation tensors held
            assert len(injector._fragmentation_tensors) > 0
        finally:
            injector.release()


class TestMemoryLeak:
    """Tests for memory leak functionality."""

    def test_start_leak_sets_rate(self):
        """start_leak sets the leak rate."""
        injector = GPUMemoryInjector()
        injector.start_leak(100)
        assert injector._leak_rate_mb == 100
        assert injector._leak_active is True

    def test_start_leak_negative_rate_handled(self):
        """start_leak handles negative rate."""
        injector = GPUMemoryInjector()
        injector.start_leak(-50)
        assert injector._leak_rate_mb == 0
        assert injector._leak_active is True

    def test_stop_leak_stops_without_releasing(self):
        """stop_leak stops leak without releasing memory."""
        injector = GPUMemoryInjector()
        injector.start_leak(100)
        injector.stop_leak()
        assert injector._leak_active is False
        # Leak rate preserved (for potential restart)
        assert injector._leak_rate_mb == 100

    def test_step_leak_returns_zero_when_inactive(self):
        """step_leak returns 0 when leak not active."""
        injector = GPUMemoryInjector()
        result = injector.step_leak()
        assert result == 0

    def test_step_leak_returns_zero_when_rate_zero(self):
        """step_leak returns 0 when rate is zero."""
        injector = GPUMemoryInjector()
        injector.start_leak(0)
        result = injector.step_leak()
        assert result == 0

    def test_step_leak_returns_zero_without_cuda(self):
        """step_leak returns 0 when CUDA not available."""
        injector = GPUMemoryInjector()
        injector.start_leak(100)
        with patch.object(injector, "_is_cuda_available", return_value=False):
            result = injector.step_leak()
            assert result == 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_step_leak_allocates_memory(self):
        """step_leak allocates memory when active."""
        injector = GPUMemoryInjector()
        try:
            injector.start_leak(10)  # 10 MB per step
            result = injector.step_leak()
            assert result > 0
            assert len(injector._leak_tensors) > 0
        finally:
            injector.release()

    def test_get_leak_total_mb(self):
        """get_leak_total_mb returns correct value."""
        injector = GPUMemoryInjector()
        # No leak yet
        assert injector.get_leak_total_mb() == 0


class TestRelease:
    """Tests for release method."""

    def test_release_clears_all_tensors(self):
        """release clears all tensor lists."""
        injector = GPUMemoryInjector()
        # Simulate some allocations by adding to internal lists
        injector._allocated_tensors.append(MagicMock())
        injector._leak_tensors.append(MagicMock())
        injector._fragmentation_tensors.append(MagicMock())
        injector._leak_active = True
        injector._leak_rate_mb = 100

        injector.release()

        assert len(injector._allocated_tensors) == 0
        assert len(injector._leak_tensors) == 0
        assert len(injector._fragmentation_tensors) == 0
        assert injector._leak_active is False
        assert injector._leak_rate_mb == 0.0


class TestGetMemoryStats:
    """Tests for get_memory_stats method."""

    def test_returns_zeros_without_cuda(self):
        """get_memory_stats returns zeros when CUDA not available."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            stats = injector.get_memory_stats()
            assert stats["total"] == 0
            assert stats["allocated"] == 0
            assert stats["free"] == 0
            assert stats["pressure_allocated"] == 0
            assert stats["leak_allocated"] == 0
            assert stats["fragmentation_allocated"] == 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_returns_valid_stats_with_cuda(self):
        """get_memory_stats returns valid stats with CUDA."""
        injector = GPUMemoryInjector()
        stats = injector.get_memory_stats()
        assert stats["total"] > 0
        assert stats["free"] >= 0
        assert stats["allocated"] >= 0


class TestMemoryPressureStrategy:
    """Tests for MEMORY_PRESSURE strategy."""

    def test_strategy_reads_config_ratio(self):
        """MEMORY_PRESSURE strategy reads pressure_ratio from config."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            parameters={"pressure_ratio": 0.5},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "allocate_pressure", return_value=True) as mock_pressure:
            result = injector()
            mock_pressure.assert_called_once_with(0.5)
            assert result == "result"

    def test_strategy_uses_default_ratio(self):
        """MEMORY_PRESSURE strategy uses default ratio of 0.9."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            parameters={},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "allocate_pressure", return_value=True) as mock_pressure:
            injector()
            mock_pressure.assert_called_once_with(0.9)


class TestOOMSimulationStrategy:
    """Tests for OOM_SIMULATION strategy."""

    def test_strategy_reads_config_headroom(self):
        """OOM_SIMULATION strategy reads leave_headroom_mb from config."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OOM_SIMULATION,
            trigger=trigger,
            parameters={"leave_headroom_mb": 500},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "trigger_oom") as mock_oom:
            injector()
            mock_oom.assert_called_once_with(500)

    def test_strategy_uses_default_headroom(self):
        """OOM_SIMULATION strategy uses default headroom of 0."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OOM_SIMULATION,
            trigger=trigger,
            parameters={},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "trigger_oom") as mock_oom:
            injector()
            mock_oom.assert_called_once_with(0)


class TestFragmentationStrategy:
    """Tests for MEMORY_FRAGMENTATION strategy."""

    def test_strategy_reads_config_count(self):
        """MEMORY_FRAGMENTATION strategy reads fragment_count from config."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_FRAGMENTATION,
            trigger=trigger,
            parameters={"fragment_count": 500},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "cause_fragmentation", return_value=500) as mock_frag:
            result = injector()
            mock_frag.assert_called_once_with(500)
            assert result == "result"

    def test_strategy_uses_default_count(self):
        """MEMORY_FRAGMENTATION strategy uses default count of 1000."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_FRAGMENTATION,
            trigger=trigger,
            parameters={},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "cause_fragmentation", return_value=1000) as mock_frag:
            injector()
            mock_frag.assert_called_once_with(1000)


class TestMemoryLeakStrategy:
    """Tests for MEMORY_LEAK strategy."""

    def test_strategy_reads_config_rate(self):
        """MEMORY_LEAK strategy reads leak_rate_mb from config."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_LEAK,
            trigger=trigger,
            parameters={"leak_rate_mb": 50},
        )
        injector.set_config(config)
        injector.set_step(0)

        with (
            patch.object(injector, "start_leak") as mock_start,
            patch.object(injector, "step_leak", return_value=50 * 1024 * 1024) as mock_step,
        ):
            result = injector()
            mock_start.assert_called_once_with(50)
            mock_step.assert_called_once()
            assert result == "result"

    def test_strategy_uses_default_rate(self):
        """MEMORY_LEAK strategy uses default rate of 100."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_LEAK,
            trigger=trigger,
            parameters={},
        )
        injector.set_config(config)
        injector.set_step(0)

        with (
            patch.object(injector, "start_leak") as mock_start,
            patch.object(injector, "step_leak", return_value=100 * 1024 * 1024),
        ):
            injector()
            mock_start.assert_called_once_with(100)


class TestGPUMemoryInjectorIntegration:
    """Integration tests for GPUMemoryInjector."""

    def test_step_based_trigger(self):
        """Injector respects step-based triggers."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=10)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            parameters={"pressure_ratio": 0.1},
        )
        injector.set_config(config)

        with patch.object(injector, "allocate_pressure", return_value=True) as mock_pressure:
            # Step 3: outside range, should not inject
            injector.set_step(3)
            injector()
            mock_pressure.assert_not_called()

            # Step 7: inside range, should inject
            injector.set_step(7)
            injector()
            mock_pressure.assert_called_once()

    def test_periodic_trigger(self):
        """Injector respects periodic triggers."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=3)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_FRAGMENTATION,
            trigger=trigger,
            parameters={"fragment_count": 10},
        )
        injector.set_config(config)

        with patch.object(injector, "cause_fragmentation", return_value=10) as mock_frag:
            # Step 1: not a multiple of 3
            injector.set_step(1)
            injector()
            assert mock_frag.call_count == 0

            # Step 3: multiple of 3, should inject
            injector.set_step(3)
            injector()
            assert mock_frag.call_count == 1

            # Step 4: not a multiple of 3
            injector.set_step(4)
            injector()
            assert mock_frag.call_count == 1

            # Step 6: multiple of 3, should inject
            injector.set_step(6)
            injector()
            assert mock_frag.call_count == 2

    def test_collector_recording(self):
        """Injector records injections to collector."""
        original = MagicMock(return_value="result")
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_123"

        injector = GPUMemoryInjector(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            parameters={"pressure_ratio": 0.1},
            severity="medium",
            expected_behavior="Memory fills to 10%",
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "allocate_pressure", return_value=True):
            injector()

        # Should have recorded start
        collector.record_fault_injection.assert_called_once()
        call_kwargs = collector.record_fault_injection.call_args[1]
        assert call_kwargs["fault_type"] == "memory_pressure"
        assert call_kwargs["target_layer"] == "L3"
        assert call_kwargs["severity"] == "medium"

        # Should have recorded end
        collector.record_fault_outcome.assert_called_once()
        assert collector.record_fault_outcome.call_args[0][0] == "fault_123"

    def test_failure_recording(self):
        """Injector records failures to collector."""
        original = MagicMock(return_value="result")
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_456"

        injector = GPUMemoryInjector(original, collector=collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OOM_SIMULATION,
            trigger=trigger,
        )
        injector.set_config(config)
        injector.set_step(0)

        # Make trigger_oom raise an exception
        with patch.object(injector, "trigger_oom", side_effect=RuntimeError("OOM simulated")):
            with pytest.raises(RuntimeError):
                injector()

        # Should have recorded failure outcome
        collector.record_fault_outcome.assert_called_once()
        outcome = collector.record_fault_outcome.call_args[0][1]
        assert "exception" in outcome
        assert "RuntimeError" in outcome

    def test_kwargs_passing(self):
        """Injector passes kwargs to original function."""
        original = MagicMock(return_value="result")
        injector = GPUMemoryInjector(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.MEMORY_PRESSURE,
            trigger=trigger,
            parameters={"pressure_ratio": 0.1},
        )
        injector.set_config(config)
        injector.set_step(0)

        with patch.object(injector, "allocate_pressure", return_value=True):
            injector(arg1="value1", arg2=42)

        original.assert_called_once_with(arg1="value1", arg2=42)

    def test_destructor_releases_memory(self):
        """Injector releases memory on deletion."""
        injector = GPUMemoryInjector()
        injector._allocated_tensors.append(MagicMock())

        with patch.object(injector, "release") as mock_release:
            injector.__del__()
            mock_release.assert_called_once()


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_device_returns_specified_device(self):
        """_get_device returns the specified device."""
        injector = GPUMemoryInjector(device=2)
        assert injector._get_device() == 2

    def test_is_cuda_available_returns_correct_value(self):
        """_is_cuda_available returns correct value."""
        injector = GPUMemoryInjector()
        result = injector._is_cuda_available()
        assert result == HAS_CUDA

    def test_get_free_memory_returns_zero_without_cuda(self):
        """_get_free_memory returns 0 without CUDA."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            assert injector._get_free_memory() == 0

    def test_get_total_memory_returns_zero_without_cuda(self):
        """_get_total_memory returns 0 without CUDA."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            assert injector._get_total_memory() == 0

    def test_get_allocated_memory_returns_zero_without_cuda(self):
        """_get_allocated_memory returns 0 without CUDA."""
        injector = GPUMemoryInjector()
        with patch.object(injector, "_is_cuda_available", return_value=False):
            assert injector._get_allocated_memory() == 0
