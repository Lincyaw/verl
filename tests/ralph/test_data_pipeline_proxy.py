"""
Unit tests for ralph/proxies/data_pipeline.py

Tests the DataProtoConcatProxy class and its strategies for data pipeline
fault injection.
"""

import random
from unittest.mock import MagicMock

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.data_pipeline import DataProtoConcatProxy


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry before and after each test."""
    # Store original registrations
    original = ProxyRegistry.get_all_registrations()
    yield
    # Restore original registrations
    ProxyRegistry.clear()
    for target, proxy_class in original.items():
        try:
            ProxyRegistry.register(target)(proxy_class)
        except ValueError:
            pass


@pytest.fixture
def mock_original():
    """Create a mock original function."""

    def concat_fn(items, *args, **kwargs):
        """Mock concat function that returns concatenated items."""
        if isinstance(items, list) and len(items) > 0:
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            elif isinstance(items[0], dict):
                # Concat dict values if they're tensors
                result = {}
                for key in items[0]:
                    values = [item[key] for item in items]
                    if isinstance(values[0], torch.Tensor):
                        result[key] = torch.cat(values, dim=0)
                    else:
                        result[key] = values
                return result
        return items

    return concat_fn


def create_config(
    strategy: StrategyType,
    parameters: dict = None,
    enabled: bool = True,
    trigger_type: TriggerType = TriggerType.ONE_SHOT,
    trigger_step: int = 0,
) -> FaultConfig:
    """Helper to create a FaultConfig."""
    trigger = TriggerConfig(
        type=trigger_type,
        at_step=trigger_step,
    )
    return FaultConfig(
        id=f"test-{strategy.value}",
        strategy=strategy,
        trigger=trigger,
        parameters=parameters or {},
        enabled=enabled,
        severity="medium",
        expected_behavior="test",
    )


class TestDataProtoConcatProxyRegistration:
    """Tests for DataProtoConcatProxy registration."""

    def test_proxy_is_registered(self):
        """Test that the proxy is registered in the registry."""
        # Import to trigger registration
        from ralph.proxies.data_pipeline import DataProtoConcatProxy

        assert ProxyRegistry.is_registered("DataProto.concat")

    def test_get_proxy_returns_correct_class(self):
        """Test that get_proxy returns DataProtoConcatProxy."""
        from ralph.proxies.data_pipeline import DataProtoConcatProxy

        proxy_class = ProxyRegistry.get_proxy("DataProto.concat")
        assert proxy_class is DataProtoConcatProxy

    def test_supported_strategies(self):
        """Test that the proxy supports expected strategies."""
        expected = {
            StrategyType.DATA_MISMATCH,
            StrategyType.LOST_ITEMS,
            StrategyType.DUPLICATE_ITEMS,
            StrategyType.WRONG_ORDER,
        }
        assert DataProtoConcatProxy.SUPPORTED_STRATEGIES == expected


class TestDataProtoConcatProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer(self, mock_original):
        """Test that _get_layer returns 'DataPipeline'."""
        proxy = DataProtoConcatProxy(mock_original)
        assert proxy._get_layer() == "DataPipeline"

    def test_call_without_config(self, mock_original):
        """Test that calling without config passes through to original."""
        proxy = DataProtoConcatProxy(mock_original)
        items = [torch.randn(2, 4), torch.randn(3, 4)]
        result = proxy(items)
        assert result.shape == (5, 4)

    def test_call_with_disabled_config(self, mock_original):
        """Test that disabled config passes through to original."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.LOST_ITEMS, enabled=False)
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4), torch.randn(3, 4)]
        result = proxy(items)
        # Should pass through without modification
        assert result.shape == (5, 4)

    def test_unsupported_strategy_raises_error(self, mock_original):
        """Test that unsupported strategy raises ValueError."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.DELAY)  # Not supported
        with pytest.raises(ValueError, match="not supported"):
            proxy.set_config(config)


class TestDataMismatchStrategy:
    """Tests for DATA_MISMATCH strategy."""

    def test_data_mismatch_increases_batch_size(self, mock_original):
        """Test that positive mismatch_ratio increases batch size."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.DATA_MISMATCH, parameters={"mismatch_ratio": 0.5}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(4, 4), torch.randn(4, 4)]
        result = proxy(items)
        # Original would be (8, 4), with 0.5 ratio should add ~4 items
        assert result.shape[0] > 8
        assert result.shape[1] == 4

    def test_data_mismatch_decreases_batch_size(self, mock_original):
        """Test that negative mismatch_ratio decreases batch size."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.DATA_MISMATCH, parameters={"mismatch_ratio": -0.25}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(4, 4), torch.randn(4, 4)]
        result = proxy(items)
        # Original would be (8, 4), with -0.25 ratio should have ~6 items
        assert result.shape[0] < 8
        assert result.shape[1] == 4

    def test_data_mismatch_default_ratio(self, mock_original):
        """Test that default mismatch_ratio is 0.1."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.DATA_MISMATCH, parameters={})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(10, 4), torch.randn(10, 4)]
        result = proxy(items)
        # Default ratio 0.1 should add ~2 items to 20
        assert result.shape[0] >= 20

    def test_data_mismatch_with_dict_result(self, mock_original):
        """Test data mismatch with dict containing tensors."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.DATA_MISMATCH, parameters={"mismatch_ratio": 0.5}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [
            {"data": torch.randn(4, 4), "labels": torch.randn(4)},
            {"data": torch.randn(4, 4), "labels": torch.randn(4)},
        ]
        result = proxy(items)
        # Both tensors should be modified
        assert result["data"].shape[0] > 8
        assert result["labels"].shape[0] > 8


class TestLostItemsStrategy:
    """Tests for LOST_ITEMS strategy."""

    def test_lost_items_drops_items(self, mock_original):
        """Test that LOST_ITEMS drops some input items."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.LOST_ITEMS, parameters={"drop_ratio": 0.5}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4) for _ in range(10)]
        result = proxy(items)
        # With 0.5 drop ratio, should have ~5 items (10 total) = ~10 rows
        # Original would be 20 rows
        assert result.shape[0] < 20
        assert result.shape[1] == 4

    def test_lost_items_default_ratio(self, mock_original):
        """Test that default drop_ratio is 0.2."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.LOST_ITEMS, parameters={})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(1, 4) for _ in range(10)]
        result = proxy(items)
        # With 0.2 drop ratio, should drop ~2 items
        assert result.shape[0] == 8  # 10 - 2 dropped

    def test_lost_items_keeps_at_least_one(self, mock_original):
        """Test that at least one item is kept."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.LOST_ITEMS, parameters={"drop_ratio": 1.0}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4), torch.randn(2, 4)]
        result = proxy(items)
        # Should keep at least one item
        assert result.shape[0] >= 2

    def test_lost_items_ratio_clamped(self, mock_original):
        """Test that drop_ratio is clamped to [0, 1]."""
        proxy = DataProtoConcatProxy(mock_original)

        # Negative ratio should be clamped to 0
        config = create_config(
            StrategyType.LOST_ITEMS, parameters={"drop_ratio": -0.5}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4) for _ in range(5)]
        result = proxy(items)
        # Clamped to 0, no items dropped
        assert result.shape[0] == 10


class TestDuplicateItemsStrategy:
    """Tests for DUPLICATE_ITEMS strategy."""

    def test_duplicate_items_adds_items(self, mock_original):
        """Test that DUPLICATE_ITEMS adds duplicate items."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.DUPLICATE_ITEMS, parameters={"duplicate_ratio": 0.5}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4) for _ in range(10)]
        result = proxy(items)
        # With 0.5 ratio, should duplicate ~5 items -> 15 items total -> 30 rows
        assert result.shape[0] > 20
        assert result.shape[1] == 4

    def test_duplicate_items_default_ratio(self, mock_original):
        """Test that default duplicate_ratio is 0.2."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.DUPLICATE_ITEMS, parameters={})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(1, 4) for _ in range(10)]
        result = proxy(items)
        # With 0.2 ratio, should duplicate ~2 items -> 12 items
        assert result.shape[0] == 12

    def test_duplicate_items_preserves_content(self, mock_original):
        """Test that duplicated items have same content."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(
            StrategyType.DUPLICATE_ITEMS, parameters={"duplicate_ratio": 1.0}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        # Use easily identifiable items
        items = [torch.full((1, 4), float(i)) for i in range(3)]
        result = proxy(items)
        # Should have 6 items (3 original + 3 duplicates)
        assert result.shape[0] == 6

    def test_duplicate_items_ratio_clamped(self, mock_original):
        """Test that duplicate_ratio is clamped to [0, 1]."""
        proxy = DataProtoConcatProxy(mock_original)

        # Ratio > 1 should be clamped to 1
        config = create_config(
            StrategyType.DUPLICATE_ITEMS, parameters={"duplicate_ratio": 2.0}
        )
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(1, 4) for _ in range(5)]
        result = proxy(items)
        # Clamped to 1.0, all 5 items duplicated -> 10 items
        assert result.shape[0] == 10


class TestWrongOrderStrategy:
    """Tests for WRONG_ORDER strategy."""

    def test_wrong_order_shuffles_items(self, mock_original):
        """Test that WRONG_ORDER shuffles input order."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 42})
        proxy.set_config(config)
        proxy.set_step(0)

        # Create items with identifiable values
        items = [torch.full((1, 4), float(i)) for i in range(10)]
        result = proxy(items)

        # The order should be different from original
        expected_original = torch.cat(items, dim=0)
        # With shuffling, the result should differ (almost certainly with seed=42)
        assert result.shape == expected_original.shape
        # Check that at least some values are in different positions
        assert not torch.allclose(result, expected_original)

    def test_wrong_order_reproducible_with_seed(self, mock_original):
        """Test that shuffle is reproducible with same seed."""
        items = [torch.full((1, 4), float(i)) for i in range(10)]

        # First call
        proxy1 = DataProtoConcatProxy(mock_original)
        config1 = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 123})
        proxy1.set_config(config1)
        proxy1.set_step(0)
        result1 = proxy1(items)

        # Second call with same seed
        proxy2 = DataProtoConcatProxy(mock_original)
        config2 = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 123})
        proxy2.set_config(config2)
        proxy2.set_step(0)
        result2 = proxy2(items)

        assert torch.allclose(result1, result2)

    def test_wrong_order_different_with_different_seed(self, mock_original):
        """Test that shuffle differs with different seeds."""
        items = [torch.full((1, 4), float(i)) for i in range(10)]

        proxy1 = DataProtoConcatProxy(mock_original)
        config1 = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 42})
        proxy1.set_config(config1)
        proxy1.set_step(0)
        result1 = proxy1(items)

        proxy2 = DataProtoConcatProxy(mock_original)
        config2 = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 99})
        proxy2.set_config(config2)
        proxy2.set_step(0)
        result2 = proxy2(items)

        # Results should differ with different seeds
        assert not torch.allclose(result1, result2)

    def test_wrong_order_single_item_no_change(self, mock_original):
        """Test that single item is not modified."""
        proxy = DataProtoConcatProxy(mock_original)
        config = create_config(StrategyType.WRONG_ORDER, parameters={})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4)]
        result = proxy(items)
        assert result.shape == (2, 4)


class TestExtractItems:
    """Tests for _extract_items helper method."""

    def test_extract_from_list_arg(self, mock_original):
        """Test extracting items from list first argument."""
        proxy = DataProtoConcatProxy(mock_original)
        items = [1, 2, 3]
        result = proxy._extract_items((items,), {})
        assert result == [1, 2, 3]

    def test_extract_from_tuple_arg(self, mock_original):
        """Test extracting items from tuple first argument."""
        proxy = DataProtoConcatProxy(mock_original)
        items = (1, 2, 3)
        result = proxy._extract_items((items,), {})
        assert result == [1, 2, 3]

    def test_extract_from_data_protos_kwarg(self, mock_original):
        """Test extracting items from data_protos keyword argument."""
        proxy = DataProtoConcatProxy(mock_original)
        items = [1, 2, 3]
        result = proxy._extract_items((), {"data_protos": items})
        assert result == [1, 2, 3]

    def test_extract_from_items_kwarg(self, mock_original):
        """Test extracting items from items keyword argument."""
        proxy = DataProtoConcatProxy(mock_original)
        items = [1, 2, 3]
        result = proxy._extract_items((), {"items": items})
        assert result == [1, 2, 3]

    def test_extract_multiple_positional_args(self, mock_original):
        """Test extracting items from multiple positional arguments."""
        proxy = DataProtoConcatProxy(mock_original)
        result = proxy._extract_items((1, 2, 3), {})
        assert result == [1, 2, 3]

    def test_extract_empty_returns_none(self, mock_original):
        """Test that empty arguments returns None."""
        proxy = DataProtoConcatProxy(mock_original)
        result = proxy._extract_items((), {})
        assert result is None


class TestCallWithModifiedItems:
    """Tests for _call_with_modified_items helper method."""

    def test_call_with_list_first_arg(self, mock_original):
        """Test calling with modified list when original used list arg."""
        call_args = []

        def tracking_fn(items, *args, **kwargs):
            call_args.append((items, args, kwargs))
            return items

        proxy = DataProtoConcatProxy(tracking_fn)
        modified = ["a", "b", "c"]
        proxy._call_with_modified_items(([1, 2],), {}, modified)

        assert call_args[0][0] == modified

    def test_call_with_kwarg(self, mock_original):
        """Test calling with modified items when original used kwarg."""
        call_args = []

        def tracking_fn(*args, **kwargs):
            call_args.append((args, kwargs))
            return kwargs.get("data_protos", [])

        proxy = DataProtoConcatProxy(tracking_fn)
        modified = ["a", "b", "c"]
        proxy._call_with_modified_items((), {"data_protos": [1, 2]}, modified)

        assert call_args[0][1]["data_protos"] == modified


class TestDataProtoConcatProxyIntegration:
    """Integration tests for DataProtoConcatProxy."""

    def test_step_based_trigger(self, mock_original):
        """Test that step-based trigger works correctly."""
        proxy = DataProtoConcatProxy(mock_original)
        trigger = TriggerConfig(
            type=TriggerType.STEP_BASED,
            start_step=5,
            end_step=10,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.LOST_ITEMS,
            trigger=trigger,
            parameters={"drop_ratio": 0.5},
            enabled=True,
            severity="medium",
            expected_behavior="test",
        )
        proxy.set_config(config)

        items = [torch.randn(1, 4) for _ in range(10)]

        # Step 3: Should not trigger
        proxy.set_step(3)
        result = proxy(items)
        assert result.shape[0] == 10  # No items dropped

        # Step 7: Should trigger
        proxy.set_step(7)
        result = proxy(items)
        assert result.shape[0] < 10  # Items dropped

    def test_periodic_trigger(self, mock_original):
        """Test that periodic trigger works correctly."""
        proxy = DataProtoConcatProxy(mock_original)
        trigger = TriggerConfig(
            type=TriggerType.PERIODIC,
            every_n_steps=3,
        )
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DUPLICATE_ITEMS,
            trigger=trigger,
            parameters={"duplicate_ratio": 0.5},
            enabled=True,
            severity="medium",
            expected_behavior="test",
        )
        proxy.set_config(config)

        items = [torch.randn(1, 4) for _ in range(10)]

        # Step 2: Should not trigger
        proxy.set_step(2)
        result = proxy(items)
        assert result.shape[0] == 10

        # Step 3: Should trigger
        proxy.set_step(3)
        result = proxy(items)
        assert result.shape[0] > 10

        # Step 6: Should trigger again
        proxy.set_step(6)
        result = proxy(items)
        assert result.shape[0] > 10

    def test_collector_recording(self, mock_original):
        """Test that fault injections are recorded to collector."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"

        proxy = DataProtoConcatProxy(mock_original, collector)
        config = create_config(StrategyType.LOST_ITEMS, parameters={"drop_ratio": 0.3})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(1, 4) for _ in range(10)]
        proxy(items)

        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()

    def test_args_kwargs_passing(self, mock_original):
        """Test that additional args and kwargs are passed through."""
        received_kwargs = {}

        def tracking_fn(items, *args, **kwargs):
            received_kwargs.update(kwargs)
            return torch.cat(items, dim=0)

        proxy = DataProtoConcatProxy(tracking_fn)
        config = create_config(StrategyType.WRONG_ORDER, parameters={"seed": 42})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(2, 4), torch.randn(2, 4)]
        proxy(items, extra_param="value")

        # kwargs should be passed through to the original function
        assert "extra_param" in received_kwargs
        assert received_kwargs["extra_param"] == "value"

    def test_failure_recording(self, mock_original):
        """Test that failures are recorded to collector."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault-123"

        def failing_fn(*args, **kwargs):
            raise RuntimeError("Test failure")

        proxy = DataProtoConcatProxy(failing_fn, collector)
        config = create_config(StrategyType.LOST_ITEMS, parameters={})
        proxy.set_config(config)
        proxy.set_step(0)

        items = [torch.randn(1, 4)]

        with pytest.raises(RuntimeError):
            proxy(items)

        # Failure should be recorded
        collector.record_fault_outcome.assert_called_once()
        # Check the call args - can be positional or keyword
        call_args = collector.record_fault_outcome.call_args
        if call_args.kwargs:
            outcome = call_args.kwargs.get("outcome", "")
        else:
            # Positional args: (fault_id, outcome, duration_ms)
            outcome = str(call_args.args[1]) if len(call_args.args) >= 2 else ""
        # Outcome should indicate exception/failure
        assert "exception" in outcome.lower() or "failed" in outcome.lower()
