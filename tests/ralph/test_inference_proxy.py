"""
Unit tests for Inference proxies.

Tests cover GenerateProxy with all 5 supported strategies:
- DELAY: Standard delay before generation
- GENERATION_TIMEOUT: Very long delay simulating hung generation
- EMPTY_RESPONSE: Returns empty sequences
- TRUNCATED_OUTPUT: Limits output to shorter than expected length
- GARBAGE_OUTPUT: Returns random tokens instead of generated text
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from ralph.core.config import FaultConfig, StrategyType, TriggerConfig, TriggerType
from ralph.core.registry import ProxyRegistry
from ralph.proxies.inference import GenerateProxy


class TestGenerateProxyRegistration:
    """Tests for proxy registration."""

    def test_registered_with_vllm_generate_target(self):
        """GenerateProxy is registered for 'vllm.generate' target."""
        assert ProxyRegistry.is_registered("vllm.generate")
        assert ProxyRegistry.get_proxy("vllm.generate") is GenerateProxy

    def test_supported_strategies(self):
        """GenerateProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("vllm.generate")
        expected = {
            StrategyType.DELAY,
            StrategyType.GENERATION_TIMEOUT,
            StrategyType.EMPTY_RESPONSE,
            StrategyType.TRUNCATED_OUTPUT,
            StrategyType.GARBAGE_OUTPUT,
        }
        assert strategies == expected


class TestGenerateProxyBasics:
    """Tests for basic proxy functionality."""

    def test_get_layer_returns_inference(self):
        """_get_layer returns 'Inference'."""
        proxy = GenerateProxy(lambda: None)
        assert proxy._get_layer() == "Inference"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10, dtype=torch.long)})
        proxy = GenerateProxy(original)
        result = proxy()
        original.assert_called_once()
        assert "sequences" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        original_result = {"sequences": torch.ones(2, 10, dtype=torch.long)}
        original = MagicMock(return_value=original_result)
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once()
        torch.testing.assert_close(result["sequences"], original_result["sequences"])

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        proxy = GenerateProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by GenerateProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy uses delay_seconds from config."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},  # Very short delay for testing
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy()
            mock_sleep.assert_called_once_with(0.01)
            original.assert_called_once()

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay_seconds of 10.0."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},  # No delay_seconds specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(10.0)


class TestGenerationTimeoutStrategy:
    """Tests for GENERATION_TIMEOUT strategy."""

    def test_generation_timeout_with_config(self):
        """GENERATION_TIMEOUT uses timeout_seconds from config."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GENERATION_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_seconds": 60.0},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy()
            mock_sleep.assert_called_once_with(60.0)
            original.assert_called_once()

    def test_generation_timeout_default_timeout(self):
        """GENERATION_TIMEOUT uses default timeout_seconds of 300.0."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GENERATION_TIMEOUT,
            trigger=trigger,
            parameters={},  # No timeout_seconds specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(300.0)


class TestEmptyResponseStrategy:
    """Tests for EMPTY_RESPONSE strategy."""

    def test_empty_response_tensor(self):
        """EMPTY_RESPONSE returns empty tensor for tensor output."""
        original = MagicMock(return_value={"sequences": torch.ones(2, 10, dtype=torch.long)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Sequence dimension should be 0
        assert result["sequences"].shape[-1] == 0
        assert result["sequences"].dtype == torch.long

    def test_empty_response_string(self):
        """EMPTY_RESPONSE returns empty string for string output."""
        original = MagicMock(return_value={"text": "Hello, world!"})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["text"] == ""

    def test_empty_response_list(self):
        """EMPTY_RESPONSE returns empty items for list output."""
        original = MagicMock(return_value={"outputs": [
            torch.ones(10, dtype=torch.long),
            torch.ones(15, dtype=torch.long),
        ]})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Each output should be empty
        for output in result["outputs"]:
            assert output.numel() == 0

    def test_empty_response_preserves_other_keys(self):
        """EMPTY_RESPONSE preserves non-output keys."""
        original = MagicMock(return_value={
            "sequences": torch.ones(2, 10),
            "attention_mask": torch.ones(2, 10),
            "prompt_length": 5,
        })
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Other keys preserved
        assert result["prompt_length"] == 5
        assert "attention_mask" in result


class TestTruncatedOutputStrategy:
    """Tests for TRUNCATED_OUTPUT strategy."""

    def test_truncated_output_tensor(self):
        """TRUNCATED_OUTPUT limits tensor output length."""
        original = MagicMock(return_value={"sequences": torch.arange(100)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={"max_tokens": 5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["sequences"].shape[0] == 5
        torch.testing.assert_close(result["sequences"], torch.arange(5))

    def test_truncated_output_default_max_tokens(self):
        """TRUNCATED_OUTPUT uses default max_tokens of 1."""
        original = MagicMock(return_value={"sequences": torch.arange(100)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={},  # No max_tokens specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should truncate to 1 token
        assert result["sequences"].shape[0] == 1

    def test_truncated_output_list_of_tokens(self):
        """TRUNCATED_OUTPUT truncates list of token IDs."""
        original = MagicMock(return_value={"token_ids": list(range(100))})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={"max_tokens": 10},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert len(result["token_ids"]) == 10
        assert result["token_ids"] == list(range(10))

    def test_truncated_output_string(self):
        """TRUNCATED_OUTPUT truncates string output."""
        original = MagicMock(return_value={"text": "A" * 100})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={"max_tokens": 5},  # 5 tokens * 4 chars = 20 chars
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should truncate to ~20 characters (5 tokens * 4 chars approx)
        assert len(result["text"]) == 20

    def test_truncated_output_2d_tensor(self):
        """TRUNCATED_OUTPUT truncates 2D tensor along sequence dimension."""
        original = MagicMock(return_value={"sequences": torch.ones(4, 50)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={"max_tokens": 10},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Batch dimension preserved, sequence truncated
        assert result["sequences"].shape == (4, 10)


class TestGarbageOutputStrategy:
    """Tests for GARBAGE_OUTPUT strategy."""

    def test_garbage_output_tensor(self):
        """GARBAGE_OUTPUT replaces tensor with random tokens."""
        torch.manual_seed(42)
        original = MagicMock(return_value={"sequences": torch.ones(2, 10, dtype=torch.long)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
            parameters={"vocab_size": 32000},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should have same shape
        assert result["sequences"].shape == (2, 10)
        # Should be different from original (all ones)
        assert not torch.all(result["sequences"] == 1)
        # Should be within vocab range
        assert torch.all(result["sequences"] >= 0)
        assert torch.all(result["sequences"] < 32000)

    def test_garbage_output_default_vocab_size(self):
        """GARBAGE_OUTPUT uses default vocab_size of 32000."""
        torch.manual_seed(42)
        original = MagicMock(return_value={"sequences": torch.ones(10, dtype=torch.long)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
            parameters={},  # No vocab_size specified
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should be within default vocab range
        assert torch.all(result["sequences"] >= 0)
        assert torch.all(result["sequences"] < 32000)

    def test_garbage_output_list_of_tokens(self):
        """GARBAGE_OUTPUT replaces list of token IDs with random tokens."""
        import random
        random.seed(42)
        original = MagicMock(return_value={"token_ids": [1, 2, 3, 4, 5]})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
            parameters={"vocab_size": 1000, "seed": 42},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should have same length
        assert len(result["token_ids"]) == 5
        # Should be within vocab range
        assert all(0 <= t < 1000 for t in result["token_ids"])

    def test_garbage_output_string(self):
        """GARBAGE_OUTPUT replaces string with random text."""
        import random
        random.seed(42)
        original = MagicMock(return_value={"generated_text": "Hello, world!"})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
            parameters={"seed": 42},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Should have similar length
        assert len(result["generated_text"]) == len("Hello, world!")
        # Should be different
        assert result["generated_text"] != "Hello, world!"

    def test_garbage_output_preserves_shape(self):
        """GARBAGE_OUTPUT preserves tensor shape."""
        torch.manual_seed(42)
        original = MagicMock(return_value={"sequences": torch.ones(3, 5, 20, dtype=torch.long)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
            parameters={"vocab_size": 50000},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["sequences"].shape == (3, 5, 20)

    def test_garbage_output_preserves_other_keys(self):
        """GARBAGE_OUTPUT preserves non-output keys."""
        torch.manual_seed(42)
        original = MagicMock(return_value={
            "sequences": torch.ones(10, dtype=torch.long),
            "attention_mask": torch.ones(10),
            "prompt_length": 5,
        })
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GARBAGE_OUTPUT,
            trigger=trigger,
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Other keys preserved
        assert result["prompt_length"] == 5
        assert "attention_mask" in result


class TestGenerateProxyIntegration:
    """Integration tests for GenerateProxy."""

    def test_step_based_trigger(self):
        """Proxy only triggers within step range."""
        original = MagicMock(return_value={"sequences": torch.ones(10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=10)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Before range - original called normally
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["sequences"], torch.ones(10))

        # In range - empty response
        proxy.set_step(7)
        result = proxy()
        assert result["sequences"].numel() == 0

        # After range - original called normally
        proxy.set_step(15)
        result = proxy()
        torch.testing.assert_close(result["sequences"], torch.ones(10))

    def test_periodic_trigger(self):
        """Proxy triggers every N steps."""
        original = MagicMock(return_value={"sequences": torch.ones(10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
        )
        proxy.set_config(config)

        # Step 0 - triggers
        proxy.set_step(0)
        result = proxy()
        assert result["sequences"].numel() == 0

        # Step 3 - doesn't trigger
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["sequences"], torch.ones(10))

        # Step 5 - triggers
        proxy.set_step(5)
        result = proxy()
        assert result["sequences"].numel() == 0

    def test_collector_recording(self):
        """Proxy records to collector when provided."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_123"

        original = MagicMock(return_value={"sequences": torch.ones(10)})
        proxy = GenerateProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.EMPTY_RESPONSE,
            trigger=trigger,
            severity="high",
            expected_behavior="Returns empty sequences",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        # Should have recorded the injection
        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()

    def test_collector_records_failure(self):
        """Proxy records failure when strategy raises exception."""
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_456"

        def failing_original():
            raise RuntimeError("Generation failed")

        proxy = GenerateProxy(failing_original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.GENERATION_TIMEOUT,
            trigger=trigger,
            parameters={"timeout_seconds": 0.001},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep"):  # Skip actual delay
            with pytest.raises(RuntimeError):
                proxy()

        # Should have recorded failure outcome
        collector.record_fault_outcome.assert_called_once()
        call_args = collector.record_fault_outcome.call_args
        assert "failure" in call_args[0][1].lower() or "error" in call_args[1].get("outcome", "").lower()

    def test_args_passed_to_original(self):
        """Arguments are passed correctly to original function."""
        original = MagicMock(return_value={"sequences": torch.ones(10)})
        proxy = GenerateProxy(original)

        # No config - should pass through
        proxy("prompt", max_tokens=100, temperature=0.7)
        original.assert_called_with("prompt", max_tokens=100, temperature=0.7)

    def test_kwargs_passed_to_original_with_strategy(self):
        """kwargs are passed to original even when strategy executes."""
        original = MagicMock(return_value={"sequences": torch.ones(10)})
        proxy = GenerateProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.TRUNCATED_OUTPUT,
            trigger=trigger,
            parameters={"max_tokens": 5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy("prompt", max_tokens=100, temperature=0.7)

        # Original should have been called with args
        original.assert_called_with("prompt", max_tokens=100, temperature=0.7)


# ============================================================================
# UpdateWeightsProxy Tests
# ============================================================================


class TestUpdateWeightsProxyRegistration:
    """Tests for UpdateWeightsProxy registration."""

    def test_registered_with_rollout_update_weights_target(self):
        """UpdateWeightsProxy is registered for 'rollout.update_weights' target."""
        assert ProxyRegistry.is_registered("rollout.update_weights")
        from ralph.proxies.inference import UpdateWeightsProxy
        assert ProxyRegistry.get_proxy("rollout.update_weights") is UpdateWeightsProxy

    def test_supported_strategies(self):
        """UpdateWeightsProxy declares correct supported strategies."""
        strategies = ProxyRegistry.get_supported_strategies("rollout.update_weights")
        expected = {
            StrategyType.DELAY,
            StrategyType.WEIGHT_MISMATCH,
            StrategyType.PARTIAL_UPDATE,
            StrategyType.CORRUPT_WEIGHTS,
            StrategyType.OLD_WEIGHTS,
        }
        assert strategies == expected


class TestUpdateWeightsProxyBasics:
    """Tests for basic UpdateWeightsProxy functionality."""

    def test_get_layer_returns_inference(self):
        """_get_layer returns 'Inference'."""
        from ralph.proxies.inference import UpdateWeightsProxy
        proxy = UpdateWeightsProxy(lambda: None)
        assert proxy._get_layer() == "Inference"

    def test_call_without_config_calls_original(self):
        """Proxy calls original when no config is set."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {"layer1.weight": torch.ones(10, 10)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        result = proxy()
        original.assert_called_once()
        assert "layer1.weight" in result

    def test_call_with_disabled_config_calls_original(self):
        """Proxy calls original when config is disabled."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {"layer1.weight": torch.ones(10, 10)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            enabled=False,
        )
        proxy.set_config(config)
        proxy.set_step(0)
        result = proxy()
        original.assert_called_once()
        torch.testing.assert_close(result["layer1.weight"], original_weights["layer1.weight"])

    def test_unsupported_strategy_raises_error(self):
        """Setting an unsupported strategy raises ValueError."""
        from ralph.proxies.inference import UpdateWeightsProxy
        proxy = UpdateWeightsProxy(lambda: None)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OBJECT_LOST,  # Not supported by UpdateWeightsProxy
            trigger=trigger,
        )
        with pytest.raises(ValueError) as exc_info:
            proxy.set_config(config)
        assert "not supported" in str(exc_info.value)


class TestUpdateWeightsDelayStrategy:
    """Tests for DELAY strategy."""

    def test_delay_strategy_with_config(self):
        """DELAY strategy uses delay_seconds from config."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original = MagicMock(return_value={"weight": torch.ones(10)})
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={"delay_seconds": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            result = proxy()
            mock_sleep.assert_called_once_with(0.01)
            original.assert_called_once()

    def test_delay_strategy_default_delay(self):
        """DELAY strategy uses default delay_seconds of 10.0."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original = MagicMock(return_value={"weight": torch.ones(10)})
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.DELAY,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        with patch("time.sleep") as mock_sleep:
            proxy()
            mock_sleep.assert_called_once_with(10.0)


class TestWeightMismatchStrategy:
    """Tests for WEIGHT_MISMATCH strategy."""

    def test_weight_mismatch_alters_shape(self):
        """WEIGHT_MISMATCH alters tensor shape."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {"layer.weight": torch.ones(100, 100)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WEIGHT_MISMATCH,
            trigger=trigger,
            parameters={"mismatch_ratio": 0.2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Shape should be altered (either increased or decreased)
        assert result["layer.weight"].shape[-1] != 100

    def test_weight_mismatch_default_ratio(self):
        """WEIGHT_MISMATCH uses default mismatch_ratio of 0.1."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {"layer.weight": torch.ones(100, 100)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WEIGHT_MISMATCH,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Shape should be altered by ~10%
        new_size = result["layer.weight"].shape[-1]
        assert new_size != 100
        assert 80 <= new_size <= 120  # Within reasonable range

    def test_weight_mismatch_handles_dict(self):
        """WEIGHT_MISMATCH handles dict of weights."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {
            "layer1.weight": torch.ones(50, 50),
            "layer2.weight": torch.ones(100, 100),
        }
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.WEIGHT_MISMATCH,
            trigger=trigger,
            parameters={"mismatch_ratio": 0.2},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Both layers should be modified
        assert "layer1.weight" in result
        assert "layer2.weight" in result


class TestPartialUpdateStrategy:
    """Tests for PARTIAL_UPDATE strategy."""

    def test_partial_update_zeros_some_weights(self):
        """PARTIAL_UPDATE zeros out some weights."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {
            "layer1.weight": torch.ones(10, 10),
            "layer2.weight": torch.ones(10, 10),
            "layer3.weight": torch.ones(10, 10),
            "layer4.weight": torch.ones(10, 10),
        }
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_UPDATE,
            trigger=trigger,
            parameters={"update_ratio": 0.5},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Some weights should be zeros, some should be ones
        zero_count = sum(1 for v in result.values() if torch.all(v == 0))
        ones_count = sum(1 for v in result.values() if torch.all(v == 1))
        assert zero_count > 0  # At least some weights zeroed
        assert ones_count > 0  # At least some weights preserved

    def test_partial_update_default_ratio(self):
        """PARTIAL_UPDATE uses default update_ratio of 0.5."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {f"layer{i}.weight": torch.ones(10, 10) for i in range(10)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_UPDATE,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # About half should be zeroed
        zero_count = sum(1 for v in result.values() if torch.all(v == 0))
        assert 3 <= zero_count <= 7  # Roughly half

    def test_partial_update_specific_keys(self):
        """PARTIAL_UPDATE can skip specific keys."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original_weights = {
            "layer1.weight": torch.ones(10, 10),
            "layer2.weight": torch.ones(10, 10),
            "layer3.weight": torch.ones(10, 10),
        }
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.PARTIAL_UPDATE,
            trigger=trigger,
            parameters={"keys_to_skip": ["layer1.weight", "layer3.weight"]},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Specified keys should be zeros
        assert torch.all(result["layer1.weight"] == 0)
        assert torch.all(result["layer3.weight"] == 0)
        # Unspecified key should be preserved
        assert torch.all(result["layer2.weight"] == 1)


class TestCorruptWeightsStrategy:
    """Tests for CORRUPT_WEIGHTS strategy."""

    def test_corrupt_weights_adds_noise(self):
        """CORRUPT_WEIGHTS adds Gaussian noise to weights."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original_weights = {"layer.weight": torch.ones(100, 100)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Weights should be different from original
        assert not torch.all(result["layer.weight"] == 1)
        # Mean should still be close to 1
        assert abs(result["layer.weight"].mean().item() - 1.0) < 0.05

    def test_corrupt_weights_default_noise_scale(self):
        """CORRUPT_WEIGHTS uses default noise_scale of 0.01."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original_weights = {"layer.weight": torch.ones(1000)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Std dev of difference should be close to 0.01
        diff_std = (result["layer.weight"] - 1.0).std().item()
        assert 0.005 < diff_std < 0.02

    def test_corrupt_weights_handles_multiple_tensors(self):
        """CORRUPT_WEIGHTS corrupts all tensors in dict."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original_weights = {
            "layer1.weight": torch.ones(50, 50),
            "layer2.weight": torch.zeros(100, 100),
        }
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        # Both should have noise
        assert not torch.all(result["layer1.weight"] == 1)
        assert not torch.all(result["layer2.weight"] == 0)

    def test_corrupt_weights_preserves_shape(self):
        """CORRUPT_WEIGHTS preserves tensor shapes."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original_weights = {"layer.weight": torch.ones(3, 5, 7)}
        original = MagicMock(return_value=original_weights)
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.1},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        result = proxy()

        assert result["layer.weight"].shape == (3, 5, 7)


class TestOldWeightsStrategy:
    """Tests for OLD_WEIGHTS strategy."""

    def test_old_weights_returns_previous_weights(self):
        """OLD_WEIGHTS returns previously stored weights."""
        from ralph.proxies.inference import UpdateWeightsProxy
        call_count = [0]

        def mock_update_weights():
            call_count[0] += 1
            return {"layer.weight": torch.ones(10) * call_count[0]}

        proxy = UpdateWeightsProxy(mock_update_weights)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=1)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OLD_WEIGHTS,
            trigger=trigger,
            parameters={"staleness_steps": 1},
        )
        proxy.set_config(config)

        # First call - no history, returns current
        proxy.set_step(0)
        result1 = proxy()
        assert result1["layer.weight"].mean().item() == 1

        # Second call - returns weights from first call
        proxy.set_step(1)
        result2 = proxy()
        assert result2["layer.weight"].mean().item() == 1  # Old weights from call 1

        # Third call - returns weights from second call
        proxy.set_step(2)
        result3 = proxy()
        assert result3["layer.weight"].mean().item() == 2  # Old weights from call 2

    def test_old_weights_default_staleness(self):
        """OLD_WEIGHTS uses default staleness_steps of 1."""
        from ralph.proxies.inference import UpdateWeightsProxy
        call_count = [0]

        def mock_update_weights():
            call_count[0] += 1
            return {"layer.weight": torch.ones(10) * call_count[0]}

        proxy = UpdateWeightsProxy(mock_update_weights)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=1)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OLD_WEIGHTS,
            trigger=trigger,
            parameters={},  # No staleness_steps specified
        )
        proxy.set_config(config)

        # Build up history
        proxy.set_step(0)
        proxy()
        proxy.set_step(1)
        result = proxy()

        # Should return 1-step-old weights
        assert result["layer.weight"].mean().item() == 1

    def test_old_weights_multiple_staleness(self):
        """OLD_WEIGHTS can return weights from multiple steps ago."""
        from ralph.proxies.inference import UpdateWeightsProxy
        call_count = [0]

        def mock_update_weights():
            call_count[0] += 1
            return {"layer.weight": torch.ones(10) * call_count[0]}

        proxy = UpdateWeightsProxy(mock_update_weights)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=1)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.OLD_WEIGHTS,
            trigger=trigger,
            parameters={"staleness_steps": 3},
        )
        proxy.set_config(config)

        # Build up history
        for step in range(5):
            proxy.set_step(step)
            proxy()

        # Next call should return 3-step-old weights
        proxy.set_step(5)
        result = proxy()
        assert result["layer.weight"].mean().item() == 3  # From call 3 (5 - staleness 2 = 3)


class TestUpdateWeightsProxyIntegration:
    """Integration tests for UpdateWeightsProxy."""

    def test_step_based_trigger(self):
        """Proxy only triggers within step range."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original = MagicMock(return_value={"weight": torch.ones(100)})
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.STEP_BASED, start_step=5, end_step=10)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)

        # Before range - original called normally
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["weight"], torch.ones(100))

        # In range - corrupted
        proxy.set_step(7)
        result = proxy()
        assert not torch.all(result["weight"] == 1)

        # After range - original called normally
        proxy.set_step(15)
        result = proxy()
        torch.testing.assert_close(result["weight"], torch.ones(100))

    def test_periodic_trigger(self):
        """Proxy triggers every N steps."""
        from ralph.proxies.inference import UpdateWeightsProxy
        torch.manual_seed(42)
        original = MagicMock(return_value={"weight": torch.ones(100)})
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.PERIODIC, every_n_steps=5)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.5},
        )
        proxy.set_config(config)

        # Step 0 - triggers
        proxy.set_step(0)
        result = proxy()
        assert not torch.all(result["weight"] == 1)

        # Step 3 - doesn't trigger
        proxy.set_step(3)
        result = proxy()
        torch.testing.assert_close(result["weight"], torch.ones(100))

        # Step 5 - triggers
        proxy.set_step(5)
        result = proxy()
        assert not torch.all(result["weight"] == 1)

    def test_collector_recording(self):
        """Proxy records to collector when provided."""
        from ralph.proxies.inference import UpdateWeightsProxy
        collector = MagicMock()
        collector.record_fault_injection.return_value = "fault_789"

        original = MagicMock(return_value={"weight": torch.ones(10)})
        proxy = UpdateWeightsProxy(original, collector)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            severity="high",
            expected_behavior="Corrupts weight tensors",
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy()

        # Should have recorded the injection
        collector.record_fault_injection.assert_called_once()
        collector.record_fault_outcome.assert_called_once()

    def test_args_passed_to_original(self):
        """Arguments are passed correctly to original function."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original = MagicMock(return_value={"weight": torch.ones(10)})
        proxy = UpdateWeightsProxy(original)

        # No config - should pass through
        proxy({"new_weights": torch.ones(10)}, version=42)
        original.assert_called_with({"new_weights": torch.ones(10)}, version=42)

    def test_kwargs_passed_to_original_with_strategy(self):
        """kwargs are passed to original even when strategy executes."""
        from ralph.proxies.inference import UpdateWeightsProxy
        original = MagicMock(return_value={"weight": torch.ones(10)})
        proxy = UpdateWeightsProxy(original)
        trigger = TriggerConfig(type=TriggerType.ONE_SHOT, at_step=0)
        config = FaultConfig(
            id="test",
            strategy=StrategyType.CORRUPT_WEIGHTS,
            trigger=trigger,
            parameters={"noise_scale": 0.01},
        )
        proxy.set_config(config)
        proxy.set_step(0)

        proxy({"new_weights": torch.ones(10)}, version=42)

        # Original should have been called with args
        original.assert_called_with({"new_weights": torch.ones(10)}, version=42)
