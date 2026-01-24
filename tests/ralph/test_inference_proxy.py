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
