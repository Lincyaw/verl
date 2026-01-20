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


"""Unit tests for hierarchical recovery manager."""

import time
import unittest
from unittest.mock import MagicMock, patch

from verl.fault_injection.base import FaultContext, FaultResult, FaultStatus
from verl.fault_injection.config import FaultLayer
from verl.fault_injection.recovery.base import (
    BaseRecoveryStrategy,
    RecoveryContext,
    RecoveryDecision,
    RecoveryDecisionEngine,
    RecoveryMode,
    RecoveryPriority,
    RecoveryResult,
    RecoveryStatus,
)
from verl.fault_injection.recovery.hierarchical import (
    HierarchicalRecoveryManager,
    IntelligentRecoveryDecisionEngine,
)


class MockRecoveryStrategy(BaseRecoveryStrategy):
    """Mock recovery strategy for testing."""

    def __init__(self, name: str, priority: RecoveryPriority, should_succeed: bool = True):
        super().__init__()
        self._name = name
        self._priority = priority
        self.should_succeed = should_succeed

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> RecoveryPriority:
        return self._priority

    def can_recover(self, context: RecoveryContext) -> bool:
        return True

    def execute(self, context: RecoveryContext) -> RecoveryResult:
        if self.should_succeed:
            return RecoveryResult(
                recovery_id=f"{self.name}_recovery",
                status=RecoveryStatus.SUCCESS,
                strategy_name=self.name,
                start_time=time.time(),
                end_time=time.time(),
                metadata={"mock": True},
            )
        else:
            return RecoveryResult(
                recovery_id=f"{self.name}_recovery",
                status=RecoveryStatus.FAILED,
                strategy_name=self.name,
                start_time=time.time(),
                end_time=time.time(),
                error=Exception("Mock failure"),
                metadata={"mock": True},
            )


class TestHierarchicalRecoveryManager(unittest.TestCase):
    """Test hierarchical recovery manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.fault_result = FaultResult(
            fault_id="test_fault_001",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "process_kill"},
        )

        self.fault_context = FaultContext(
            worker_id="worker_001",
            rank=0,
            host="localhost",
            process_type="actor",
            layer=FaultLayer.WORKER,
            metadata={"pid": 12345},
        )

        self.recovery_context = RecoveryContext(
            fault_result=self.fault_result,
            fault_context=self.fault_context,
            attempt_count=0,
            max_attempts=3,
        )

    def test_basic_recovery(self):
        """Test basic recovery flow."""
        # Create strategies
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH),
        ]

        # Create decision engine
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.return_value = RecoveryDecision(
            strategy_name="retry",
            priority=RecoveryPriority.MEDIUM,
            mode=RecoveryMode.AUTOMATIC,
            confidence=0.8,
            reason="Test decision",
        )
        decision_engine.get_strategy.return_value = strategies[0]

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Execute recovery
        result = manager.handle_fault(self.fault_result, self.fault_context)

        # Verify results
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "retry")
        decision_engine.decide_recovery.assert_called_once()
        decision_engine.get_strategy.assert_called_once_with("retry")

    def test_recovery_with_multiple_attempts(self):
        """Test recovery with multiple attempts."""
        # Create strategies that fail then succeed
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM, should_succeed=False),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH, should_succeed=True),
        ]

        # Create decision engine that returns different strategies
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.side_effect = [
            RecoveryDecision(
                strategy_name="retry",
                priority=RecoveryPriority.MEDIUM,
                mode=RecoveryMode.AUTOMATIC,
                confidence=0.8,
                reason="First attempt",
            ),
            RecoveryDecision(
                strategy_name="restart",
                priority=RecoveryPriority.HIGH,
                mode=RecoveryMode.AUTOMATIC,
                confidence=0.9,
                reason="Second attempt",
            ),
        ]
        decision_engine.get_strategy.side_effect = strategies

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Execute recovery
        result = manager.handle_fault(self.fault_result, self.fault_context)

        # Verify results
        self.assertEqual(result.status, RecoveryStatus.SUCCESS)
        self.assertEqual(result.strategy_name, "restart")
        self.assertEqual(decision_engine.decide_recovery.call_count, 2)

    def test_recovery_timeout(self):
        """Test recovery timeout."""

        # Create slow strategy
        class SlowStrategy(BaseRecoveryStrategy):
            @property
            def name(self):
                return "slow"

            @property
            def priority(self):
                return RecoveryPriority.MEDIUM

            def can_recover(self, context):
                return True

            def execute(self, context):
                time.sleep(2)  # Sleep longer than timeout
                return RecoveryResult(
                    recovery_id="slow_recovery",
                    status=RecoveryStatus.SUCCESS,
                    strategy_name=self.name,
                    start_time=time.time(),
                    end_time=time.time(),
                )

        strategies = [SlowStrategy()]

        # Create decision engine
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.return_value = RecoveryDecision(
            strategy_name="slow",
            priority=RecoveryPriority.MEDIUM,
            mode=RecoveryMode.AUTOMATIC,
            confidence=0.8,
            reason="Test",
        )
        decision_engine.get_strategy.return_value = strategies[0]

        # Create recovery manager with short timeout
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=1,
            recovery_timeout=0.5,  # Short timeout
        )

        # Execute recovery
        result = manager.handle_fault(self.fault_result, self.fault_context)

        # Verify timeout
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertIn("timeout", result.error.args[0])

    def test_manual_recovery_mode(self):
        """Test manual recovery mode."""
        # Create strategy
        strategies = [MockRecoveryStrategy("manual", RecoveryPriority.CRITICAL)]

        # Create decision engine
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.return_value = RecoveryDecision(
            strategy_name="manual",
            priority=RecoveryPriority.CRITICAL,
            mode=RecoveryMode.MANUAL,
            confidence=0.5,
            reason="Manual intervention needed",
        )
        decision_engine.get_strategy.return_value = strategies[0]

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Mock manual approval (simulate user approval)
        with patch.object(manager, "_request_manual_approval", return_value=True):
            # Execute recovery
            result = manager.handle_fault(self.fault_result, self.fault_context, mode=RecoveryMode.MANUAL)

            # Verify results
            self.assertEqual(result.status, RecoveryStatus.SUCCESS)
            self.assertEqual(result.strategy_name, "manual")

    def test_manual_recovery_denied(self):
        """Test manual recovery when denied."""
        # Create decision engine
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.return_value = RecoveryDecision(
            strategy_name="manual",
            priority=RecoveryPriority.CRITICAL,
            mode=RecoveryMode.MANUAL,
            confidence=0.5,
            reason="Manual intervention needed",
        )

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Mock manual approval denial
        with patch.object(manager, "_request_manual_approval", return_value=False):
            # Execute recovery
            result = manager.handle_fault(self.fault_result, self.fault_context, mode=RecoveryMode.MANUAL)

            # Verify cancellation
            self.assertEqual(result.status, RecoveryStatus.CANCELLED)
            self.assertEqual(result.metadata["reason"], "Manual approval denied")

    def test_should_not_attempt_recovery(self):
        """Test when recovery should not be attempted."""
        # Already recovered fault
        self.fault_result.status = FaultStatus.RECOVERED

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=MagicMock(),
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Execute recovery
        result = manager.handle_fault(self.fault_result, self.fault_context)

        # Verify no recovery attempted
        self.assertEqual(result.status, RecoveryStatus.CANCELLED)
        self.assertIn("Recovery not needed", result.metadata["reason"])

    def test_recovery_statistics(self):
        """Test recovery statistics."""
        # Create successful strategies
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH),
        ]

        # Create decision engine
        decision_engine = MagicMock(spec=RecoveryDecisionEngine)
        decision_engine.decide_recovery.return_value = RecoveryDecision(
            strategy_name="retry",
            priority=RecoveryPriority.MEDIUM,
            mode=RecoveryMode.AUTOMATIC,
            confidence=0.8,
            reason="Test",
        )
        decision_engine.get_strategy.return_value = strategies[0]

        # Create recovery manager
        manager = HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=3,
            recovery_timeout=10.0,
        )

        # Execute multiple recoveries
        for _ in range(3):
            manager.handle_fault(self.fault_result, self.fault_context)

        # Get statistics
        stats = manager.get_recovery_statistics()

        # Verify statistics
        self.assertEqual(stats["total_recoveries"], 3)
        self.assertEqual(stats["successful_recoveries"], 3)
        self.assertEqual(stats["failed_recoveries"], 0)
        self.assertEqual(stats["success_rate"], 1.0)


class TestIntelligentRecoveryDecisionEngine(unittest.TestCase):
    """Test intelligent recovery decision engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.fault_result = FaultResult(
            fault_id="test_fault_001",
            status=FaultStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"fault_type": "process_kill"},
        )

        self.fault_context = FaultContext(
            worker_id="worker_001",
            rank=0,
            host="localhost",
            process_type="actor",
            layer=FaultLayer.WORKER,
            metadata={"pid": 12345},
        )

        self.recovery_context = RecoveryContext(
            fault_result=self.fault_result,
            fault_context=self.fault_context,
            attempt_count=0,
            max_attempts=3,
        )

    def test_decide_recovery(self):
        """Test recovery decision making."""
        # Create strategies
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH),
        ]

        # Create decision engine
        engine = IntelligentRecoveryDecisionEngine(strategies)

        # Make decision
        decision = engine.decide_recovery(self.recovery_context)

        # Verify decision
        self.assertIsNotNone(decision)
        self.assertIn(decision.strategy_name, ["retry", "restart"])
        self.assertGreater(decision.confidence, 0)
        self.assertGreater(len(decision.reason), 0)

    def test_evaluate_strategies(self):
        """Test strategy evaluation."""
        # Create strategies
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH),
            MockRecoveryStrategy("manual", RecoveryPriority.CRITICAL),
        ]

        # Create decision engine
        engine = IntelligentRecoveryDecisionEngine(strategies)

        # Evaluate strategies
        decisions = engine.evaluate_strategies(self.recovery_context)

        # Verify decisions
        self.assertEqual(len(decisions), 3)

        # Check ordering by priority
        priorities = [d.priority.value for d in decisions]
        self.assertEqual(priorities, sorted(priorities))

    def test_no_recoverable_strategies(self):
        """Test when no strategies can recover."""

        # Create strategies that can't recover
        class NonRecoverableStrategy(BaseRecoveryStrategy):
            @property
            def name(self):
                return "non_recoverable"

            @property
            def priority(self):
                return RecoveryPriority.MEDIUM

            def can_recover(self, context):
                return False

            def execute(self, context):
                return RecoveryResult(
                    recovery_id="non_recoverable",
                    status=RecoveryStatus.FAILED,
                    strategy_name=self.name,
                    start_time=time.time(),
                    end_time=time.time(),
                )

        strategies = [NonRecoverableStrategy()]
        engine = IntelligentRecoveryDecisionEngine(strategies)

        # Make decision
        decision = engine.decide_recovery(self.recovery_context)

        # Verify no recovery possible
        self.assertEqual(decision.strategy_name, "none")
        self.assertEqual(decision.confidence, 0.0)

    def test_layer_specific_bonus(self):
        """Test layer-specific strategy scoring."""
        # Create strategies
        strategies = [
            MockRecoveryStrategy("ray_recovery", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("worker_recovery", RecoveryPriority.MEDIUM),
        ]

        # Create decision engine
        engine = IntelligentRecoveryDecisionEngine(strategies)

        # Test orchestration layer (should prefer ray_recovery)
        self.fault_context.layer = FaultLayer.ORCHESTRATION
        decision = engine.decide_recovery(self.recovery_context)
        self.assertEqual(decision.strategy_name, "ray_recovery")

        # Test worker layer (should prefer worker_recovery)
        self.fault_context.layer = FaultLayer.WORKER
        decision = engine.decide_recovery(self.recovery_context)
        self.assertEqual(decision.strategy_name, "worker_recovery")

    def test_historical_success_rate(self):
        """Test historical success rate calculation."""
        # Create recovery history
        history = []
        for i in range(5):
            history.append(
                RecoveryResult(
                    recovery_id=f"recovery_{i}",
                    status=RecoveryStatus.SUCCESS if i < 3 else RecoveryStatus.FAILED,
                    strategy_name="retry",
                    start_time=time.time(),
                    end_time=time.time(),
                )
            )

        # Create strategies
        strategies = [MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM)]

        # Create decision engine with history
        engine = IntelligentRecoveryDecisionEngine(strategies, recovery_history=history)

        # Check historical success rate
        success_rate = engine._get_historical_success_rate("retry")
        self.assertEqual(success_rate, 0.6)  # 3 out of 5 successful

    def test_preferred_alternative(self):
        """Test using preferred alternative from metadata."""
        # Create strategies
        strategies = [
            MockRecoveryStrategy("retry", RecoveryPriority.MEDIUM),
            MockRecoveryStrategy("restart", RecoveryPriority.HIGH),
        ]

        # Create decision engine
        engine = IntelligentRecoveryDecisionEngine(strategies)

        # Set preferred alternative
        self.recovery_context.metadata["preferred_alternative"] = "restart"

        # Make decision
        decision = engine.decide_recovery(self.recovery_context)

        # Should use preferred alternative
        self.assertEqual(decision.strategy_name, "restart")


if __name__ == "__main__":
    unittest.main()
