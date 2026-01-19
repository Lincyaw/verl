"""Hierarchical recovery management system."""

import logging
import time
from typing import Any, Dict, List, Optional

from ..base import FaultContext, FaultResult, FaultStatus
from ..config import FaultLayer, FaultType
from .base import (
    BaseRecoveryStrategy,
    RecoveryContext,
    RecoveryDecision,
    RecoveryDecisionEngine,
    RecoveryMode,
    RecoveryPriority,
    RecoveryResult,
    RecoveryStatus,
)

logger = logging.getLogger(__name__)


class HierarchicalRecoveryManager:
    """Manages hierarchical recovery strategies across different fault layers."""

    def __init__(
        self,
        decision_engine: RecoveryDecisionEngine,
        max_attempts: int = 3,
        recovery_timeout: float = 300.0,
        enable_preventive_recovery: bool = True,
    ):
        self.decision_engine = decision_engine
        self.max_attempts = max_attempts
        self.recovery_timeout = recovery_timeout
        self.enable_preventive_recovery = enable_preventive_recovery
        self._active_recoveries: Dict[str, RecoveryContext] = {}
        self._recovery_history: List[RecoveryResult] = []

    def handle_fault(
        self,
        fault_result: FaultResult,
        fault_context: FaultContext,
        mode: RecoveryMode = RecoveryMode.AUTOMATIC,
    ) -> RecoveryResult:
        """Handle a fault and initiate recovery if needed."""
        recovery_id = f"recovery_{fault_result.fault_id}_{int(time.time())}"

        # Check if we should attempt recovery
        if not self._should_attempt_recovery(fault_result, fault_context):
            return RecoveryResult(
                recovery_id=recovery_id,
                status=RecoveryStatus.CANCELLED,
                strategy_name="none",
                start_time=time.time(),
                end_time=time.time(),
                metadata={"reason": "Recovery not needed or disabled"},
            )

        # Create recovery context
        recovery_context = RecoveryContext(
            fault_result=fault_result,
            fault_context=fault_context,
            max_attempts=self.max_attempts,
        )

        # Store active recovery
        self._active_recoveries[recovery_id] = recovery_context

        try:
            # Execute recovery
            result = self._execute_recovery(recovery_context, mode)
            self._recovery_history.append(result)
            return result

        finally:
            # Clean up
            del self._active_recoveries[recovery_id]

    def _should_attempt_recovery(self, fault_result: FaultResult, fault_context: FaultContext) -> bool:
        """Determine if recovery should be attempted."""
        # Don't recover if fault already recovered
        if fault_result.status == FaultStatus.RECOVERED:
            return False

        # Don't recover if fault is still in progress
        if fault_result.status == FaultStatus.INJECTING:
            return False

        # Check if fault layer supports recovery
        if fault_context.layer in [FaultLayer.UI]:
            # UI layer faults might not need recovery
            return False

        return True

    def _execute_recovery(self, context: RecoveryContext, mode: RecoveryMode) -> RecoveryResult:
        """Execute recovery with multiple attempts."""
        start_time = time.time()
        last_error = None

        while context.attempt_count < context.max_attempts:
            context.attempt_count += 1

            # Check timeout
            if time.time() - start_time > self.recovery_timeout:
                return RecoveryResult(
                    recovery_id=f"recovery_{context.fault_result.fault_id}",
                    status=RecoveryStatus.FAILED,
                    strategy_name="timeout",
                    start_time=start_time,
                    end_time=time.time(),
                    error=Exception("Recovery timeout exceeded"),
                    metadata={"attempts": context.attempt_count},
                )

            # Get recovery decision
            decision = self.decision_engine.decide_recovery(context)

            # Check mode
            if mode == RecoveryMode.MANUAL and context.attempt_count == 1:
                # In manual mode, require approval for first attempt
                if not self._request_manual_approval(decision):
                    return RecoveryResult(
                        recovery_id=f"recovery_{context.fault_result.fault_id}",
                        status=RecoveryStatus.CANCELLED,
                        strategy_name=decision.strategy_name,
                        start_time=start_time,
                        end_time=time.time(),
                        metadata={"reason": "Manual approval denied"},
                    )

            # Execute recovery strategy
            strategy = self.decision_engine.get_strategy(decision.strategy_name)
            if strategy is None:
                return RecoveryResult(
                    recovery_id=f"recovery_{context.fault_result.fault_id}",
                    status=RecoveryStatus.FAILED,
                    strategy_name=decision.strategy_name,
                    start_time=start_time,
                    end_time=time.time(),
                    error=Exception(f"Strategy {decision.strategy_name} not found"),
                )

            # Execute strategy with checks
            result = strategy._execute_with_checks(context)
            result.recovery_id = f"recovery_{context.fault_result.fault_id}_attempt_{context.attempt_count}"

            # Check if successful
            if result.status == RecoveryStatus.SUCCESS:
                return RecoveryResult(
                    recovery_id=f"recovery_{context.fault_result.fault_id}",
                    status=RecoveryStatus.SUCCESS,
                    strategy_name=decision.strategy_name,
                    start_time=start_time,
                    end_time=time.time(),
                    metadata={
                        "attempts": context.attempt_count,
                        "final_strategy": decision.strategy_name,
                        "confidence": decision.confidence,
                    },
                )

            # Store error for next iteration
            last_error = result.error

            # Check if we should try alternatives
            if result.next_recovery and context.attempt_count < context.max_attempts:
                # Add alternative to metadata for next decision
                context.metadata["preferred_alternative"] = result.next_recovery

        # All attempts failed
        return RecoveryResult(
            recovery_id=f"recovery_{context.fault_result.fault_id}",
            status=RecoveryStatus.FAILED,
            strategy_name="all_strategies",
            start_time=start_time,
            end_time=time.time(),
            error=last_error or Exception("All recovery attempts failed"),
            metadata={"attempts": context.attempt_count},
        )

    def _request_manual_approval(self, decision: RecoveryDecision) -> bool:
        """Request manual approval for recovery."""
        # In a real implementation, this would interact with a UI or API
        # For now, we'll log and return True (approved)
        logger.warning(
            f"Manual approval requested for recovery strategy: {decision.strategy_name}\n"
            f"Reason: {decision.reason}\n"
            f"Confidence: {decision.confidence:.2f}\n"
            f"Alternatives: {decision.alternatives}"
        )
        # In production, this would wait for user input
        return True

    def enable_preventive_recovery_for_layer(self, layer: FaultLayer) -> None:
        """Enable preventive recovery for a specific layer."""
        if not self.enable_preventive_recovery:
            return

        # Set up monitoring and early detection for the layer
        logger.info(f"Enabling preventive recovery for layer: {layer.value}")

        # In a real implementation, this would:
        # 1. Set up health checks
        # 2. Configure early warning thresholds
        # 3. Register preventive recovery handlers

    def get_recovery_history(self, limit: int = 100) -> List[RecoveryResult]:
        """Get recent recovery history."""
        return self._recovery_history[-limit:]

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self._recovery_history:
            return {}

        total = len(self._recovery_history)
        successful = sum(1 for r in self._recovery_history if r.status == RecoveryStatus.SUCCESS)
        failed = sum(1 for r in self._recovery_history if r.status == RecoveryStatus.FAILED)

        return {
            "total_recoveries": total,
            "successful_recoveries": successful,
            "failed_recoveries": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_recovery_time": sum(
                r.duration for r in self._recovery_history if r.duration
            )
            / total
            if total > 0
            else 0.0,
        }


class IntelligentRecoveryDecisionEngine(RecoveryDecisionEngine):
    """Intelligent decision engine that considers fault context and history."""

    def __init__(self, strategies: List[BaseRecoveryStrategy], recovery_history: List[RecoveryResult] = None):
        super().__init__(strategies)
        self.recovery_history = recovery_history or []

    def decide_recovery(
        self, context: RecoveryContext, available_strategies: List[str] = None
    ) -> RecoveryDecision:
        """Make intelligent recovery decision based on context and history."""
        # Evaluate all strategies
        decisions = self.evaluate_strategies(context)

        if not decisions:
            # No strategy can recover
            return RecoveryDecision(
                strategy_name="none",
                priority=RecoveryPriority.CRITICAL,
                mode=RecoveryMode.MANUAL,
                confidence=0.0,
                reason="No recovery strategy can handle this fault",
                alternatives=[],
            )

        # Apply intelligent selection
        best_decision = self._select_best_strategy(decisions, context)

        # Check if we should use an alternative from previous attempt
        if "preferred_alternative" in context.metadata:
            alt_name = context.metadata["preferred_alternative"]
            alt_decision = next((d for d in decisions if d.strategy_name == alt_name), None)
            if alt_decision:
                best_decision = alt_decision

        return best_decision

    def _select_best_strategy(self, decisions: List[RecoveryDecision], context: RecoveryContext) -> RecoveryDecision:
        """Select the best strategy based on multiple factors."""
        # Score each decision
        scored_decisions = []
        for decision in decisions:
            score = self._calculate_strategy_score(decision, context)
            scored_decisions.append((score, decision))

        # Sort by score (highest first)
        scored_decisions.sort(key=lambda x: x[0], reverse=True)

        # Return the best decision
        return scored_decisions[0][1]

    def _calculate_strategy_score(self, decision: RecoveryDecision, context: RecoveryContext) -> float:
        """Calculate a score for a recovery decision."""
        score = 0.0

        # Base score from confidence
        score += decision.confidence * 0.4

        # Priority bonus
        priority_bonus = (5 - decision.priority.value) * 0.1
        score += priority_bonus

        # Historical success rate
        hist_success = self._get_historical_success_rate(decision.strategy_name)
        score += hist_success * 0.3

        # Layer-specific adjustments
        layer_bonus = self._get_layer_specific_bonus(decision.strategy_name, context.fault_context.layer)
        score += layer_bonus

        # Attempt penalty
        attempt_penalty = min(0.2, context.attempt_count * 0.05)
        score -= attempt_penalty

        return max(0.0, min(1.0, score))

    def _get_historical_success_rate(self, strategy_name: str) -> float:
        """Get historical success rate for a strategy."""
        relevant_recoveries = [r for r in self.recovery_history if r.strategy_name == strategy_name]
        if not relevant_recoveries:
            return 0.5  # Default for unknown strategies

        successful = sum(1 for r in relevant_recoveries if r.status == RecoveryStatus.SUCCESS)
        return successful / len(relevant_recoveries)

    def _get_layer_specific_bonus(self, strategy_name: str, layer: Optional[FaultLayer]) -> float:
        """Get layer-specific bonus for a strategy."""
        if not layer:
            return 0.0

        # Define layer-strategy compatibility
        compatibility = {
            FaultLayer.ORCHESTRATION: ["ray_recovery", "restart", "retry"],
            FaultLayer.WORKER: ["process_recovery", "restart", "retry", "checkpoint_recovery"],
            FaultLayer.ENGINE: ["restart", "checkpoint_recovery", "degradation"],
            FaultLayer.INFERENCE: ["restart", "degradation", "retry"],
        }

        compatible_strategies = compatibility.get(layer, [])
        if strategy_name in compatible_strategies:
            return 0.1

        return 0.0