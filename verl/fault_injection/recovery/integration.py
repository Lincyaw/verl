"""Integration of recovery system with fault injection orchestrator."""

import logging
from typing import Any, Dict, List, Optional

from ..base import FaultContext, FaultResult, FaultStatus
from ..config import FaultInjectionConfig, RecoveryConfig, RecoveryStrategyConfig
from ..orchestrator import FaultOrchestrator
from .base import BaseRecoveryStrategy, RecoveryContext, RecoveryDecisionEngine
from .hierarchical import HierarchicalRecoveryManager, IntelligentRecoveryDecisionEngine
from .strategies import (
    CheckpointRecoveryStrategy,
    DegradationRecoveryStrategy,
    IgnoreRecoveryStrategy,
    ManualRecoveryStrategy,
    ProcessRecoveryStrategy,
    RayRecoveryStrategy,
    RedeployRecoveryStrategy,
    ResourceRecoveryStrategy,
    RestartRecoveryStrategy,
    RetryRecoveryStrategy,
)

logger = logging.getLogger(__name__)


class RecoveryOrchestratorIntegration:
    """Integrates recovery system with fault injection orchestrator."""

    def __init__(self, orchestrator: FaultOrchestrator, config: RecoveryConfig):
        self.orchestrator = orchestrator
        self.config = config
        self.recovery_manager = self._create_recovery_manager(config)
        self._recovery_enabled = config.enabled

    def _create_recovery_manager(self, config: RecoveryConfig) -> HierarchicalRecoveryManager:
        """Create recovery manager with configured strategies."""
        # Create strategies based on configuration
        strategies = self._create_strategies(config.strategies)

        # Create decision engine
        decision_engine = IntelligentRecoveryDecisionEngine(strategies)

        # Create recovery manager
        return HierarchicalRecoveryManager(
            decision_engine=decision_engine,
            max_attempts=config.max_recovery_attempts,
            recovery_timeout=config.recovery_timeout_seconds,
            enable_preventive_recovery=config.enable_preventive_recovery,
        )

    def _create_strategies(self, strategy_configs: List[RecoveryStrategyConfig]) -> List[BaseRecoveryStrategy]:
        """Create recovery strategies from configuration."""
        strategies = []
        strategy_map = {
            "ignore": IgnoreRecoveryStrategy,
            "retry": RetryRecoveryStrategy,
            "restart": RestartRecoveryStrategy,
            "process_recovery": ProcessRecoveryStrategy,
            "resource_recovery": ResourceRecoveryStrategy,
            "checkpoint_recovery": CheckpointRecoveryStrategy,
            "degradation": DegradationRecoveryStrategy,
            "manual": ManualRecoveryStrategy,
            "ray_recovery": RayRecoveryStrategy,
            "redeploy": RedeployRecoveryStrategy,
        }

        for config in strategy_configs:
            if not config.enabled:
                continue

            strategy_class = strategy_map.get(config.name)
            if strategy_class:
                strategy = strategy_class(config.parameters)
                strategies.append(strategy)
            else:
                logger.warning(f"Unknown recovery strategy: {config.name}")

        return strategies

    def on_fault_completed(self, fault_result: FaultResult, fault_context: FaultContext) -> Optional[FaultResult]:
        """Handle fault completion and initiate recovery if needed."""
        if not self._recovery_enabled:
            return None

        # Check if recovery is needed
        if fault_result.status not in [FaultStatus.COMPLETED, FaultStatus.FAILED]:
            return None

        # Determine recovery mode
        mode = self._get_recovery_mode()

        # Initiate recovery
        logger.info(f"Initiating recovery for fault: {fault_result.fault_id}")
        recovery_result = self.recovery_manager.handle_fault(fault_result, fault_context, mode)

        # Update fault result with recovery information
        if recovery_result.status.value == "success":
            fault_result.status = FaultStatus.RECOVERED
            fault_result.metadata["recovery"] = {
                "strategy": recovery_result.strategy_name,
                "duration": recovery_result.duration,
                "attempts": recovery_result.metadata.get("attempts", 1),
            }
        else:
            fault_result.metadata["recovery_failure"] = {
                "strategy": recovery_result.strategy_name,
                "error": str(recovery_result.error) if recovery_result.error else None,
                "attempts": recovery_result.metadata.get("attempts", 1),
            }

        return fault_result

    def _get_recovery_mode(self) -> Any:
        """Get recovery mode from configuration."""
        from .base import RecoveryMode

        mode_map = {
            "automatic": RecoveryMode.AUTOMATIC,
            "manual": RecoveryMode.MANUAL,
            "semi_automatic": RecoveryMode.SEMI_AUTOMATIC,
        }

        return mode_map.get(self.config.mode, RecoveryMode.AUTOMATIC)

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return self.recovery_manager.get_recovery_statistics()

    def enable_preventive_recovery(self, layer: Optional[str] = None) -> None:
        """Enable preventive recovery for specific layer or all layers."""
        if layer:
            from ..config import FaultLayer
            fault_layer = FaultLayer(layer)
            self.recovery_manager.enable_preventive_recovery_for_layer(fault_layer)
        else:
            # Enable for all layers
            for fault_layer in FaultLayer:
                self.recovery_manager.enable_preventive_recovery_for_layer(fault_layer)

    def update_config(self, config: RecoveryConfig) -> None:
        """Update recovery configuration."""
        self.config = config
        self._recovery_enabled = config.enabled
        if config.enabled:
            self.recovery_manager = self._create_recovery_manager(config)


def create_recovery_integration(orchestrator: FaultOrchestrator, config: FaultInjectionConfig) -> RecoveryOrchestratorIntegration:
    """Factory function to create recovery integration."""
    return RecoveryOrchestratorIntegration(orchestrator, config.recovery)