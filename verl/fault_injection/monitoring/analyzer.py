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


"""Fault impact analysis for the fault injection system."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..base import FaultLayer, FaultResult, FaultStatus

logger = logging.getLogger(__name__)


@dataclass
class ImpactMetrics:
    """Metrics for fault impact analysis."""

    affected_components: set[str] = field(default_factory=set)
    affected_layers: set[FaultLayer] = field(default_factory=set)
    fault_count: int = 0
    failure_rate: float = 0.0
    recovery_rate: float = 0.0
    avg_impact_duration: float = 0.0
    max_impact_duration: float = 0.0
    cascade_probability: float = 0.0
    system_degradation: float = 0.0  # 0-1 scale


@dataclass
class ImpactAnalysisResult:
    """Result of fault impact analysis."""

    analysis_id: str
    timestamp: datetime
    fault_id: str
    fault_type: str
    impact_metrics: ImpactMetrics
    affected_services: list[str] = field(default_factory=list)
    cascade_risk: str = "low"  # low, medium, high, critical
    recommendations: list[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0-1 scale


@dataclass
class ComponentDependency:
    """Represents a component dependency."""

    component: str
    depends_on: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    layer: Optional[FaultLayer] = None
    criticality: str = "medium"  # low, medium, high, critical


class FaultImpactAnalyzer:
    """Analyzes the impact of faults on the system."""

    def __init__(self):
        # Component dependency graph
        self._dependencies: dict[str, ComponentDependency] = {}
        self._layer_dependencies: dict[FaultLayer, set[FaultLayer]] = {}
        self._historical_faults: list[FaultResult] = []
        self._impact_patterns: dict[str, Any] = {}

        # Initialize default dependencies
        self._initialize_default_dependencies()

    def _initialize_default_dependencies(self) -> None:
        """Initialize default component dependencies for verl."""
        # UI Layer dependencies
        self.add_dependency("hydra_config", layer=FaultLayer.UI, criticality="high")
        self.add_dependency("ray_init", layer=FaultLayer.UI, depends_on=["ray_cluster"], criticality="critical")

        # Orchestration Layer dependencies
        self.add_dependency("ray_cluster", layer=FaultLayer.ORCHESTRATION, criticality="critical")
        self.add_dependency("gcs", layer=FaultLayer.ORCHESTRATION, depends_on=["ray_cluster"], criticality="critical")
        self.add_dependency(
            "actor_system", layer=FaultLayer.ORCHESTRATION, depends_on=["ray_cluster"], criticality="high"
        )
        self.add_dependency(
            "resource_pool", layer=FaultLayer.ORCHESTRATION, depends_on=["actor_system"], criticality="medium"
        )

        # Worker Layer dependencies
        self.add_dependency(
            "fsdp_workers", layer=FaultLayer.WORKER, depends_on=["actor_system", "resource_pool"], criticality="high"
        )
        self.add_dependency(
            "megatron_workers",
            layer=FaultLayer.WORKER,
            depends_on=["actor_system", "resource_pool"],
            criticality="high",
        )
        self.add_dependency(
            "gradient_sync",
            layer=FaultLayer.WORKER,
            depends_on=["fsdp_workers", "megatron_workers"],
            criticality="high",
        )

        # Engine Layer dependencies
        self.add_dependency(
            "training_engine",
            layer=FaultLayer.ENGINE,
            depends_on=["fsdp_workers", "megatron_workers"],
            criticality="critical",
        )
        self.add_dependency(
            "checkpoint_system", layer=FaultLayer.ENGINE, depends_on=["training_engine"], criticality="high"
        )
        self.add_dependency("nccl_comm", layer=FaultLayer.ENGINE, depends_on=["training_engine"], criticality="high")

        # Inference Layer dependencies
        self.add_dependency(
            "vllm_backend", layer=FaultLayer.INFERENCE, depends_on=["training_engine"], criticality="medium"
        )
        self.add_dependency(
            "sglang_backend", layer=FaultLayer.INFERENCE, depends_on=["training_engine"], criticality="medium"
        )
        self.add_dependency(
            "kv_cache", layer=FaultLayer.INFERENCE, depends_on=["vllm_backend", "sglang_backend"], criticality="medium"
        )

        # Cross-layer dependencies
        self._layer_dependencies = {
            FaultLayer.UI: {FaultLayer.ORCHESTRATION},
            FaultLayer.ORCHESTRATION: set(),
            FaultLayer.WORKER: {FaultLayer.ORCHESTRATION},
            FaultLayer.ENGINE: {FaultLayer.WORKER, FaultLayer.ORCHESTRATION},
            FaultLayer.INFERENCE: {FaultLayer.ENGINE, FaultLayer.ORCHESTRATION},
        }

    def add_dependency(
        self,
        component: str,
        depends_on: Optional[list[str]] = None,
        dependents: Optional[list[str]] = None,
        layer: Optional[FaultLayer] = None,
        criticality: str = "medium",
    ) -> None:
        """Add a component dependency."""
        if component not in self._dependencies:
            self._dependencies[component] = ComponentDependency(
                component=component,
                layer=layer,
                criticality=criticality,
            )

        dep = self._dependencies[component]
        if depends_on:
            dep.depends_on.extend(depends_on)
        if dependents:
            dep.dependents.extend(dependents)
        if layer:
            dep.layer = layer
        if criticality:
            dep.criticality = criticality

    def analyze_fault_impact(
        self, fault_result: FaultResult, historical_window_hours: int = 24
    ) -> ImpactAnalysisResult:
        """Analyze the impact of a specific fault."""
        logger.info(f"Analyzing impact of fault: {fault_result.fault_id}")

        # Get affected components
        affected_components = self._get_affected_components(fault_result)

        # Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(fault_result, affected_components, historical_window_hours)

        # Assess cascade risk
        cascade_risk = self._assess_cascade_risk(fault_result, affected_components)

        # Generate recommendations
        recommendations = self._generate_recommendations(fault_result, impact_metrics)

        # Calculate confidence score
        confidence = self._calculate_confidence(fault_result, historical_window_hours)

        return ImpactAnalysisResult(
            analysis_id=f"analysis_{fault_result.fault_id}_{int(time.time())}",
            timestamp=datetime.now(),
            fault_id=fault_result.fault_id,
            fault_type=fault_result.metadata.get("fault_type", "unknown"),
            impact_metrics=impact_metrics,
            affected_services=list(affected_components),
            cascade_risk=cascade_risk,
            recommendations=recommendations,
            confidence_score=confidence,
        )

    def _get_affected_components(self, fault_result: FaultResult) -> set[str]:
        """Get components affected by the fault."""
        affected = set()

        # Direct impact based on fault type and layer
        layer = fault_result.metadata.get("layer")
        fault_type = fault_result.metadata.get("fault_type")

        # Map fault types to components
        component_map = {
            # UI Layer
            "hydra_config_error": {"hydra_config"},
            "ray_init_failure": {"ray_init"},
            "cli_arg_error": {"hydra_config"},
            "env_var_error": {"hydra_config"},
            # Orchestration Layer
            "ray_cluster_failure": {"ray_cluster", "gcs", "actor_system"},
            "actor_crash": {"actor_system", "resource_pool"},
            "gcs_failure": {"gcs", "actor_system"},
            "network_partition": {"ray_cluster", "actor_system"},
            # Worker Layer
            "fsdp_sync_failure": {"fsdp_workers", "gradient_sync"},
            "megatron_sync_failure": {"megatron_workers", "gradient_sync"},
            "gradient_sync_timeout": {"gradient_sync"},
            "cuda_oom_worker": {"fsdp_workers", "megatron_workers"},
            "worker_crash": {"fsdp_workers", "megatron_workers"},
            # Engine Layer
            "engine_init_failure": {"training_engine", "checkpoint_system"},
            "checkpoint_corruption": {"checkpoint_system"},
            "nccl_failure": {"nccl_comm", "training_engine"},
            "device_mesh_error": {"training_engine"},
            # Inference Layer
            "inference_oom": {"kv_cache", "vllm_backend", "sglang_backend"},
            "scheduler_deadlock": {"vllm_backend", "sglang_backend"},
            "compilation_failure": {"vllm_backend", "sglang_backend"},
        }

        # Get directly affected components
        direct_components = component_map.get(fault_type, set())
        affected.update(direct_components)

        # Add transitive dependencies
        for component in list(affected):
            if component in self._dependencies:
                dep = self._dependencies[component]

                # Add dependents (components that depend on this one)
                for dependent in dep.dependents:
                    affected.add(dependent)

                # Add dependencies (components this one depends on)
                for dependency in dep.depends_on:
                    affected.add(dependency)

        # Add layer-based components
        if layer:
            layer_components = {c for c, dep in self._dependencies.items() if dep.layer == layer}
            affected.update(layer_components)

        return affected

    def _calculate_impact_metrics(
        self, fault_result: FaultResult, affected_components: set[str], historical_window_hours: int
    ) -> ImpactMetrics:
        """Calculate impact metrics for the fault."""
        metrics = ImpactMetrics()

        # Basic metrics
        metrics.affected_components = affected_components
        metrics.fault_count = 1
        metrics.affected_layers = {
            self._dependencies[c].layer
            for c in affected_components
            if c in self._dependencies and self._dependencies[c].layer
        }

        # Duration impact
        duration = fault_result.duration or 0
        metrics.avg_impact_duration = duration
        metrics.max_impact_duration = duration

        # Failure and recovery rates
        if fault_result.status == FaultStatus.FAILED:
            metrics.failure_rate = 1.0
        else:
            metrics.failure_rate = 0.0

        if fault_result.recovery_time_ms is not None:
            metrics.recovery_rate = 1.0

        # Cascade probability
        metrics.cascade_probability = self._calculate_cascade_probability(fault_result, affected_components)

        # System degradation
        metrics.system_degradation = self._calculate_system_degradation(fault_result, affected_components)

        return metrics

    def _calculate_cascade_probability(self, fault_result: FaultResult, affected_components: set[str]) -> float:
        """Calculate the probability of fault cascade."""
        # Base probability
        base_probability = 0.1

        # Increase based on number of affected components
        component_factor = min(len(affected_components) * 0.1, 0.5)

        # Increase based on fault layer
        layer_multipliers = {
            FaultLayer.UI: 0.5,
            FaultLayer.ORCHESTRATION: 0.8,
            FaultLayer.WORKER: 0.6,
            FaultLayer.ENGINE: 0.7,
            FaultLayer.INFERENCE: 0.3,
        }
        layer_multiplier = layer_multipliers.get(fault_result.metadata.get("layer"), 0.5)

        # Increase based on criticality of affected components
        criticality_score = 0
        for component in affected_components:
            if component in self._dependencies:
                criticality = self._dependencies[component].criticality
                if criticality == "critical":
                    criticality_score += 0.3
                elif criticality == "high":
                    criticality_score += 0.2
                elif criticality == "medium":
                    criticality_score += 0.1

        # Calculate final probability
        probability = base_probability + component_factor
        probability *= layer_multiplier
        probability += min(criticality_score, 0.4)

        return min(probability, 1.0)

    def _calculate_system_degradation(self, fault_result: FaultResult, affected_components: set[str]) -> float:
        """Calculate system degradation level."""
        degradation = 0.0

        # Base degradation based on fault status
        if fault_result.status == FaultStatus.FAILED:
            degradation = 0.3
        elif fault_result.status == FaultStatus.COMPLETED:
            degradation = 0.1

        # Increase based on affected components
        for component in affected_components:
            if component in self._dependencies:
                criticality = self._dependencies[component].criticality
                if criticality == "critical":
                    degradation += 0.2
                elif criticality == "high":
                    degradation += 0.15
                elif criticality == "medium":
                    degradation += 0.1

        # Cap at 1.0
        return min(degradation, 1.0)

    def _assess_cascade_risk(self, fault_result: FaultResult, affected_components: set[str]) -> str:
        """Assess the cascade risk level."""
        # Calculate cascade probability
        cascade_prob = self._calculate_cascade_probability(fault_result, affected_components)

        # Determine risk level
        if cascade_prob > 0.7:
            return "critical"
        elif cascade_prob > 0.5:
            return "high"
        elif cascade_prob > 0.3:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, fault_result: FaultResult, impact_metrics: ImpactMetrics) -> list[str]:
        """Generate recommendations based on fault impact."""
        recommendations = []

        # High cascade risk recommendations
        if impact_metrics.cascade_probability > 0.5:
            recommendations.append("Consider isolating affected components to prevent cascade")
            recommendations.append("Monitor dependent components closely for secondary failures")

        # High degradation recommendations
        if impact_metrics.system_degradation > 0.5:
            recommendations.append("System degradation is significant - consider failover")
            recommendations.append("Review resource allocation and scaling policies")

        # Low recovery rate recommendations
        if impact_metrics.recovery_rate < 0.5:
            recommendations.append("Recovery mechanisms appear ineffective - review recovery strategies")
            recommendations.append("Consider implementing additional recovery mechanisms")

        # Critical component impact recommendations
        critical_components = {
            c
            for c in impact_metrics.affected_components
            if c in self._dependencies and self._dependencies[c].criticality == "critical"
        }
        if critical_components:
            recommendations.append(f"Critical components affected: {', '.join(critical_components)}")
            recommendations.append("Prioritize restoration of critical components")

        # Layer-specific recommendations
        if FaultLayer.ORCHESTRATION in impact_metrics.affected_layers:
            recommendations.append("Orchestration layer affected - check Ray cluster health")
        if FaultLayer.WORKER in impact_metrics.affected_layers:
            recommendations.append("Worker layer affected - verify worker processes and resources")
        if FaultLayer.ENGINE in impact_metrics.affected_layers:
            recommendations.append("Engine layer affected - check training engine status")
        if FaultLayer.INFERENCE in impact_metrics.affected_layers:
            recommendations.append("Inference layer affected - verify inference backends")

        return recommendations

    def _calculate_confidence(self, fault_result: FaultResult, historical_window_hours: int) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence
        confidence = 0.7

        # Increase based on historical data availability
        cutoff = datetime.now() - timedelta(hours=historical_window_hours)
        recent_faults = [f for f in self._historical_faults if f.timestamp >= cutoff]

        if len(recent_faults) > 10:
            confidence += 0.2
        elif len(recent_faults) > 5:
            confidence += 0.1

        # Increase based on fault information completeness
        if fault_result.target_info:
            confidence += 0.1
        if fault_result.error_message:
            confidence += 0.05

        # Cap at 1.0
        return min(confidence, 1.0)

    def add_historical_fault(self, fault_result: FaultResult) -> None:
        """Add a fault to historical data for pattern analysis."""
        self._historical_faults.append(fault_result)

        # Keep only recent faults (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self._historical_faults = [f for f in self._historical_faults if f.timestamp >= cutoff]

    def get_impact_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get a summary of fault impacts over a time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_faults = [f for f in self._historical_faults if f.timestamp >= cutoff]

        if not recent_faults:
            return {
                "total_faults": 0,
                "affected_layers": {},
                "cascade_risk": "low",
                "avg_degradation": 0.0,
                "recommendations": [],
            }

        # Analyze each fault
        analyses = []
        for fault in recent_faults:
            analysis = self.analyze_fault_impact(fault, hours)
            analyses.append(analysis)

        # Aggregate metrics
        total_faults = len(analyses)
        affected_layers = {}
        cascade_risks = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        total_degradation = 0.0

        for analysis in analyses:
            # Count by layer
            for layer in analysis.impact_metrics.affected_layers:
                affected_layers[layer.value] = affected_layers.get(layer.value, 0) + 1

            # Count cascade risks
            cascade_risks[analysis.cascade_risk] += 1

            # Sum degradation
            total_degradation += analysis.impact_metrics.system_degradation

        # Calculate recommendations
        recommendations = set()
        for analysis in analyses:
            recommendations.update(analysis.recommendations)

        return {
            "total_faults": total_faults,
            "affected_layers": affected_layers,
            "cascade_risk_distribution": cascade_risks,
            "avg_degradation": total_degradation / total_faults,
            "most_common_risk": max(cascade_risks, key=cascade_risks.get),
            "recommendations": list(recommendations)[:5],  # Top 5 recommendations
        }
