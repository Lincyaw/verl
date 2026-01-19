"""Scenario configuration loader and validator."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .config import (
    BaseFaultConfig,
    FaultConfig,
    FaultDependencyConfig,
    FaultLayer,
    FaultScenarioConfig,
    FaultScenarioTemplate,
    FaultTrigger,
    FaultTriggerConfig,
    FaultType,
)


logger = logging.getLogger(__name__)


class ScenarioConfigLoader:
    """Loads and validates fault scenario configurations."""

    def __init__(self):
        self._fault_config_map = {
            # Process faults
            FaultType.PROCESS_KILL: "ProcessFaultConfig",
            FaultType.PROCESS_EXIT: "ProcessFaultConfig",
            FaultType.PROCESS_HANG: "ProcessFaultConfig",
            # Network faults
            FaultType.NETWORK_DELAY: "NetworkFaultConfig",
            FaultType.NETWORK_LOSS: "NetworkFaultConfig",
            FaultType.NETWORK_PARTITION: "NetworkFaultConfig",
            # Resource faults
            FaultType.MEMORY_OOM: "ResourceFaultConfig",
            FaultType.MEMORY_LEAK: "ResourceFaultConfig",
            FaultType.DISK_FULL: "ResourceFaultConfig",
            FaultType.FD_EXHAUSTION: "ResourceFaultConfig",
            # CUDA faults
            FaultType.CUDA_OOM: "CudaFaultConfig",
            FaultType.CUDA_ERROR: "CudaFaultConfig",
            FaultType.NCCL_FAILURE: "CudaFaultConfig",
            # Configuration faults
            FaultType.CONFIG_CORRUPTION: "ConfigFaultConfig",
            FaultType.ENV_VAR_MISSING: "ConfigFaultConfig",
            # UI faults
            FaultType.HYDRA_CONFIG_ERROR: "UIFaultConfig",
            FaultType.RAY_INIT_FAILURE: "UIFaultConfig",
            FaultType.CLI_ARG_ERROR: "UIFaultConfig",
            FaultType.ENV_VAR_ERROR: "UIFaultConfig",
            FaultType.UI_FREEZE: "UIFaultConfig",
            FaultType.UI_CRASH: "UIFaultConfig",
            # Orchestration faults
            FaultType.RAY_CLUSTER_FAILURE: "OrchestrationFaultConfig",
            FaultType.ACTOR_CRASH: "OrchestrationFaultConfig",
            FaultType.ACTOR_DEATH: "OrchestrationFaultConfig",
            FaultType.RESOURCE_EXHAUSTION: "OrchestrationFaultConfig",
            FaultType.TASK_SCHEDULING_FAILURE: "OrchestrationFaultConfig",
            FaultType.RESOURCE_POOL_FAULT: "OrchestrationFaultConfig",
            FaultType.GCS_FAILURE: "OrchestrationFaultConfig",
            FaultType.PLACEMENT_GROUP_FAULT: "OrchestrationFaultConfig",
            # Worker faults
            FaultType.FSDP_SYNC_FAILURE: "WorkerFaultConfig",
            FaultType.FSDP_SHARDING_ERROR: "WorkerFaultConfig",
            FaultType.MEGATRON_SYNC_FAILURE: "WorkerFaultConfig",
            FaultType.MEGATRON_PIPELINE_ERROR: "WorkerFaultConfig",
            FaultType.GRADIENT_SYNC_TIMEOUT: "WorkerFaultConfig",
            FaultType.GRADIENT_NAN: "WorkerFaultConfig",
            FaultType.PARAMETER_NAN: "WorkerFaultConfig",
            FaultType.CUDA_OOM_WORKER: "WorkerFaultConfig",
            FaultType.CUDA_MEMORY_FRAGMENTATION: "WorkerFaultConfig",
            FaultType.WORKER_CRASH: "WorkerFaultConfig",
            FaultType.WORKER_HANG: "WorkerFaultConfig",
            # Engine faults
            FaultType.ENGINE_INIT_FAILURE: "EngineFaultConfig",
            FaultType.ENGINE_HANG: "EngineFaultConfig",
            FaultType.CHECKPOINT_CORRUPTION: "EngineFaultConfig",
            FaultType.DEVICE_MESH_ERROR: "EngineFaultConfig",
            FaultType.PRECISION_ERROR: "EngineFaultConfig",
            FaultType.DEVICE_MAP_ERROR: "EngineFaultConfig",
            # Inference faults
            FaultType.INFERENCE_OOM: "InferenceFaultConfig",
            FaultType.SCHEDULER_DEADLOCK: "InferenceFaultConfig",
            FaultType.COMPILATION_FAILURE: "InferenceFaultConfig",
        }

    def load_scenario_from_yaml(self, path: Union[str, Path]) -> FaultScenarioConfig:
        """Load a fault scenario from YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle multiple scenarios in one file
        if len(data) == 1:
            scenario_name = list(data.keys())[0]
            scenario_data = data[scenario_name]
        else:
            raise ValueError("YAML file must contain exactly one scenario")

        return self.parse_scenario_config(scenario_name, scenario_data)

    def load_scenarios_from_yaml(self, path: Union[str, Path]) -> Dict[str, FaultScenarioConfig]:
        """Load multiple fault scenarios from YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        scenarios = {}
        for scenario_name, scenario_data in data.items():
            scenarios[scenario_name] = self.parse_scenario_config(scenario_name, scenario_data)

        return scenarios

    def parse_scenario_config(self, name: str, data: Dict[str, Any]) -> FaultScenarioConfig:
        """Parse scenario configuration from dictionary."""

        # Parse faults
        faults = []
        for fault_data in data.get("faults", []):
            fault = self.parse_fault_config(fault_data)
            faults.append(fault)

        # Parse dependencies
        dependencies = []
        for dep_data in data.get("dependencies", []):
            dep = FaultDependencyConfig(
                fault_name=dep_data["fault_name"],
                condition=dep_data["condition"],
                delay_seconds=dep_data.get("delay_seconds")
            )
            dependencies.append(dep)

        return FaultScenarioConfig(
            name=name,
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            faults=faults,
            dependencies=dependencies,
            cascade_mode=data.get("cascade_mode", "none"),
            cascade_interval_seconds=data.get("cascade_interval_seconds", 5.0),
            max_cascade_depth=data.get("max_cascade_depth", 3),
            stop_on_failure=data.get("stop_on_failure", False)
        )

    def parse_fault_config(self, data: Dict[str, Any]) -> FaultConfig:
        """Parse fault configuration from dictionary."""

        # Extract basic fields
        name = data["name"]
        fault_type = FaultType(data["type"])
        layer = FaultLayer(data["layer"])

        # Create base config
        base_config = BaseFaultConfig(
            name=name,
            layer=layer,
            type=fault_type,
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            duration_seconds=data.get("duration_seconds"),
            recovery_seconds=data.get("recovery_seconds"),
            metadata=data.get("metadata", {})
        )

        # Parse trigger config
        if "trigger" in data:
            trigger_data = data["trigger"]
            base_config.trigger = FaultTriggerConfig(
                type=FaultTrigger(trigger_data["type"]),
                delay_seconds=trigger_data.get("delay_seconds"),
                count_threshold=trigger_data.get("count_threshold"),
                probability=trigger_data.get("probability"),
                condition=trigger_data.get("condition")
            )

        # Parse target config
        if "target" in data:
            target_data = data["target"]
            from .config import FaultTarget, FaultTargetConfig
            base_config.target = FaultTargetConfig(
                mode=FaultTarget(target_data["mode"]),
                ranks=target_data.get("ranks"),
                hosts=target_data.get("hosts"),
                process_types=target_data.get("process_types"),
                count=target_data.get("count", 1)
            )

        # Add fault-specific fields
        for key, value in data.items():
            if key not in ["name", "type", "layer", "enabled", "description",
                          "duration_seconds", "recovery_seconds", "metadata",
                          "trigger", "target"]:
                setattr(base_config, key, value)

        return base_config

    def save_scenario_to_yaml(self, scenario: FaultScenarioConfig, path: Union[str, Path]) -> None:
        """Save scenario configuration to YAML file."""
        path = Path(path)

        data = {
            scenario.name: {
                "description": scenario.description,
                "enabled": scenario.enabled,
                "cascade_mode": scenario.cascade_mode,
                "cascade_interval_seconds": scenario.cascade_interval_seconds,
                "max_cascade_depth": scenario.max_cascade_depth,
                "stop_on_failure": scenario.stop_on_failure,
                "faults": [self.fault_to_dict(fault) for fault in scenario.faults],
                "dependencies": [
                    {
                        "fault_name": dep.fault_name,
                        "condition": dep.condition,
                        "delay_seconds": dep.delay_seconds
                    }
                    for dep in scenario.dependencies
                ]
            }
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def fault_to_dict(self, fault: FaultConfig) -> Dict[str, Any]:
        """Convert fault config to dictionary."""
        data = {
            "name": fault.name,
            "type": fault.type.value,
            "layer": fault.layer.value,
            "enabled": fault.enabled,
        }

        if fault.description:
            data["description"] = fault.description

        if fault.duration_seconds:
            data["duration_seconds"] = fault.duration_seconds

        if fault.recovery_seconds:
            data["recovery_seconds"] = fault.recovery_seconds

        # Add trigger
        if hasattr(fault, 'trigger') and fault.trigger:
            data["trigger"] = {
                "type": fault.trigger.type.value
            }
            if fault.trigger.delay_seconds:
                data["trigger"]["delay_seconds"] = fault.trigger.delay_seconds
            if fault.trigger.count_threshold:
                data["trigger"]["count_threshold"] = fault.trigger.count_threshold
            if fault.trigger.probability:
                data["trigger"]["probability"] = fault.trigger.probability
            if fault.trigger.condition:
                data["trigger"]["condition"] = fault.trigger.condition

        # Add target
        if hasattr(fault, 'target') and fault.target:
            data["target"] = {
                "mode": fault.target.mode.value
            }
            if fault.target.ranks:
                data["target"]["ranks"] = fault.target.ranks
            if fault.target.hosts:
                data["target"]["hosts"] = fault.target.hosts
            if fault.target.process_types:
                data["target"]["process_types"] = fault.target.process_types
            if fault.target.count != 1:
                data["target"]["count"] = fault.target.count

        # Add fault-specific fields
        for key, value in fault.__dict__.items():
            if key not in ["name", "layer", "type", "enabled", "description",
                          "duration_seconds", "recovery_seconds", "metadata",
                          "trigger", "target", "_target", "_trigger"]:
                if value is not None and not key.startswith('_'):
                    data[key] = value

        return data

    def create_template_library(self) -> Dict[str, FaultScenarioTemplate]:
        """Create a library of common fault scenario templates."""

        templates = {}

        # System failure cascade template
        templates["system_failure_cascade"] = FaultScenarioTemplate(
            name="system_failure_cascade",
            description="Simulate cascading system failures",
            category="system",
            faults=[
                {
                    "name": "memory_pressure",
                    "type": "memory_oom",
                    "layer": "orchestration",
                    "memory_mb": 4096,
                    "target": {"mode": "random", "count": 2}
                },
                {
                    "name": "network_partition",
                    "type": "network_partition",
                    "layer": "orchestration",
                    "partition_type": "partial",
                    "partition_duration": 30.0,
                    "target": {"mode": "rank", "ranks": [0, 1]}
                },
                {
                    "name": "worker_crash",
                    "type": "worker_crash",
                    "layer": "worker",
                    "worker_type": "actor",
                    "crash_method": "exception",
                    "target": {"mode": "random", "count": 1}
                }
            ],
            dependencies=[
                {
                    "fault_name": "network_partition",
                    "condition": "on_failure",
                    "delay_seconds": 5.0
                },
                {
                    "fault_name": "worker_crash",
                    "condition": "after",
                    "delay_seconds": 10.0
                }
            ],
            cascade_config={
                "mode": "tree",
                "interval": 5.0,
                "max_depth": 3
            }
        )

        # Training disruption template
        templates["training_disruption"] = FaultScenarioTemplate(
            name="training_disruption",
            description="Simulate common training disruptions",
            category="training",
            faults=[
                {
                    "name": "gradient_nan",
                    "type": "gradient_nan",
                    "layer": "worker",
                    "gradient_nan_probability": 1.0,
                    "target": {"mode": "all"}
                },
                {
                    "name": "checkpoint_corruption",
                    "type": "checkpoint_corruption",
                    "layer": "engine",
                    "corruption_type": "random_bytes",
                    "checkpoint_path": "/tmp/checkpoint.pt",
                    "target": {"mode": "random", "count": 1}
                },
                {
                    "name": "nccl_timeout",
                    "type": "nccl_failure",
                    "layer": "engine",
                    "nccl_error_type": "timeout",
                    "nccl_timeout_ms": 30000,
                    "target": {"mode": "random", "count": 2}
                }
            ],
            dependencies=[
                {
                    "fault_name": "checkpoint_corruption",
                    "condition": "on_success",
                    "delay_seconds": 60.0
                }
            ],
            cascade_config={
                "mode": "linear",
                "interval": 30.0
            }
        )

        # Inference degradation template
        templates["inference_degradation"] = FaultScenarioTemplate(
            name="inference_degradation",
            description="Simulate inference service degradation",
            category="inference",
            faults=[
                {
                    "name": "inference_oom",
                    "type": "inference_oom",
                    "layer": "inference",
                    "backend": "vllm",
                    "kv_cache_size_mb": 2048,
                    "target": {"mode": "random", "count": 1}
                },
                {
                    "name": "scheduler_deadlock",
                    "type": "scheduler_deadlock",
                    "layer": "inference",
                    "deadlock_type": "request_queue",
                    "target": {"mode": "all"}
                },
                {
                    "name": "compilation_failure",
                    "type": "compilation_failure",
                    "layer": "inference",
                    "compilation_stage": "graph_capture",
                    "target": {"mode": "random", "count": 1}
                }
            ],
            dependencies=[],
            cascade_config={
                "mode": "burst",
                "interval": 0.0
            }
        )

        return templates