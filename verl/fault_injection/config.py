"""Fault injection configuration."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml


class TriggerType(Enum):
    PROBABILITY = "probability"
    TIMED = "timed"
    COUNT = "count"


class FaultType(Enum):
    # Network
    NETWORK_DELAY = "network_delay"
    NETWORK_LOSS = "network_loss"
    NETWORK_PARTITION = "network_partition"
    PORT_BLOCK = "port_block"
    # Process
    PROCESS_KILL = "process_kill"
    PROCESS_EXIT = "process_exit"
    # System
    DISK_FULL = "disk_full"
    FD_EXHAUST = "fd_exhaust"
    MEMORY_PRESSURE = "memory_pressure"
    # Code
    CODE_SLEEP = "code_sleep"
    CODE_OOM = "code_oom"
    CODE_NAN = "code_nan"
    ENV_MODIFY = "env_modify"


@dataclass
class TriggerConfig:
    type: TriggerType = TriggerType.TIMED
    probability: float = 1.0
    after_seconds: float = 0.0
    count: int = 1


@dataclass
class FaultSpec:
    type: FaultType
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    target_rank: Optional[int] = None
    params: dict = field(default_factory=dict)


@dataclass
class FaultConfig:
    name: str = "default"
    description: str = ""
    faults: list[FaultSpec] = field(default_factory=list)
    log_output_dir: str = "./fault_logs"

    @classmethod
    def from_yaml(cls, path: str) -> "FaultConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        faults = []
        for fault_data in data.get("faults", []):
            trigger_data = fault_data.get("trigger", {})
            trigger = TriggerConfig(
                type=TriggerType(trigger_data.get("type", "timed")),
                probability=trigger_data.get("probability", 1.0),
                after_seconds=_parse_duration(trigger_data.get("after", "0s")),
                count=trigger_data.get("count", 1),
            )
            faults.append(
                FaultSpec(
                    type=FaultType(fault_data["type"]),
                    trigger=trigger,
                    target_rank=fault_data.get("target_rank"),
                    params=fault_data.get("params", {}),
                )
            )

        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            faults=faults,
            log_output_dir=data.get("log_output_dir", "./fault_logs"),
        )


def _parse_duration(s: str) -> float:
    """Parse duration string like '60s', '5m', '1h' to seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().lower()
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    return float(s)
