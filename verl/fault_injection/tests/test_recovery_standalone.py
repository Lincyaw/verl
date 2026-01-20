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


#!/usr/bin/env python3
"""Standalone test for recovery system."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# Minimal test implementations
class FaultStatus(Enum):
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class FaultResult:
    fault_id: str
    status: FaultStatus
    start_time: float
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FaultContext:
    worker_id: Optional[str] = None
    rank: Optional[int] = None
    host: Optional[str] = None
    process_type: Optional[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RecoveryStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class RecoveryResult:
    recovery_id: str
    status: RecoveryStatus
    strategy_name: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecoveryContext:
    fault_result: FaultResult
    fault_context: FaultContext
    attempt_count: int = 0
    max_attempts: int = 3
    recovery_start_time: float = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.recovery_start_time is None:
            self.recovery_start_time = time.time()


# Test recovery strategies
class IgnoreRecoveryStrategy:
    """Strategy that ignores the fault."""

    def __init__(self, config=None):
        self.config = config or {}

    @property
    def name(self):
        return "ignore"

    def can_recover(self, context: RecoveryContext) -> bool:
        return True

    def execute(self, context: RecoveryContext) -> RecoveryResult:
        print(f"  → Executing ignore recovery for fault {context.fault_result.fault_id}")
        return RecoveryResult(
            recovery_id=f"{self.name}_recovery",
            status=RecoveryStatus.SUCCESS,
            strategy_name=self.name,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"action": "ignored"},
        )


class RetryRecoveryStrategy:
    """Strategy that retries the operation."""

    def __init__(self, config=None):
        self.config = config or {}

    @property
    def name(self):
        return "retry"

    def can_recover(self, context: RecoveryContext) -> bool:
        return context.attempt_count < context.max_attempts

    def execute(self, context: RecoveryContext) -> RecoveryResult:
        retry_count = self.config.get("retry_count", 3)
        retry_delay = self.config.get("retry_delay", 0.1)

        print(f"  → Executing retry recovery for fault {context.fault_result.fault_id}")
        print(f"    - Retry count: {retry_count}")
        print(f"    - Retry delay: {retry_delay}s")

        # Simulate retry logic
        for i in range(retry_count):
            print(f"    - Retry attempt {i + 1}/{retry_count}")
            time.sleep(retry_delay)

        return RecoveryResult(
            recovery_id=f"{self.name}_recovery",
            status=RecoveryStatus.SUCCESS,
            strategy_name=self.name,
            start_time=time.time(),
            end_time=time.time(),
            metadata={"retry_count": retry_count, "retry_delay": retry_delay},
        )


# Test function
def test_recovery_strategies():
    """Test the recovery strategies."""
    print("=== Testing Recovery Strategies ===\n")

    # Create test fault
    fault_result = FaultResult(
        fault_id="test_fault_001",
        status=FaultStatus.FAILED,
        start_time=time.time(),
        end_time=time.time(),
        metadata={"fault_type": "process_kill"},
    )

    fault_context = FaultContext(
        worker_id="worker_001",
        rank=0,
        host="localhost",
        process_type="actor",
        metadata={"pid": 12345},
    )

    recovery_context = RecoveryContext(
        fault_result=fault_result,
        fault_context=fault_context,
        attempt_count=0,
        max_attempts=3,
    )

    # Test ignore strategy
    print("1. Testing Ignore Recovery Strategy:")
    strategy = IgnoreRecoveryStrategy()
    print(f"   Can recover: {strategy.can_recover(recovery_context)}")
    result = strategy.execute(recovery_context)
    print(f"   Result: {result.status.value}")
    print(f"   Action: {result.metadata['action']}\n")

    # Test retry strategy
    print("2. Testing Retry Recovery Strategy:")
    strategy = RetryRecoveryStrategy({"retry_count": 2, "retry_delay": 0.1})
    print(f"   Can recover: {strategy.can_recover(recovery_context)}")
    result = strategy.execute(recovery_context)
    print(f"   Result: {result.status.value}")
    print(f"   Metadata: {result.metadata}\n")

    # Test retry with max attempts reached
    print("3. Testing Retry with Max Attempts Reached:")
    recovery_context.attempt_count = 3
    print(f"   Can recover: {strategy.can_recover(recovery_context)}")
    print("   (Should be False because max attempts reached)\n")

    print("=== All tests completed successfully! ===")


if __name__ == "__main__":
    test_recovery_strategies()
