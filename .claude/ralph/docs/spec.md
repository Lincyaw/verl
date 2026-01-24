# Ralph: 面向分布式 RL 系统 (Verl) 的全栈运行时故障注入与 AIOps 数据集构建

## 1. 任务目标 (Objective)

构建一套**非侵入式的运行时故障注入系统 (Ralph)**，针对分布式强化学习框架 **Verl** 及其核心依赖栈（**Ray, PyTorch/FSDP**），在单机多卡环境下进行受控的故障模拟。

核心目的是**诱导（Induce）**而非**伪造（Fabricate）**系统行为：通过在运行时劫持控制流或数据流，迫使 Verl 及其底层组件进入既有的异常处理分支、容错恢复路径或崩溃路径，从而捕获真实、高保真的日志与指标，构建用于 AIOps/RCA 研究的标准数据集。

---

## 2. 系统架构 (System Architecture)

```
+-----------------------------------------------------------------------------------+
|                              RALPH FAULT INJECTION SYSTEM                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +----------------+     +-------------------+     +-------------------------+     |
|  |  YAML Config   |---->| Injection Engine  |---->|  Transparent Proxies    |     |
|  |  (Scenarios)   |     | (Orchestrator)    |     |  (Per-Layer Hooks)      |     |
|  +----------------+     +-------------------+     +-------------------------+     |
|                               |                           |                       |
|                               v                           v                       |
|                    +-------------------+         +-------------------+            |
|                    | Trigger Scheduler |         | Proxy Registry    |            |
|                    | (Step/Prob/Time)  |         | (Active Hooks)    |            |
|                    +-------------------+         +-------------------+            |
|                                                                                   |
+-----------------------------------------------------------------------------------+
        |                           |                           |
        v                           v                           v
+---------------+         +------------------+         +------------------+
|  L0: Ray      |         |  L1: FSDP/Dist   |         |  L2: Verl        |
|  Scheduling   |         |  Communication   |         |  Business Logic  |
+---------------+         +------------------+         +------------------+
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |  L3: Resource Layer           |
                    |  (GPU Memory, CPU, KV Cache)  |
                    +-------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                          DUAL-STREAM DATA COLLECTION                              |
+-----------------------------------------------------------------------------------+
|  Stream A: System Telemetry          |  Stream B: Ground Truth Labels             |
|  - Ray logs (/tmp/ray/*)             |  - Injection timestamp                     |
|  - verl Tracking metrics             |  - Fault type, target, severity            |
|  - NCCL/CUDA logs                    |  - Expected vs actual behavior             |
|  - Throughput (TPS)                  |  - Recovery actions                        |
+--------------------------------------+--------------------------------------------+
                                    |
                                    v
                    +-------------------------------+
                    |  Unified Dataset (JSONL)      |
                    |  For AIOps/RCA Research       |
                    +-------------------------------+
```

---

## 3. 环境设定 (Environment Setup)

| 配置项 | 值 |
|--------|-----|
| **目标系统 (SUT)** | Verl (volcengine/verl) |
| **运行模式** | Single Node, Multi-GPU |
| **依赖栈** | Ray Core, PyTorch FSDP, vLLM/SGLang |
| **约束条件** | 不依赖 K8s/Chaos Mesh，纯 Python Monkey Patching |

### 技术栈层级

| 层级 | 组件 | verl 中的核心模块 |
|------|------|------------------|
| **L2** | Verl Application | `RayPPOTrainer`, `ActorRolloutRefWorker`, `RewardManager` |
| **L1** | PyTorch Distributed | FSDP Engine, `torch.distributed.*` |
| **L0** | Ray Scheduling | `RayWorkerGroup`, `@register` decorator |
| **L-1** | Infrastructure | CUDA, Linux OS, Hardware |

---

## 4. 故障注入点概览 (Injection Points Overview)

详细的注入点规格、策略和实现 checklist 请参见 [injection_checklist.md](./injection_checklist.md)。

### 注入层级分类

| 层级 | 类别 | 注入点数量 | 策略数量 |
|------|------|-----------|---------|
| **L0** | Ray 调度层 | 3 | 15 |
| **L1** | PyTorch 分布式通信层 | 3 | 13 |
| **L2** | Verl 业务逻辑层 | 5 | 20 |
| **L3** | 资源层 | 3 | 7 |
| **扩展** | 算法层 (GAE/GRPO/KL) | 5 | 17 |
| **扩展** | 数据管道层 (DataProto) | 4 | 13 |
| **扩展** | 推理引擎层 (vLLM/SGLang) | 4 | 15 |
| **扩展** | Critic Worker | 2 | 8 |
| **扩展** | 多轮对话/Agent | 3 | 11 |
| **扩展** | 分词与序列处理 | 4 | 14 |
| **扩展** | Megatron 特定 | 4 | 16 |
| **扩展** | 优化器与 LR 调度 | 3 | 11 |
| **总计** | - | **37** | **169** |

### 实现优先级

| 优先级 | 注入点 | 描述 |
|--------|--------|------|
| **P0** | ray.get, all_reduce, RewardManager, save_checkpoint | 核心路径，最易触发系统异常 |
| **P1** | GAE, KL, generate, update_weights | 算法正确性、推理引擎 |
| **P2** | DataProto, CriticWorker, Optimizer | 数据流、Critic、优化 |
| **P3** | Megatron, Agent, Tokenization | 特定场景、边缘功能 |

---

## 5. 架构设计: Mixin + 继承模式 (Architecture Design)

### 5.1 设计原则

采用 **Mixin + 继承** 模式实现故障注入策略，核心原则：

1. **通用策略复用**: 通过 Mixin 类提供可复用的通用故障操作（delay, corrupt, exception）
2. **特有策略扩展**: 每个 Proxy 类可以实现注入点特有的策略
3. **类型安全**: 每个 Proxy 明确声明支持的策略类型
4. **可测试性**: Mixin 可以独立测试
5. **渐进式开发**: 先实现通用 Mixin，再逐步添加特有策略

### 5.2 策略分类

| 策略类型 | 实现方式 | 适用范围 | 示例 |
|----------|----------|----------|------|
| **通用策略** | Mixin 类 | 所有注入点 | `delay`, `corrupt_tensor`, `raise_exception`, `skip` |
| **特有策略** | Proxy 子类方法 | 特定注入点 | `reward_flip`, `object_lost`, `deadlock` |

### 5.3 项目结构

```
ralph/
├── core/
│   ├── __init__.py
│   ├── config.py                    # FaultConfig, TriggerConfig
│   ├── engine.py                    # InjectionEngine (编排器)
│   ├── scheduler.py                 # TriggerScheduler
│   └── registry.py                  # ProxyRegistry (代理注册表)
│
├── mixins/                          # 通用策略 Mixin
│   ├── __init__.py
│   ├── delay.py                     # DelayMixin
│   ├── tensor.py                    # TensorCorruptionMixin
│   ├── exception.py                 # ExceptionMixin
│   ├── skip.py                      # SkipMixin
│   └── result.py                    # ResultModificationMixin
│
├── proxies/                         # 具体代理实现
│   ├── __init__.py
│   ├── base.py                      # BaseProxy
│   ├── l0_ray.py                    # Ray 层代理
│   ├── l1_distributed.py            # FSDP/Distributed 层代理
│   ├── l2_verl.py                   # Verl 业务层代理
│   ├── l3_resource.py               # 资源层代理
│   ├── algorithm.py                 # 算法层代理 (GAE, KL)
│   ├── data_pipeline.py             # 数据管道代理
│   ├── inference.py                 # 推理引擎代理
│   └── megatron.py                  # Megatron 特定代理
│
├── collectors/                      # 数据采集
│   ├── __init__.py
│   ├── dual_stream.py
│   ├── metrics_hook.py
│   └── log_parser.py
│
└── cli/
    ├── __init__.py
    └── main.py
```

### 5.4 核心类实现

#### 5.4.1 配置类 (FaultConfig)

```python
# ralph/core/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class TriggerType(Enum):
    ONE_SHOT = "one_shot"          # 在指定 step 触发一次
    STEP_BASED = "step_based"      # 在 step 范围内触发
    PROBABILISTIC = "probabilistic" # 概率触发
    PERIODIC = "periodic"          # 周期性触发

class StrategyType(Enum):
    # 通用策略 (由 Mixin 提供)
    DELAY = "delay"
    CORRUPT_TENSOR = "corrupt_tensor"
    INJECT_NAN = "inject_nan"
    INJECT_INF = "inject_inf"
    RAISE_EXCEPTION = "raise_exception"
    SKIP = "skip"
    REPEAT = "repeat"
    MODIFY_RESULT = "modify_result"

    # 特有策略 (由 Proxy 子类实现)
    # Ray 特有
    OBJECT_LOST = "object_lost"
    PARTIAL_FAILURE = "partial_failure"
    WORKER_DEATH = "worker_death"
    # FSDP 特有
    DEADLOCK = "deadlock"
    # Reward 特有
    REWARD_FLIP = "reward_flip"
    CONSTANT_REWARD = "constant_reward"
    # ... 更多特有策略

@dataclass
class TriggerConfig:
    """触发条件配置"""
    type: TriggerType
    at_step: Optional[int] = None           # ONE_SHOT
    start_step: Optional[int] = None        # STEP_BASED
    end_step: Optional[int] = None          # STEP_BASED
    probability: float = 1.0                # PROBABILISTIC
    every_n_steps: Optional[int] = None     # PERIODIC

    def should_trigger(self, current_step: int, rng=None) -> bool:
        if self.type == TriggerType.ONE_SHOT:
            return current_step == self.at_step
        elif self.type == TriggerType.STEP_BASED:
            return self.start_step <= current_step <= self.end_step
        elif self.type == TriggerType.PROBABILISTIC:
            import random
            r = rng if rng else random
            return r.random() < self.probability
        elif self.type == TriggerType.PERIODIC:
            return current_step % self.every_n_steps == 0
        return False

@dataclass
class FaultConfig:
    """故障注入配置"""
    id: str                                  # 唯一标识
    strategy: StrategyType                   # 策略类型
    trigger: TriggerConfig                   # 触发条件
    parameters: Dict[str, Any] = field(default_factory=dict)  # 策略参数
    enabled: bool = True                     # 是否启用
    severity: str = "medium"                 # low, medium, high, critical
    expected_behavior: str = ""              # 预期行为描述
```

#### 5.4.2 通用策略 Mixin

```python
# ralph/mixins/delay.py
import time
from typing import Any

class DelayMixin:
    """提供延迟注入能力"""

    def _apply_delay(self, seconds: float) -> None:
        """在操作前/后添加延迟"""
        time.sleep(seconds)

    def _strategy_delay(self, *args, **kwargs) -> Any:
        """延迟策略: 先延迟，再执行原操作"""
        delay_seconds = self._config.parameters.get("delay_seconds", 10)
        self._apply_delay(delay_seconds)
        return self._original(*args, **kwargs)
```

```python
# ralph/mixins/tensor.py
import torch
from typing import Union

class TensorCorruptionMixin:
    """提供张量损坏能力"""

    def _corrupt_tensor(self, tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(tensor.float()) * noise_scale
        return tensor + noise.to(tensor.dtype)

    def _inject_nan(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """注入 NaN 值"""
        mask = torch.rand_like(tensor.float()) < ratio
        tensor = tensor.clone()
        tensor[mask] = float('nan')
        return tensor

    def _inject_inf(self, tensor: torch.Tensor, ratio: float,
                    positive: bool = True) -> torch.Tensor:
        """注入 Inf 值"""
        mask = torch.rand_like(tensor.float()) < ratio
        tensor = tensor.clone()
        tensor[mask] = float('inf') if positive else float('-inf')
        return tensor

    def _scale_tensor(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """缩放张量"""
        return tensor * scale

    def _strategy_corrupt_tensor(self, *args, **kwargs):
        """张量损坏策略: 执行原操作，损坏结果中的张量"""
        result = self._original(*args, **kwargs)
        noise_scale = self._config.parameters.get("noise_scale", 0.1)
        return self._corrupt_result_tensors(result, noise_scale)

    def _corrupt_result_tensors(self, result, noise_scale: float):
        """递归损坏结果中的张量"""
        if isinstance(result, torch.Tensor):
            return self._corrupt_tensor(result, noise_scale)
        elif isinstance(result, dict):
            return {k: self._corrupt_result_tensors(v, noise_scale)
                    for k, v in result.items()}
        elif isinstance(result, (list, tuple)):
            corrupted = [self._corrupt_result_tensors(v, noise_scale) for v in result]
            return type(result)(corrupted)
        return result
```

```python
# ralph/mixins/exception.py
from typing import Type, Optional

class ExceptionMixin:
    """提供异常注入能力"""

    # 预定义的异常映射
    EXCEPTION_MAP = {
        "IOError": IOError,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "PermissionError": PermissionError,
        "TimeoutError": TimeoutError,
        "FileNotFoundError": FileNotFoundError,
        "OSError": OSError,
    }

    def _raise_exception(self,
                         exc_type: Union[str, Type[Exception]],
                         message: str,
                         **exc_kwargs) -> None:
        """抛出指定异常"""
        if isinstance(exc_type, str):
            exc_class = self.EXCEPTION_MAP.get(exc_type, RuntimeError)
        else:
            exc_class = exc_type
        raise exc_class(message, **exc_kwargs)

    def _strategy_raise_exception(self, *args, **kwargs):
        """异常策略: 抛出配置的异常"""
        exc_type = self._config.parameters.get("exception_type", "RuntimeError")
        message = self._config.parameters.get("message", "Ralph injected fault")
        self._raise_exception(exc_type, message)
```

```python
# ralph/mixins/skip.py
from typing import Any, Optional

class SkipMixin:
    """提供跳过/重复操作能力"""

    def _skip_operation(self, default_return: Any = None) -> Any:
        """跳过操作，返回默认值"""
        return default_return

    def _repeat_operation(self, times: int, *args, **kwargs) -> Any:
        """重复执行操作"""
        result = None
        for _ in range(times):
            result = self._original(*args, **kwargs)
        return result

    def _strategy_skip(self, *args, **kwargs):
        """跳过策略"""
        default = self._config.parameters.get("default_return", None)
        return self._skip_operation(default)

    def _strategy_repeat(self, *args, **kwargs):
        """重复策略"""
        times = self._config.parameters.get("repeat_times", 2)
        return self._repeat_operation(times, *args, **kwargs)
```

```python
# ralph/mixins/result.py
from typing import Any, Callable

class ResultModificationMixin:
    """提供结果修改能力"""

    def _modify_result(self, result: Any, modifier: Callable[[Any], Any]) -> Any:
        """使用自定义函数修改结果"""
        return modifier(result)

    def _negate_result(self, result: Any) -> Any:
        """取反结果 (适用于数值/张量)"""
        import torch
        if isinstance(result, (int, float)):
            return -result
        elif isinstance(result, torch.Tensor):
            return -result
        elif isinstance(result, dict):
            return {k: self._negate_result(v) for k, v in result.items()}
        return result

    def _set_constant(self, result: Any, value: Any) -> Any:
        """将结果设为常数"""
        import torch
        if isinstance(result, torch.Tensor):
            return torch.full_like(result, value)
        elif isinstance(result, dict):
            return {k: self._set_constant(v, value) for k, v in result.items()}
        return value
```

#### 5.4.3 BaseProxy 基类

```python
# ralph/proxies/base.py
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Set, Dict
import functools

from ralph.core.config import FaultConfig, StrategyType
from ralph.collectors import DualStreamCollector

class BaseProxy(ABC):
    """
    所有透明代理的基类。

    子类需要:
    1. 继承所需的 Mixin 类
    2. 实现 _get_layer() 方法
    3. 声明 SUPPORTED_STRATEGIES
    4. 实现特有策略方法 (_strategy_xxx)
    """

    # 子类需要声明支持的策略类型
    SUPPORTED_STRATEGIES: Set[StrategyType] = set()

    def __init__(self, original_fn: Callable, collector: DualStreamCollector):
        self._original = original_fn
        self._collector = collector
        self._config: Optional[FaultConfig] = None
        self._current_step = 0

        # 构建策略方法映射
        self._strategy_methods: Dict[StrategyType, Callable] = {}
        self._build_strategy_map()

    def _build_strategy_map(self):
        """构建策略类型到方法的映射"""
        for strategy in self.SUPPORTED_STRATEGIES:
            method_name = f"_strategy_{strategy.value}"
            if hasattr(self, method_name):
                self._strategy_methods[strategy] = getattr(self, method_name)

    def set_config(self, config: FaultConfig):
        """设置故障配置"""
        if config.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Strategy {config.strategy} not supported by {self.__class__.__name__}. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )
        self._config = config

    def set_step(self, step: int):
        """设置当前训练步数"""
        self._current_step = step

    def __call__(self, *args, **kwargs) -> Any:
        """透明代理调用入口"""
        # 检查是否应该注入
        if self._should_inject():
            fault_id = self._record_injection_start()
            try:
                result = self._execute_strategy(*args, **kwargs)
                self._record_injection_end(fault_id, "success")
                return result
            except Exception as e:
                self._record_injection_end(fault_id, f"exception: {type(e).__name__}: {e}")
                raise

        # 正常执行
        return self._original(*args, **kwargs)

    def _should_inject(self) -> bool:
        """检查是否应该注入故障"""
        if not self._config or not self._config.enabled:
            return False
        return self._config.trigger.should_trigger(self._current_step)

    def _execute_strategy(self, *args, **kwargs) -> Any:
        """执行配置的策略"""
        strategy = self._config.strategy

        if strategy in self._strategy_methods:
            return self._strategy_methods[strategy](*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Strategy {strategy} declared but not implemented in {self.__class__.__name__}"
            )

    def _record_injection_start(self) -> str:
        """记录注入开始"""
        return self._collector.record_fault_injection(
            fault_type=self._config.strategy.value,
            target_layer=self._get_layer(),
            target_function=self._get_target_name(),
            severity=self._config.severity,
            parameters=self._config.parameters,
            expected_behavior=self._config.expected_behavior,
        )

    def _record_injection_end(self, fault_id: str, outcome: str):
        """记录注入结束"""
        self._collector.record_fault_outcome(fault_id, outcome, duration_ms=0)

    @abstractmethod
    def _get_layer(self) -> str:
        """返回注入点所属层级 (L0, L1, L2, L3, Algorithm, etc.)"""
        raise NotImplementedError

    def _get_target_name(self) -> str:
        """返回目标函数名"""
        return getattr(self._original, '__name__', str(self._original))
```

#### 5.4.4 具体代理实现示例

```python
# ralph/proxies/l0_ray.py
import ray
from ralph.proxies.base import BaseProxy
from ralph.mixins.delay import DelayMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.core.config import StrategyType

class RayGetProxy(BaseProxy, DelayMixin, ExceptionMixin, TensorCorruptionMixin):
    """
    ray.get 的透明代理

    支持的策略:
    - 通用: delay, raise_exception, corrupt_tensor
    - 特有: object_lost, partial_failure, timeout
    """

    SUPPORTED_STRATEGIES = {
        # 通用策略 (来自 Mixin)
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
        StrategyType.CORRUPT_TENSOR,
        # 特有策略
        StrategyType.OBJECT_LOST,
        StrategyType.PARTIAL_FAILURE,
    }

    def _get_layer(self) -> str:
        return "L0"

    # ========== 特有策略实现 ==========

    def _strategy_object_lost(self, futures, timeout=None):
        """Ray 特有: 模拟 ObjectLostError"""
        ref = futures[0] if isinstance(futures, list) else futures
        raise ray.exceptions.ObjectLostError(
            object_ref=ref,
            owner_address="ralph_simulated",
            call_site="ralph_fault_injection"
        )

    def _strategy_partial_failure(self, futures, timeout=None):
        """Ray 特有: 部分结果失败"""
        results = self._original(futures, timeout)
        fail_ratio = self._config.parameters.get("fail_ratio", 0.5)
        import random
        for i in range(len(results)):
            if random.random() < fail_ratio:
                results[i] = None
        return results
```

```python
# ralph/proxies/l1_distributed.py
import torch
import torch.distributed as dist
from ralph.proxies.base import BaseProxy
from ralph.mixins.delay import DelayMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.core.config import StrategyType

class AllReduceProxy(BaseProxy, DelayMixin, TensorCorruptionMixin):
    """
    torch.distributed.all_reduce 的透明代理

    支持的策略:
    - 通用: delay, corrupt_tensor, inject_nan, inject_inf
    - 特有: deadlock, nccl_timeout
    """

    SUPPORTED_STRATEGIES = {
        # 通用策略
        StrategyType.DELAY,
        StrategyType.CORRUPT_TENSOR,
        StrategyType.INJECT_NAN,
        StrategyType.INJECT_INF,
        # 特有策略
        StrategyType.DEADLOCK,
    }

    def _get_layer(self) -> str:
        return "L1"

    # ========== 特有策略实现 ==========

    def _strategy_deadlock(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        """FSDP 特有: 让指定 rank 死锁"""
        import time
        deadlock_rank = self._config.parameters.get("deadlock_rank", 0)
        if dist.get_rank() == deadlock_rank:
            while True:
                time.sleep(1)  # 无限挂起
        return self._original(tensor, op, group, async_op)

    # 通用策略需要适配参数
    def _strategy_inject_nan(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        """注入 NaN 到张量"""
        nan_ratio = self._config.parameters.get("nan_ratio", 0.001)
        tensor = self._inject_nan(tensor, nan_ratio)
        return self._original(tensor, op, group, async_op)

    def _strategy_inject_inf(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        """注入 Inf 到张量"""
        inf_ratio = self._config.parameters.get("inf_ratio", 0.001)
        tensor = self._inject_inf(tensor, inf_ratio)
        return self._original(tensor, op, group, async_op)
```

```python
# ralph/proxies/l2_verl.py
import torch
from ralph.proxies.base import BaseProxy
from ralph.mixins.delay import DelayMixin
from ralph.mixins.tensor import TensorCorruptionMixin
from ralph.mixins.exception import ExceptionMixin
from ralph.mixins.result import ResultModificationMixin
from ralph.core.config import StrategyType

class RewardManagerProxy(BaseProxy, DelayMixin, TensorCorruptionMixin, ResultModificationMixin):
    """
    RewardManager.__call__ 的透明代理

    支持的策略:
    - 通用: delay, corrupt_tensor
    - 特有: reward_flip, constant_reward, sparse_reward, reward_scale
    """

    SUPPORTED_STRATEGIES = {
        # 通用策略
        StrategyType.DELAY,
        StrategyType.CORRUPT_TENSOR,
        # 特有策略
        StrategyType.REWARD_FLIP,
        StrategyType.CONSTANT_REWARD,
    }

    def _get_layer(self) -> str:
        return "L2"

    # ========== 特有策略实现 ==========

    def _strategy_reward_flip(self, data, return_dict=False):
        """Reward 特有: 奖励取反"""
        result = self._original(data, return_dict)
        return self._negate_result(result)

    def _strategy_constant_reward(self, data, return_dict=False):
        """Reward 特有: 返回常数奖励"""
        result = self._original(data, return_dict)
        constant = self._config.parameters.get("constant_value", 0.0)
        return self._set_constant(result, constant)


class CheckpointSaveProxy(BaseProxy, DelayMixin, ExceptionMixin):
    """
    FSDPCheckpointManager.save_checkpoint 的透明代理
    """

    SUPPORTED_STRATEGIES = {
        StrategyType.DELAY,
        StrategyType.RAISE_EXCEPTION,
    }

    def _get_layer(self) -> str:
        return "L2"

    def _strategy_raise_exception(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """重写异常策略，提供更友好的错误消息"""
        exc_type = self._config.parameters.get("exception_type", "IOError")
        message = self._config.parameters.get(
            "message",
            f"Ralph: Failed to save checkpoint to {local_path}"
        )
        self._raise_exception(exc_type, message)
```

### 5.5 策略注册与发现

```python
# ralph/core/registry.py
from typing import Dict, Type, Set
from ralph.proxies.base import BaseProxy
from ralph.core.config import StrategyType

class ProxyRegistry:
    """代理注册表: 管理注入点到代理类的映射"""

    _registry: Dict[str, Type[BaseProxy]] = {}

    @classmethod
    def register(cls, target: str):
        """装饰器: 注册代理类"""
        def decorator(proxy_class: Type[BaseProxy]):
            cls._registry[target] = proxy_class
            return proxy_class
        return decorator

    @classmethod
    def get_proxy(cls, target: str) -> Type[BaseProxy]:
        """获取指定注入点的代理类"""
        if target not in cls._registry:
            raise KeyError(f"No proxy registered for target: {target}")
        return cls._registry[target]

    @classmethod
    def get_supported_strategies(cls, target: str) -> Set[StrategyType]:
        """获取指定注入点支持的策略"""
        proxy_class = cls.get_proxy(target)
        return proxy_class.SUPPORTED_STRATEGIES

    @classmethod
    def list_targets(cls) -> list:
        """列出所有已注册的注入点"""
        return list(cls._registry.keys())

# 使用示例
@ProxyRegistry.register("ray.get")
class RayGetProxy(BaseProxy, DelayMixin, ExceptionMixin):
    ...

@ProxyRegistry.register("torch.distributed.all_reduce")
class AllReduceProxy(BaseProxy, DelayMixin, TensorCorruptionMixin):
    ...

@ProxyRegistry.register("RewardManager.__call__")
class RewardManagerProxy(BaseProxy, DelayMixin, ResultModificationMixin):
    ...
```

### 5.6 架构优势总结

| 特性 | 实现方式 | 优势 |
|------|----------|------|
| **代码复用** | Mixin 类 | `delay`, `corrupt_tensor` 等通用策略只实现一次 |
| **类型安全** | `SUPPORTED_STRATEGIES` 声明 | 配置时即可验证策略是否支持 |
| **灵活扩展** | 继承 + 方法重写 | 特有策略可自由实现，不影响其他代理 |
| **独立测试** | Mixin 可单独测试 | 提高测试覆盖率和可维护性 |
| **配置验证** | 注册表 + 策略声明 | 加载 YAML 时即可验证配置合法性 |
| **渐进开发** | 分层结构 | 先实现 Mixin，再实现 Proxy，最后添加特有策略 |

---

## 6. YAML 配置规范 (Configuration Schema)

```yaml
# ralph_config.yaml
version: "1.0"
experiment:
  name: "nccl_timeout_resilience_test"
  description: "Test system resilience to NCCL communication timeouts"
  seed: 42

global:
  enabled: true
  log_level: "DEBUG"
  output_dir: "/experiments/ralph/run_001"

data_collection:
  stream_a:  # System telemetry
    enabled: true
    sources:
      - type: "ray_logs"
        path: "/tmp/ray/session_latest/logs"
      - type: "verl_metrics"
        tracking_backend: "wandb"
      - type: "stdout_stderr"
        capture: true
    output:
      format: "jsonl"
      path: "${global.output_dir}/telemetry.jsonl"

  stream_b:  # Ground truth labels
    enabled: true
    output:
      format: "jsonl"
      path: "${global.output_dir}/labels.jsonl"

scenarios:
  # L0 - Ray Layer
  - id: "ray_object_lost"
    layer: "L0"
    target: "ray.get"
    enabled: true
    fault_type: "object_lost"
    trigger:
      type: "one_shot"
      at_step: 100
    parameters:
      affected_workers: [0]
    expected_behavior: "ObjectLostError, Ray retry mechanism"
    severity: "critical"

  # L1 - FSDP Layer
  - id: "nccl_timeout"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    enabled: true
    fault_type: "nccl_timeout"
    trigger:
      type: "one_shot"
      at_step: 150
    parameters:
      delay_seconds: 1800
    expected_behavior: "NCCL timeout, process group failure"
    severity: "critical"

  - id: "gradient_corruption"
    layer: "L1"
    target: "torch.distributed.all_reduce"
    enabled: true
    fault_type: "gradient_corruption"
    trigger:
      type: "step_based"
      start_step: 200
      end_step: 210
    parameters:
      noise_scale: 0.1
    expected_behavior: "Training instability, loss spike"
    severity: "medium"

  # L2 - Verl Business Layer
  - id: "wrong_reward"
    layer: "L2"
    target: "RewardManager.__call__"
    enabled: true
    fault_type: "wrong_reward"
    trigger:
      type: "step_based"
      start_step: 100
      end_step: 120
    parameters:
      noise_scale: 10.0
    expected_behavior: "Policy learning degradation"
    severity: "medium"

  - id: "checkpoint_failure"
    layer: "L2"
    target: "FSDPCheckpointManager.save_checkpoint"
    enabled: true
    fault_type: "io_error"
    trigger:
      type: "probabilistic"
      probability: 0.3
      min_step: 50
    parameters: {}
    expected_behavior: "Checkpoint skip, continue training"
    severity: "high"

  # L3 - Resource Layer
  - id: "gpu_oom"
    layer: "L3"
    target: "gpu_memory"
    enabled: true
    fault_type: "memory_pressure"
    trigger:
      type: "step_based"
      start_step: 80
      end_step: 90
    parameters:
      pressure_ratio: 0.95
    expected_behavior: "CUDA OOM, batch reduction"
    severity: "high"
```

---

## 7. 项目结构 (Project Structure)

```
verl/
├── ralph/                              # 故障注入模块
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                   # YAML 配置解析与验证
│   │   ├── engine.py                   # 注入引擎/编排器
│   │   ├── scheduler.py                # 触发调度 (step/prob/time)
│   │   └── registry.py                 # 代理注册与管理
│   │
│   ├── proxies/
│   │   ├── __init__.py
│   │   ├── base.py                     # 基础代理类
│   │   ├── l0_ray.py                   # Ray 层代理
│   │   ├── l1_distributed.py           # FSDP/distributed 代理
│   │   ├── l2_verl.py                  # Verl 业务逻辑代理
│   │   └── l3_resource.py              # 资源层代理
│   │
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── dual_stream.py              # 双流数据采集器
│   │   ├── metrics_hook.py             # verl Tracking 集成
│   │   └── log_parser.py               # Ray/NCCL 日志解析
│   │
│   ├── triggers/
│   │   ├── __init__.py
│   │   ├── step_based.py               # 基于 Step 的触发器
│   │   ├── probabilistic.py            # 概率触发器
│   │   └── compound.py                 # 复合触发条件
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py                     # CLI 入口
│
├── configs/ralph/                      # 预设配置
│   ├── default.yaml
│   ├── nccl_timeout_test.yaml
│   ├── checkpoint_resilience.yaml
│   └── oom_recovery.yaml
│
└── tests/ralph/                        # 测试
    ├── test_proxies.py
    ├── test_triggers.py
    ├── test_collectors.py
    └── test_e2e.py
```
