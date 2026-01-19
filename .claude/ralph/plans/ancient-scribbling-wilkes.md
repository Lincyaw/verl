# verl分层故障注入系统设计

## 1. 总体架构设计

基于verl的5层架构，设计分层故障注入系统：

```
verl Fault Injection Framework
├── Layer 1: User Interface Faults
│   ├── Configuration faults (Hydra config corruption)
│   ├── CLI argument faults
│   └── Environment setup faults
├── Layer 2: Orchestration Faults
│   ├── Ray cluster faults
│   ├── Resource pool faults
│   └── Task scheduling faults
├── Layer 3: Worker Faults
│   ├── FSDP worker faults
│   ├── Megatron worker faults
│   └── Engine worker faults
├── Layer 4: Engine Faults
│   ├── Model loading faults
│   ├── Device mesh faults
│   └── Communication faults
└── Layer 5: Inference Faults
    ├── vLLM faults
    ├── SGLang faults
    └── HF rollout faults
```

## 2. 各层故障注入设计

### Layer 1: 用户界面层故障

**故障类型：**
1. **配置故障**
   - Hydra配置解析错误
   - 缺失必要配置项
   - 配置值类型错误
   - 循环引用配置

2. **环境故障**
   - Ray初始化失败
   - CUDA环境变量错误
   - Python路径问题
   - 依赖包版本冲突

**注入点：**
- `main_ppo.py:main()` - 配置加载阶段
- `verl/utils/ray_utils.py` - Ray初始化
- `hydra/config_loader` - 配置解析

### Layer 2: 编排层故障

**故障类型：**
1. **Ray集群故障**
   - Actor崩溃
   - GCS连接失败
   - 资源不足
   - 节点间网络分区

2. **资源池故障**
   - GPU分配失败
   - 内存不足
   - 资源泄露
   - 资源竞争

3. **任务调度故障**
   - 任务超时
   - 任务死锁
   - 优先级反转
   - 调度器崩溃

**注入点：**
- `verl/trainer/ppo/ray_trainer.py` - RayTrainer
- `verl/single_controller/ray/base.py` - ResourcePool
- `ray/actor.py` - Actor生命周期

### Layer 3: 工作器层故障

**FSDP工作器故障：**
1. **分片故障**
   - 参数分片不均
   - 梯度同步失败
   - All-reduce挂起
   - 分片策略冲突

2. **内存故障**
   - CUDA OOM
   - 内存碎片
   - 显存泄露
   - 缓存未命中

**Megatron工作器故障：**
1. **并行故障**
   - 流水线气泡
   - 张量并行不同步
   - 专家并行路由错误
   - 通信死锁

2. **检查点故障**
   - 检查点损坏
   - 加载失败
   - 版本不兼容
   - 部分加载

**注入点：**
- `verl/workers/fsdp_workers.py` - FSDPWorker
- `verl/workers/megatron_workers.py` - MegatronWorker
- `torch/distributed/fsdp/` - FSDP核心

### Layer 4: 引擎层故障

**故障类型：**
1. **模型加载故障**
   - 权重文件损坏
   - 模型结构不匹配
   - 精度转换错误
   - 设备映射失败

2. **设备网格故障**
   - GPU拓扑检测错误
   - 设备ID冲突
   - 跨节点通信失败
   - NCCL初始化错误

3. **通信故障**
   - NCCL超时
   - 通信缓冲区溢出
   - 集合通信不匹配
   - 量化通信错误

**注入点：**
- `verl/workers/engine/base.py` - BaseEngine
- `verl/workers/engine/fsdp_engine.py` - FSDPEngine
- `verl/workers/engine/megatron_engine.py` - MegatronEngine

### Layer 5: 推理层故障

**vLLM故障：**
1. **KV缓存故障**
   - 缓存分配失败
   - 缓存驱逐错误
   - 缓存污染
   - 内存碎片

2. **调度故障**
   - 请求饿死
   - 优先级反转
   - 批处理错误
   - 预填充延迟

**SGLang故障：**
1. **图编译故障**
   - 图构建失败
   - 优化pass错误
   - 内存规划失败
   - 内核生成错误

2. **RadixAttention故障**
   - 树结构损坏
   - 缓存一致性错误
   - 前缀匹配失败
   - 内存池耗尽

**注入点：**
- `verl/workers/rollout/vllm_rollout/` - vLLM后端
- `verl/workers/rollout/sglang_rollout/` - SGLang后端
- `vllm/core/scheduler.py` - vLLM调度器

## 3. 故障恢复机制设计

### 3.1 分层恢复策略

#### Layer 1: 用户界面层恢复
```python
class UILayerRecovery:
    """用户界面层故障恢复"""

    def recover_config_error(self, error_type: str, config: Dict) -> Dict:
        """配置错误恢复"""
        if error_type == "missing_key":
            # 使用默认值或从备份配置恢复
            return self.merge_with_default_config(config)
        elif error_type == "type_mismatch":
            # 类型转换或回退到兼容类型
            return self.cast_config_types(config)
        elif error_type == "circular_reference":
            # 打破循环引用
            return self.resolve_circular_refs(config)

    def recover_ray_init_failure(self, retry_count: int = 3) -> bool:
        """Ray初始化失败恢复"""
        for i in range(retry_count):
            try:
                # 清理环境
                self.cleanup_ray_env()
                # 延迟重试（指数退避）
                time.sleep(2 ** i)
                # 尝试重新初始化
                ray.init(address="auto", ignore_reinit_error=True)
                return True
            except Exception as e:
                if i == retry_count - 1:
                    # 降级到本地模式
                    return self.fallback_to_local_mode()
        return False
```

#### Layer 2: 编排层恢复
```python
class OrchestrationLayerRecovery:
    """编排层故障恢复"""

    def __init__(self, ray_trainer):
        self.ray_trainer = ray_trainer
        self.actor_recovery_strategies = {
            "actor_crash": self.recover_actor_crash,
            "resource_exhausted": self.recover_resource_exhaustion,
            "gcs_failure": self.recover_gcs_failure
        }

    def recover_actor_crash(self, actor_name: str, worker_group) -> bool:
        """Actor崩溃恢复"""
        try:
            # 1. 检测失败的Actor
            failed_actors = self.detect_failed_actors(worker_group)

            # 2. 重启失败的Actor
            for actor in failed_actors:
                # 保存Actor状态（如果可能）
                state = self.save_actor_state(actor)

                # 重启Actor
                new_actor = self.restart_actor(actor, actor_name)

                # 恢复状态
                if state:
                    self.restore_actor_state(new_actor, state)

                # 重新加入工作组
                worker_group.replace_actor(actor, new_actor)

            # 3. 重新同步集群状态
            self.resync_cluster_state()
            return True

        except Exception as e:
            # 降级策略：减少并行度
            return self.fallback_to_reduced_parallelism()

    def recover_resource_exhaustion(self, resource_type: str) -> bool:
        """资源耗尽恢复"""
        recovery_actions = [
            # 1. 清理未使用的资源
            self.cleanup_unused_resources,
            # 2. 调整批处理大小
            lambda: self.adjust_batch_size(0.8),
            # 3. 释放缓存
            self.clear_caches,
            # 4. 动态调整并行度
            lambda: self.adjust_parallelism(0.7),
            # 5. 请求更多资源（如果可能）
            self.request_additional_resources
        ]

        for action in recovery_actions:
            try:
                if action():
                    return True
            except Exception:
                continue

        # 最终降级：暂停非关键任务
        return self.suspend_non_critical_tasks()

    def recover_gcs_failure(self) -> bool:
        """GCS故障恢复"""
        # 1. 等待GCS恢复
        if self.wait_for_gcs_recovery(timeout=60):
            return True

        # 2. 重新连接Ray集群
        ray.shutdown()
        time.sleep(5)

        try:
            ray.init(address="auto", ignore_reinit_error=True)
            # 3. 重建Actor系统
            self.rebuild_actor_system()
            return True
        except Exception:
            # 降级到本地执行模式
            return self.enable_local_execution_mode()
```

#### Layer 3: 工作器层恢复
```python
class WorkerLayerRecovery:
    """工作器层故障恢复"""

    def __init__(self, worker_group):
        self.worker_group = worker_group
        self.checkpoint_manager = CheckpointManager()

    def recover_fsdp_sharding_fault(self, failed_rank: int) -> bool:
        """FSDP分片故障恢复"""
        # 1. 重新初始化FSDP状态
        self.reinitialize_fsdp_state(failed_rank)

        # 2. 重新同步参数
        self.resync_model_parameters(failed_rank)

        # 3. 验证分片一致性
        if not self.verify_sharding_consistency():
            # 使用检查点恢复
            return self.recover_from_checkpoint()

        return True

    def recover_megatron_pipeline_fault(self, failed_stage: int) -> bool:
        """Megatron流水线故障恢复"""
        # 1. 暂停流水线执行
        self.pause_pipeline_execution()

        # 2. 重新初始化失败阶段
        self.reinitialize_pipeline_stage(failed_stage)

        # 3. 重新同步流水线状态
        self.resync_pipeline_state()

        # 4. 恢复执行
        self.resume_pipeline_execution()

        # 5. 验证数据一致性
        return self.verify_pipeline_consistency()

    def recover_gradient_sync_failure(self) -> bool:
        """梯度同步失败恢复"""
        # 1. 检测同步失败的梯度
        failed_gradients = self.detect_failed_gradient_sync()

        # 2. 重新计算失败的梯度
        for param in failed_gradients:
            # 清空梯度
            param.grad = None
            # 重新计算
            self.recompute_gradient(param)

        # 3. 重试同步
        return self.retry_gradient_sync()

    def recover_cuda_oom(self, oom_rank: int) -> bool:
        """CUDA OOM恢复"""
        recovery_steps = [
            # 1. 清理未使用的张量
            lambda: self.cleanup_unused_tensors(oom_rank),
            # 2. 清空缓存
            lambda: torch.cuda.empty_cache(),
            # 3. 减少批处理大小
            lambda: self.reduce_batch_size(0.5),
            # 4. 启用梯度累积
            lambda: self.enable_gradient_accumulation(2),
            # 5. 使用检查点技术
            lambda: self.enable_checkpointing()
        ]

        for step in recovery_steps:
            try:
                if step():
                    # 重试失败的操作
                    return self.retry_failed_operation()
            except Exception:
                continue

        # 最终手段：重启工作器
        return self.restart_worker(oom_rank)
```

#### Layer 4: 引擎层恢复
```python
class EngineLayerRecovery:
    """引擎层故障恢复"""

    def __init__(self, engine):
        self.engine = engine
        self.model_registry = ModelRegistry()

    def recover_model_loading_fault(self, fault_type: str) -> bool:
        """模型加载故障恢复"""
        if fault_type == "corrupted_checkpoint":
            # 尝试从备份恢复
            backup_path = self.find_backup_checkpoint()
            if backup_path:
                return self.load_from_backup(backup_path)

            # 尝试修复损坏的检查点
            return self.repair_corrupted_checkpoint()

        elif fault_type == "incompatible_version":
            # 使用版本转换器
            converter = self.get_version_converter()
            return converter.convert_and_load()

        elif fault_type == "device_mapping_failed":
            # 重新检测设备拓扑
            self.redetect_device_topology()
            # 使用更保守的映射策略
            return self.use_conservative_mapping()

    def recover_nccl_timeout(self, timeout_rank: int) -> bool:
        """NCCL超时恢复"""
        # 1. 调整NCCL参数
        self.adjust_nccl_parameters({
            "NCCL_TIMEOUT": "600",  # 增加到10分钟
            "NCCL_P2P_DISABLE": "1",  # 禁用P2P
            "NCCL_IB_DISABLE": "1"   # 禁用InfiniBand
        })

        # 2. 重新初始化通信器
        self.reinitialize_communicator(timeout_rank)

        # 3. 验证通信
        if self.verify_communication():
            return True

        # 4. 降级到更简单的通信模式
        return self.fallback_to_simple_communication()

    def recover_device_mesh_fault(self) -> bool:
        """设备网格故障恢复"""
        # 1. 重新计算设备布局
        self.recalculate_device_layout()

        # 2. 使用更简单的并行策略
        simplified_strategy = self.simplify_parallel_strategy()

        # 3. 重新初始化设备网格
        return self.reinitialize_device_mesh(simplified_strategy)
```

#### Layer 5: 推理层恢复
```python
class InferenceLayerRecovery:
    """推理层故障恢复"""

    def __init__(self, rollout_backend):
        self.backend = rollout_backend
        self.recovery_strategies = {
            "vllm": VLLMRecovery(),
            "sglang": SGLangRecovery(),
            "hf": HFRolloutRecovery()
        }

    def recover_vllm_kv_cache_oom(self) -> bool:
        """vLLM KV缓存OOM恢复"""
        vllm_recovery = self.recovery_strategies["vllm"]

        # 1. 调整缓存分配策略
        vllm_recovery.adjust_allocation_strategy("conservative")

        # 2. 减少最大序列长度
        vllm_recovery.reduce_max_sequence_length(0.8)

        # 3. 启用缓存压缩
        vllm_recovery.enable_cache_compression()

        # 4. 调整批处理大小
        vllm_recovery.reduce_batch_size(0.7)

        # 5. 重启vLLM引擎
        return vllm_recovery.restart_engine()

    def recover_sglang_compilation_fault(self, error_info: str) -> bool:
        """SGLang编译故障恢复"""
        sglang_recovery = self.recovery_strategies["sglang"]

        # 1. 简化计算图
        sglang_recovery.simplify_computation_graph()

        # 2. 禁用有问题的优化pass
        problematic_passes = self.identify_problematic_passes(error_info)
        sglang_recovery.disable_optimization_passes(problematic_passes)

        # 3. 使用更保守的编译选项
        sglang_recovery.use_conservative_compilation_options()

        # 4. 重新编译
        return sglang_recovery.recompile_graph()

    def recover_request_scheduling_deadlock(self) -> bool:
        """请求调度死锁恢复"""
        # 1. 检测死锁的请求
        deadlocked_requests = self.detect_scheduling_deadlock()

        # 2. 强制释放部分请求
        for request in deadlocked_requests[:len(deadlocked_requests)//2]:
            self.force_release_request(request)

        # 3. 调整调度策略
        self.adjust_scheduling_strategy("fair_share")

        # 4. 重新调度剩余的请求
        return self.reschedule_remaining_requests()
```

### 3.2 智能恢复决策引擎

```python
class IntelligentRecoveryEngine:
    """智能恢复决策引擎"""

    def __init__(self):
        self.recovery_history = RecoveryHistory()
        self.ml_predictor = RecoveryMLPredictor()
        self.cost_calculator = RecoveryCostCalculator()

    def decide_recovery_strategy(self, fault_info: Dict) -> RecoveryStrategy:
        """基于历史数据和ML模型选择最佳恢复策略"""

        # 1. 快速模式匹配
        similar_cases = self.recovery_history.find_similar_cases(fault_info)
        if similar_cases and similar_cases[0].success_rate > 0.9:
            return similar_cases[0].strategy

        # 2. ML模型预测
        ml_prediction = self.ml_predictor.predict_success_probability(
            fault_info,
            self.get_available_strategies()
        )

        # 3. 成本效益分析
        strategy_costs = {}
        for strategy in self.get_available_strategies():
            success_prob = ml_prediction[strategy.name]
            recovery_cost = self.cost_calculator.calculate_cost(strategy)
            impact_cost = self.estimate_impact_cost(fault_info)

            # 期望成本 = 恢复成本 + (1-成功率) * 影响成本
            expected_cost = recovery_cost + (1 - success_prob) * impact_cost
            strategy_costs[strategy] = expected_cost

        # 4. 选择期望成本最低的策略
        best_strategy = min(strategy_costs.items(), key=lambda x: x[1])[0]

        # 5. 记录决策
        self.record_decision(fault_info, best_strategy)

        return best_strategy

    def adapt_strategy(self, strategy: RecoveryStrategy, feedback: RecoveryFeedback) -> RecoveryStrategy:
        """根据执行反馈调整策略"""

        if feedback.success:
            # 成功：记录并可能加速后续类似恢复
            self.recovery_history.record_success(strategy, feedback)
            return strategy
        else:
            # 失败：分析原因并调整
            failure_analysis = self.analyze_failure(strategy, feedback)

            # 根据失败原因调整策略参数
            adjusted_strategy = self.adjust_strategy_parameters(
                strategy,
                failure_analysis
            )

            # 如果调整无效，尝试完全不同的策略
            if not self.is_strategy_viable(adjusted_strategy):
                return self.select_alternative_strategy(strategy)

            return adjusted_strategy
```

### 3.3 恢复协调器

```python
class RecoveryCoordinator:
    """恢复协调器，统一管理所有恢复活动"""

    def __init__(self):
        self.layer_recoveries = {
            "ui": UILayerRecovery(),
            "orchestration": OrchestrationLayerRecovery(),
            "worker": WorkerLayerRecovery(),
            "engine": EngineLayerRecovery(),
            "inference": InferenceLayerRecovery()
        }
        self.decision_engine = IntelligentRecoveryEngine()
        self.metrics_collector = RecoveryMetricsCollector()
        self.notification_service = RecoveryNotificationService()

    def handle_fault(self, fault_event: FaultEvent) -> RecoveryResult:
        """处理故障事件"""

        # 1. 记录故障
        self.metrics_collector.record_fault(fault_event)

        # 2. 快速评估
        severity = self.assess_fault_severity(fault_event)

        # 3. 决定是否启动恢复
        if severity < FaultSeverity.MINOR:
            return RecoveryResult.no_action_needed()

        # 4. 选择恢复策略
        strategy = self.decision_engine.decide_recovery_strategy(fault_info)

        # 5. 执行恢复
        recovery_result = self.execute_recovery(strategy, fault_event)

        # 6. 监控恢复过程
        self.monitor_recovery(recovery_result)

        # 7. 处理恢复结果
        if recovery_result.success:
            self.handle_successful_recovery(recovery_result)
        else:
            self.handle_failed_recovery(recovery_result)

        # 8. 通知相关方
        self.notification_service.notify_recovery_result(recovery_result)

        return recovery_result

    def execute_recovery(self, strategy: RecoveryStrategy, fault_event: FaultEvent) -> RecoveryResult:
        """执行恢复策略"""

        recovery_id = self.generate_recovery_id()
        start_time = time.time()

        try:
            # 1. 准备恢复环境
            self.prepare_recovery_environment(strategy)

            # 2. 执行分层恢复
            for layer in strategy.target_layers:
                layer_recovery = self.layer_recoveries[layer]
                recovery_method = getattr(layer_recovery, strategy.method_name)

                # 执行恢复
                layer_result = recovery_method(**strategy.parameters)

                # 检查是否成功
                if not layer_result:
                    # 尝试备选策略
                    fallback_strategy = strategy.get_fallback()
                    if fallback_strategy:
                        return self.execute_recovery(fallback_strategy, fault_event)
                    else:
                        raise RecoveryFailedException(f"Layer {layer} recovery failed")

            # 3. 验证恢复结果
            if self.verify_recovery(fault_event):
                return RecoveryResult.success(
                    recovery_id=recovery_id,
                    duration=time.time() - start_time,
                    strategy=strategy
                )
            else:
                raise RecoveryFailedException("Recovery verification failed")

        except Exception as e:
            return RecoveryResult.failure(
                recovery_id=recovery_id,
                duration=time.time() - start_time,
                strategy=strategy,
                error=str(e)
            )
```

### 3.4 预防性恢复机制

```python
class PreventiveRecoveryManager:
    """预防性恢复管理器，在故障发生前采取行动"""

    def __init__(self):
        self.health_monitor = SystemHealthMonitor()
        self.risk_predictor = RiskPredictor()
        self.recovery_scheduler = RecoveryScheduler()

    def start_preventive_monitoring(self):
        """启动预防性监控"""

        # 1. 系统健康检查
        self.health_monitor.start_continuous_monitoring(
            metrics=["memory_usage", "gpu_utilization", "network_latency", "disk_space"],
            callback=self.health_check_callback
        )

        # 2. 风险预测
        self.risk_predictor.start_prediction_loop(
            interval=60,  # 每分钟预测一次
            callback=self.risk_prediction_callback
        )

    def health_check_callback(self, health_status: HealthStatus):
        """健康检查回调"""

        # 内存压力预防
        if health_status.memory_usage > 0.85:
            self.schedule_preventive_action(
                action="reduce_memory_footprint",
                priority=ActionPriority.HIGH,
                params={"target_usage": 0.7}
            )

        # GPU利用率异常预防
        if health_status.gpu_utilization_variance > 0.5:
            self.schedule_preventive_action(
                action="balance_gpu_load",
                priority=ActionPriority.MEDIUM,
                params={"max_variance": 0.2}
            )

    def risk_prediction_callback(self, risk_assessment: RiskAssessment):
        """风险预测回调"""

        if risk_assessment.overall_risk > 0.7:
            # 高风险：立即执行预防性恢复
            self.execute_immediate_preventive_recovery(risk_assessment)
        elif risk_assessment.overall_risk > 0.4:
            # 中等风险：计划预防性恢复
            self.schedule_preventive_recovery(risk_assessment)

    def execute_immediate_preventive_recovery(self, risk_assessment: RiskAssessment):
        """执行立即预防性恢复"""

        for risk in risk_assessment.risks:
            if risk.severity == RiskSeverity.HIGH:
                # 根据风险类型执行相应的预防性恢复
                if risk.type == RiskType.MEMORY_EXHAUSTION:
                    self.prevent_memory_exhaustion()
                elif risk.type == RiskType.NETWORK_PARTITION:
                    self.prevent_network_partition()
                elif risk.type == RiskType.GRADIENT_EXPLOSION:
                    self.prevent_gradient_explosion()
```

## 4. 配置驱动的故障场景

### 4.1 场景配置格式

```yaml
fault_injection:
  enabled: true
  scenarios:
    - name: "network_partition_training"
      description: "模拟训练中的网络分区"
      triggers:
        - type: "step"
          value: 100
        - type: "time"
          value: "5m"
      faults:
        - layer: "orchestration"
          type: "ray_actor_crash"
          target: "critic_worker"
          probability: 0.3
          duration: "30s"
          recovery: "auto_restart"

        - layer: "engine"
          type: "nccl_timeout"
          target: "actor_rollout"
          delay: "10s"
          recovery: "retry_with_backoff"
```

### 4.2 生产环境故障场景示例

#### 场景1：大规模训练中的网络分区
```yaml
# scenarios/network_partition_large_scale.yaml
name: "large_scale_network_partition"
description: "模拟大规模训练中跨节点的网络分区故障"
tags: ["production", "network", "critical"]

triggers:
  - type: "step"
    value: 500  # 在第500步触发
  - type: "metric"
    metric: "gpu_memory_usage"
    threshold: 0.9
    duration: "2m"  # 持续2分钟

faults:
  # Ray Actor网络分区
  - layer: "orchestration"
    type: "ray_actor_network_partition"
    target: "actor_rollout_workers"
    scope:
      nodes: ["node-2", "node-3"]  # 特定节点
      probability: 0.8
    duration: "30s"
    recovery:
      strategy: "graceful_restart"
      timeout: "5m"
      retry_count: 3

  # NCCL通信超时
  - layer: "engine"
    type: "nccl_timeout"
    target: "all_reduce_operations"
    parameters:
      timeout_ms: 30000
      ranks: [0, 1, 2, 3]  # 特定rank
    recovery:
      strategy: "nccl_reinitialize"
      fallback: "reduce_parallelism"

  # 梯度同步失败
  - layer: "worker"
    type: "gradient_sync_failure"
    target: "fsdp_workers"
    probability: 0.5
    parameters:
      sync_type: "all_reduce"
      error_rate: 0.1
    recovery:
      strategy: "gradient_recomputation"
      backup_strategy: "checkpoint_recovery"

# 监控配置
monitoring:
  metrics:
    - name: "recovery_time"
      type: "histogram"
    - name: "data_loss"
      type: "counter"

  alerts:
    - condition: "recovery_time > 300s"
      severity: "critical"
      action: "page_oncall"
```

#### 场景2：vLLM推理服务OOM故障
```yaml
# scenarios/vllm_oom_cascade.yaml
name: "vllm_oom_cascade_failure"
description: "模拟vLLM推理服务中的级联OOM故障"
tags: ["inference", "memory", "cascade"]

triggers:
  - type: "load"
    requests_per_second: 1000  # 高负载触发
  - type: "time"
    cron: "0 14 * * *"  # 每天下午2点

faults:
  # KV缓存OOM
  - layer: "inference"
    type: "vllm_kv_cache_oom"
    target: "vllm_rollout_backend"
    parameters:
      memory_threshold: 0.95
      allocation_failure_rate: 0.3
    cascade:
      enabled: true
      spread_to: ["scheduler", "worker"]
      delay: "5s"
    recovery:
      strategy: "adaptive_batch_reduction"
      parameters:
        batch_size_reduction: 0.5
        max_seq_len_reduction: 0.8

  # 调度器死锁
  - layer: "inference"
    type: "request_scheduler_deadlock"
    target: "vllm_scheduler"
    condition: "kv_cache_oom > 5"  # 在5次OOM后触发
    parameters:
      max_wait_time: 60s
      affected_requests: 100
    recovery:
      strategy: "scheduler_restart"
      grace_period: "30s"

  # 工作器崩溃
  - layer: "orchestration"
    type: "ray_actor_crash"
    target: "inference_workers"
    condition: "scheduler_deadlock > 1"
    probability: 0.7
    recovery:
      strategy: "rolling_restart"
      batch_size: 2
      delay_between_batches: "10s"

# 验证配置
validation:
  success_criteria:
    - "p99_latency < 2s"
    - "error_rate < 1%"
    - "recovery_time < 120s"
```

#### 场景3：FSDP训练中的梯度爆炸
```yaml
# scenarios/fsdp_gradient_explosion.yaml
name: "fsdp_gradient_explosion_with_recovery"
description: "模拟FSDP训练中的梯度爆炸及自动恢复"
tags: ["training", "fsdp", "gradient", "numerical"]

triggers:
  - type: "step"
    value: 1000
  - type: "metric"
    metric: "gradient_norm"
    threshold: 100.0

faults:
  # 梯度爆炸
  - layer: "worker"
    type: "gradient_explosion"
    target: "fsdp_actor_worker"
    parameters:
      explosion_factor: 1000  # 梯度乘以1000
      affected_parameters: ["lm_head.weight", "transformer.wte.weight"]
    detection:
      gradient_norm_threshold: 50.0
      check_interval: 10  # 每10步检查
    recovery:
      strategy: "gradient_clipping"
      parameters:
        max_norm: 1.0
        norm_type: 2.0
      fallback: "rollback_optimizer"

  # 损失NaN
  - layer: "worker"
    type: "loss_nan"
    target: "training_step"
    condition: "gradient_explosion > 3"  # 3次梯度爆炸后
    recovery:
      strategy: "mixed_precision_recovery"
      parameters:
        reduce_amp_scale: true
        enable_loss_scaling: true

  # 优化器状态损坏
  - layer: "worker"
    type: "optimizer_state_corruption"
    target: "adam_optimizer"
    condition: "loss_nan > 1"
    parameters:
      corruption_type: "momentum_nan"
    recovery:
      strategy: "optimizer_reinitialization"
      preserve_lr_schedule: true

# 高级选项
advanced:
  # 自动调参
  auto_tune:
    enabled: true
    parameters: ["learning_rate", "batch_size", "gradient_accumulation_steps"]

  # 检查点策略
  checkpointing:
    before_fault: true
    after_recovery: true
    max_checkpoints: 5
```

#### 场景4：混合并行训练中的通信死锁
```yaml
# scenarios/hybrid_parallel_deadlock.yaml
name: "hybrid_parallel_communication_deadlock"
description: "模拟Megatron+DDP混合并行中的通信死锁"
tags: ["megatron", "communication", "deadlock", "hybrid"]

triggers:
  - type: "step"
    value: 2000
  - type: "metric"
    metric: "communication_stall_time"
    threshold: 30s

faults:
  # 张量并行通信死锁
  - layer: "worker"
    type: "tensor_parallel_deadlock"
    target: "megatron_tensor_parallel_group"
    parameters:
      deadlock_rank: 2
      deadlock_op: "all_reduce"
      timeout: 60s
    recovery:
      strategy: "communicator_recreation"
      parameters:
        preserve_order: true
        timeout_multiplier: 2

  # 流水线并行气泡
  - layer: "worker"
    type: "pipeline_parallel_bubble"
    target: "megatron_pipeline_stage"
    parameters:
      bubble_size: 4
      affected_stages: [1, 2, 3]
    recovery:
      strategy: "dynamic_rebalancing"
      parameters:
        rebalance_interval: 100
        threshold: 0.2

  # DDP AllReduce挂起
  - layer: "worker"
    type: "ddp_allreduce_hang"
    target: "data_parallel_group"
    condition: "tensor_parallel_deadlock > 0"
    parameters:
      hang_rank: 0
      collective_op: "allreduce"
    recovery:
      strategy: "collective_restart"
      synchronous: true
      timeout: 120s

# 性能监控
performance_monitoring:
  metrics:
    - "throughput_drop_percentage"
    - "gpu_idle_time"
    - "communication_efficiency"

  slo:
    max_throughput_drop: 20%
    max_gpu_idle_time: 15%
    min_communication_efficiency: 80%
```

### 4.3 动态故障配置

```python
class DynamicFaultConfigurator:
    """动态故障配置管理器"""

    def __init__(self):
        self.active_configs = {}
        self.config_templates = self.load_templates()
        self.environment_detector = EnvironmentDetector()

    def generate_context_aware_config(self, base_config: Dict, context: Dict) -> Dict:
        """根据运行时上下文生成故障配置"""

        # 1. 检测环境特征
        env_features = self.environment_detector.detect()

        # 2. 根据环境调整故障参数
        adjusted_config = self.adjust_config_for_environment(base_config, env_features)

        # 3. 根据训练阶段调整
        training_phase = context.get("training_phase", "initial")
        phase_adjusted_config = self.adjust_for_training_phase(adjusted_config, training_phase)

        # 4. 根据集群规模调整
        cluster_size = context.get("num_gpus", 1)
        scale_adjusted_config = self.adjust_for_cluster_size(phase_adjusted_config, cluster_size)

        return scale_adjusted_config

    def adjust_config_for_environment(self, config: Dict, env: Dict) -> Dict:
        """根据环境特征调整配置"""

        # 网络环境
        if env.get("network_latency_ms", 0) > 10:
            # 高延迟环境，增加超时时间
            config = self.increase_timeouts(config, factor=2.0)

        # GPU内存
        if env.get("gpu_memory_gb", 32) < 24:
            # 小内存GPU，调整内存相关故障
            config = self.adjust_memory_faults(config, scale=0.7)

        # InfiniBand可用性
        if not env.get("infiniband_available", False):
            # 无IB网络，调整通信故障
            config = self.adjust_communication_faults(config, use_tcp=True)

        return config

    def create_progressive_fault_sequence(self, base_fault: Dict, steps: int = 5) -> List[Dict]:
        """创建渐进式故障序列"""

        sequence = []
        for i in range(steps):
            step_config = deepcopy(base_fault)

            # 逐步增加故障强度
            if "probability" in step_config:
                step_config["probability"] = base_fault["probability"] * (i + 1) / steps

            if "severity" in step_config:
                step_config["severity"] = min(1.0, base_fault["severity"] * (i + 1) / steps * 2)

            # 添加触发条件
            step_config["trigger_condition"] = {
                "type": "step",
                "value": 100 * (i + 1)
            }

            sequence.append(step_config)

        return sequence
```

### 4.4 场景组合与依赖

```yaml
# scenarios/complex_cascade_scenario.yaml
name: "complex_cascade_with_dependencies"
description: "复杂的多层故障级联场景，展示故障依赖关系"

# 定义故障组件
components:
  - name: "primary_fault"
    type: "ray_actor_crash"
    layer: "orchestration"
    target: "critic_worker"
    parameters:
      crash_probability: 0.8

  - name: "secondary_fault"
    type: "gradient_sync_failure"
    layer: "worker"
    target: "actor_worker"
    depends_on: "primary_fault"
    dependency_type: "after_n_occurrences"
    dependency_params:
      n: 2
      within_time: "5m"

  - name: "tertiary_fault"
    type: "vllm_kv_cache_oom"
    layer: "inference"
    target: "rollout_backend"
    depends_on: ["primary_fault", "secondary_fault"]
    dependency_type: "after_all"
    delay: "30s"

# 恢复策略链
recovery_chain:
  - trigger: "primary_fault"
    action: "restart_actor"
    success_criteria: "actor_healthy"

  - trigger: "secondary_fault"
    action: "reinitialize_communication"
    depends_on: "primary_fault_recovery"

  - trigger: "tertiary_fault"
    action: "restart_inference"
    parameters:
      rolling_restart: true
      batch_size: 1

# 验证配置
verification:
  - type: "metric"
    metric: "overall_health_score"
    condition: "> 0.9"

  - type: "custom"
    script: "verify_no_data_loss.py"

  - type: "time"
    max_recovery_time: "600s"
```

## 5. 监控和观测系统

### 5.1 指标收集

1. **系统指标**
   - CPU/内存/磁盘使用率
   - GPU利用率、显存使用
   - 网络带宽、延迟
   - 进程状态

2. **应用指标**
   - 训练吞吐量
   - 推理延迟
   - 批处理大小
   - 收敛速度

3. **故障指标**
   - 故障注入次数
   - 恢复成功率
   - 故障影响范围
   - 恢复时间

### 5.2 可视化仪表板

```python
class FaultInjectionDashboard:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()

    def create_dashboard(self):
        return {
            "system_overview": self.create_system_panel(),
            "training_metrics": self.create_training_panel(),
            "fault_injection": self.create_fault_panel(),
            "recovery_status": self.create_recovery_panel()
        }
```

### 5.3 实时监控实现

```python
class RealtimeFaultMonitor:
    """实时故障监控系统"""

    def __init__(self):
        self.metrics_buffer = MetricsBuffer(max_size=10000)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()

    def start_monitoring(self, config: Dict):
        """启动实时监控"""

        # 1. 启动指标收集线程
        self.start_metrics_collection(config["metrics"])

        # 2. 启动异常检测
        self.start_anomaly_detection(config["anomaly_detection"])

        # 3. 启动告警管理
        self.start_alert_management(config["alerts"])

        # 4. 启动实时仪表板
        self.start_dashboard_server(config["dashboard"])

    def process_fault_event(self, event: FaultEvent):
        """处理故障事件"""

        # 1. 记录事件
        self.record_event(event)

        # 2. 更新指标
        self.update_metrics(event)

        # 3. 检测异常模式
        anomalies = self.detect_anomalies(event)

        # 4. 触发告警
        if anomalies:
            self.trigger_alerts(anomalies)

        # 5. 更新仪表板
        self.update_dashboard(event)

    def create_live_dashboard(self) -> Dashboard:
        """创建实时仪表板"""

        return Dashboard(
            panels=[
                # 系统健康面板
                Panel(
                    title="System Health",
                    type="gauge",
                    queries=[
                        Query("cpu_usage", "avg", "1m"),
                        Query("memory_usage", "avg", "1m"),
                        Query("gpu_utilization", "avg", "1m")
                    ],
                    thresholds=[0.7, 0.85, 0.95]
                ),

                # 故障注入面板
                Panel(
                    title="Fault Injection Activity",
                    type="timeseries",
                    queries=[
                        Query("faults_injected", "rate", "1m"),
                        Query("faults_recovered", "rate", "1m"),
                        Query("recovery_time", "avg", "5m")
                    ]
                ),

                # 训练性能面板
                Panel(
                    title="Training Performance",
                    type="timeseries",
                    queries=[
                        Query("throughput", "avg", "1m"),
                        Query("loss", "last", "1m"),
                        Query("gradient_norm", "avg", "1m")
                    ]
                ),

                # 推理性能面板
                Panel(
                    title="Inference Performance",
                    type="timeseries",
                    queries=[
                        Query("request_latency_p50", "avg", "1m"),
                        Query("request_latency_p99", "avg", "1m"),
                        Query("requests_per_second", "rate", "1m")
                    ]
                )
            ]
        )
```

### 5.4 告警系统

```python
class IntelligentAlertManager:
    """智能告警管理器"""

    def __init__(self):
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_history = AlertHistory()
        self.correlation_engine = AlertCorrelationEngine()

    def create_alert_rule(self, rule_config: Dict) -> AlertRule:
        """创建告警规则"""

        return AlertRule(
            name=rule_config["name"],
            condition=self.parse_condition(rule_config["condition"]),
            severity=rule_config["severity"],
            window=rule_config.get("window", "5m"),
            cooldown=rule_config.get("cooldown", "10m"),
            actions=self.create_actions(rule_config["actions"]),
            metadata=rule_config.get("metadata", {})
        )

    def evaluate_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
        """评估告警条件"""

        triggered_alerts = []

        for rule in self.alert_rules.values():
            # 1. 检查条件
            if rule.evaluate(metrics):
                # 2. 检查冷却时间
                if not self.is_in_cooldown(rule):
                    # 3. 创建告警
                    alert = self.create_alert(rule, metrics)
                    triggered_alerts.append(alert)

                    # 4. 记录历史
                    self.alert_history.record(alert)

        # 5. 关联分析
        correlated_alerts = self.correlation_engine.correlate(triggered_alerts)

        return correlated_alerts

    def send_notification(self, alert: Alert):
        """发送告警通知"""

        # 1. 根据严重级别选择通道
        channels = self.select_notification_channels(alert.severity)

        # 2. 格式化消息
        message = self.format_alert_message(alert)

        # 3. 发送通知
        for channel in channels:
            try:
                channel.send(message)
            except Exception as e:
                self.log_notification_failure(channel, alert, e)

    def create_recovery_suggestions(self, alert: Alert) -> List[str]:
        """生成恢复建议"""

        suggestions = []

        # 基于告警类型和历史数据生成建议
        if alert.type == "memory_exhaustion":
            suggestions.extend([
                "考虑减少批处理大小",
                "启用梯度累积",
                "检查内存泄露",
                "考虑使用模型并行"
            ])
        elif alert.type == "communication_timeout":
            suggestions.extend([
                "检查网络连接",
                "增加超时时间",
                "考虑使用更稳定的通信后端",
                "检查是否有防火墙限制"
            ])

        return suggestions
```

## 6. 具体实现代码示例

### verl工作器故障注入器

```python
class VerlWorkerFaultInjector:
    """针对verl工作器的故障注入器"""

    def __init__(self, worker_type: str):
        self.worker_type = worker_type
        self.injection_points = self._get_injection_points()

    def inject_fsdp_sharding_fault(self, rank: int, delay: float = 0.1):
        """注入FSDP分片故障"""
        @inject_at("torch.distributed.fsdp.FullyShardedDataParallel.__init__")
        def faulty_init(self, *args, **kwargs):
            if dist.get_rank() == rank:
                time.sleep(delay)  # 模拟分片延迟
                if random.random() < 0.1:
                    raise RuntimeError(f"FSDP sharding failed on rank {rank}")
            return original_init(self, *args, **kwargs)

    def inject_megatron_pipeline_fault(self, stage_id: int):
        """注入Megatron流水线故障"""
        @inject_at("megatron.core.pipeline_parallel.schedules.forward_step")
        def faulty_forward(step_id, *args, **kwargs):
            if step_id == stage_id and random.random() < 0.05:
                raise RuntimeError(f"Pipeline stage {stage_id} failed")
            return original_forward(step_id, *args, **kwargs)
```

### Ray编排层故障注入

```python
class RayOrchestrationFaultInjector:
    """针对Ray编排层的故障注入"""

    def inject_actor_placement_fault(self, actor_class: str, probability: float):
        """注入Actor放置故障"""
        @inject_at("ray.actor.ActorClassRemote._remote")
        def faulty_placement(self, args=None, kwargs=None, **options):
            if random.random() < probability:
                # 模拟资源不足
                options["resources"] = {"non_existent_resource": 1}
            return original_remote(self, args, kwargs, **options)

    def inject_resource_exhaustion(self, resource_type: str, exhaustion_level: float):
        """注入资源耗尽故障"""
        @inject_at("ray.cluster_resources")
        def mock_resources():
            resources = original_cluster_resources()
            if resource_type in resources:
                resources[resource_type] *= (1 - exhaustion_level)
            return resources
```

### 推理后端故障注入

```python
class InferenceBackendFaultInjector:
    """针对推理后端的故障注入"""

    def inject_vllm_kv_cache_fault(self, max_oom_count: int = 3):
        """注入vLLM KV缓存OOM故障"""
        oom_count = 0

        @inject_at("vllm.core.block_manager.BlockManager.allocate")
        def faulty_allocate(self, *args, **kwargs):
            nonlocal oom_count
            if oom_count < max_oom_count and random.random() < 0.2:
                oom_count += 1
                raise torch.cuda.OutOfMemoryError("Simulated KV cache OOM")
            return original_allocate(self, *args, **kwargs)

    def inject_sglang_compilation_fault(self, failure_rate: float = 0.1):
        """注入SGLang编译故障"""
        @inject_at("sglang.compiler.compile")
        def faulty_compile(graph, *args, **kwargs):
            if random.random() < failure_rate:
                raise RuntimeError("Graph compilation failed due to unsupported operation")
            return original_compile(graph, *args, **kwargs)
```

### 配置驱动的故障管理器

```python
class ConfigDrivenFaultManager:
    """基于配置的故障注入管理器"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.injectors = self.initialize_injectors()
        self.active_scenarios = {}

    def load_config(self, path: str) -> Dict:
        """加载故障注入配置"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_injectors(self) -> Dict[str, BaseFaultInjector]:
        """初始化各层故障注入器"""
        return {
            "ui": UserInterfaceFaultInjector(),
            "orchestration": RayOrchestrationFaultInjector(),
            "worker": VerlWorkerFaultInjector(),
            "engine": EngineFaultInjector(),
            "inference": InferenceBackendFaultInjector()
        }

    def start_scenario(self, scenario_name: str):
        """启动故障场景"""
        scenario = self.find_scenario(scenario_name)
        if not scenario:
            raise ValueError(f"Scenario {scenario_name} not found")

        # 设置触发器
        for trigger in scenario.get("triggers", []):
            self.setup_trigger(trigger, scenario)

        # 激活故障
        for fault in scenario.get("faults", []):
            self.activate_fault(fault)

        self.active_scenarios[scenario_name] = scenario

    def setup_trigger(self, trigger: Dict, scenario: Dict):
        """设置故障触发器"""
        trigger_type = trigger["type"]

        if trigger_type == "step":
            # 训练步数触发
            @inject_at("verl.trainer.ppo.ppo_ray_trainer.train")
            def step_trigger(self, step):
                if step >= trigger["value"]:
                    self.start_scenario(scenario["name"])
                return original_train(self, step)

        elif trigger_type == "time":
            # 时间触发
            delay = self.parse_time(trigger["value"])
            threading.Timer(delay, lambda: self.start_scenario(scenario["name"])).start()

        elif trigger_type == "metric":
            # 指标阈值触发
            self.setup_metric_monitor(trigger["metric"], trigger["threshold"],
                                    lambda: self.start_scenario(scenario["name"]))
```

## 7. 集成到verl的示例

### 修改main_ppo.py支持故障注入

```python
def main(config):
    # 初始化故障注入管理器
    if config.get("fault_injection", {}).get("enabled", False):
        fault_manager = ConfigDrivenFaultManager(config.fault_injection.config_path)

        # 启动配置的故障场景
        for scenario in config.fault_injection.get("auto_start_scenarios", []):
            fault_manager.start_scenario(scenario)

    # 原有的main函数逻辑
    ray.init(address=config.ray.get("address", None), ignore_reinit_error=True)
    # ...
```

### 在RayTrainer中添加故障注入钩子

```python
class RayPPOTrainer:
    def __init__(self, config, fault_manager=None):
        self.config = config
        self.fault_manager = fault_manager

        # 注册故障注入钩子
        if fault_manager:
            self.register_fault_hooks()

    def register_fault_hooks(self):
        """注册故障注入钩子"""
        # 在关键操作前后添加检查点
        self.actor_rollout_worker_group = self.fault_manager.wrap_worker_group(
            self.actor_rollout_worker_group, "actor"
        )

        self.critic_worker_group = self.fault_manager.wrap_worker_group(
            self.critic_worker_group, "critic"
        )
```

## 8. 实施计划

### 第一阶段：核心集成
1. 将现有故障注入代码集成到verl核心组件
2. 添加配置支持到Hydra配置系统
3. 实现基础监控指标收集

### 第二阶段：高级功能
1. 实现动态故障注入
2. 添加自动恢复机制
3. 创建可视化仪表板

### 第三阶段：生产就绪
1. 性能优化和开销最小化
2. 完整的测试覆盖
3. 文档和示例

## 9. 关键文件路径

- 主要入口：`verl/trainer/main_ppo.py`
- 编排层：`verl/trainer/ppo/ray_trainer.py`
- 工作器层：`verl/workers/fsdp_workers.py`, `verl/workers/megatron_workers.py`
- 引擎层：`verl/workers/engine/`
- 推理层：`verl/workers/rollout/`
- 配置系统：`verl/utils/config/`
- 现有故障注入：`verl_fault_inject.py`, `fault_injector.py`