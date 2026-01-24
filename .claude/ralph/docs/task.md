# Ralph 注入点 Checklist

本文档详细描述每个注入点的位置、功能、注入策略和预期行为，作为实现和验证的 checklist。

---

## 1. L0 Ray 调度层

### 1.1 ray.get

**位置与功能**
- **文件**: `verl/single_controller/ray/base.py`
- **行号**: 777 (在 `execute_all_sync` 方法中调用)
- **函数签名**: `ray.get(futures: List[ObjectRef], timeout: Optional[float] = None) -> List[Any]`
- **功能**: 阻塞等待 Ray ObjectRef 列表完成，返回实际结果。是 verl 中所有同步 Worker 调用的核心。
- **调用链**: `RayPPOTrainer.fit()` → `worker_group.update_actor()` → `execute_all_sync()` → `ray.get()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 1.1.1 | `delay` | 在调用原始 `ray.get` 前 `time.sleep(N)` | step=100, delay=30s |
| 1.1.2 | `object_lost` | 抛出 `ray.exceptions.ObjectLostError` | probability=0.01 |
| 1.1.3 | `timeout` | 抛出 `ray.exceptions.GetTimeoutError` | step=150 |
| 1.1.4 | `partial_failure` | 将返回列表中部分结果置为 `None` | fail_ratio=0.3 |
| 1.1.5 | `corrupt_result` | 修改返回的 DataProto 中的张量值 | noise_scale=0.1 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `delay` | 训练步骤变慢，TPS 下降 | `perf/throughput` 下降, `perf/time_per_step` 增加 |
| `object_lost` | Ray 任务失败，触发重试或 Job 崩溃 | Ray 日志: "Object lost", Worker 状态变为 DEAD |
| `timeout` | 超时异常，训练中断 | 异常堆栈, `ray.exceptions.GetTimeoutError` |
| `partial_failure` | 部分 Worker 结果丢失，可能导致数据不完整 | 数据校验失败, batch 大小异常 |
| `corrupt_result` | 静默数据损坏，可能导致训练不稳定 | Loss 异常波动, 梯度范数异常 |

**实现 Checklist**
- [ ] 创建 `RayGetProxy` 类
- [ ] 实现 `delay` 策略
- [ ] 实现 `object_lost` 策略
- [ ] 实现 `timeout` 策略
- [ ] 实现 `partial_failure` 策略
- [ ] 实现 `corrupt_result` 策略
- [ ] 编写单元测试
- [ ] 集成测试验证

---

### 1.2 ray.put

**位置与功能**
- **文件**: `verl/utils/ray_utils.py` (在 `parallel_put` 函数中)
- **行号**: ~65
- **函数签名**: `ray.put(value: Any) -> ObjectRef`
- **功能**: 将 Python 对象放入 Ray Object Store，返回 ObjectRef。用于跨 Worker 数据传输。
- **调用链**: `RayPPOTrainer` → `parallel_put(data_list)` → `ray.put(item)` for each item

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 1.2.1 | `delay` | 在调用前延迟 | step_range=[50,60] |
| 1.2.2 | `corrupt_tensor` | 修改张量数据添加噪声 | corruption_ratio=0.01 |
| 1.2.3 | `store_full` | 抛出 `ObjectStoreFullError` | probability=0.05 |
| 1.2.4 | `silent_drop` | 返回无效的 ObjectRef（不实际存储） | probability=0.001 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `delay` | 数据传输变慢 | `perf/throughput` 下降 |
| `corrupt_tensor` | 数据静默损坏 | Loss 异常, 梯度异常 |
| `store_full` | Object Store 满，任务失败 | Ray 日志: "Object store full" |
| `silent_drop` | 后续 `ray.get` 失败 | `ObjectLostError` 在 get 时触发 |

**实现 Checklist**
- [ ] 创建 `RayPutProxy` 类
- [ ] 实现 `delay` 策略
- [ ] 实现 `corrupt_tensor` 策略
- [ ] 实现 `store_full` 策略
- [ ] 实现 `silent_drop` 策略
- [ ] 编写单元测试

---

### 1.3 execute_all_sync / execute_all_async

**位置与功能**
- **文件**: `verl/single_controller/ray/base.py`
- **行号**: 766-807
- **函数签名**:
  - `execute_all_sync(method_name: str, *args, **kwargs) -> List[Any]`
  - `execute_all_async(method_name: str, *args, **kwargs) -> List[ObjectRef]`
- **功能**: 在所有 Worker 上执行指定方法。`sync` 版本等待所有结果，`async` 版本返回 futures。
- **调用链**: `RayPPOTrainer` → `actor_rollout_wg.generate_sequences()` → `execute_all_sync("generate_sequences", data)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 1.3.1 | `worker_death` | 调用 `ray.kill(worker)` 杀掉特定 Worker | kill_worker_idx=1, at_step=200 |
| 1.3.2 | `straggler` | 让特定 Worker 的调用延迟 | straggler_idx=0, delay=60s |
| 1.3.3 | `skip_worker` | 跳过某个 Worker 不执行 | skip_idx=2 |
| 1.3.4 | `duplicate_call` | 对某个 Worker 重复调用 | duplicate_idx=0 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `worker_death` | Worker 崩溃，Ray 检测到 Actor 死亡 | Ray 日志: "Actor died", Worker 状态 DEAD |
| `straggler` | 训练被最慢 Worker 拖慢 | `perf/time_per_step` 大幅增加 |
| `skip_worker` | 结果列表长度不匹配，数据处理失败 | `IndexError` 或数据校验失败 |
| `duplicate_call` | 可能导致状态不一致 | 梯度累积错误, Loss 异常 |

**实现 Checklist**
- [ ] 创建 `ExecuteAllProxy` 类
- [ ] 实现 `worker_death` 策略
- [ ] 实现 `straggler` 策略
- [ ] 实现 `skip_worker` 策略
- [ ] 实现 `duplicate_call` 策略
- [ ] 编写单元测试

---

## 2. L1 FSDP 分布式通信层

### 2.1 torch.distributed.all_reduce

**位置与功能**
- **文件**: `verl/utils/torch_functional.py` (在 `distributed_mean_max_min_std` 等函数中)
- **行号**: ~924 (示例调用点)
- **函数签名**: `torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)`
- **功能**: 在所有 rank 间对张量执行 reduce 操作（默认求和）。用于梯度同步、指标聚合。
- **调用链**: `FSDPEngine.training_step()` → backward → FSDP gradient sync → `all_reduce`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 2.1.1 | `nccl_timeout` | `time.sleep(1800)` 超过 NCCL 超时 | at_step=150 |
| 2.1.2 | `gradient_corruption` | `tensor.add_(noise)` 添加噪声 | noise_scale=0.1, steps=[100,110] |
| 2.1.3 | `nan_injection` | 将部分元素设为 `float('nan')` | nan_ratio=0.001 |
| 2.1.4 | `inf_injection` | 将部分元素设为 `float('inf')` | inf_ratio=0.001 |
| 2.1.5 | `deadlock` | 特定 rank 无限等待 | deadlock_rank=0 |
| 2.1.6 | `scale_corruption` | 将张量乘以大/小系数 | scale_factor=1e6 或 1e-8 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `nccl_timeout` | NCCL Watchdog 触发超时，进程组崩溃 | NCCL 日志: "Watchdog timeout", 进程退出 |
| `gradient_corruption` | 梯度被污染，训练不稳定 | `actor/grad_norm` 异常, Loss 波动 |
| `nan_injection` | NaN 传播，Loss 变为 NaN | `actor/pg_loss` = NaN |
| `inf_injection` | Inf 导致梯度爆炸 | `actor/grad_norm` = Inf |
| `deadlock` | 训练完全卡死 | TPS=0, 进程无响应 |
| `scale_corruption` | 梯度过大/过小 | 梯度爆炸/消失 |

**实现 Checklist**
- [ ] 创建 `AllReduceProxy` 类
- [ ] 实现 `nccl_timeout` 策略
- [ ] 实现 `gradient_corruption` 策略
- [ ] 实现 `nan_injection` 策略
- [ ] 实现 `inf_injection` 策略
- [ ] 实现 `deadlock` 策略
- [ ] 实现 `scale_corruption` 策略
- [ ] 编写单元测试

---

### 2.2 torch.distributed.all_gather / allgather_dict_tensors

**位置与功能**
- **文件**: `verl/utils/torch_functional.py`
- **行号**: 407-447
- **函数签名**:
  - `torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)`
  - `allgather_dict_tensors(tensors: dict, size: int, group, dim: int = 0) -> dict`
- **功能**: 收集所有 rank 的张量并拼接。用于收集分布式训练的输出。
- **调用链**: `FSDPEngineWithValueHead.compute_values()` → `allgather_dict_tensors(outputs)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 2.2.1 | `corrupt_gathered` | 修改特定 rank 的收集结果 | corrupt_ranks=[0,2], factor=0.5 |
| 2.2.2 | `missing_rank` | 将某 rank 数据置零 | missing_rank=1 |
| 2.2.3 | `shape_mismatch` | 修改张量形状导致拼接失败 | target_rank=0 |
| 2.2.4 | `delay` | 延迟 gather 操作 | delay=30s |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `corrupt_gathered` | 部分数据被污染 | 计算结果异常 |
| `missing_rank` | 部分数据缺失 | Value 估计偏差 |
| `shape_mismatch` | `torch.cat` 失败 | `RuntimeError: Sizes of tensors must match` |
| `delay` | 通信变慢 | TPS 下降 |

**实现 Checklist**
- [ ] 创建 `AllGatherProxy` 类
- [ ] 实现 `corrupt_gathered` 策略
- [ ] 实现 `missing_rank` 策略
- [ ] 实现 `shape_mismatch` 策略
- [ ] 实现 `delay` 策略
- [ ] 编写单元测试

---

### 2.3 torch.distributed.barrier

**位置与功能**
- **文件**: `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- **行号**: 178, 208, 298, 359 (多处调用)
- **函数签名**: `torch.distributed.barrier(group=None, async_op=False, device_ids=None)`
- **功能**: 同步所有 rank，确保所有进程到达同一点后再继续。用于 checkpoint 保存前后的同步。
- **调用链**: `FSDPCheckpointManager.save_checkpoint()` → `barrier()` → 保存文件

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 2.3.1 | `barrier_timeout` | 特定 rank 延迟到达 barrier | hang_rank=0, hang_duration=600s |
| 2.3.2 | `barrier_skip` | 跳过 barrier 调用 | probability=0.1 |
| 2.3.3 | `async_desync` | 添加随机延迟造成去同步 | max_desync_delay=5s |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `barrier_timeout` | 其他 rank 在 barrier 处等待超时 | NCCL 超时日志 |
| `barrier_skip` | 可能导致竞态条件 | 文件损坏, 数据不一致 |
| `async_desync` | 进程间轻微去同步 | checkpoint 文件时间戳不一致 |

**实现 Checklist**
- [ ] 创建 `BarrierProxy` 类
- [ ] 实现 `barrier_timeout` 策略
- [ ] 实现 `barrier_skip` 策略
- [ ] 实现 `async_desync` 策略
- [ ] 编写单元测试

---

## 3. L2 Verl 业务逻辑层

### 3.1 RewardManager.__call__ (NaiveRewardManager)

**位置与功能**
- **文件**: `verl/workers/reward_manager/naive.py`
- **行号**: 46-122
- **函数签名**: `__call__(data: DataProto, return_dict: bool = False) -> torch.Tensor | dict`
- **功能**: 计算每个样本的奖励分数。遍历 batch 中的每个样本，解码响应，调用 `compute_score` 计算奖励。
- **调用链**: `RayPPOTrainer.fit()` → `reward_fn(batch)` → `NaiveRewardManager.__call__(data)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 3.1.1 | `wrong_reward` | 返回随机噪声替代真实奖励 | noise_scale=10.0, steps=[100,120] |
| 3.1.2 | `reward_flip` | 将奖励符号取反 | probability=0.5 |
| 3.1.3 | `constant_reward` | 返回常数奖励 | constant_value=0.0 |
| 3.1.4 | `delayed_reward` | 延迟奖励计算 | delay=10s |
| 3.1.5 | `sparse_reward` | 大部分奖励置零，只保留少数 | keep_ratio=0.1 |
| 3.1.6 | `reward_scale` | 放大/缩小奖励值 | scale_factor=100 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_reward` | 策略学习方向完全错误 | `actor/pg_loss` 不下降, 验证奖励下降 |
| `reward_flip` | 策略向相反方向优化 | 验证性能持续恶化 |
| `constant_reward` | 策略无法学习（无梯度信号） | `actor/pg_loss` 接近零且不变 |
| `delayed_reward` | 训练变慢 | `perf/time_per_step` 增加 |
| `sparse_reward` | 学习信号稀疏，收敛变慢 | 收敛曲线平缓 |
| `reward_scale` | 梯度过大/过小 | `actor/grad_norm` 异常 |

**实现 Checklist**
- [ ] 创建 `RewardManagerProxy` 类
- [ ] 实现 `wrong_reward` 策略
- [ ] 实现 `reward_flip` 策略
- [ ] 实现 `constant_reward` 策略
- [ ] 实现 `delayed_reward` 策略
- [ ] 实现 `sparse_reward` 策略
- [ ] 实现 `reward_scale` 策略
- [ ] 编写单元测试

---

### 3.2 FSDPCheckpointManager.save_checkpoint

**位置与功能**
- **文件**: `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- **行号**: 180-363
- **函数签名**: `save_checkpoint(local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None)`
- **功能**: 保存 FSDP 模型的 checkpoint，包括模型参数、优化器状态、学习率调度器状态。
- **调用链**: `RayPPOTrainer.fit()` → `CheckpointHandler.save_checkpoint(step)` → `FSDPCheckpointManager.save_checkpoint()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 3.2.1 | `io_error` | 抛出 `IOError` | at_steps=[100,200], probability=0.3 |
| 3.2.2 | `disk_full` | 抛出 `OSError(errno.ENOSPC)` | at_step=150 |
| 3.2.3 | `permission_denied` | 抛出 `PermissionError` | at_step=100 |
| 3.2.4 | `partial_save` | 只保存部分文件后失败 | truncate_probability=0.5 |
| 3.2.5 | `slow_io` | 极慢的保存速度 | delay=120s |
| 3.2.6 | `corrupt_file` | 保存后损坏文件内容 | corruption_bytes=1024 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `io_error` | checkpoint 保存失败，训练继续 | 日志: "Checkpoint save failed" |
| `disk_full` | 磁盘满错误 | `OSError: No space left on device` |
| `permission_denied` | 权限错误 | `PermissionError` 堆栈 |
| `partial_save` | checkpoint 不完整 | `latest_checkpointed_iteration.txt` 不更新 |
| `slow_io` | 保存时间异常长 | 保存操作耗时增加 |
| `corrupt_file` | 后续加载失败 | `load_checkpoint` 时 `RuntimeError` |

**实现 Checklist**
- [ ] 创建 `CheckpointSaveProxy` 类
- [ ] 实现 `io_error` 策略
- [ ] 实现 `disk_full` 策略
- [ ] 实现 `permission_denied` 策略
- [ ] 实现 `partial_save` 策略
- [ ] 实现 `slow_io` 策略
- [ ] 实现 `corrupt_file` 策略
- [ ] 编写单元测试

---

### 3.3 FSDPCheckpointManager.load_checkpoint

**位置与功能**
- **文件**: `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- **行号**: 98-176
- **函数签名**: `load_checkpoint(local_path: str, hdfs_path: str = None, del_local_after_load=False)`
- **功能**: 加载 FSDP 模型的 checkpoint，恢复训练状态。
- **调用链**: `RayPPOTrainer.__init__()` → `CheckpointHandler.load_checkpoint()` → `FSDPCheckpointManager.load_checkpoint()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 3.3.1 | `file_not_found` | 抛出 `FileNotFoundError` | always |
| 3.3.2 | `corrupt_state_dict` | 加载后修改模型参数 | noise_scale=0.01 |
| 3.3.3 | `version_mismatch` | 抛出版本不匹配错误 | always |
| 3.3.4 | `partial_load` | 只加载部分参数 | skip_keys=["lm_head"] |
| 3.3.5 | `wrong_optimizer_state` | 损坏优化器状态 | corruption_ratio=0.1 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `file_not_found` | 无法恢复，从头训练 | 日志: "Checkpoint not found" |
| `corrupt_state_dict` | 模型参数被污染，性能下降 | 验证准确率骤降 |
| `version_mismatch` | 加载失败 | `RuntimeError: version mismatch` |
| `partial_load` | 部分参数未恢复 | 模型行为异常 |
| `wrong_optimizer_state` | 优化器状态错误，训练不稳定 | Loss 震荡 |

**实现 Checklist**
- [ ] 创建 `CheckpointLoadProxy` 类
- [ ] 实现 `file_not_found` 策略
- [ ] 实现 `corrupt_state_dict` 策略
- [ ] 实现 `version_mismatch` 策略
- [ ] 实现 `partial_load` 策略
- [ ] 实现 `wrong_optimizer_state` 策略
- [ ] 编写单元测试

---

### 3.4 RayPPOTrainer._update_actor

**位置与功能**
- **文件**: `verl/trainer/ppo/ray_trainer.py`
- **行号**: 1283-1317
- **函数签名**: `_update_actor(batch: DataProto) -> DataProto`
- **功能**: 执行 Actor 模型的策略更新，包括前向传播、计算 PPO loss、反向传播、优化器步骤。
- **调用链**: `RayPPOTrainer.fit()` → `self._update_actor(batch)` → `actor_rollout_wg.update_actor(batch)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 3.4.1 | `nan_input` | 将输入 batch 中的部分值设为 NaN | nan_ratio=0.001 |
| 3.4.2 | `exploding_gradients` | 放大输入值导致梯度爆炸 | scale_factor=1e6 |
| 3.4.3 | `vanishing_gradients` | 缩小输入值导致梯度消失 | scale_factor=1e-8 |
| 3.4.4 | `skip_update` | 跳过更新步骤 | probability=0.1 |
| 3.4.5 | `double_update` | 重复执行更新 | probability=0.05 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `nan_input` | NaN 传播到 loss 和梯度 | `actor/pg_loss` = NaN |
| `exploding_gradients` | 梯度爆炸，可能触发梯度裁剪 | `actor/grad_norm` 极大 |
| `vanishing_gradients` | 梯度消失，模型不更新 | `actor/grad_norm` ≈ 0 |
| `skip_update` | 模型参数不更新 | 参数不变 |
| `double_update` | 参数更新过度 | 训练不稳定 |

**实现 Checklist**
- [ ] 创建 `UpdateActorProxy` 类
- [ ] 实现 `nan_input` 策略
- [ ] 实现 `exploding_gradients` 策略
- [ ] 实现 `vanishing_gradients` 策略
- [ ] 实现 `skip_update` 策略
- [ ] 实现 `double_update` 策略
- [ ] 编写单元测试

---

### 3.5 RayPPOTrainer.fit (训练主循环)

**位置与功能**
- **文件**: `verl/trainer/ppo/ray_trainer.py`
- **行号**: 1349+
- **函数签名**: `fit()`
- **功能**: PPO 训练的主循环，协调数据加载、rollout、reward 计算、advantage 估计、actor/critic 更新。
- **调用链**: `main_ppo.py` → `trainer.fit()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 3.5.1 | `stall` | 在特定位置 `time.sleep(infinity)` | at_step=100 |
| 3.5.2 | `infinite_loop` | 重复执行某个步骤 | loop_step=50 |
| 3.5.3 | `skip_step` | 跳过训练步骤 | skip_steps=[100,101,102] |
| 3.5.4 | `early_stop` | 提前结束训练 | stop_at_step=50 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `stall` | 训练完全卡死 | TPS=0, 进程无响应 |
| `infinite_loop` | 重复同一步骤 | step 不增加 |
| `skip_step` | 跳过部分训练 | step 跳跃 |
| `early_stop` | 训练提前结束 | 训练结束日志 |

**实现 Checklist**
- [ ] 创建 `TrainingLoopProxy` 类
- [ ] 实现 `stall` 策略
- [ ] 实现 `infinite_loop` 策略
- [ ] 实现 `skip_step` 策略
- [ ] 实现 `early_stop` 策略
- [ ] 编写单元测试

---

## 4. L3 资源层

### 4.1 GPU Memory

**位置与功能**
- **目标**: CUDA 显存分配
- **功能**: 通过分配大张量模拟显存压力

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 4.1.1 | `memory_pressure` | 分配张量占用指定比例显存 | pressure_ratio=0.9 |
| 4.1.2 | `oom_simulation` | 填满显存触发 OOM | leave_headroom=100MB |
| 4.1.3 | `fragmentation` | 频繁分配/释放造成碎片 | fragment_count=1000 |
| 4.1.4 | `memory_leak` | 持续分配不释放 | leak_rate=100MB/step |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `memory_pressure` | 显存紧张，可能触发 OOM | `torch.cuda.memory_allocated` 增加 |
| `oom_simulation` | `RuntimeError: CUDA out of memory` | OOM 错误日志 |
| `fragmentation` | 分配失败即使有足够总显存 | OOM 但 `memory_reserved` < 总显存 |
| `memory_leak` | 显存逐渐耗尽 | `memory_allocated` 单调增加 |

**实现 Checklist**
- [ ] 创建 `GPUMemoryInjector` 类
- [ ] 实现 `memory_pressure` 策略
- [ ] 实现 `oom_simulation` 策略
- [ ] 实现 `fragmentation` 策略
- [ ] 实现 `memory_leak` 策略
- [ ] 编写单元测试

---

### 4.2 KV Cache (vLLM/SGLang)

**位置与功能**
- **文件**: `verl/workers/rollout/vllm/vllm_rollout.py`
- **目标**: `rollout.resume(tags=['weights', 'kv_cache'])`, `rollout.release()`
- **功能**: 管理推理引擎的 KV cache 分配和释放

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 4.2.1 | `kv_cache_fail` | `resume` 时抛出 `RuntimeError` | probability=0.1 |
| 4.2.2 | `kv_cache_corrupt` | 修改 KV cache 内容 | corruption_ratio=0.01 |
| 4.2.3 | `release_fail` | `release` 时不实际释放 | probability=0.1 |
| 4.2.4 | `partial_resume` | 只恢复部分 KV cache | resume_ratio=0.5 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `kv_cache_fail` | 推理引擎无法启动 | `RuntimeError` 在 generate 时 |
| `kv_cache_corrupt` | 生成质量下降 | 生成文本乱码/重复 |
| `release_fail` | 显存不释放，后续 OOM | 切换到 trainer_mode 时 OOM |
| `partial_resume` | 部分序列无法正确 decode | 生成结果异常 |

**实现 Checklist**
- [ ] 创建 `KVCacheInjector` 类
- [ ] 实现 `kv_cache_fail` 策略
- [ ] 实现 `kv_cache_corrupt` 策略
- [ ] 实现 `release_fail` 策略
- [ ] 实现 `partial_resume` 策略
- [ ] 编写单元测试

---

### 4.3 CPU Stress

**位置与功能**
- **目标**: 系统 CPU 资源
- **功能**: 通过多线程计算模拟 CPU 满载

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 4.3.1 | `cpu_stress` | 启动多个 CPU-bound 线程 | num_workers=8, load=95% |
| 4.3.2 | `io_contention` | 大量磁盘 IO 操作 | io_threads=4 |
| 4.3.3 | `memory_stress` | 大量内存分配 | alloc_size=10GB |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `cpu_stress` | 数据加载变慢，Python GIL 竞争 | `perf/throughput` 下降 |
| `io_contention` | 文件操作变慢 | checkpoint 保存变慢 |
| `memory_stress` | 系统内存紧张，可能 OOM Killer | 进程被杀 |

**实现 Checklist**
- [ ] 创建 `CPUStressInjector` 类
- [ ] 实现 `cpu_stress` 策略
- [ ] 实现 `io_contention` 策略
- [ ] 实现 `memory_stress` 策略
- [ ] 编写单元测试

---

## 5. 算法层

### 5.1 compute_gae_advantage_return (GAE)

**位置与功能**
- **文件**: `verl/trainer/ppo/core_algos.py`
- **函数签名**: `compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam)`
- **功能**: 使用 GAE 算法计算优势值和回报。通过 TD-error 和指数衰减计算时序差分优势。
- **调用链**: `RayPPOTrainer.compute_advantage()` → `compute_gae_advantage_return()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 5.1.1 | `wrong_advantage` | 返回随机噪声 | noise_scale=1.0 |
| 5.1.2 | `zero_advantage` | 返回全零优势 | always |
| 5.1.3 | `inverted_advantage` | 将优势取反 | always |
| 5.1.4 | `scaled_advantage` | 放大/缩小优势值 | scale=10 或 0.1 |
| 5.1.5 | `delayed_advantage` | 将优势向后偏移 | delay_steps=5 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_advantage` | 策略更新方向随机 | Loss 不收敛 |
| `zero_advantage` | 策略不更新（无梯度） | `actor/pg_loss` ≈ 0 |
| `inverted_advantage` | 策略向相反方向优化 | 验证性能持续下降 |
| `scaled_advantage` | 梯度过大/过小 | 训练不稳定或停滞 |
| `delayed_advantage` | 时序信用分配错误 | 学习效果差 |

**实现 Checklist**
- [ ] 创建 `GAEProxy` 类
- [ ] 实现 `wrong_advantage` 策略
- [ ] 实现 `zero_advantage` 策略
- [ ] 实现 `inverted_advantage` 策略
- [ ] 实现 `scaled_advantage` 策略
- [ ] 实现 `delayed_advantage` 策略
- [ ] 编写单元测试

---

### 5.2 compute_grpo_outcome_advantage (GRPO)

**位置与功能**
- **文件**: `verl/trainer/ppo/core_algos.py`
- **函数签名**: `compute_grpo_outcome_advantage(token_level_rewards, response_mask, uid)`
- **功能**: GRPO 算法的组内归一化优势计算。对同一 prompt 的多个响应进行组内标准化。
- **调用链**: `RayPPOTrainer.compute_advantage()` → `compute_grpo_outcome_advantage()` (当 adv_estimator=GRPO)

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 5.2.1 | `wrong_grouping` | 错误地分组样本 | shuffle_uids=True |
| 5.2.2 | `wrong_normalization` | 使用错误的均值/标准差 | bias=1.0, scale=0.5 |
| 5.2.3 | `skip_normalization` | 跳过组内归一化 | always |
| 5.2.4 | `single_sample_groups` | 让每个样本单独成组 | always |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_grouping` | 优势计算基于错误的参照组 | 学习不稳定 |
| `wrong_normalization` | 优势值偏离真实分布 | 梯度方差增大 |
| `skip_normalization` | 优势值未标准化 | 梯度尺度不一致 |
| `single_sample_groups` | 所有优势变为 0 | 策略不更新 |

**实现 Checklist**
- [ ] 创建 `GRPOProxy` 类
- [ ] 实现 `wrong_grouping` 策略
- [ ] 实现 `wrong_normalization` 策略
- [ ] 实现 `skip_normalization` 策略
- [ ] 实现 `single_sample_groups` 策略
- [ ] 编写单元测试

---

### 5.3 apply_kl_penalty

**位置与功能**
- **文件**: `verl/trainer/ppo/core_algos.py`
- **函数签名**: `apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty: str)`
- **功能**: 计算 KL 散度并应用 KL 惩罚到奖励中。`reward = score - beta * KL`
- **调用链**: `RayPPOTrainer.fit()` → `apply_kl_penalty(batch, kl_ctrl, kl_penalty_type)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 5.3.1 | `wrong_kl` | 返回错误的 KL 值 | scale=10 |
| 5.3.2 | `zero_kl` | KL 惩罚始终为 0 | always |
| 5.3.3 | `extreme_kl` | 返回极大的 KL 值 | value=1000 |
| 5.3.4 | `negative_kl` | 返回负的 KL 值（鼓励偏离） | always |
| 5.3.5 | `delayed_kl` | 在计算前添加延迟 | delay=5s |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_kl` | KL 惩罚不准确 | 策略偏离程度与惩罚不匹配 |
| `zero_kl` | 无 KL 约束，策略可能过度偏离 | `rollout_corr/kl` 持续增大 |
| `extreme_kl` | 惩罚过重，奖励变为负 | `token_level_rewards` 大部分为负 |
| `negative_kl` | 鼓励策略偏离参考 | 策略快速崩溃 |
| `delayed_kl` | 训练变慢 | `perf/time_per_step` 增加 |

**实现 Checklist**
- [ ] 创建 `KLPenaltyProxy` 类
- [ ] 实现 `wrong_kl` 策略
- [ ] 实现 `zero_kl` 策略
- [ ] 实现 `extreme_kl` 策略
- [ ] 实现 `negative_kl` 策略
- [ ] 实现 `delayed_kl` 策略
- [ ] 编写单元测试

---

### 5.4 AdaptiveKLController.update

**位置与功能**
- **文件**: `verl/trainer/ppo/kl_ctrl.py`
- **函数签名**: `update(current_kl: float, n_steps: int)`
- **功能**: 根据当前 KL 与目标 KL 的差异，自适应调整 beta 系数。
- **调用链**: `apply_kl_penalty()` → `kl_ctrl.update(current_kl, n_steps)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 5.4.1 | `controller_stuck` | `update` 方法不修改 `self.value` | always |
| 5.4.2 | `oscillating_beta` | 让 beta 在极值间震荡 | min=0.001, max=100 |
| 5.4.3 | `wrong_direction` | beta 调整方向相反 | always |
| 5.4.4 | `extreme_sensitivity` | 让 controller 对 KL 变化过度敏感 | sensitivity=100 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `controller_stuck` | beta 不自适应，KL 失控 | `rollout_corr/kl` 持续增大/减小 |
| `oscillating_beta` | beta 剧烈波动，训练不稳定 | Loss 震荡 |
| `wrong_direction` | beta 调整方向错误 | KL 与 beta 同向变化 |
| `extreme_sensitivity` | beta 频繁剧烈变化 | 训练不稳定 |

**实现 Checklist**
- [ ] 创建 `KLControllerProxy` 类
- [ ] 实现 `controller_stuck` 策略
- [ ] 实现 `oscillating_beta` 策略
- [ ] 实现 `wrong_direction` 策略
- [ ] 实现 `extreme_sensitivity` 策略
- [ ] 编写单元测试

---

## 6. 数据管道层

### 6.1 DataProto.concat

**位置与功能**
- **文件**: `verl/protocol/protocol.py`
- **函数签名**: `concat(data_protos: List[DataProto]) -> DataProto`
- **功能**: 将多个 DataProto 对象沿 batch 维度拼接。
- **调用链**: `RayWorkerGroup.collect()` → `DataProto.concat(results)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 6.1.1 | `data_mismatch` | 返回 batch 和 non_tensor_batch 大小不一致的结果 | always |
| 6.1.2 | `lost_items` | 丢失部分输入 DataProto | drop_ratio=0.1 |
| 6.1.3 | `duplicate_items` | 重复某些 DataProto | duplicate_idx=0 |
| 6.1.4 | `wrong_order` | 打乱拼接顺序 | shuffle=True |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `data_mismatch` | 后续处理时索引错误 | `IndexError` 或 `KeyError` |
| `lost_items` | 数据丢失 | batch size 不匹配预期 |
| `duplicate_items` | 数据重复 | batch size 超过预期 |
| `wrong_order` | 数据对应关系错乱 | Loss 计算结果异常 |

**实现 Checklist**
- [ ] 创建 `DataProtoConcatProxy` 类
- [ ] 实现 `data_mismatch` 策略
- [ ] 实现 `lost_items` 策略
- [ ] 实现 `duplicate_items` 策略
- [ ] 实现 `wrong_order` 策略
- [ ] 编写单元测试

---

### 6.2 DataProto.chunk

**位置与功能**
- **文件**: `verl/protocol/protocol.py`
- **函数签名**: `chunk(chunks: int) -> List[DataProto]`
- **功能**: 将 DataProto 沿 batch 维度分割为多个块。
- **调用链**: `RayWorkerGroup.dispatch()` → `data.chunk(world_size)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 6.2.1 | `uneven_split` | 返回大小不均匀的块 | variance=0.5 |
| 6.2.2 | `lost_chunks` | 返回少于预期数量的块 | drop_last=True |
| 6.2.3 | `empty_chunk` | 某些块为空 | empty_idx=2 |
| 6.2.4 | `overlapping_chunks` | 块之间有重叠数据 | overlap_ratio=0.1 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `uneven_split` | Worker 负载不均衡 | 某些 Worker 明显更慢 |
| `lost_chunks` | Worker 数量与数据块不匹配 | `IndexError` |
| `empty_chunk` | 某 Worker 收到空数据 | 空 batch 处理错误 |
| `overlapping_chunks` | 数据被重复处理 | 梯度计算错误 |

**实现 Checklist**
- [ ] 创建 `DataProtoChunkProxy` 类
- [ ] 实现 `uneven_split` 策略
- [ ] 实现 `lost_chunks` 策略
- [ ] 实现 `empty_chunk` 策略
- [ ] 实现 `overlapping_chunks` 策略
- [ ] 编写单元测试

---

### 6.3 DataProto.to (设备传输)

**位置与功能**
- **文件**: `verl/protocol/protocol.py`
- **函数签名**: `to(device: str | torch.device) -> DataProto`
- **功能**: 将 DataProto 中的张量移动到指定设备。
- **调用链**: Worker 初始化 → `data.to(f"cuda:{rank}")`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 6.3.1 | `device_mismatch` | 返回错误设备上的张量 | wrong_device="cuda:0" when should be "cuda:1" |
| 6.3.2 | `transfer_timeout` | 设备传输时延迟 | delay=30s |
| 6.3.3 | `partial_transfer` | 只传输部分张量 | skip_keys=["values"] |
| 6.3.4 | `corrupt_on_transfer` | 传输时损坏数据 | noise_scale=0.01 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `device_mismatch` | 设备不匹配错误 | `RuntimeError: Expected all tensors to be on the same device` |
| `transfer_timeout` | 数据准备变慢 | `perf/time_per_step` 增加 |
| `partial_transfer` | 缺少必要数据 | `KeyError` |
| `corrupt_on_transfer` | 数据静默损坏 | 计算结果异常 |

**实现 Checklist**
- [ ] 创建 `DataProtoToProxy` 类
- [ ] 实现 `device_mismatch` 策略
- [ ] 实现 `transfer_timeout` 策略
- [ ] 实现 `partial_transfer` 策略
- [ ] 实现 `corrupt_on_transfer` 策略
- [ ] 编写单元测试

---

### 6.4 rearrange_micro_batches (动态 Batching)

**位置与功能**
- **文件**: `verl/utils/seqlen_balancing.py`
- **函数签名**: `rearrange_micro_batches(batch, max_token_len, ...)`
- **功能**: 根据序列长度动态分组 micro-batch，优化 GPU 利用率。
- **调用链**: `prepare_dynamic_batch()` → `rearrange_micro_batches()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 6.4.1 | `unbalanced_batches` | 让 micro-batch 大小极度不均 | imbalance_factor=10 |
| 6.4.2 | `lost_samples` | 丢失部分样本 | drop_ratio=0.05 |
| 6.4.3 | `wrong_sequence_length` | 返回错误的序列长度信息 | length_bias=100 |
| 6.4.4 | `excessive_padding` | 强制过多 padding | pad_to=max_length |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `unbalanced_batches` | GPU 利用率低，某些 micro-batch 非常慢 | micro-batch 处理时间方差大 |
| `lost_samples` | 数据丢失 | 训练数据量减少 |
| `wrong_sequence_length` | 内存估算错误 | 可能 OOM |
| `excessive_padding` | 浪费计算资源 | TPS 下降 |

**实现 Checklist**
- [ ] 创建 `MicroBatchProxy` 类
- [ ] 实现 `unbalanced_batches` 策略
- [ ] 实现 `lost_samples` 策略
- [ ] 实现 `wrong_sequence_length` 策略
- [ ] 实现 `excessive_padding` 策略
- [ ] 编写单元测试

---

## 7. 推理引擎层

### 7.1 vLLM/SGLang generate

**位置与功能**
- **文件**:
  - vLLM: `verl/workers/rollout/vllm/vllm_rollout.py`
  - SGLang: `verl/workers/rollout/sglang/sglang_rollout.py`
- **函数签名**: `generate(prompts, sampling_params) -> responses`
- **功能**: 使用 LLM 生成响应序列。
- **调用链**: `ActorRolloutRefWorker.generate_sequences()` → `rollout.generate()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 7.1.1 | `generation_timeout` | 生成时添加长延迟 | delay=300s |
| 7.1.2 | `empty_response` | 返回空响应 | probability=0.1 |
| 7.1.3 | `truncated_output` | 截断生成结果 | max_new_tokens=10 |
| 7.1.4 | `repetitive_output` | 返回重复内容 | repeat_token="the" |
| 7.1.5 | `garbage_output` | 返回随机 token | always |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `generation_timeout` | rollout 阶段超时 | `perf/generation_time` 极大 |
| `empty_response` | 空响应无法计算奖励 | 奖励计算错误 |
| `truncated_output` | 生成不完整 | 平均响应长度下降 |
| `repetitive_output` | 生成质量差 | 奖励低 |
| `garbage_output` | 完全随机输出 | 奖励极低/异常 |

**实现 Checklist**
- [ ] 创建 `GenerateProxy` 类
- [ ] 实现 `generation_timeout` 策略
- [ ] 实现 `empty_response` 策略
- [ ] 实现 `truncated_output` 策略
- [ ] 实现 `repetitive_output` 策略
- [ ] 实现 `garbage_output` 策略
- [ ] 编写单元测试

---

### 7.2 update_weights (权重同步)

**位置与功能**
- **文件**: `verl/workers/rollout/*/`
- **函数签名**: `update_weights(state_dict: dict)`
- **功能**: 将训练后的模型权重同步到推理引擎。
- **调用链**: `ActorRolloutRefWorker.rollout_mode()` → `rollout.update_weights(state_dict)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 7.2.1 | `weight_mismatch` | 传入形状不匹配的权重 | always |
| 7.2.2 | `partial_update` | 只更新部分权重 | skip_layers=["lm_head"] |
| 7.2.3 | `sync_timeout` | 同步时超时 | delay=600s |
| 7.2.4 | `corrupt_weights` | 损坏权重值 | noise_scale=0.1 |
| 7.2.5 | `old_weights` | 使用旧的权重 | use_step=current_step-10 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `weight_mismatch` | 权重加载失败 | `RuntimeError: size mismatch` |
| `partial_update` | 推理使用不完整权重 | 生成质量下降 |
| `sync_timeout` | 模式切换超时 | `perf/mode_switch_time` 极大 |
| `corrupt_weights` | 推理使用损坏权重 | 生成乱码 |
| `old_weights` | 推理使用过时策略 | off-policy 程度增大 |

**实现 Checklist**
- [ ] 创建 `UpdateWeightsProxy` 类
- [ ] 实现 `weight_mismatch` 策略
- [ ] 实现 `partial_update` 策略
- [ ] 实现 `sync_timeout` 策略
- [ ] 实现 `corrupt_weights` 策略
- [ ] 实现 `old_weights` 策略
- [ ] 编写单元测试

---

### 7.3 SamplingParams (采样参数)

**位置与功能**
- **文件**: `verl/workers/rollout/*/`
- **功能**: 控制生成时的采样策略（temperature, top_p, top_k 等）。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 7.3.1 | `wrong_temperature` | 修改 temperature 参数 | temp=100 或 temp=0.001 |
| 7.3.2 | `wrong_top_p` | 修改 top_p 参数 | top_p=0.01 或 top_p=0.9999 |
| 7.3.3 | `deterministic` | 强制 greedy 解码 | temperature=0, top_k=1 |
| 7.3.4 | `high_randomness` | 强制高随机性 | temperature=10, top_p=1.0 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_temperature` | 采样分布异常 | 生成多样性过高/过低 |
| `wrong_top_p` | 采样范围异常 | 生成质量变化 |
| `deterministic` | 所有生成相同 | 组内奖励方差为 0 |
| `high_randomness` | 生成极度随机 | 奖励方差极大 |

**实现 Checklist**
- [ ] 创建 `SamplingParamsProxy` 类
- [ ] 实现 `wrong_temperature` 策略
- [ ] 实现 `wrong_top_p` 策略
- [ ] 实现 `deterministic` 策略
- [ ] 实现 `high_randomness` 策略
- [ ] 编写单元测试

---

## 8. Worker 层

### 8.1 CriticWorker.compute_values

**位置与功能**
- **文件**: `verl/workers/critic/`
- **函数签名**: `compute_values(data: DataProto) -> DataProto`
- **功能**: 使用 Critic 网络计算每个 token 的 value 估计。
- **调用链**: `RayPPOTrainer.fit()` → `critic_wg.compute_values(batch)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 8.1.1 | `wrong_values` | 返回随机 value | noise_scale=10 |
| 8.1.2 | `constant_values` | 返回常数 value | value=0.0 |
| 8.1.3 | `delayed_values` | 延迟返回 | delay=30s |
| 8.1.4 | `nan_values` | 返回 NaN value | probability=0.01 |
| 8.1.5 | `inverted_values` | value 取反 | always |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_values` | GAE 计算错误 | 优势估计偏差大 |
| `constant_values` | GAE 变为简单折扣回报 | TD-error 异常 |
| `delayed_values` | 训练变慢 | `perf/time_per_step` 增加 |
| `nan_values` | NaN 传播到优势 | `advantages` 含 NaN |
| `inverted_values` | 优势计算方向错误 | 策略学习方向错误 |

**实现 Checklist**
- [ ] 创建 `ComputeValuesProxy` 类
- [ ] 实现 `wrong_values` 策略
- [ ] 实现 `constant_values` 策略
- [ ] 实现 `delayed_values` 策略
- [ ] 实现 `nan_values` 策略
- [ ] 实现 `inverted_values` 策略
- [ ] 编写单元测试

---

### 8.2 CriticWorker.update_critic

**位置与功能**
- **文件**: `verl/workers/critic/`
- **函数签名**: `update_critic(data: DataProto) -> DataProto`
- **功能**: 更新 Critic 网络参数。
- **调用链**: `RayPPOTrainer.fit()` → `critic_wg.update_critic(batch)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 8.2.1 | `nan_loss` | 让 loss 变为 NaN | inject_nan_ratio=0.01 |
| 8.2.2 | `frozen_critic` | 阻止参数更新 | always |
| 8.2.3 | `wrong_loss` | 计算错误的 loss | loss_scale=100 |
| 8.2.4 | `skip_update` | 跳过更新 | probability=0.5 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `nan_loss` | Critic 训练失败 | `critic/vf_loss` = NaN |
| `frozen_critic` | Value 估计不改善 | `critic/vf_loss` 不下降 |
| `wrong_loss` | Critic 学习错误信号 | Value 估计质量差 |
| `skip_update` | Critic 部分未更新 | 更新频率下降 |

**实现 Checklist**
- [ ] 创建 `UpdateCriticProxy` 类
- [ ] 实现 `nan_loss` 策略
- [ ] 实现 `frozen_critic` 策略
- [ ] 实现 `wrong_loss` 策略
- [ ] 实现 `skip_update` 策略
- [ ] 编写单元测试

---

## 9. Multi-Turn & Agent 层

### 9.1 ToolAgentLoop._call_tool

**位置与功能**
- **文件**: `verl/workers/agent/tool_agent_loop.py`
- **函数签名**: `_call_tool(tool_call) -> tool_response`
- **功能**: 执行单个工具调用，返回工具执行结果。
- **调用链**: `ToolAgentLoop._handle_processing_tools_state()` → `_call_tool(tool_call)`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 9.1.1 | `tool_timeout` | 工具执行超时 | delay=300s |
| 9.1.2 | `tool_exception` | 工具抛出异常 | error_type="RuntimeError" |
| 9.1.3 | `wrong_result` | 返回错误结果 | result="incorrect answer" |
| 9.1.4 | `empty_result` | 返回空结果 | always |
| 9.1.5 | `malformed_result` | 返回格式错误的结果 | corrupt_json=True |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `tool_timeout` | Agent 循环卡住 | 单轮时间极长 |
| `tool_exception` | 异常传播，可能终止对话 | 错误日志 |
| `wrong_result` | Agent 基于错误信息继续 | 最终答案错误 |
| `empty_result` | Agent 可能重试或困惑 | 对话轮数增加 |
| `malformed_result` | 解析失败 | 解析错误日志 |

**实现 Checklist**
- [ ] 创建 `CallToolProxy` 类
- [ ] 实现 `tool_timeout` 策略
- [ ] 实现 `tool_exception` 策略
- [ ] 实现 `wrong_result` 策略
- [ ] 实现 `empty_result` 策略
- [ ] 实现 `malformed_result` 策略
- [ ] 编写单元测试

---

### 9.2 ToolAgentLoop 状态机

**位置与功能**
- **文件**: `verl/workers/agent/tool_agent_loop.py`
- **功能**: 管理 Agent 的状态转换 (PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED)。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 9.2.1 | `state_corruption` | 将状态设为无效值 | state="INVALID" |
| 9.2.2 | `infinite_loop` | 阻止状态转换到 TERMINATED | always |
| 9.2.3 | `skip_state` | 跳过某个状态 | skip="PROCESSING_TOOLS" |
| 9.2.4 | `wrong_transition` | 执行错误的状态转换 | transition="PENDING→TERMINATED" |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `state_corruption` | 状态机崩溃 | 异常日志 |
| `infinite_loop` | Agent 无限循环 | 对话轮数达到上限 |
| `skip_state` | 跳过工具处理 | 工具未被调用 |
| `wrong_transition` | 对话异常终止 | 对话提前结束 |

**实现 Checklist**
- [ ] 创建 `StateMachineProxy` 类
- [ ] 实现 `state_corruption` 策略
- [ ] 实现 `infinite_loop` 策略
- [ ] 实现 `skip_state` 策略
- [ ] 实现 `wrong_transition` 策略
- [ ] 编写单元测试

---

### 9.3 ToolParser.parse

**位置与功能**
- **文件**: `verl/workers/agent/tool_parser.py`
- **函数签名**: `parse(response: str) -> List[ToolCall]`
- **功能**: 从 LLM 响应中解析工具调用。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 9.3.1 | `parse_failure` | 解析时抛出异常 | probability=0.1 |
| 9.3.2 | `wrong_tool_name` | 返回错误的工具名 | replace_with="unknown_tool" |
| 9.3.3 | `wrong_arguments` | 返回错误的参数 | corrupt_args=True |
| 9.3.4 | `extra_tool_calls` | 添加额外的工具调用 | extra_count=2 |
| 9.3.5 | `missing_tool_calls` | 丢失部分工具调用 | drop_ratio=0.5 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `parse_failure` | 工具调用被忽略 | 工具执行次数下降 |
| `wrong_tool_name` | 调用不存在的工具 | 工具查找失败 |
| `wrong_arguments` | 工具执行失败或返回错误 | 工具返回错误 |
| `extra_tool_calls` | 执行多余工具 | 工具调用次数增加 |
| `missing_tool_calls` | 缺少必要工具调用 | 任务完成失败 |

**实现 Checklist**
- [ ] 创建 `ToolParserProxy` 类
- [ ] 实现 `parse_failure` 策略
- [ ] 实现 `wrong_tool_name` 策略
- [ ] 实现 `wrong_arguments` 策略
- [ ] 实现 `extra_tool_calls` 策略
- [ ] 实现 `missing_tool_calls` 策略
- [ ] 编写单元测试

---

## 10. 序列处理层

### 10.1 log_probs_from_logits_response

**位置与功能**
- **文件**: `verl/utils/torch_functional.py`
- **函数签名**: `log_probs_from_logits_response(logits, input_ids, response_mask)`
- **功能**: 从 logits 计算响应部分的 log probabilities。
- **调用链**: `ActorRolloutRefWorker.compute_log_prob()` → `log_probs_from_logits_response()`

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 10.1.1 | `wrong_log_probs` | 返回错误的 log probs | noise_scale=1.0 |
| 10.1.2 | `numerical_instability` | 注入 log(0) 导致 -inf | probability=0.001 |
| 10.1.3 | `constant_log_probs` | 返回常数 log prob | value=-1.0 |
| 10.1.4 | `shifted_log_probs` | 偏移 log probs | shift=5.0 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_log_probs` | 重要性采样权重错误 | PPO 比率异常 |
| `numerical_instability` | -inf 传播 | Loss 变为 -inf |
| `constant_log_probs` | 所有 action 等概率 | 比率恒为 1 |
| `shifted_log_probs` | 比率系统性偏大/偏小 | 训练不稳定 |

**实现 Checklist**
- [ ] 创建 `LogProbsProxy` 类
- [ ] 实现 `wrong_log_probs` 策略
- [ ] 实现 `numerical_instability` 策略
- [ ] 实现 `constant_log_probs` 策略
- [ ] 实现 `shifted_log_probs` 策略
- [ ] 编写单元测试

---

### 10.2 postprocess_data (Padding/Truncation)

**位置与功能**
- **文件**: `verl/utils/torch_functional.py`
- **函数签名**: `postprocess_data(input_ids, attention_mask, max_length, truncation, left_pad)`
- **功能**: 对序列进行 padding 和 truncation 处理。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 10.2.1 | `wrong_padding` | 使用错误的 pad token | pad_token_id=0 (应为其他) |
| 10.2.2 | `wrong_truncation` | 截断时丢失关键信息 | truncate_prompt=True |
| 10.2.3 | `asymmetric_padding` | left/right padding 混用 | random_side=True |
| 10.2.4 | `excessive_truncation` | 过度截断 | max_length=10 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_padding` | attention 计算错误 | 模型输出异常 |
| `wrong_truncation` | 输入信息丢失 | 生成质量下降 |
| `asymmetric_padding` | 位置编码混乱 | 模型输出乱码 |
| `excessive_truncation` | 大部分输入被截断 | 任务失败率增加 |

**实现 Checklist**
- [ ] 创建 `PostprocessDataProxy` 类
- [ ] 实现 `wrong_padding` 策略
- [ ] 实现 `wrong_truncation` 策略
- [ ] 实现 `asymmetric_padding` 策略
- [ ] 实现 `excessive_truncation` 策略
- [ ] 编写单元测试

---

## 11. Megatron 特定层

### 11.1 MegatronEngine.optimizer_step

**位置与功能**
- **文件**: `verl/workers/engine/megatron/transformer_impl.py`
- **函数签名**: `optimizer_step()`
- **功能**: 执行 Megatron 分布式优化器的更新步骤。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 11.1.1 | `gradient_overflow` | 让梯度溢出 | scale_grads=1e10 |
| 11.1.2 | `nan_params` | 将部分参数设为 NaN | nan_ratio=0.001 |
| 11.1.3 | `skip_step` | 跳过优化步骤 | probability=0.1 |
| 11.1.4 | `wrong_lr` | 使用错误的学习率 | lr_scale=100 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `gradient_overflow` | 梯度裁剪触发或训练失败 | `grad_norm` 极大 |
| `nan_params` | NaN 传播到前向传播 | 输出变为 NaN |
| `skip_step` | 参数未更新 | Loss 不下降 |
| `wrong_lr` | 学习过快/过慢 | 训练不稳定/停滞 |

**实现 Checklist**
- [ ] 创建 `MegatronOptimizerProxy` 类
- [ ] 实现 `gradient_overflow` 策略
- [ ] 实现 `nan_params` 策略
- [ ] 实现 `skip_step` 策略
- [ ] 实现 `wrong_lr` 策略
- [ ] 编写单元测试

---

### 11.2 Pipeline/Tensor Parallelism

**位置与功能**
- **文件**: `verl/workers/engine/megatron/`
- **功能**: Megatron 的流水线并行和张量并行通信。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 11.2.1 | `pp_stage_failure` | 让某个 PP stage 失败 | fail_stage=2 |
| 11.2.2 | `tp_desync` | TP rank 间去同步 | desync_rank=0 |
| 11.2.3 | `wrong_micro_batch_routing` | micro-batch 路由错误 | swap_stages=[0,1] |
| 11.2.4 | `activation_corruption` | 损坏 PP 间传递的激活 | noise_scale=0.1 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `pp_stage_failure` | 流水线中断 | 异常日志 |
| `tp_desync` | TP rank 结果不一致 | 最终输出错误 |
| `wrong_micro_batch_routing` | micro-batch 处理顺序错误 | 计算结果错误 |
| `activation_corruption` | 前向传播结果错误 | Loss 异常 |

**实现 Checklist**
- [ ] 创建 `ParallelismProxy` 类
- [ ] 实现 `pp_stage_failure` 策略
- [ ] 实现 `tp_desync` 策略
- [ ] 实现 `wrong_micro_batch_routing` 策略
- [ ] 实现 `activation_corruption` 策略
- [ ] 编写单元测试

---

## 12. 优化器与学习率调度层

### 12.1 optimizer.step

**位置与功能**
- **目标**: PyTorch 优化器的 `step()` 方法
- **功能**: 使用计算的梯度更新模型参数。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 12.1.1 | `skip_step` | 不执行 step | probability=0.1 |
| 12.1.2 | `double_step` | 执行两次 step | probability=0.05 |
| 12.1.3 | `corrupted_momentum` | 损坏优化器动量 | noise_scale=0.1 |
| 12.1.4 | `reset_state` | 重置优化器状态 | probability=0.01 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `skip_step` | 参数未更新 | 参数不变 |
| `double_step` | 参数更新过度 | 训练不稳定 |
| `corrupted_momentum` | 优化方向受干扰 | Loss 波动 |
| `reset_state` | 丢失优化历史 | 收敛变慢 |

**实现 Checklist**
- [ ] 创建 `OptimizerStepProxy` 类
- [ ] 实现 `skip_step` 策略
- [ ] 实现 `double_step` 策略
- [ ] 实现 `corrupted_momentum` 策略
- [ ] 实现 `reset_state` 策略
- [ ] 编写单元测试

---

### 12.2 lr_scheduler.step

**位置与功能**
- **目标**: 学习率调度器的 `step()` 方法
- **功能**: 按照调度策略更新学习率。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 12.2.1 | `wrong_lr` | 设置错误的学习率 | lr=1e-2 (过大) |
| 12.2.2 | `lr_spike` | 学习率突然增大 | spike_factor=100 |
| 12.2.3 | `lr_zero` | 学习率变为 0 | always |
| 12.2.4 | `skip_schedule` | 跳过调度更新 | probability=0.5 |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `wrong_lr` | 学习过快/过慢 | 训练不稳定/停滞 |
| `lr_spike` | 突然大幅更新 | Loss 突然增大 |
| `lr_zero` | 学习停止 | 参数不变 |
| `skip_schedule` | 学习率不按预期变化 | `actor/lr` 不变 |

**实现 Checklist**
- [ ] 创建 `LRSchedulerProxy` 类
- [ ] 实现 `wrong_lr` 策略
- [ ] 实现 `lr_spike` 策略
- [ ] 实现 `lr_zero` 策略
- [ ] 实现 `skip_schedule` 策略
- [ ] 编写单元测试

---

### 12.3 GradScaler (AMP)

**位置与功能**
- **目标**: PyTorch AMP 的 GradScaler
- **功能**: 管理混合精度训练的梯度缩放。

**注入策略**

| ID | 策略名称 | 实现方式 | 触发条件示例 |
|----|----------|----------|--------------|
| 12.3.1 | `scale_overflow` | 让 scale 因子溢出 | force_scale=1e38 |
| 12.3.2 | `inf_gradients` | 让 unscale 后梯度为 inf | inject_inf=True |
| 12.3.3 | `wrong_scale` | 使用错误的 scale 因子 | scale=1e-10 |
| 12.3.4 | `disable_scaling` | 禁用缩放 | enabled=False |

**预期行为与遥测指标**

| 策略 | 预期系统行为 | 可观测指标 |
|------|-------------|-----------|
| `scale_overflow` | scale 变为 inf，训练失败 | Loss = inf |
| `inf_gradients` | 跳过更新 | 频繁跳过 step |
| `wrong_scale` | 梯度太小无效 | 训练停滞 |
| `disable_scaling` | 可能数值下溢 | fp16 精度问题 |

**实现 Checklist**
- [ ] 创建 `GradScalerProxy` 类
- [ ] 实现 `scale_overflow` 策略
- [ ] 实现 `inf_gradients` 策略
- [ ] 实现 `wrong_scale` 策略
- [ ] 实现 `disable_scaling` 策略
- [ ] 编写单元测试

---

## 总计

| 类别 | 注入点数量 | 策略总数 |
|------|-----------|---------|
| L0 Ray 调度层 | 3 | 14 |
| L1 FSDP 分布式层 | 3 | 14 |
| L2 Verl 业务层 | 5 | 27 |
| L3 资源层 | 3 | 12 |
| 算法层 | 4 | 19 |
| 数据管道层 | 4 | 16 |
| 推理引擎层 | 3 | 15 |
| Worker 层 | 2 | 9 |
| Multi-Turn/Agent 层 | 3 | 15 |
| 序列处理层 | 2 | 8 |
| Megatron 特定层 | 2 | 8 |
| 优化器/调度器层 | 3 | 12 |
| **总计** | **37** | **169** |

---

## 实现优先级

### P0 - 核心 (立即实现)
- [ ] 1.1 ray.get
- [ ] 2.1 all_reduce
- [ ] 3.1 RewardManager
- [ ] 3.2 save_checkpoint

### P1 - 重要 (第二阶段)
- [ ] 5.1 GAE
- [ ] 5.3 apply_kl_penalty
- [ ] 7.1 generate
- [ ] 8.1 compute_values

### P2 - 扩展 (第三阶段)
- [ ] 6.x DataProto 相关
- [ ] 9.x Agent 相关
- [ ] 10.x 序列处理相关

### P3 - 高级 (第四阶段)
- [ ] 11.x Megatron 相关
- [ ] 12.x 优化器相关
