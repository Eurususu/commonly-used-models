# 代码审查报告 (第三轮 · 最终)

> 审查范围：两轮修复后的 `dinov3_det` 全量训练管线 (26 个源文件)
> 审查日期：2026-06-02
> 训练状态：正常运行，Loss 数值合理，种子 42 下完全可复现

---

## ✅ 第二轮问题修复确认

| 原编号 | 问题 | 修复状态 |
|--------|------|----------|
| V2-1 | ReduceLROnPlateau double-step | ✅ 已加 `else:` 分支 |
| V2-2 | PostProcess 参数传错 | ✅ 已改为 `(outputs, orig_target_sizes)` |
| V2-3 | BaseValidator 缺 is_main_process | ✅ 已添加参数并传递 |
| V2-4 | attn_mask.contiguous() None 崩溃 | ✅ 已加 `if attn_mask is not None` 守卫 |
| V2-6 | scheduler resume 当 scheduler 为 None 崩溃 | ⬜ 未修复 |
| V2-7 | 硬编码 backbone 权重路径 | ⬜ 未修复 |
| V2-8 | seed_worker 缺 torch.manual_seed | ✅ 已添加 |
| V2-9 | RandomResize 缺 .clone() | ✅ 已添加 |
| V2-10 | val_logs 非主进程过时 | ⬜ 未修复（当前不影响运行） |
| V2-12 | GlobalRpeCrossAttention 掩码 dtype | ⬜ 未修复 |

---

## 🔴 仍需关注的问题

### 1. 硬编码 backbone 预训练权重路径 (迁移性)

**文件:** [train.py:130](tools/train.py#L130)

```python
weight_path = "/home/jia/anktechDrive/研发部/共享/算法模型/dinov3/vit_backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
```

三轮审查中持续存在。在其他环境/机器上无法运行。建议移入 YAML 配置：

```yaml
model:
  backbone_weights: "/path/to/weights.pth"
```

---

### 2. Resume 时 scheduler 可能为 None

**文件:** [train.py:243](tools/train.py#L243)

```python
if 'scheduler_state_dict' in checkpoint:
    scheduler.load_state_dict(...)  # 若本次运行未配置 scheduler，此处 crash
```

如果上次训练有 scheduler，但这次没有，`scheduler` 为 `None` 会 `AttributeError`。

**修复：** 加 `and scheduler is not None`。

---

### 3. `GlobalRpeCrossAttention` 中 padding mask 仍为 int64

**文件:** [detr_transformer_decoder.py:422-424](layers/detr_transformer_decoder.py#L422-L424)

`GlobalCrossAttention` 已修复为布尔掩码，但同文件的 `GlobalRpeCrossAttention` 仍使用旧式：

```python
attn_mask += input_padding_mask[:, None, None] * -100  # bool * -100 = int64
```

RPE 场景需要 float 加法掩码（叠加 RPE bias），但缺少 `.float()` 转换。当前训练中使用 `global_rpe_decomp` decoder type，**此代码路径会被执行**。

**修复：**
```python
attn_mask += input_padding_mask[:, None, None].float() * -100.0
```

---

## 🟡 低优先级 / 设计建议

### 4. `dinov3_vit.py` forward() 非训练模式处理 List 输入

**文件:** [dinov3_vit.py:330-335](models/dinov3_vit.py#L330-L335)

当输入为 `List[Tensor]`（多裁剪推理）且 `is_training=False` 时，`ret` 是 `List[Dict]`，`ret["key"]` 会 `TypeError`。当前检测管线只传单个 Tensor，不影响训练，但作为模型公共接口有隐患。

### 5. `val_logs` 在非主 DDP 进程上保持默认值

**文件:** [tasks.py:160-175](engine/tasks.py#L160-L175)

DDP 广播了 `map_50_95` 标量但未广播 `val_logs` 字典。非主进程返回的 `val_logs` 始终是 `{"mAP_50_95": 0.0}`。当前 `train_engine.py` 只在主进程使用 `val_logs`，暂无实际问题。

---

## 📊 三轮修复总结

| 轮次 | 严重问题 | 重要问题 | 一般问题 |
|------|---------|---------|---------|
| V1 (初始) | 6 | 10 | 14 |
| V2 (第二轮) | 5 | 5 | 4 |
| V3 (最终) | **0** | **2** | **3** |

**当前代码质量评价：** 经过两轮修复，训练管线的核心逻辑（数据加载、变换、模型前向传播、损失计算、匹配器、优化器、梯度裁剪、DDP 支持、断点恢复、COCO mAP 评估）均正确实现。剩余 5 个问题中只有 #3（RPE 掩码 dtype）会在训练中被实际执行到，建议优先修复。
