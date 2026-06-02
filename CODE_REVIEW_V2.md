# 代码审查报告 (第二轮)

> 审查范围：上一轮问题修复后的 `dinov3_det` 训练管线全量代码
> 审查日期：2026-06-02
> 前置状态：训练正常启动运行，Loss 数值比修复前更稳定合理

---

## ✅ 上一轮问题修复确认

| 原编号 | 问题 | 修复状态 |
|--------|------|----------|
| 1 | RandomHorizontalFlip 框翻转逻辑 | ✅ 已正确修复为 `w - boxes[:, [2, 0]]` |
| 2 | matcher neginf 替换方向 | ✅ 已改为 `neginf=100.0` |
| 3 | mask_toekn 拼写 | ✅ 已修正 |
| 4 | forward_features_list 提前返回 | ✅ 已移出循环 |
| 5 | attn_mask dtype | ✅ 已改为布尔掩码 |
| 6 | new_reference_points 未定义 | ✅ 已添加 fallback |
| 7 | Resize clone | ✅ 已加 `.clone()` |
| 8 | dict_to_namespace 就地修改 | ✅ 已用字典推导式隔离 |
| 9 | Resume 不完整 | ✅ 已恢复 optimizer/scheduler/epoch |
| 11 | 共享参数对象 | ✅ 已添加清晰注释说明 |
| 12 | ReduceLROnPlateau 不支持 | ✅ 已添加支持（但引入新 bug，见下文） |
| 14 | 缺少梯度裁剪 | ✅ 已添加 clip_grad_norm_ |
| 15 | WarmupOneCycleLR momentum=0 | ⬜ 未修复 |
| 16 | dinov3_huge_plus 缺 fp32 | ✅ 已添加 |
| 17 | DataLoader 除零 | ✅ 已添加检查 |
| 19 | PositionEmbeddingLearned 50 | ⬜ 未修复 |
| 20 | _max_by_axis 就地修改 | ⬜ 未修复 |
| 21 | 直接访问 _module_dict | ✅ 已改为公共 API |
| 23 | 无随机种子 | ✅ 已添加 set_seed + seed_worker |
| 25 | 只保存最优模型 | ✅ 已添加 checkpoint_latest |
| 26 | 权重键名解析脆弱 | ✅ 已重写为三步精确匹配 |
| 27 | build_task 类型标注 | ✅ 已改为 Any |

---

## 🔴 严重问题 (新引入 / 新发现)

### 1. ReduceLROnPlateau 调度器被 double-step

**文件:** [train_engine.py:120-127](engine/train_engine.py#L120-L127)

```python
if self.scheduler:
    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if self.val_loader:
            self.scheduler.step(main_metric)    # ← 第一次 step
        else:
            raise ValueError(...)
    # 简化处理，统一 step
    self.scheduler.step()                        # ← 第二次 step（无条件执行！）
```

`self.scheduler.step()` 在 `if isinstance(ReduceLROnPlateau)` 块**外部**，导致 ReduceLROnPlateau 每个 epoch 被 step 两次。其他调度器不受影响。

**修复：** 加 `else:`

```python
if self.scheduler:
    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if self.val_loader:
            self.scheduler.step(main_metric)
        else:
            raise ValueError(...)
    else:
        self.scheduler.step()
```

---

### 2. PostProcess 调用时参数顺序错误

**文件:** [tasks.py:114-116](engine/tasks.py#L114-L116)

```python
orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
batch_res = self.postprocessor(outputs, targets, orig_target_sizes)
#                                    ^^^^^^^ 传了 targets（list of dicts）作为 target_sizes
```

`PostProcess.forward` 签名是 `(self, outputs, target_sizes, original_target_sizes=None)`。这里把 `targets`（一个 dict 列表）传给了 `target_sizes`，运行时会因 `target_sizes.shape` 而崩溃。

**修复：**
```python
batch_res = self.postprocessor(outputs, orig_target_sizes)
```

---

### 3. `BaseValidator` 缺少 `is_main_process` 属性

**文件:** [val_engine.py:7](engine/val_engine.py#L7) 和 [tasks.py:97-108](engine/tasks.py#L97-L108)

`BaseValidator.__init__` 不接受也不设置 `is_main_process`。但 `DetectionValidator.evaluate()` 多处使用 `self.is_main_process`（line 108、163）。当 `DetectionTrainer` 构造 `DetectionValidator` 时也没有传递此参数：

```python
# tasks.py:208-213
DetectionValidator(model=self.model, dataloader=self.val_loader, 
                   criterion=self.criterion, device=self.device)
# ← 没有 is_main_process
```

在单卡模式下由于是默认进程（rank 0），即使没有该属性也不会报错——但这只是碰巧。在 DDP 多卡环境下，非主进程访问 `self.is_main_process` 会 `AttributeError`。

**修复：** 在 `BaseValidator.__init__` 中添加 `is_main_process=True` 参数并存储。

---

### 4. `attn_mask.contiguous()` 在无 padding 时崩溃

**文件:** [detr_transformer_decoder.py:65](layers/detr_transformer_decoder.py#L65)

```python
attn_mask = None                           # line 54
if input_padding_mask is not None:
    attn_mask = ~input_padding_mask[:, None, None]
                                            # 如果 input_padding_mask 是 None，attn_mask 仍是 None
attn_mask = attn_mask.contiguous()          # line 65: NoneType has no attribute 'contiguous'
```

当 `input_padding_mask` 为 `None`（即图像无 padding），`attn_mask` 保持 `None`，随后调用 `.contiguous()` 直接崩溃。

**修复：**
```python
if attn_mask is not None:
    attn_mask = attn_mask.contiguous()
```

---

### 5. `dinov3_vit.py` 的 `forward()` 在非训练模式处理 List 输入时崩溃

**文件:** [dinov3_vit.py:330-335](models/dinov3_vit.py#L330-L335)

```python
def forward(self, *args, is_training=False, **kwargs):
    ret = self.forward_features(*args, **kwargs)
    if is_training:
        return ret
    else:
        return self.head(ret["x_norm_clstoken"])  # ← ret 可能是 List[Dict]
```

`forward_features` 在输入是 `List[Tensor]` 时返回 `List[Dict]`。对 list 做 `["key"]` 索引会 `TypeError`。虽然检测管线只传单个 Tensor，不影响当前训练，但这是模型的公共接口缺陷。

**修复：**
```python
if not is_training:
    if isinstance(ret, list):
        ret = ret[0]
    return self.head(ret["x_norm_clstoken"])
```

---

## 🟠 重要问题

### 6. Resume 时 scheduler 可能为 None 但 checkpoint 包含 scheduler 状态

**文件:** [train.py:243](tools/train.py#L243)

```python
if 'scheduler_state_dict' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # scheduler 可能是 None
```

如果用户上一次训练用了 scheduler，但这次决定不用（config 中删掉了 scheduler），`scheduler` 是 `None`，这行会 `AttributeError`。

**修复：**
```python
if 'scheduler_state_dict' in checkpoint and scheduler is not None:
```

---

### 7. 硬编码 backbone 预训练权重路径仍然存在

**文件:** [train.py:130](tools/train.py#L130)

```python
weight_path = "/home/jia/anktechDrive/研发部/共享/算法模型/dinov3/vit_backbone/..."
```

上一轮报告过，仍然未改。建议移到 YAML 配置的 `model.backbone_weights` 字段中。

---

### 8. `seed_worker` 缺少 `torch.manual_seed`

**文件:** [dataloaders.py:12-17](dataset/dataloaders.py#L12-L17)

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # ← 缺少 torch.manual_seed(worker_seed)
```

如果 `__getitem__` 中有使用 `torch.rand` / `torch.randn` 的数据增强，这些操作将不受种子控制，导致不可复现。

**修复：** 添加 `torch.manual_seed(worker_seed)`。

---

### 9. `RandomResize` 仍然缺少 `.clone()`

**文件:** [transforms.py:297](dataset/transforms.py#L297)

`Resize` 已修复（加了 `.clone()`），但 `RandomResize` 中仍然是：

```python
boxes = target["boxes"]          # 没有 .clone()
boxes[:, 0::2] *= scale_w        # 就地修改
```

与 `_crop` 和 `Resize` 的防御性风格不一致，在 `RandomSelect` 分支中可能导致框被重复缩放。

---

### 10. `val_logs` 在非主进程上是过时的默认值

**文件:** [tasks.py:160-175](engine/tasks.py#L160-L175)

DDP 广播了 `map_50_95` 标量，但没有广播 `val_logs` 字典。非主进程返回的 `val_logs` 始终是 `{"mAP_50_95": 0.0, "mAP_50": 0.0}`。当前 `train_engine.py` 只在主进程使用 `val_logs`，所以暂无实际问题，但如果未来其他代码读取非主进程的 `val_logs` 会出错。

---

## 🟡 一般问题

### 11. `DetectionValidator` 无效 `is_coco` 参数

**文件:** [tasks.py:236](engine/tasks.py#L236)

```python
return self.validator.evaluate(is_coco=True)
```

`evaluate(is_coco=True)` 硬编码传 `True`，意味着只在 COCO 数据集上可用。如果换用其他检测数据集（如 VOC），`COCO_80_TO_90_REVERSE` 映射会产生错误类别 ID。建议从配置中读取此参数。

---

### 12. `GlobalRpeCrossAttention` 中 `input_padding_mask` 仍用旧式加法掩码

**文件:** [detr_transformer_decoder.py:422-424](layers/detr_transformer_decoder.py#L422-L424)

`GlobalCrossAttention` 已改为布尔掩码，但同文件的 `GlobalRpeCrossAttention` 仍使用 `input_padding_mask[:, None, None] * -100`（int64 掩码）。RPE 场景下需要加法掩码（叠加 RPE bias），所以这里确实应该用 float 加法掩码，但需要显式 `.float()` 转换：

```python
attn_mask += input_padding_mask[:, None, None].float() * -100.0
```

---

### 13. 注释残留与格式问题

| 文件 | 行号 | 问题 |
|------|------|------|
| [transforms.py:97-100](dataset/transforms.py#L97-L100) | — | 注释仍说"归一化 [cx, cy, w, h]"，实际已改为绝对 xyxy |
| [train_engine.py:130](engine/train_engine.py#L130) | — | 拼写 `heckpoint_latest.pth` → `checkpoint_latest.pth` |
| [train_engine.py:145](engine/train_engine.py#L145) | — | 文件末尾无换行符 |
| [tasks.py:10](engine/tasks.py#L10) | — | `__all__` 缺少 `DetectionTrainer`、`DetectionValidator` |

---

## 📊 问题汇总

| 严重程度 | 数量 | 关键项 |
|---------|------|-------|
| 🔴 严重 | 5 | ReduceLROnPlateau double-step、PostProcess 参数顺序、is_main_process 缺失、attn_mask None 崩溃、forward List 崩溃 |
| 🟠 重要 | 5 | scheduler resume 崩溃、硬编码路径、seed_worker 缺 torch seed、RandomResize clone、val_logs 过时 |
| 🟡 一般 | 4 | is_coco 硬编码、RPE 掩码 dtype、注释残留、格式问题 |

**建议优先修复顺序：** 问题 1 → 2 → 3 → 4 → 6 → 8
