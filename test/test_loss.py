"""测试 loss 注册和构建功能"""
import torch
from loss import build_loss, list_losses

# 列出所有可用的损失函数
print("可用损失函数:", list_losses())

# 测试构建不同的损失函数
print("\n=== 测试构建损失函数 ===")

# 1. 测试 CrossEntropy (需要查看实际的类名)
try:
    loss_fn = build_loss("CrossEntropyLoss")
    print(f"CrossEntropyLoss: {loss_fn}")
except Exception as e:
    print(f"CrossEntropyLoss 失败: {e}")

# 2. 测试 LabelSmoothingCrossEntropy
try:
    loss_fn = build_loss("LabelSmoothingCrossEntropy", smoothing=0.1)
    print(f"LabelSmoothingCrossEntropy: {loss_fn}")
except Exception as e:
    print(f"LabelSmoothingCrossEntropy 失败: {e}")

# 3. 测试 SoftTargetCrossEntropy
try:
    loss_fn = build_loss("SoftTargetCrossEntropy")
    print(f"SoftTargetCrossEntropy: {loss_fn}")
except Exception as e:
    print(f"SoftTargetCrossEntropy 失败: {e}")

# 4. 测试带参数构建并 forward
print("\n=== 测试前向传播 ===")
target = torch.randint(0, 10, (4,))
pred = torch.randn(4, 10)

loss_fn = build_loss("LabelSmoothingCrossEntropy", smoothing=0.1)
loss = loss_fn(pred, target)
print(f"LabelSmoothingCrossEntropy loss: {loss.item():.4f}")

# 5. 测试不存在的损失函数
print("\n=== 测试错误处理 ===")
try:
    build_loss("non_existent_loss")
except ValueError as e:
    print(f"正确捕获异常: {e}")
