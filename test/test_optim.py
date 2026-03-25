"""测试优化器注册和构建功能"""
import torch
import torch.nn as nn
from optim import build_optimizer, list_optimizers

# 列出所有可用的优化器
print("可用优化器:", list_optimizers())

# 创建一个简单的模型用于测试
model = nn.Linear(10, 2)

# 1. 测试 SGD 优化器
print("\n=== 测试 SGD ===")
optimizer = build_optimizer("SGD", params=model.parameters(), lr=0.01)
print(f"SGD: {optimizer}")
print(f"  lr: {optimizer.param_groups[0]['lr']}")

# 2. 测试 Adam 优化器
print("\n=== 测试 Adam ===")
optimizer = build_optimizer("Adam", params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(f"Adam: {optimizer}")
print(f"  lr: {optimizer.param_groups[0]['lr']}")

# 3. 测试 AdamW 优化器
print("\n=== 测试 AdamW ===")
optimizer = build_optimizer("AdamW", params=model.parameters(), lr=0.001, weight_decay=0.01)
print(f"AdamW: {optimizer}")
print(f"  weight_decay: {optimizer.param_groups[0]['weight_decay']}")

# 4. 测试 RMSprop 优化器
print("\n=== 测试 RMSprop ===")
optimizer = build_optimizer("RMSprop", params=model.parameters(), lr=0.01, alpha=0.99)
print(f"RMSprop: {optimizer}")

# 5. 测试优化器实际更新参数
print("\n=== 测试参数更新 ===")
optimizer = build_optimizer("SGD", params=model.parameters(), lr=0.1)

# 保存更新前的参数
params_before = [p.clone() for p in model.parameters()]

# 前向传播
x = torch.randn(1, 10)
target = torch.tensor([1])
criterion = nn.CrossEntropyLoss()
output = model(x)
loss = criterion(output, target)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()

# 检查参数是否更新
params_after = list(model.parameters())
updated = not all(torch.allclose(b, a) for b, a in zip(params_before, params_after))
print(f"参数已更新: {updated}")

# 6. 测试不存在的优化器
print("\n=== 测试错误处理 ===")
try:
    build_optimizer("non_existent_optimizer", params=model.parameters())
except ValueError as e:
    print(f"正确捕获异常: {e}")