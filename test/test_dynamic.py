import torch
from models import build_model
import yaml
with open("./config/model/dynamic_model.yaml") as f:
    layers_cfg = yaml.safe_load(f)

# model = build_model("dynamic_graph_model", layers_cfg=layers_cfg["model"]["kwargs"]["layers_cfg"])

# ✅ 为什么 **cfg["model"]["kwargs"] 是神级操作？
"""
场景推演：

    如果是动态网络 YAML： kwargs 字典是 {"layers_cfg": [...]}。
    解包后等价于：build_model("dynamic_graph_model", layers_cfg=[...])。完美契合！

    如果是 ResNet YAML： kwargs 字典是 {"num_classes": 10}。
    解包后等价于：build_model("resnet18", num_classes=10)。完美契合！

    如果是不需要任何参数的模型（比如空 kwargs）：
    解包后什么都没有，等价于：build_model("vgg16")。依然完美！
"""
model = build_model(layers_cfg["model"]["name"], **layers_cfg['model'].get('kwargs', {}))

print(model)

result = model(torch.randn(1, 3, 224, 224))
print(result.shape)




# 1. 读取 YAML
with open("config/model/unet_custom.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# 2. 构建模型 (直接调用工厂函数，它会自动触发你的 DynamicGraphModel)
model = build_model(cfg["model"]["name"], **cfg["model"].get("kwargs", {}))
print(model)

# 3. 终极测试：来一个假数据跑一次前向传播！
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)

print(f"🎉 前向传播成功！输出形状为: {output.shape}") 
print(f"预期输出: {torch.Size([2, 1, 224, 224])}")

