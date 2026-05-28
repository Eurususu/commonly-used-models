import torch
import numpy as np
from PIL import Image

def inspect_struct(obj, indent=0):
    pad = " " * indent
    if isinstance(obj, dict):
        print(f"{pad}[字典 Dict] 包含 {len(obj)} 个键: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{pad}  ├── 键 '{k}':")
            inspect_struct(v, indent + 6)
            
    elif isinstance(obj, list):
        print(f"{pad}[列表 List] 长度: {len(obj)}")
        if len(obj) > 0:
            print(f"{pad}  ├── 取第一个元素作为代表:")
            inspect_struct(obj[0], indent + 6)
            
    elif isinstance(obj, tuple):
        print(f"{pad}[元组 Tuple] 长度: {len(obj)}")
        for i, v in enumerate(obj):
            print(f"{pad}  ├── 第 {i} 个元素:")
            inspect_struct(v, indent + 6)
            
    elif torch.is_tensor(obj):
        print(f"{pad}[张量 Tensor] 形状: {list(obj.shape)}, 数据类型: {obj.dtype}, 所在设备: {obj.device}")
        
    elif isinstance(obj, np.ndarray):  # 👈 新增：NumPy 数组判断
        print(f"{pad}[NumPy 数组 ndarray] 形状: {list(obj.shape)}, 数据类型: {obj.dtype}")

    elif isinstance(obj, Image.Image): # 👈 新增：完美支持 PIL 图像
        print(f"{pad}[PIL 图像] 尺寸 (宽x高): {obj.size}, 模式: {obj.mode}")
        
    elif hasattr(obj, '__dict__'):
        # 可能是某种自定义对象 (比如 Detectron2 的 Instances)
        print(f"{pad}[自定义对象 {type(obj).__name__}] 属性: {list(vars(obj).keys())}")
        
    else:
        # 针对基础数据类型（如 int, float, str, bool），不仅打印类型，顺便把值也打印出来，方便调试
        if isinstance(obj, (int, float, str, bool)):
            print(f"{pad}[基础类型 {type(obj).__name__}] 值: {obj}")
        else:
            print(f"{pad}[其他类型] {type(obj)}")
            

if __name__ == "__main__":
    # 可以用来查看输出的数据结构,
    # 这里假设 outputs 是你的输出数据
    outputs = "xxx"
    inspect_struct(outputs)


    # 也可以用来查看训练数据和验证数据dataloader的数据结构
    train_loader = []
    for i, batch_data in enumerate(train_loader):
        print(f"\n=== 探诊第 {i} 个 Batch 的数据结构 ===")
        inspect_struct(batch_data)


    # 也可以用来查看dataset的数据结构
    train_dataset = []
    print("\n=== 探诊单个样本的数据结构 ===")
    sample = train_dataset[0]  # 取第一张图
    inspect_struct(sample)

