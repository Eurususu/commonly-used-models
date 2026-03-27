import torch
import torch.nn as nn

__all__ = ["Concat", "Add"]

class Concat(nn.Module):
    """特征融合层：将一个列表中的多个张量按通道维度拼接"""
    def __init__(self, dim=1, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, x: list | tuple):
        # x 必须是一个包含多个 tensor 的列表，例如 [tensor1, tensor2]
        return torch.cat(x, dim=self.dim)
    

class Add(nn.Module):
    """残差相加层：将列表中的多个张量逐元素相加"""
    def __init__(self, alpha=1, **kwargs):
        super().__init__()
        self.alpha = alpha # 显示传入alpha
        # self.add_kwargs = kwargs # 隐式传入alpha，需要翻阅文档，知道有哪些参数可以传入

    def forward(self, x: list | tuple):
        if len(x) != 2:
            raise ValueError(f"Add 层如果使用 torch.add，仅支持 2 个输入，但收到了 {len(x)} 个！")
        # return torch.add(*x, **self.add_kwargs) # 隐式传入alpha，需要翻阅文档，知道有哪些参数可以传入
        return torch.add(*x, alpha=self.alpha) # 显示传入alpha