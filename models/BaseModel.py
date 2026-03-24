"""统一的模型注册表和工厂函数"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import importlib

# ============================================================
# 统一接口：BaseModel
# ============================================================

class BaseModel(nn.Module):
    """所有模型的基类，定义统一接口"""

    def __init__(self, num_classes: int = 1, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """返回模型的默认配置，供外部调用"""
        return {"num_classes": 1}

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息（参数量等）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_classes": getattr(self, 'num_classes', 1)
        }
