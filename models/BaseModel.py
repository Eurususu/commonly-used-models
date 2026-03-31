"""统一的模型注册表和工厂函数"""

import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging

# ============================================================
# 统一接口：BaseModel
# ============================================================

__all__ = ["BaseModel"]

class BaseModel(nn.Module):
    """所有模型的基类，定义统一接口"""

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"BaseModel 收到了额外的参数 {kwargs}，但这些参数将被忽略！")

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息（参数量等）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }
