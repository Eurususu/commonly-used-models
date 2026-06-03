"""统一的模型基类与工厂函数"""

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
            logging.warning(f"{self.__class__.__name__} 收到了额外的参数 {kwargs}，但这些参数将被忽略！")

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息（参数量等）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }

    @staticmethod
    def _init_weights(module):
        """通用权重初始化：Kaiming (Conv2d) + 常量 (BatchNorm2d) + 正态 (Linear)"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
