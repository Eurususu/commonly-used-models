"""统一的模型注册表和工厂函数"""

import torch.nn as nn
from typing import Optional, Dict, Any, Callable

# ============================================================
# 统一接口：BaseModel
# ============================================================

class BaseModel(nn.Module):
    """所有模型的基类，定义统一接口"""

    def __init__(self):
        super().__init__()

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息（参数量等）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }
