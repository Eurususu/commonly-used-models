import torch.nn as nn
from typing import Optional, Dict, Any, Callable


# 模型注册表
# 表示字典的值是一个可以调用的对象，且返回一个 nn.Module
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
def register_model(name: str):
    """装饰器：注册模型到全局注册表"""
    # 将内部函数的参数名改为 obj，表示它既可以是 cls 也可以是 func
    def decorator(obj):
        MODEL_REGISTRY[name.lower()] = obj
        return obj
    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """根据名称构建模型"""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """列出所有可用模型"""
    return list(MODEL_REGISTRY.keys())