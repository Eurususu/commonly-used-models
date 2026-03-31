import torch.nn as nn
from utils.registry import Registry


# 创建模型注册表
MODEL_REGISTRY = Registry("models")

__all__ = ['register_model', 'build_model', 'list_models']


def register_model(name: str = None, force: bool = False):
    """装饰器：注册模型到全局注册表"""
    return MODEL_REGISTRY.register(name=name, force=force)


def build_model(name: str, **kwargs) -> nn.Module:
    """根据名称构建模型"""
    return MODEL_REGISTRY.build(name, **kwargs)


def list_models() -> list:
    """列出所有可用模型"""
    return MODEL_REGISTRY.keys()