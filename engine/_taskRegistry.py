import torch.nn as nn
from utils.registry import Registry
from typing import Any


# 创建模型注册表
TASK_REGISTRY = Registry("tasks")

__all__ = ['register_task', 'build_task', 'list_tasks']


def register_task(name: str = None, force: bool = False):
    """装饰器：注册任务到全局注册表"""
    return TASK_REGISTRY.register(name=name, force=force)


def build_task(name: str,*args, **kwargs) -> Any:
    """根据名称构建任务"""
    return TASK_REGISTRY.build(name, *args, **kwargs)


def list_tasks() -> list:
    """列出所有可用任务"""
    return TASK_REGISTRY.keys()