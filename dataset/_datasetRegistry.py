from utils.registry import Registry

DATASET_REGISTRY = Registry("datasets")

__all__ = ['register_dataset', 'build_dataset', 'list_datasets']

def build_dataset(name: str, **kwargs):
    """根据名称构建数据集"""
    return DATASET_REGISTRY.build(name, **kwargs)

def list_datasets() -> list:
    """返回所有数据集名称"""
    return DATASET_REGISTRY.keys()

def register_dataset(name: str = None, force: bool = False):
    """注册数据集"""
    return DATASET_REGISTRY.register(name=name, force=force)