from utils.registry import Registry

OPTIM_REGISTRY = Registry("optimizers")

def register_optimizer(name: str = None):
    """装饰器：注册优化器到全局注册表"""
    return OPTIM_REGISTRY.register(name)

def build_optimizer(name: str, **kwargs):
    """根据名称构建优化器"""
    return OPTIM_REGISTRY.build(name, **kwargs)

def list_optimizers() -> list:
    """列出所有注册的优化器"""
    return OPTIM_REGISTRY.keys()