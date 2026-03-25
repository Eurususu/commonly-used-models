from utils.registry import Registry

LOSS_REGISTRY = Registry("losses")

def register_loss(name: str = None):
    """装饰器：注册损失函数到全局注册表"""
    return LOSS_REGISTRY.register(name)

def build_loss(name: str, **kwargs):
    """根据名称构建损失函数"""
    return LOSS_REGISTRY.build(name, **kwargs)

def list_losses() -> list:
    """返回所有损失函数的名称"""
    return LOSS_REGISTRY.keys()
