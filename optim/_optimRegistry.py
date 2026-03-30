from utils.registry import Registry

OPTIM_REGISTRY = Registry("optimizers")

__all__ = ['register_optimizer', 'build_optimizer', 'list_optimizers']

def register_optimizer(name: str = None):
    """装饰器：注册优化器到全局注册表"""
    return OPTIM_REGISTRY.register(name)

def build_optimizer(params, name: str, **kwargs):
    """根据名称构建优化器"""
    if params is None:
        raise ValueError("构建optimizer 必须传入模型参数")
    
    name = name.lower()
    if name not in OPTIM_REGISTRY._module_dict:
        available = ", ".join(OPTIM_REGISTRY.keys())
        raise ValueError(f"❌ 未知优化器: {name}。可用选项: {available}")
    
    # 拿到优化器类
    optimizer_cls = OPTIM_REGISTRY._module_dict[name]
    
    # 将参数和额外的 kwargs 一起传进去实例化
    return optimizer_cls(params, **kwargs)

def list_optimizers() -> list:
    """列出所有注册的优化器"""
    return OPTIM_REGISTRY.keys()