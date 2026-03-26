from utils.registry import Registry

SCHEDULER_REGISTRY = Registry("schedules")

def register_scheduler(name: str = None):
    """注册学习率调度器"""
    return SCHEDULER_REGISTRY.register(name)

def build_scheduler(optimizer, name: str, **kwargs):
    """
    学习率调度器专属工厂函数
    注意：第一个参数必须接收已经实例化的 optimizer 对象
    """
    if optimizer is None:
        raise ValueError("构建 Scheduler 必须传入 optimizer 参数！")
    
    if name is None or name.lower() == "none":
        return None # 允许用户选择不使用调度器

    name = name.lower()
    if name not in SCHEDULER_REGISTRY._module_dict:
        available = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"❌ 未知调度器: {name}。可用选项: {available}")
    
    # 拿到调度器类
    scheduler_cls = SCHEDULER_REGISTRY._module_dict[name]
    
    # 将 optimizer 和额外的 kwargs 一起传进去实例化
    return scheduler_cls(optimizer, **kwargs)


def list_schedulers() -> list:
    """列出所有注册的学习率调度器"""
    return SCHEDULER_REGISTRY.keys()

