from utils.registry import Registry
import torchvision.transforms as T

TRANSFORMS_REGISTRY = Registry('transforms')

__all__ = ['register_transform', 'build_transforms', 'list_transforms']

def build_transforms(transform_cfgs: list):
    """
    根据配置列表构建数据增强 Compose 流水线
    
    Args:
        transform_cfgs (list): 包含多个 transform 配置的列表。
        例如:
        [
            {"name": "resize", "kwargs": {"size": [256, 256]}},
            {"name": "to_tensor", "kwargs": {}},
            {"name": "normalize", "kwargs": {"mean": [0.5,0.5,0.5], "std": [0.5,0.5,0.5]}}
        ]
    """
    if not transform_cfgs:
        return None
        
    transform_list = []
    for cfg in transform_cfgs:
        name = cfg.get("name").lower()
        kwargs = cfg.get("kwargs", {})
        
        if name not in TRANSFORMS_REGISTRY.keys():
            raise ValueError(f"未知的 Transform: {name}。可用列表: {TRANSFORMS_REGISTRY.keys()}")
            
        # 拿到具体的类并实例化
        transform_cls = TRANSFORMS_REGISTRY._module_dict[name]
        transform_list.append(transform_cls(**kwargs))
        
    # 用 torchvision 的 Compose 把它们串起来
    return T.Compose(transform_list)



def list_transforms() -> list:
    """返回所有数据变换的名称"""
    return TRANSFORMS_REGISTRY.keys()


def register_transform(name: str = None):
    """注册数据变换"""
    return TRANSFORMS_REGISTRY.register(name)