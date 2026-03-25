from typing import Dict, Callable, Any

class Registry:
    """一个通用的注册表类，可以用来注册任何组件（Model, Loss, Optim 等）"""
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Callable] = {}

    def register(self, name: str = None):
        """装饰器：注册模块"""
        def decorator(obj: Callable):
            target_name = name if name is not None else obj.__name__
            target_name = target_name.lower()
            if target_name in self._module_dict:
                raise KeyError(f"模块 {target_name} 已经在 {self.name} 中注册过了！")
            self._module_dict[target_name] = obj
            return obj
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        """工厂函数：根据名字构建对象"""
        name = name.lower()
        if name not in self._module_dict:
            available = ", ".join(self._module_dict.keys())
            raise ValueError(f"在 {self.name} 中找不到: {name}. 可用选项: {available}")
        return self._module_dict[name](**kwargs)

    def keys(self):
        return list(self._module_dict.keys())