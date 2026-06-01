from typing import Dict, Callable, Any, List
import logging

class Registry:
    """一个通用的注册表类，可以用来注册任何组件（Model, Loss, Optim 等）"""
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Callable] = {}

    def register(self, name: str = None, force: bool = False):
        """装饰器：注册模块"""
        def decorator(obj: Callable):
            target_name = name if name is not None else getattr(obj, '__name__', str(obj))
            target_name = target_name.lower()
            if target_name in self._module_dict:
                if not force:
                    raise KeyError(f"模块 {target_name} 已经在 {self.name} 中注册过了！")
                else:
                    logging.warning(f"⚠️ 警告: 模块 {target_name} 已经在 {self.name} 中注册过了，将被强制覆盖。")
            self._module_dict[target_name] = obj
            return obj
        return decorator
    
    def get(self, name: str) -> Callable:
        """获取已注册的类/函数本身，而不实例化"""
        name = name.lower()
        if name not in self._module_dict:
            available = ", ".join(self._module_dict.keys())
            raise ValueError(f"❌ 在 {self.name} 中找不到: '{name}'. 可用选项: {available}")
        return self._module_dict[name]

    def build(self, name: str, *args, **kwargs) -> Any:
        """工厂函数：根据名字构建对象"""
        target_obj = self.get(name)
        return target_obj(*args, **kwargs)

    def keys(self) -> List[str]:
        """返回所有已注册模块的名称列表"""
        return list(self._module_dict.keys())

    def __contains__(self, name: str) -> bool:
        """支持 'in' 操作符，例如：if 'resnet' in registry:"""
        return name.lower() in self._module_dict