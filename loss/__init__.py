import os
import importlib
from ._lossRegistry import register_loss, build_loss, list_losses



current_dir = os.path.dirname(__file__) # 获取当前目录路径
for filename in os.listdir(current_dir):
    # 筛选出普通的 Python 文件（排除 __init__.py 这种以 _ 开头的文件）
    if filename.endswith('.py') and not filename.startswith('_'):
        # 去掉 ".py" 后缀，拿到模块名，比如 "resnet.py" 变成 "resnet"
        module_name = filename[:-3]

        # 这行代码等价于手动写的：from . import module_name
        importlib.import_module(f".{module_name}", package=__name__)
            
            
__all__ = [
    "build_loss",
    "list_losses",
    "register_loss"
]