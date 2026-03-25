"""
模型包的初始化文件
在这里触发所有子模块的加载，完成模型注册
"""
import os
import importlib

# 1. 暴露常用的工厂函数和基类，方便外部直接从包名导入
from ._modelRegistry import build_model, list_models, register_model
from .BaseModel import BaseModel

# # 2. 自动导入各个具体的模型文件
# # 只要这些文件被 import，它们里面的 @register_model 就会执行，把模型塞进 registry 字典里
# from . import unet
# from . import resnet
# from . import vgg

# 2. 【核心魔法】自动扫描并动态导入当前目录下的所有模型文件
current_dir = os.path.dirname(__file__) # 获取当前目录路径
for filename in os.listdir(current_dir):
    # 筛选出普通的 Python 文件（排除 __init__.py 这种以 _ 开头的文件）
    if filename.endswith('.py') and not filename.startswith('_'):
        # 去掉 ".py" 后缀，拿到模块名，比如 "resnet.py" 变成 "resnet"
        module_name = filename[:-3]

        # 为了严谨，我们跳过基础组件文件，只导入具体的模型文件
        if module_name not in ["BaseModel"]:
            # 这行代码等价于手动写的：from . import module_name
            importlib.import_module(f".{module_name}", package=__name__)
            

# 3. 声明对外开放的 API
__all__ = [
    'build_model',
    'list_models',
    'register_model',
    'BaseModel'
]

