"""
模型包的初始化文件
在这里触发所有子模块的加载，完成模型注册
"""
from utils.auto_import import auto_scan_and_import

# 1. 暴露常用的工厂函数和基类，方便外部直接从包名导入
from ._taskRegistry import build_task, list_tasks, register_task

auto_scan_and_import(
    caller_file=__file__,
    caller_package=__name__,
    exclude=[]
)
            

# 3. 声明对外开放的 API
__all__ = [
    'build_task',
    'list_tasks',
    'register_task',
]

