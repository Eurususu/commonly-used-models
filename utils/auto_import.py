# utils/importer.py
import os
import importlib
from typing import List, Optional

def auto_scan_and_import(caller_file: str, caller_package: str, exclude: Optional[List[str]] = None):
    """
    自动扫描当前目录并动态导入 Python 模块，用于触发注册表。
    
    Args:
        caller_file (str): 调用者所在的文件路径，传入 __file__ 即可。
        caller_package (str): 调用者所在的包名，传入 __name__ 即可。
        exclude (List[str], optional): 需要排除的模块名称列表（不带 .py 后缀）。默认为空。
    """
    # 处理可变默认参数的最佳实践
    if exclude is None:
        exclude = []

    # 1. 解析调用者所在的目录
    current_dir = os.path.dirname(caller_file)

    # 2. 遍历该目录
    for filename in os.listdir(current_dir):
        # 筛选普通的 Python 文件，跳过 _ 开头的文件
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]

            # 3. 如果不在排除列表中，则进行动态导入
            if module_name not in exclude:
                importlib.import_module(f".{module_name}", package=caller_package)