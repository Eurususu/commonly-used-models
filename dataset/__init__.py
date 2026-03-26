# dataset/__init__.py
from utils.auto_import import auto_scan_and_import
from ._datasetRegistry import build_dataset, register_dataset, list_datasets
from ._transformsRegistry import build_transforms, register_transform, list_transforms
from .dataloaders import create_dataloader

# 自动扫描当前目录，触发所有 Dataset 的注册装饰器
# 排除掉注册表文件和 dataloader 工厂文件
auto_scan_and_import(
    caller_file=__file__, 
    caller_package=__name__, 
    exclude=["dataloaders"]
)

# 统一对外暴露 API
__all__ = [
    'build_dataset',
    'register_dataset',
    'list_datasets',
    'build_transforms',
    'register_transform',
    'list_transforms',
    'create_dataloader',
]