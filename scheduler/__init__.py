from ._schedulerRegistry import build_scheduler, register_scheduler, list_schedulers
from utils.auto_import import auto_scan_and_import

# 自动扫描当前目录，触发所有 Scheduler 的注册装饰器
auto_scan_and_import(
    caller_file=__file__, 
    caller_package=__name__, 
)


__all__ = [
    'build_scheduler',
    'register_scheduler',
    'list_schedulers'
]