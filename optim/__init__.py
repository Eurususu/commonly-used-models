from utils.auto_import import auto_scan_and_import
from ._optimRegistry import register_optimizer, build_optimizer, list_optimizers


auto_scan_and_import(
    caller_file=__file__,
    caller_package=__name__,
)

__all__ = [
    "build_optimizer",
    "list_optimizers",
    "register_optimizer"
]