from dataset import list_datasets
from models import list_models
from optim import list_optimizers
from loss import list_losses
from scheduler import list_schedulers
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store_true", help="列出所有可用的数据集")
    parser.add_argument("--models", action="store_true", help="列出所有可用的模型")
    parser.add_argument("--losses", action="store_true", help="列出所有可用的损失函数")
    parser.add_argument("--optimizers", action="store_true", help="列出所有可用的优化器")
    parser.add_argument("--schedulers", action="store_true", help="列出所有可用的调度器")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.datasets:
        print("Available datasets:",list_datasets())
    if args.models:
        print("Available models:",list_models())
    if args.losses:
        print("Available losses:",list_losses())
    if args.optimizers:
        print("Available optimizers:",list_optimizers())
    if args.schedulers:
        print("Available schedulers:",list_schedulers())
