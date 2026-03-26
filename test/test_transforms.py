import yaml
from dataset import build_transforms, list_transforms

print(f"📋 可用的 Transforms 列表: {list_transforms()}")

cfg = yaml.safe_load(open("config/dataset/seg_data.yaml"))
transforms = build_transforms(cfg["train_transforms"])

print(transforms)