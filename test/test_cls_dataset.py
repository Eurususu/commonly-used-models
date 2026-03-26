import os
import shutil
import torch
import numpy as np
from PIL import Image
import yaml


from dataset import create_dataloader, build_transforms, list_datasets
print(f"📋 可用的数据集列表: {list_datasets()}")



cfg = yaml.safe_load(open("config/dataset/cls_data.yaml"))

try:
    # 构建 Transform 流水线
    print("\n🔧 正在组装 Transforms 流水线...")
    train_transforms = build_transforms(cfg["train_transforms"])
    print(train_transforms)

    # 将组装好的 transform 塞进 dataset 的配置中
    cfg["dataset"]["kwargs"]["transforms"] = train_transforms

    # 构建 DataLoader
    print("\n🚀 正在创建 DataLoader...")
    train_loader = create_dataloader(
        dataset_name = cfg["dataset"]["name"],
        dataset_cfg = cfg["dataset"]["kwargs"],
        loader_cfg = cfg["train_loader"]
    )

    # 抽取一个 Batch 进行终极验证
    print("\n🎉 成功创建 DataLoader！正在抽取一个 Batch 进行验证...")

    images, labels = next(iter(train_loader))

    print(f"✅ 成功读取 Batch！")
    print(f"🖼️ Images Tensor 形状: {images.shape}  (期望: [16, 3, 224, 224])")
    print(f"🏷️ Labels Tensor 形状: {labels.shape}  (期望: [16])")
    print(f"🏷️ Labels 内容: {labels}")

    # 验证数值范围 (经过 Normalize 后，通常在 -3 到 3 之间)
    print(f"📊 Images 数值范围: Min = {images.min().item():.4f}, Max = {images.max().item():.4f}")

except Exception as e:
    print(f"\n❌ 测试失败！捕获到异常: {e}")
    raise e


print(f"\n🏁 数据流测试完美结束！")