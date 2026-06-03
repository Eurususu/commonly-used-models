import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from ._datasetRegistry import register_dataset

__all__ = ["COCODetectionDataset"]

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

# 🌟 新增：COCO 官方 90 类 ID 到连续 80 类索引的映射表
COCO_90_TO_80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

@register_dataset("det_coco_dataset")
class COCODetectionDataset(Dataset):
    """
    标准的 COCO 格式目标检测数据集，专为 DETR 适配目标字典格式
    """
    def __init__(self, data_dir, ann_file, transforms=None, **kwargs):
        super().__init__()
        if kwargs:
            import logging
            logging.warning(f"COCODetectionDataset 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
            
        if COCO is None:
            raise ImportError("❌ 找不到 pycocotools，请先运行: pip install pycocotools")
            
        self.data_dir = data_dir
        self.transforms = transforms
        
        print(f"⏳ 正在加载 COCO 标注文件: {ann_file} ...")
        self.coco = COCO(ann_file)
        
        # 获取所有包含标注的图像 ID 列表
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(f"✅ 成功加载，共有 {len(self.ids)} 张有效图像。")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. 获取图像基础信息
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        
        # 2. 读取图像
        # 使用 PIL 读取，因为 torchvision 的 transforms 对 PIL Image 支持最好
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # 3. 加载该图像对应的所有标注框
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_anns = self.coco.loadAnns(ann_ids)

        # 4. 解析并重组为 DETR 需要的 Target 字典
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in coco_anns:
            # 过滤掉无效标注
            if 'ignore' in ann and ann['ignore'] == 1:
                continue
            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                continue

            # 🌟 修复：将原始的 category_id 映射到 0~79
            raw_cat_id = ann['category_id']
            if raw_cat_id not in COCO_90_TO_80:
                continue # 如果遇到奇怪的 ID 直接跳过
            mapped_label = COCO_90_TO_80[raw_cat_id]
                
            # COCO 默认格式是 [x_min, y_min, width, height]
            x, y, bw, bh = ann['bbox']
            
            # 为了适配前几回合我们写的 Resize 数据增强，
            # 我们先将其转换为绝对坐标格式 [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + bw, y + bh])

            # 🌟 把映射后的连续索引存入 labels
            labels.append(mapped_label)

            area.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # 组装为 Tensor 格式
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # 记录图像原始尺寸和当前尺寸 (DETR 的后处理和 mAP 计算强依赖这两个字段)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # 5. 应用我们上一回合自定义的支持 (image, target) 双输入的 Transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
