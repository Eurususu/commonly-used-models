import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from ._datasetRegistry import register_dataset


__all__ = ["ClassificationDataset"]


@register_dataset("cls_dataset")
class ClassificationDataset(Dataset):
    """
    标准的图像分类数据集 (基于 ImageFolder 结构)
    """
    def __init__(self, data_dir, transforms=None, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        
        # 1. 扫描 data_dir 下的所有子文件夹，提取类别名称
        if not os.path.exists(self.data_dir):
            raise ValueError(f"❌ 找不到数据目录: {self.data_dir}")
            
        # 获取所有文件夹的名字，并排个序（保证每次类别的索引是一致的）
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                               if os.path.isdir(os.path.join(self.data_dir, d))])
        
        if not self.classes:
            raise ValueError(f"❌ 目录 {self.data_dir} 下没有找到任何类别文件夹！")

        # 2. 建立 类别名 -> 索引 的映射字典 (比如 {'cats': 0, 'dogs': 1, 'birds': 2})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 3. 遍历所有文件夹，收集所有的 (图片路径, 类别索引) 元组
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            
            # 支持的图片后缀
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            
            for root, _, fnames in sorted(os.walk(cls_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(valid_extensions):
                        path = os.path.join(root, fname)
                        idx = self.class_to_idx[cls_name]
                        self.samples.append((path, idx))
                        
        print(f"📊 成功加载分类数据集！共发现 {len(self.classes)} 个类别，{len(self.samples)} 张图片。")
        print(f"🏷️ 类别映射: {self.class_to_idx}")

    def __len__(self):
        # 返回数据集的总大小
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 获取当前索引的图片路径和标签
        img_path, label = self.samples[idx]
        
        # 2. 读取图片 (使用 PIL)
        # 必须使用 .convert('RGB')，防止读到 RGBA (4通道) 或者是灰度图 (1通道) 导致报错
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 警告: 读取图片失败 {img_path}, 错误: {e}")
            # 遇到坏图时，一种简单的容错机制是返回另一张随机图
            # 这里为了严谨，直接抛出异常，实际训练时可以 catch 或者清洗数据
            raise e
            
        # 3. 数据增强与预处理
        # 分类任务的 transform 极其简单，只需要对 image 做变换即可，不用管 label
        if self.transforms is not None:
            image = self.transforms(image)
            
        # 4. 返回 Tensor 格式的图片和整型的标签
        return image, label
