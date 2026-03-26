# dataset/segmentation.py
import os
import torch
from torch.utils.data import Dataset
# 假设你以后可能会用到 PIL 或 cv2 读取图片
# from PIL import Image 
from ._datasetRegistry import register_dataset

@register_dataset("seg_dataset")
class SegmentationDataset(Dataset):
    """
    标准的图像分割数据集
    """
    def __init__(self, data_dir, image_folder="images", mask_folder="masks", transform=None, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_folder)
        self.mask_dir = os.path.join(data_dir, mask_folder)
        self.transform = transform
        
        # 获取所有图片的文件名 (假设原图和 mask 名字一样)
        # 实际工程中建议加个 try-except 或者 assert 检查路径是否存在
        if os.path.exists(self.image_dir):
            self.images = sorted(os.listdir(self.image_dir))
        else:
            self.images = []
            print(f"⚠️ 警告: 找不到图片目录 {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # --- 这里写具体的读取逻辑 ---
        # 比如: 
        # image = Image.open(img_path).convert("RGB")
        # mask = Image.open(mask_path).convert("L")
        
        # 为了演示，我们先返回假数据 (Dummy Tensors)
        # 假设是 3 通道 224x224 的图，和 1 通道的 Mask
        image = torch.randn(3, 224, 224) 
        mask = torch.randint(0, 2, (1, 224, 224)).float()
        
        # 如果有数据增强，就应用它
        if self.transform is not None:
            # 注意：分割任务的数据增强通常需要同时对 img 和 mask 做相同的空间变换！
            pass 
            
        return image, mask