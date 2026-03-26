import os
import shutil
import numpy as np
from PIL import Image

def create_dummy_imagenet(root_dir="./dummy_imagenet", num_classes=3, imgs_per_class=100):
    """
    创建一个迷你的假 ImageNet 数据集目录，用于快速测试
    """
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    
    classes = [f"class_{i:02d}" for i in range(num_classes)]
    
    print(f"📦 正在构建虚拟 ImageNet 数据集 -> {root_dir}")
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(imgs_per_class):
            # 生成一张随机的 RGB 噪声图片 (大小 300x300)
            random_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            img = Image.fromarray(random_img)
            img_path = os.path.join(cls_dir, f"img_{i:03d}.jpg")
            img.save(img_path)
            
    print(f"✅ 虚拟数据集创建完毕！包含 {num_classes} 个类别，每个类别 {imgs_per_class} 张图片。")
    return root_dir


data_dir = create_dummy_imagenet()
print(f"虚拟数据集路径: {data_dir}")