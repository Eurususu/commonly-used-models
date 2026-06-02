from ._transformsRegistry import register_transform
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
from ._transformsRegistry import build_transforms


__all__ = ["ColorJitter", "Resize", "RandomHorizontalFlip", "ToTensor", "Normalize", 
           "CenterCrop", "RandomCrop", "RandomSelect", "RandomResize", "RandomSizeCrop",
           "FormatDETR"]

@register_transform("color_jitter")
class ColorJitter:
    """
    支持概率触发的色彩抖动 (像素级增强)
    """
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        # 在内部实例化官方的 ColorJitter 来干脏活
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target=None):
        # 按概率决定是否触发
        if random.random() < self.p:
            image = self.jitter(image)
            
        # ⚠️ 架构师法则重申：ColorJitter 是像素级增强，绝对不要修改 target(几何坐标)！
        return image, target
    
@register_transform("resize")
class Resize:
    def __init__(self, size):
        """
        size: 可以是一个整数 (短边缩放)，也可以是一个元组 (h, w)
        """
        self.size = size

    def __call__(self, image, target=None):
        # 1. 记录原始图像的宽高 (兼容 PIL Image 和 Tensor)
        if isinstance(image, torch.Tensor):
            h_orig, w_orig = image.shape[-2:]
        else:
            w_orig, h_orig = image.size 

        # 2. 调整图像大小
        image = F.resize(image, self.size)

        # 3. 如果没有 target (比如推理阶段)，直接返回
        if target is None:
            return image, None

        # 4. 获取新的图像宽高
        if isinstance(image, torch.Tensor):
            h_new, w_new = image.shape[-2:]
        else:
            w_new, h_new = image.size

        # DETR 通常需要记录每张图片的真实/原始尺寸 (用于后续反推绝对坐标和 mAP 计算)
        target["size"] = torch.tensor([h_new, w_new])

        # 5. 处理边界框 (非常关键！)
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            
            # 💡 假设1：如果你的 boxes 是绝对像素坐标 [x_min, y_min, x_max, y_max]
            # 我们需要计算缩放比例并应用到框上
            scale_w = w_new / w_orig
            scale_h = h_new / h_orig
            
            # 注意：如果你的数据集输出的已经是 0~1 的归一化坐标，
            # 那么不需要乘 scale_w 和 scale_h，这部分可以直接跳过！
            # (通常标准做法是：Dataset 输出绝对坐标 -> Resize 缩放绝对坐标 -> 最后 Normalize 时转成 0~1)
            
            boxes[:, 0::2] *= scale_w  # x1, x2 乘上宽度缩放比
            boxes[:, 1::2] *= scale_h  # y1, y2 乘上高度缩放比
            
            target["boxes"] = boxes
            
        return image, target

'''
核心原则：
在Transforms的整个旅途中，保持绝对坐标 [x1, y1, x2, y2]，
只在喂给 Loss 函数的前一秒（也就是 FormatDETR），再把它变成归一化坐标！
'''
@register_transform("random_horizontal_flip")
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            # 1. 翻转图片
            image = F.hflip(image)
            
            # 2. 同步翻转目标框 (假设框是归一化的 [cx, cy, w, h])
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone() # 防御性编程，避免原地修改报错
                # 把中心点 x 坐标翻转 (因为是 0~1 归一化，所以直接用 1 - cx)
                if isinstance(image, torch.Tensor):
                    w = image.shape[-1]
                else:
                    w = image.size[0]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
                
        return image, target

@register_transform("to_tensor")
class ToTensor:
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target

@register_transform("normalize")
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def _crop(image, target, region):
    """
    底层核心裁剪逻辑，负责同步裁剪图像和真实标签
    region 格式: (top, left, height, width)
    """
    i, j, h, w = region
    
    # 1. 裁剪图像本身
    image = F.crop(image, i, j, h, w)
    
    if target is None:
        return image, target

    # 2. 更新图像尺寸记录
    target["size"] = torch.tensor([h, w])

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].clone() # 防御性编程，避免原地修改报错
        
        # 3. 坐标平移 (减去左上角偏移量)
        boxes[:, 0::2] -= j  # x1, x2 减去 left
        boxes[:, 1::2] -= i  # y1, y2 减去 top
        
        # 4. 边界截断 (将框限制在新的图片范围内 0~w, 0~h)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        # 5. 目标过滤 (如果截断后 右下角 <= 左上角，说明框完全在裁剪区域外)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        
        # 应用过滤
        boxes = boxes[keep]
        target["boxes"] = boxes
        
        if "labels" in target:
            target["labels"] = target["labels"][keep]
        if "iscrowd" in target:
            target["iscrowd"] = target["iscrowd"][keep]
            
        # 6. 重新计算截断后的目标面积 (极其关键！)
        if "area" in target:
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return image, target

@register_transform("center_crop")
class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            w, h = image.size

        th, tw = self.size
        
        # 如果图片比要裁剪的尺寸还小，就直接返回原图（或者你也可以加入 Pad 逻辑）
        if w < tw or h < th:
            return image, target
            
        # 计算中心裁剪的左上角坐标
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return _crop(image, target, (i, j, th, tw))


@register_transform("random_crop")
class RandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            w, h = image.size

        th, tw = self.size
        
        # 容错处理：如果图像本身小于裁切尺寸，直接将其整体保留
        if w <= tw and h <= th:
            return image, target

        # 计算随机裁剪的上限边界，防止越界
        max_i = max(0, h - th)
        max_j = max(0, w - tw)
        
        # 随机生成左上角坐标
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
        
        # 如果图像某一边小于裁剪尺寸，使用图像本身的长度
        crop_h = min(th, h)
        crop_w = min(tw, w)

        return _crop(image, target, (i, j, crop_h, crop_w))
    
@register_transform("random_select")
class RandomSelect:
    """
    分支控制器：在两条数据增强流水线中随机选择一条执行
    """
    def __init__(self, branch_a, branch_b, p=0.5):
        self.p = p
        # 🌟 递归调用工厂函数！把 YAML 里的子列表重新变成流水线对象
        self.transform_a = build_transforms(branch_a)
        self.transform_b = build_transforms(branch_b)

    def __call__(self, image, target=None):
        if random.random() < self.p:
            return self.transform_a(image, target)
        else:
            return self.transform_b(image, target)
        
@register_transform("random_resize")
class RandomResize:
    """
    从给定的尺寸列表中随机挑选一个短边尺寸进行缩放，并限制长边不超过 max_size
    """
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        # 随机抽取一个基础尺寸
        size = random.choice(self.sizes)
        
        # 计算缩放逻辑 (限制最大边长)
        if isinstance(image, torch.Tensor):
            h_orig, w_orig = image.shape[-2:]
        else:
            w_orig, h_orig = image.size
            
        if self.max_size is not None:
            min_original_size = float(min((w_orig, h_orig)))
            max_original_size = float(max((w_orig, h_orig)))
            if max_original_size / min_original_size * size > self.max_size:
                size = int(round(self.max_size * min_original_size / max_original_size))
        
        if isinstance(image, torch.Tensor):
            h_orig, w_orig = image.shape[-2:]
        else:
            w_orig, h_orig = image.size 


        image = F.resize(image, size)


        if target is None:
            return image, None


        if isinstance(image, torch.Tensor):
            h_new, w_new = image.shape[-2:]
        else:
            w_new, h_new = image.size


        target["size"] = torch.tensor([h_new, w_new])


        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            
            scale_w = w_new / w_orig
            scale_h = h_new / h_orig
            
            
            boxes[:, 0::2] *= scale_w  # x1, x2 乘上宽度缩放比
            boxes[:, 1::2] *= scale_h  # y1, y2 乘上高度缩放比
            
            target["boxes"] = boxes
        
        return image, target

@register_transform("random_size_crop")
class RandomSizeCrop:
    """
    在给定范围内随机生成一个裁剪尺寸，然后调用你之前写的 _crop 逻辑
    """
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target=None):
        w = image.shape[-1] if isinstance(image, torch.Tensor) else image.size[0]
        h = image.shape[-2] if isinstance(image, torch.Tensor) else image.size[1]
        
        # 随机生成一个介于 min 和 max 之间的尺寸
        crop_w = random.randint(self.min_size, min(w, self.max_size))
        crop_h = random.randint(self.min_size, min(h, self.max_size))
        
        # 随机生成左上角
        i = torch.randint(0, h - crop_h + 1, size=(1,)).item()
        j = torch.randint(0, w - crop_w + 1, size=(1,)).item()
        
        # 这里需要导入你上一回合写的 _crop 函数
        return _crop(image, target, (i, j, crop_h, crop_w))
    
@register_transform("format_detr")
class FormatDETR:
    """
    DETR 数据流的最终收尾工作：
    将绝对像素的 [x1, y1, x2, y2] 转换为 0~1 归一化的 [cx, cy, w, h]
    """
    def __call__(self, image, target=None):
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            
            # 因为通常放在 ToTensor 之后，image 已经是 [C, H, W] 的张量了
            h, w = image.shape[-2:]
            
            # 1. 坐标系转换: xyxy -> cxcywh
            boxes_cxcywh = torch.zeros_like(boxes)
            boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
            boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
            boxes_cxcywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # w
            boxes_cxcywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # h
            
            # 2. 归一化: 严格除以图像的宽和高
            image_size = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
            boxes_normalized = boxes_cxcywh / image_size
            
            # 3. 截断保护: 确保没有任何坐标越界 0~1 范围 (针对某些裁剪后产生的极端浮点误差)
            target["boxes"] = boxes_normalized.clamp(min=0.0, max=1.0)
            
        return image, target
