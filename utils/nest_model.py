from typing import Optional
from torch import Tensor


'''
NestedTensor
计算机视觉的 Batch 训练中，所有的图片必须保持相同的尺寸（比如拼成一个 [B, 3, 800, 1024] 的大张量）。
但现实中，图片有长有宽。我们通常的做法是把它们放到左上角，右边和下边用 0 填充（Padding）成一样大。
但 Transformer 具有全局注意力机制，如果不加限制，模型会去计算那些无意义的“黑色填充区域”的注意力！
NestedTensor它把数据张量 (tensors) 和 布尔掩码 (mask) 强行绑定在了一起。
tensors: 补齐后的图片数据。
mask: 一个 2D 的布尔矩阵。真实图像的像素位置是 False，填充出来的黑边位置是 True
在网络里流转，真实的特征和“哪里是废料”的信息就永远绑在一起，方便下游的 Attention 模块随时忽略这些废料。
'''
__all__ = ['NestedTensor']

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return len(self.tensors)