# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn

from utils.nest_model import NestedTensor

'''
DINOv3 骨干网络里不是已经有 RoPE（旋转位置编码）了吗？为什么这里又要搞一套位置编码？
原因在于：RoPE 是相对位置编码，主要用于骨干网络提取特征；
而 DETR 的解码器（Decoder）在生成边界框时，强依赖绝对的空间坐标感（比如“这个物体在画面的左上角”）。
因此，我们必须在进入 DETR 头之前，显式地把 2D 的绝对物理坐标注入到特征图中。


'''

__all__ = ["PositionEmbeddingSine", "PositionEmbeddingLearned"]

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask # [Batch, Height, Width]
        assert mask is not None
        not_mask = ~mask
        # 主要是用来生成height和width的坐标位置，但是黑边不计算
        # cumsum 是 Cumulative Sum（累加）的缩写。参数 1 代表沿着第 1 个维度（也就是 Height，高度/行）向下进行累加。
        # 真实图像里（数字是1），往下走一步，累加值就 +1。所以第一行是 1，第二行是 2，第三行是 3……
        # 一旦你跨出了真实图像，进入了下方的 Padding 黑边区域（数字变成了 0），无论往下走多少行，累加值都会停留在最后那个真实的坐标上（+0），不再增加！
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # 减去0.5 就是取坐标中心，如0.5, 1.5 2.5 ...
            # 提取最大值并归一化, [:, -1:, :] 是 PyTorch 的切片语法，意思是取最后一行。
            # 用当前坐标除以最大坐标，其实就是归一化，将坐标范围缩放到 [0, 1]
            # 然后再乘以2π，把 (0, 1] 的百分比比例，放大到了 (0, 2π] 
            # 为什么要映射到 2π？ 因为紧接着在下一行代码里，这些坐标就要被送进 sin() 和 cos() 三角函数里去提取频域特征了！一个完整的正弦周期正好是 2π。
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale # 
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        else:
            y_embed = (y_embed - 0.5) * self.scale
            x_embed = (x_embed - 0.5) * self.scale
        # 如果num_pos_feats=128，则dim_t = [0, 1, 2, 3, 4, 5... 127]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 2 * (dim_t // 2) 它会把数组变成：[0, 0, 2, 2, 4, 4... 126, 126] 
        # 正弦（Sin）和余弦（Cos）是成对出现的，它们共享同一个频率分母！
        # temperature在之前的实现中，temperature = 10000.0
        # 最后得到的dim_t=10000^([0, 0, 2, 2, 4, 4... 126, 126]/128)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # dim_t是128维的一个数组，x_embed[:, :, :, None]是把x_embed的第4维扩展为1维，然后除以dim_t，就是广播机制，得到[B, H, W, 128]
        # 刚才算出的那个 0∼2π 的标量坐标，被同时放进了 128 个不同尺度的坐标系里。有的坐标系刻度极细（高频），有的坐标系刻度极粗（低频）
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 0::2：切片取出所有的偶数索引（0, 2, 4...），对它们套上 .sin() 函数
        # 1::2：切片取出所有的奇数索引（1, 3, 5...），对它们套上 .cos() 函数。
        # stack 变成 [B, H, W, 64, 2] 再进行flatten变成 [B, H, W, 128]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 最后得到形状[Batch, 256, Height, Width]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        # 那个硬编码的 50 限制了它处理高分辨率图像的能力 不能超过50 x 50
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos