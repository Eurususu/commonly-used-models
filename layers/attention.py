import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from utils.cat_uncat import cat_keep_shapes, uncat_with_shapes
from typing import List, Tuple

__all__ = [
    'LinearKMaskedBias',
    'SelfAttention',
    ]


'''
Meta 的研究员在训练 DINOv3 这个百亿参数级的巨兽时发现：给 Q 和 K 同时加偏置是过度参数化的（Redundant）。两个偏置在点积时会疯狂耦合，反而容易导致训练不稳定。
因此，他们做了一个决定：保留 Q 和 V 的偏置，强行杀掉（归零）K 的偏置。 这就是 LinearKMaskedBias 存在的意义。

输出维度 o 必须能被 3 整除（Q 占前 1/3，K 占中间 1/3，V 占后 1/3）。
'''
class LinearKMaskedBias(nn.Linear):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            # 1. 先用 1 填满 mask
            bias_mask = torch.ones_like(self.bias)
            # 2. 将中间 1/3 的区域（即 K 的区域）强行置为 0
            bias_mask[o // 3 : 2 * o // 3].fill_(0)
            # 3. 注册 buffer
            self.register_buffer("bias_mask", bias_mask)
    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)
    

'''
根据前面的embeding的输出形状是[B,N,D] 通过线性层将qkv映射到qkv的3倍维度
然后取出q,k,v,做必要的transpose，加入RoPE旋转位置编码，然后计算attention

torch.nn.functional.scaled_dot_product_attention(q, k, v): 它会在底层自动调用 FlashAttention 或 xFormers 等由 CUDA 极致优化的算子，
不仅能让显存占用从 O(N^2) 降到 O(N)，还能让运算速度翻倍。

“Prefix” (前缀) 保护机制：
这是视觉模型特有的痛点。输入进来的 Token 序列（长度为 N）不仅仅包含图像 Patch，最前面通常还会有特殊的 [CLS] 标签，或者 DINOv3 特有的 [REG] (寄存器) 标签。
这些特殊的标签是没有 2D 物理坐标的！
所以代码计算了 prefix = N - sin.shape[-2]，精准地把前面的特殊 Token（q_prefix）切出来，只对后面的图像 Token 应用 RoPE 旋转，最后再把它们拼回去 (torch.cat)。

混合精度防线： RoPE 的 \sin 和 \cos 必须用高精度（通常是 FP32）计算，否则长文本/大图的位置信息会因精度截断而错乱。
代码里先把 q 和 k 强行转换成 rope 的精度（to(dtype=rope_dtype)），算完后再转回原来的类型（如 BF16）。
'''
# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            mask_k_bias: bool = False,
            device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        num_heads = dim // num_heads
        self.scale = num_heads**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        # 将embedding的输出映射到qkv也就是三倍的维度
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k


    def forward(self, x: Tensor, attn_bias=None, rope=None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x
    
    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> list[Tensor]:
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, dim=2) # [B, N, num_heads, head_dim]
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]] # [B, num_heads, N, head_dim]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope) # 注入位置编码
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v) # softmax(QK^T / sqrt(head_dim))*V
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])