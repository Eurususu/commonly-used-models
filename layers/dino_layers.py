import torch.nn as nn
from typing import Union, Callable, List, Optional, Tuple, Literal
from torch import Tensor
import torch
import math
import numpy as np
import torch.nn.functional as F
from .attention import SelfAttention
from .utils import cat_keep_shapes, uncat_with_shapes

__all__ = [
    'LayerScale',
    'ListForwardMixin',
    'Mlp',
    'PatchEmbed',
    'RMSNorm',
    'RopePositionEmbedding',
    'SwiGLUFFN',
    'SelfAttentionBlock'
]


'''
dim : 它定义了一个长度为 dim（也就是特征维度，比如 768 或 4096）的一维可学习参数（向量）。这意味着，每一个特征通道都会有一个专属的缩放系数。
init_values: 这是最关键的一点。gamma 被初始化为一个极小的值（默认 10^{-5}）。
forward: 在前向传播时，它只是做了一个简单的逐元素乘法（Element-wise multiplication）：x \times \gamma

在 Transformer 中，残差连接的标准公式是：y = x + F(x) 
其中 F(x) 是自注意力模块（Attention）或前馈网络（FFN）的输出。

当网络变得非常深（比如 DINOv3 的 40 层）时，在训练初期，各个 $F(x)$ 产生的激活值方差会随着深度急剧放大，导致梯度爆炸或消失。
深层网络在刚开始训练时，往往会因为剧烈的特征变换而“迷失方向”，难以收敛。

加入LayerScale，它可以在每个层之间进行缩放，从而解决深度网络中的梯度爆炸问题。
y = x + gamma * F(x)
因为 gamma 在初始化时极其接近 0（1e-5），所以在训练的第一步，公式约等于：y=x

在训练的最开始，无论这个网络有 10 层还是 100 层，它在数学上都等价于一个“恒等映射（Identity Mapping）”。输入 $x$ 几乎原封不动地穿过了整个网络。
随着训练的推进，反向传播会根据损失函数的需要，缓慢且平滑地更新 $\gamma$ 的值，逐渐“唤醒”每一层的功能。
网络可以自己决定：对哪些通道放大特征注入，对哪些通道继续保持静默。

DINOv3 要挑战的是 40 层的深度和 7B 的参数量。如果没有 LayerScale 强行将深层网络的初期行为“压制”成恒等映射，
这种规模的自监督学习（本身就不如监督学习稳定）几乎是不可能调通的。它提供了一种极其廉价（参数量仅增加一点点）但收益巨大的初始化稳定性。
'''

class LayerScale(nn.Module):
    def __init__(self, dim : int, init_values: Union[float, Tensor] = 1e-5,
                 inplace: bool = False, device=None) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

'''
全连接层（Linear / MLP）通常只关心最后一个维度（特征维度 C），前面的维度不管是 (Batch, Length) 还是 (Batch, Height, Width)
对线性层来说都是独立的“Token”。

cat_keep_shapes（打包）
这个函数的任务是把 x_list（比如包含了不同分辨率特征图的列表）压扁并拼在一起。
shapes: 记录下每个 Tensor 原本的样子，为了以后还原。
num_tokens: 这里用了一个非常巧妙的写法 x.select(dim=-1, index=0).numel()。意思是抛弃最后一个特征维度（C），
计算前面所有维度乘起来的总长度（即 Token 的总数量）。例如 (2, 14, 14, 768) 会算出 2 * 14 * 14 = 392。
flattened: 将每个 Tensor 展平成 (num_tokens, C) 的形状，并在第 0 维度上拼接。
多个形状各异的 Tensor 变成了一个巨大的 (Total_Tokens, C) 的二维矩阵。

uncat_with_shapes（解包）
等网络计算完之后，把那个巨大的二维矩阵拆回去。
outputs_splitted: torch.split_with_sizes根据打包时记录的 num_tokens，把大矩阵重新切分成多个小矩阵。
shapes_adjusted:  网络前向传播可能会改变特征维度（比如把 C 映射成了 C_{out}）。
所以还原时，不能完全照搬原始 shape，而是保留前面的空间维度 shape[:-1]，把最后一个维度替换成网络输出的真实维度 flattened.shape[-1]
reshape: 最后把每个小块 reshape 回原始的多维结构

Mixin（混入模式）
自己不实现具体的业务逻辑（__init__ 直接报错，没有具体的 forward）。
但是把它继承给任何一个已有的 PyTorch 模块，那个模块就瞬间获得了 forward_list 这个超能力。
它的逻辑就是：打包 -> 调用宿主类的 forward() -> 解包

为什么要打包推理再解包呢？
假设你有一个列表 x_list 包含了 5 个不同分辨率的特征图，你需要让它们都通过 Mlp
菜鸟做法: outputs = [mlp(x) for x in x_list]
在 GPU 上，每一次模型调用（哪怕是很小的矩阵相乘）都会有内核启动开销（Kernel Launch Overhead）。写 for 循环会导致 GPU 没吃饱就被频繁打断。
专业做法：outputs = mlp.forward_list(x_list)
它在底层把 5 个运算合并成了一次巨大的矩阵乘法。GPU 最喜欢的就是这种单一的、海量的矩阵运算。计算完成后再用 CPU 极快的内存操作把结果切分回来。

'''
class ListForwardMixin(object):
    def __init__(self, x: Tensor):
        raise NotImplementedError
    
    def forward_list(self, x_list: list[Tensor]) -> list[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)
    
"""
Callable 指的是“一切可以在后面加括号 () 运行的东西”。
Callable 后面的方括号用来严格定义这个函数的输入参数和返回值，格式是 Callable[[输入参数表], 返回值类型]
... (省略号)： 这是 Python 语法中一个真实的符号（叫做 Ellipsis）。在这里它代表：“我不管你调用这个函数时需要传什么参数、传几个参数，随便你

默认值是 nn.GELU，而不是 nn.GELU()
菜鸟写法：
def __init__(self, act_name: str = "gelu"):
    if act_name == "gelu":
        self.act = nn.GELU()
    elif act_name == "relu":
        self.act = nn.ReLU()
    # 以后想加个 Swish，我还得来改这段代码，极其违背“开闭原则”！
专家写法：
def __init__(self, act_layer: Callable[..., nn.Module] = nn.GELU):
    self.act = act_layer()
"""
class Mlp(nn.Module, ListForwardMixin):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
            device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

'''
PatchEmbed
将传统的二维图像矩阵，切割并转换成 Transformer 能够理解的“一维 Token 序列”。 也就是代码注释里写的 (B, C, H, W) -> (B, N, D)。

x = self.proj(x)：每进行一次卷积计算，实际上就是把当前 16x16 区域内的所有像素，通过线性映射压缩成了一个长度为 embed_dim（比如 768）的向量！
(B, 3, 224, 224) -> (B, 768, 14, 14)

x = x.flatten(2): Transformer 不需要 2D 结构，它只认“序列”。这行代码把空间维度 (14, 14) 压扁成了一个维度 196。
形状变成：(B, 768, 196)

x = x.transpose(1, 2): Transformer 的标准输入格式是 (批次大小, Token数量, 向量维度)。所以我们需要把 196 和 768 调换位置。
形状变成：(B, 196, 768) 也就是前面的 (B, N, D)

最后经过一个可选的 LayerNorm。如果 flatten_embedding 为 False，它还会把形状还原回 (B, 14, 14, 768)，
这通常是为了适配一些需要保留 2D 局部空间特征的下游任务（比如 Swin Transformer 或一些分割网络）。

公式 k = 1 / (in_chans \times patch_size^2) 是 PyTorch 线性层的标准 Kaiming 初始化变体，确保训练初期梯度的稳定。

'''
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Callable | None = None,
            flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)

        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        self.img_size = image_HW
        self.patch_size = patch_HW

        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x) # # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2) # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)# B H W C
        return x
    
    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


'''
RMSNorm：均方根归一化，已经全面取代统治Transformer的LayerNorm

传统的 LayerNorm 包含两个核心动作：
平移（Shift / Mean-centering）： 减去均值 \mu，把数据中心拉回 0。
缩放（Scale / Variance-scaling）： 除以标准差 \sigma，把数据分布压平。
2019 年，RMSNorm 的作者们提出了一个灵魂拷问：LayerNorm 到底是因为“平移”起作用，还是因为“缩放”起作用？经过大量实验，
他们得出了一个反直觉的结论：均值平移（减去均值）对模型的成功几乎没有贡献，真正让 Transformer 稳定收敛的是方差缩放。
既然均值没用，那干脆就不要算了！于是，RMSNorm 直接砍掉了均值计算，只保留了基于输入向量自身大小（均方根 RMS）的缩放操作。

公式：
x = x / sqrt(eps + mean(x^2)) * gamma

优势：
1. 计算速度更快 计算均值需要遍历一次整个特征张量，计算方差又要遍历一次，而且这两步之间存在数据依赖（必须先算完均值，才能算方差）。
RMSNorm 不需要算均值，直接算平方和再开根号。这极大地缩短了前向传播和反向传播的计算图。在实际测试中，RMSNorm 通常比 LayerNorm 快 10% 到 50% 不等。
2. 显存读写更少 在 GPU 计算中，很多时候瓶颈不是算力（FLOPs），而是内存访问（Memory Bound）。
LayerNorm 需要在 GPU 的高带宽内存（HBM）和计算单元（SRAM）之间来回搬运数据来计算均值和方差。
RMSNorm 减少了数据搬运的次数。同时，它移除了偏置参数 beta，这也省掉了一部分梯度和优化器状态（如 Adam 的动量）的显存占用。
3. 极其适合底层算子融合

使用场景：
1. LLM大语言模型
2. 视觉大模型 vit变体这种模型

不能使用的场景：
1. 基于旧模型微调（Fine-tuning）： 如果你下载了一个使用 LayerNorm 预训练好的模型（比如经典的 BERT、ResNet，或者早期的 ViT），你绝对不能在微调时把它改成 RMSNorm。
归一化方式是刻在模型权重 DNA 里的，强行更改会导致模型直接崩溃。batch norm的模型也不行
2. 特定任务：如果你的特定任务中，特征的绝对平移量（均值偏离 0 的程度）包含了非常关键的物理或语义信息，
去掉均值中心化可能会导致性能下降（但在标准的 CV 和 NLP 任务中，这种情况极其罕见）

另外为什么要先 .float() 转成单精度，算完又 .type_as(x) 转回去？
防溢出： 当模型使用 FP16 (半精度) 训练时，最大只能表示 65504。在计算 x.pow(2) 时，
如果 x 里的某个值大于 256，它的平方就会超过 FP16 的上限，导致溢出变成 NaN（Not a Number），整个模型瞬间崩溃。
将输入强制提升为 FP32 计算出精确的归一化结果后，再安全地转换回原来的精度格式（FP16/BF16），从而兼顾了数学稳定性与显存/计算速度。

'''
class RMSNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output =  self._norm(x.float()).type_as(x)
        return output * self.weight
    

'''
DINOv3 抛弃了传统的绝对位置编码，全面拥抱了 RoPE。这段代码就是 DINOv3 能够处理任意分辨率图像，并且对裁剪、缩放具有极强鲁棒性的核心密码。

assert embed_dim % (4 * num_heads) == 0: 
每个注意力头分配到的维度是 D_head = embed_dim / num_heads。
因为是二维图像，我们需要把 D_head 平分成两半，一半用来编码高度（H），一半用来编码宽度（W）。
所以各占 D_head / 2。
在 RoPE 的算法中，特征总是两两配对进行二维旋转运算的（需要生成正弦和余弦对）。
所以独特的频率周期数只需要 D_head / 2 / 2 = D_head / 4 个。这就是为什么 D_head 必须能被 4 整除。


base & min_period max_period: RoPE 是通过不同频率（周期）的正弦/余弦波来编码位置的。
base 是从自然语言处理（NLP）领域，特别是最初的 RoPE 论文和 LLaMA 模型中继承下来的经典参数（默认值通常是 10000.0）。
公式如下：
period = base ** (2 * i / (D_head // 2)) i从0到D_head // 4
对于i = 0 时，周期是 base^0 = 1
对于i = D_head // 4 时，周期是 base^(2 * D_head // 4 / (D_head // 2)) = base^1 = base
base=10000 在文本任务里很好用，因为文本是一个离散的、无限延伸的一维长序列。
但在二维图像中，图像的大小是有限的，盲目套用 10000 作为一个神秘的魔法数字，不仅不直观，而且可能导致有些维度的波长远远超过了图片的物理尺寸，完全失去意义。

min_period：代表变化最剧烈的波。它控制着最前几个特征维度。物理意义： 它负责捕获极小范围内的位置变化（比如相邻两个 Patch 之间的距离）。
max_period：代表变化最平缓的波。它控制着最后几个特征维度。物理意义： 它负责捕获全局的位置信息（比如整张图片的高和宽）
如果采用min_period和max_period方式来做的话，那么就是
base = max_period / min_period
period = base**(i) 其中i时torch.linspace从(0,1)中取D_head // 4个数

数据增强：
Shift (平移)： 给坐标加上一个随机偏移量。相当于告诉模型：“图像在视野中发生了整体平移，但物体的相对位置没变。”
Jitter (抖动)： 乘上一个独立的缩放系数，长宽缩放比例不同。相当于对图像进行了非等比的“拉伸/挤压”。
Rescale (缩放)： 乘上一个统一的缩放系数，模拟物体变大或变小。

theta计算：
公式是这样的：theta = 2 * pi * pos / period 这里的pos是网格上flatten之后的位置
coords[:, :, None]：形状是 [HW, 2, 1]。HW 是所有的图像块（Token），2 代表高度 y 和宽度 x 两个维度的坐标。
periods[None, None, :]: 形状是 [1, 1, D//4], 分配给二维坐标的不同周期（波长）
相除的结果： 形状自动扩展为 [HW, 2, D//4]
对于每一张图上的每一个 Patch（共 HW 个），它的 y 坐标会被除以 D/4 个不同的周期，
它的 x 坐标也会被除以 D/4 个不同的周期。这就把一个简单的二维物理坐标，映射成了高维空间中的一堆角度。

angles = angles.flatten(1, 2): 每一个 Token 都有了一个长度为 D/2 的角度向量。[HW, D//2]
这个向量的前一半是高度 y 产生的旋转角度，后一半是宽度 x 产生的旋转角度。高度和宽度的信息被天衣无缝地拼在了一起。

angles = angles.tile(2): 意思是把刚才那个长度为 D/2 的向量，原封不动地复制粘贴一遍。[HW, D]
把角度复制一倍，是为了让角度的维度 [HW, D] 能够和 Query（查询向量）与 Key（键向量）的维度完全对齐。
这样在下一步中，就可以极其高效地直接进行逐元素乘法（Element-wise multiplication），而不需要复杂的切片和拼接。

'''

class RopePositionEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            *,
            num_heads: int,
            base: float | None = 100.0,
            min_period: float | None = None,
            max_period: float | None = None,
            normalize_coords: Literal["min", "max", "separate"] = "separate", # 下拉菜单
            shift_coords: float | None = None,
            jitter_coords: float | None = None,
            rescale_coords: float | None = None,
            dtype: torch.dtype | None = None,
            device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")
        
        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head

        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()
    
    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

         # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw
        
        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base**(2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)) # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype) # shape:[D//4] range:[0, 1]
            periods = base**exponents # range [1, max_period / min_period]
            periods = periods / base # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


'''
传统的MLP的公式如下:
FFN(x) = Activation(W1*x + b1) * W2 + b2 
先把特征 x 放大（通常放大 4 倍），经过激活函数切掉负数，然后再投影回原来的维度。只有两条边（W_1 和 W_2）

SwiGLUFFN不仅仅是一个激活函数。 它是一个完整的 前馈神经网络模块

在 2020 年，Noam Shazeer 提出了一种名为 GLU (Gated Linear Unit，门控线性单元) 的变体。
而 SwiGLUFFN 就是使用了 SiLU（也叫 Swish）激活函数的 GLU。
SwiGLUFFN 的公式如下:
SwiGLUFFN(x) = (SiLU(W_1*x + b_1) * (W_2*x + b_2)) * W_3 + b_3
第一条路（信息路，代码中的 x1）： x 经过 W_1 并通过激活函数，提取非线性特征。
第二条路（门控路，代码中的 x2）： x 经过 W_2，不加任何激活函数，直接作为一个“门控（Gate）”。
然后将两条路乘起来，最后用 W_3 整合输出。

d = int(hidden_features * 2 / 3): 在传统的 MLP 中，隐藏层维度通常是输入维度的 4 倍（参数量主要来自两块大矩阵 W_1 和 W_2）。
但是在 SwiGLU 中，我们有 W_1、W_2、W_3 三块矩阵！如果保持原来的维度，模型的整体参数量会暴增 50%
保持参数量和计算量与传统 MLP 完全一致（公平对比），作者非常聪明地把隐藏层的维度缩小了 2/3。这样 3 \times (2/3) = 2，总参数量就对齐了

swiglu_hidden_features = d + (-d % align_to): 这行代码的作用是：无论你算出来的维度 d 是多少，
它都会强行把它向上取整到 8 的倍数（在一些极端的 LLM 里，这个数字可能是 128 或 256）。
'''
class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Optional[Callable[[Tensor], Tensor]] = None,
            drop: float = 0.0,
            bias: bool = True,
            align_to: int = 8,
            device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        bd = {"bias": bias, "device": device}
        if act_layer is None:
            act_layer = F.silu
        self.act_layer = act_layer
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, **bd)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, **bd)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, **bd)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w1(x), self.w2(x)
        hidden = self.act_layer(x1) * x2
        return self.w3(hidden)
    

'''
这是DINOV3的核心模块 自注意力残差模块

先看推理模式
在推理模式下具体是这样的：
x_attn = x + ls1(attn(norm1(x)))
x_ffn = x_attn + ls2(mlp(norm2(x_attn)))
在进入 Attention 和 MLP 之前先做 LayerNorm（self.norm1 和 self.norm2）。这是大模型能稳定训练的标配。
LayerScale (ls1, ls2)： 这是自 CaiT 论文引入的技术。它在残差相加之前，给特征的每一个通道乘上一个极小的可学习参数（init_values 通常是 10^{-5} 级别）。
这能让极其深的网络（比如上百层）在训练初期几乎等同于恒等映射（Identity），从而极大地稳定百亿级参数模型的收敛。

再看训练模式 drop path
为了防止模型过拟合，传统方法会在训练时以一定的概率（比如 10%）直接把整个 Attention 块的输出强制变成 0
传统的做法是先让这批数据老老实实算完 Attention，然后再乘以一个由 0 和 1 组成的随机掩码。 这在算力上是极大的浪费（算都算完了，你把它扔了）。
DINOv3 的极速做法：只算子集！只算一部分，而不是全部算完扔掉部分
drop_path：这个参数就是扔掉样本的比例，根据这个参数算出保留的样本数量sample_subset_size
通过 torch.randperm 随机挑出sample_subset_size个样本的索引 indices_1。
residual_scale_factor：放大系数, 为了保持期望值不变
torch.index_add： 把算完的子集，乘以一个放大系数 residual_scale_factor（为了保持期望值不变）
然后极其优雅地使用 torch.index_add 按照原来的索引，把它们“嵌”回原始的残差流 x 中。


'''
class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            ffn_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = SelfAttention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
            mask_k_bias: bool = False,
            device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


