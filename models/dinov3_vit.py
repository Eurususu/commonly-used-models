import torch.nn as nn
from layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from typing import Union, Callable, List, Optional, Tuple, Literal, Any, Dict, Sequence
import logging
from functools import partial
import torch
from utils.named_apply import named_apply
from torch import Tensor

from ._modelRegistry import register_model



logger = logging.getLogger("dinov3")

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

'''
1. 怕分辨率变化？上 RoPE。
2. 怕极深网络不收敛？上 LayerScale。
3. 怕背景亮点？加 Storage Tokens。
4. 怕特征耦合？做 Untied Norms。
5. 怕 Padding 浪费算力？做 List Flattening。

4.
Untied Norms 解绑归一化: 
在传统的 Vision Transformer (ViT) 中，所有的 Token（无论你是代表全局的标签，还是代表局部的像素）在输出前，都要挤过同一个 LayerNorm 层，被同一套均值和方差进行归一化。
这太反直觉了！不同性质的数据，怎么能用同一把尺子去量呢？ 于是他们引入了这两个参数来进行精细化的“解绑”

untie_cls_and_patch_norms: 是否让 [CLS]/[REG] 标签与普通的图像 [Patch] 标签使用不同的归一化层。
[CLS] 和 [REG] 包含的是高度浓缩的全局抽象语义（比如“这是一只猫”、“这是户外场景”）
[Patch] 包含的是极其细粒度的局部视觉特征（比如“这里的边缘是弯曲的”、“这里的像素是棕色的”）

untie_global_and_local_cls_norm: 在训练阶段，是否让全局大图 (Global Crops) 和局部小图 (Local Crops) 的 [CLS] 标签使用不同的归一化层。
DINOv3 训练时使用了 Multi-Crop 技术：既喂给模型包含全貌的“大图”（如 224 x 224），也喂给模型只包含某个特写的“小图”（如 96 x 96）
大图(global)的 [CLS] 看到的是整辆汽车，小图(local)的 [CLS] 看到的可能只是一个轮胎。
如果让它们共享同一个 CLS LayerNorm，局部特征的极端分布会“污染”全局特征的归一化参数，导致模型在处理完整图像时性能下降。

[CLS]（CEO）： 掌控全局战略（全局语义）。
[Patch]（基层员工）： 负责具体执行（局部像素）。
Local [CLS]（分公司经理）： 只了解局部情况（局部视野）。
如果用同一套 KPI（同一个 LayerNorm）去考核所有人，公司必然乱套。
解绑归一化（Untying Norms）就是为不同的特征分配不同的考核标准（独立的缩放和平移参数），从而最大程度地释放了 ViT 的表征潜力。
'''

__all__ = ['', ]

class DinoVisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            pos_embed_rope_base: float = 100.0,
            pos_embed_rope_min_period: float | None = None,
            pos_embed_rope_max_period: float | None = None,
            pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
            pos_embed_rope_shift_coords: float | None = None,
            pos_embed_rope_jitter_coords: float | None = None,
            pos_embed_rope_rescale_coords: float | None = None,
            pos_embed_rope_dtype: str = "bf16",
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            ffn_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.0,
            layerscale_init: float | None = None,
            norm_layer: str = "layernorm",
            ffn_layer: str = "mlp",
            ffn_bias: bool = True,
            proj_bias: bool = True,
            n_storage_tokens: int = 0,
            mask_k_bias: bool = False,
            untie_cls_and_patch_norms: bool = False,
            untie_global_and_local_cls_norm: bool = False,
            device: Any | None = None,
            **ignored_kwargs,
    ) -> None:
        super().__init__()
        if len(ignored_kwargs):
            logger.warning(f"Ignoring kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    
    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape # 这里的H,W已经不再是原始像素了，而是经过patch_embed后的H,W，如果是224，那么H,W为14,14
        x = x.flatten(1, 2) # 将H, W拉平成HW，也就是14, 14 -> 196

        """在做掩码预训练（比如遮住图片的 50% 让模型猜），系统会传进来一个 masks 张量。
        torch.where 的作用就像“狸猫换太子”：如果这个位置的掩码是 True（被遮挡），
        就把原来的图像特征替换成一个全网共享的可学习参数 self.mask_token；如果没被遮挡，就保留原特征。"""
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_toekn.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            """
            在 DDP 模式下，PyTorch 要求：计算图里定义的所有参数，每一次前向传播都必须参与计算（哪怕梯度是 0），
            否则就会报错崩溃（"unused parameters" error）。
            当你没有传入 masks 时，self.mask_token 这个参数就被闲置了。为了骗过 PyTorch 的底层检查机制，
            作者极其聪明地写了 + 0 * self.mask_token。这让 mask_token 强行挂载到了计算图上，既不影响数值结果，又完美避免了多卡训练的崩溃！
            """
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )
        # [ 1个 CLS 标签 ] + [ 4个 Storage 标签 ] + [ 196个 Patch 标签 ]
        # 最终序列的长度变成了 1 + 4 + 196 = 201。
        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        """
        因为后面我们需要做 RoPE（旋转位置编码）！RoPE 必须知道这个序列在物理世界上是几行几列的，才能正确地赋予 2D 坐标。
        如果这里不把 (H, W) 传出去，一维的序列就彻底变成了一笔糊涂账，后面的位置编码就没法算了。
        """
        return x, (H, W)

    """
    List[Tensor] 而不是传统的 [B, C, H, W] 张量，正是为了配合我们之前讲过的 Padding-free（无填充批处理） 技术：
    列表里的每一张图片，分辨率都可以是不一样的！
    """
    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            # 所有的 2D 图片被切碎、拉平，并塞入了 [CLS] 和 [REG] 标签。
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            # 每一张图片原始的物理尺寸 (H, W) 存在了 rope 这个列表里。因为列表中每张图的尺寸可能不同，
            # 必须单独记下它们的“三围”，否则后面的位置编码就全乱套了。
            rope.append(hw_tuple)
        
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                # 动态计算 RoPE
                # 由于每张图的 (H, W) 不同，它是在每一层 Block 前，实时为每一张图单独生成对应的旋转位置编码 rope_sincos
                rope_sincos = [self.rope_embed(H=H,W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            # 解绑归一化（Untied Norms）不同的cls patch使用不同的norm
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                # x_list 里的第 2 个元素（局部小图）学生网络的输入 这里是小图的[cls]和[reg]的norm
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                # 大图也就是教师网络的输入token 大图的[cls]和[reg]的norm
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                # 普通patch的norm
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            # 不使用解绑归一化
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            '''
            x_norm_clstoken: 浓缩了整张图语义的“王牌 Token”。如果你要做图像分类，直接拿它接一个线性层就行了。
            x_storage_tokens: 那些用来吸收无效背景信息的寄存器 Token。一般下游任务不需要用它，它已经完成了身为“垃圾桶”的历史使命。
            x_norm_patchtokens: 密密麻麻的图像网格特征。这就是你之前做图像分割、目标检测时，用来还原成 2D 特征图的核心数据！
            x_prenorm / masks: 原始数据备份，供某些特殊的底层任务（比如计算掩码重建损失）使用。
            '''
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
            return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)
        
    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        
    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))
        

@register_model("dino_small")
def vit_small(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model

@register_model("dino_small_plus")
def vit_small_plus(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("ffn_layer", "swiglu")     # 使用 SwiGLU 替代普通的 MLP
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        **kwargs,
    )
    return model

@register_model("dino_base")
def vit_base(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model

@register_model("dino_large")
def vit_large(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model

# @register_model("dino_so400m")
# def vit_so400m(patch_size=16, **kwargs):
#     # 👇 补齐与官方预训练权重完全一致的魔法参数
#     kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
#     kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
#     kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

#     model = DinoVisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1152,
#         depth=27,
#         num_heads=18,
#         ffn_ratio=3.777777778,
#         **kwargs,
#     )
#     return model

@register_model("dino_huge2")
def vit_huge2(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("ffn_layer", "swiglu")     # 使用 SwiGLU 替代普通的 MLP
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6,
        **kwargs,
    )
    return model

@register_model("dino_giant2")
def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("mask_k_bias", True)       # 开启 K 的掩码偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model

@register_model("dino_7b")
def vit_7b(patch_size=16, **kwargs):
    # 👇 补齐与官方预训练权重完全一致的魔法参数
    kwargs.setdefault("n_storage_tokens", 4)     # 开启 4 个寄存器 Token
    kwargs.setdefault("ffn_layer", "swiglu")     # 使用 SwiGLU 替代普通的 MLP
    kwargs.setdefault("qkv_bias", False)         # 关闭 QKV 的偏置
    kwargs.setdefault("layerscale_init", 1e-4)   # 开启 LayerScale，通常 small 模型的初始值为 1e-4 (或 1e-5)
    kwargs.setdefault("untie_global_and_local_cls_norm", True)  # 开启局部 CLS Norm

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model






