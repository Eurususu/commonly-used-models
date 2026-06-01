import torch.nn as nn
from typing import List, Optional, Union
import torch.nn.functional as F
import torch
from layers import LayerNorm2D
from utils.nest_model import NestedTensor
from .windows import WindowsWrapper
import logging
from layers import PositionEmbeddingSine, PositionEmbeddingLearned
from enum import Enum

logger = logging.getLogger("dinov3")
from ._modelRegistry import register_model

    
'''
冻结和微调：
DINOv3 太大了，如果下游任务数据量不够，全参微调（Full Fine-tuning）很容易过拟合或显存爆炸。
train_backbone=False: 完全冻结骨干网络
blocks_to_train=["blocks.11", "blocks.10"]: 它就只训练最后两层。

为什么它要暴力拼接通道，而不是做 FPN（特征金字塔）？
因为像 DINO 这样的纯视觉 Transformer 具有单一尺度（Single-scale）的特性。
在深层网络中，因为自注意力机制的存在，每一层的空间分辨率都是 H/16 \times W/16，但不同的层捕获了不同级别的语义信息（深层懂类别，浅层懂边缘）。
沿着通道拼接（Concatenation），把不同层次的语义信息一股脑地拍在一起，交给下游的 DETR 解码器去自行挑选，是目前配合普通 ViT 做密集预测任务最有效、最流行的策略。
'''

class PositionEncoding(Enum):
    LEARNED = "learned"
    SINE = "sine"
    SINE_UNNORM = "sine_unnorm"


@register_model("detr_position_encoder")
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding == PositionEncoding.SINE:  # also called v2
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding == PositionEncoding.LEARNED:  # also called v3
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding == PositionEncoding.SINE_UNNORM:  # also called v4
        position_embedding = PositionEmbeddingSine(N_steps, normalize=False)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    position_embedding = nn.ModuleList([position_embedding for _ in range(args.num_feature_levels)])

    return position_embedding


class DINOBackbone(nn.Module):
    def __init__(
        self,
        backbone_model: nn.Module,
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        layers_to_use: Union[int, List] = 1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.blocks_to_train = blocks_to_train
        self.patch_size = self.backbone.patch_size
        self.use_layernorm = use_layernorm
        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        self.strides = [self.backbone.patch_size]

        # get embed_dim for each intermediate output
        n_all_layers = self.backbone.n_blocks
        blocks_to_take = (
            range(n_all_layers - layers_to_use, n_all_layers) if isinstance(layers_to_use, int) else layers_to_use
        )

        # if models do not define embed_dims, repeat embed_dim n_blocks times
        embed_dims = getattr(self.backbone, "embed_dims", [self.backbone.embed_dim] * self.backbone.n_blocks)
        embed_dims = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]

        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm2D(embed_dim) for embed_dim in embed_dims])

        self.num_channels = [sum(embed_dims)]
        self.layers_to_use = layers_to_use

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone.get_intermediate_layers(tensor_list.tensors, n=self.layers_to_use, reshape=True)
        if self.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        xs = [torch.cat(xs, axis=1)]

        out: list[NestedTensor] = []
        for x in xs:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out
    
class BackboneWithPositionEncoding(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        out: List[NestedTensor] = list(self[0](tensor_list))
        pos = [self[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(out)]
        return out, pos

@register_model("dinov3_backbone")
def build_backbone(backbone_model, args):
    position_embedding = build_position_encoding(args)
    train_backbone = False
    backbone = DINOBackbone(
        backbone_model, train_backbone, args.blocks_to_train, args.layers_to_use, args.backbone_use_layernorm
    )
    if args.n_windows_sqrt > 0:
        logger.info(f"Wrapping with {args.n_windows_sqrt} x {args.n_windows_sqrt} windows")
        backbone = WindowsWrapper(
            backbone, n_windows_w=args.n_windows_sqrt, n_windows_h=args.n_windows_sqrt, patch_size=backbone.patch_size
        )
    else:
        logger.info("Not wrapping with windows")

    return BackboneWithPositionEncoding(backbone, position_embedding)