import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.nest_model import NestedTensor
from utils.misc import nested_tensor_from_tensor_list
from utils.misc import inverse_sigmoid, _get_clones
from utils.box_ops import box_xyxy_to_cxcywh, delta2bbox, box_cxcywh_to_xyxy
from .dinov3_backbone import build_backbone
from .detr_head_transformer import build_transformer
from ._modelRegistry import register_model
import logging

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
"""
reparam: false
标准版 Deformable DETR
预测目标：相对偏移量 (通过 Sigmoid 压平)，天然就是 0~1 归一化的，loss函数之前不需要再除以特征图尺寸了，完美契合 DETR 的训练目标和损失函数设计。
但是收敛慢一点，依赖 inverse_sigmoid 兜底
"""
class PlainDETR(nn.Module):
    """This is the Deformable DETR module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one matching part
            num_queries_one2many: number of object queries for one-to-many matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        num_queries = num_queries_one2one + num_queries_one2many
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ]
        )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            # 🌟 True 的情况：深拷贝 
            '''
            这意味着 Decoder 的 6 层网络，拥有 6 个互相独立的回归头和分类头！ 为什么？
            因为第 1 层的任务是“从全局瞎猜一个框”，而第 6 层的任务是“对一个已经很准的框做像素级微调”。
            这 6 层预测的偏移量分布、尺度完全不同，如果强制共享权重，网络会当场精神分裂，
            因此必须给每一层分配独立的参数去学习。
            '''
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            # 🌟 False 的情况：列表浅引用
            '''
            [self.class_embed for _ in range(num_pred)] 在 Python 中生成的是一个包含相同对象引用的列表。
            这意味着，Decoder 的 6 层网络，使用的是完全相同的 1 个回归头（MLP）和分类头（Linear）！ 
            因为每一层都要从头预测绝对位置，任务性质相同，共享权重可以减少参数量，防止过拟合
            '''
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros(
            [
                self.num_queries,
                self.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            self.num_queries_one2one :,
            0 : self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_queries_one2one,
            self.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_classes_one2one, outputs_coords_one2one)
            out["aux_outputs_one2many"] = self._set_aux_loss(outputs_classes_one2many, outputs_coords_one2many)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

"""
使用了重参数的 deformable detr
输出的是绝对像素坐标
在输入到loss函数之前必须强行除以 img_size 伪装成 0~1 归一化坐标，否则 loss 计算会炸掉
收敛速度极快
"""
class PlainDETRReParam(PlainDETR):

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros(
            [
                self.num_queries,
                self.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            self.num_queries_one2one :,
            0 : self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_queries_one2one,
            self.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        # ==========================================
        # 🌟 核心修复 1：提取特征图的真实有效宽高，构建归一化分母
        # max_shape 的格式是 (valid_H, valid_W)，每个维度都是 [BS, 1, 1]
        # ==========================================
        valid_H, valid_W = max_shape
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1) # 变成 [BS, 1, 4]


        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                # 这里算出来的是绝对像素坐标！
                outputs_coord = box_xyxy_to_cxcywh(delta2bbox(reference, tmp, max_shape))

                # ==========================================
                # 🌟 核心修复 2：强行把绝对坐标按回 0~1 区间，履行 DETR 数据契约！
                # ==========================================
                outputs_coord = outputs_coord / img_size
                reference_norm = reference / img_size
            else:
                raise NotImplementedError

            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])

            # outputs_coords_old_one2one.append(reference[:, : self.num_queries_one2one])
            # outputs_coords_old_one2many.append(reference[:, self.num_queries_one2one :])
            # 上面修改为：
            # 旧坐标也必须存归一化后的，因为 aux_loss 也会用到它们算 Loss
            outputs_coords_old_one2one.append(reference_norm[:, : self.num_queries_one2one])
            outputs_coords_old_one2many.append(reference_norm[:, self.num_queries_one2one :])
            outputs_deltas_one2one.append(tmp[:, : self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
            )
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
            )

        if self.two_stage:
            # out["enc_outputs"] = {
            #     "pred_logits": enc_outputs_class,
            #     "pred_boxes": enc_outputs_coord_unact,
            #     "pred_boxes_old": output_proposals,
            #     "pred_deltas": enc_outputs_delta,
            # }
            # 上面修改为：
            # ==========================================
            # 🌟 核心修复 3：将 Two-Stage 初始化的绝对锚点也除以 img_size
            # ==========================================
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact / img_size,
                "pred_boxes_old": output_proposals / img_size,
                "pred_deltas": enc_outputs_delta,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old, outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_boxes_old": c,
                "pred_deltas": d,
            }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coord_old[:-1], outputs_deltas[:-1])
        ]


# class PostProcess(nn.Module):
#     """This module converts the model's output into the format expected by the coco api"""

#     def __init__(self, topk=100, reparam=False):
#         super().__init__()
#         self.topk = topk
#         self.reparam = reparam

#     @torch.no_grad()
#     def forward(self, outputs, target_sizes, original_target_sizes=None):
#         """Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2
#         assert not self.reparam or original_target_sizes.shape[1] == 2

#         prob = out_logits.sigmoid()
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
#         scores = topk_values
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         boxes = box_cxcywh_to_xyxy(out_bbox)
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_sizes.unbind(1)
#         if self.reparam:
#             img_h, img_w = img_h[:, None, None], img_w[:, None, None]  # [BS, 1, 1]
#             boxes[..., 0::2].clamp_(min=torch.zeros_like(img_w), max=img_w)
#             boxes[..., 1::2].clamp_(min=torch.zeros_like(img_h), max=img_h)
#             scale_h, scale_w = (original_target_sizes / target_sizes).unbind(1)
#             scale_fct = torch.stack([scale_w, scale_h, scale_w, scale_h], dim=1)
#         else:
#             scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]

#         results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

#         return results


class PostProcess(nn.Module):
    """
    既然模型层已经被我们强行拉回了 0~1 的输出规范，
    后处理模块就不再需要去处理那乱七八糟的 reparam 逻辑了，直接乘尺寸即可！
    """
    def __init__(self, topk=100): 
        super().__init__()
        self.topk = topk

    @torch.no_grad()
    def forward(self, outputs, target_sizes, original_target_sizes=None):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 统一的纯净还原逻辑：模型输出已经是 0~1 了，直接乘上真实尺寸即可
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


def _dict_to_namespace(d):
    """将嵌套字典递归转化为 SimpleNamespace 对象（仅供 dinov3_det 内部使用）"""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    else:
        return d


@register_model("dinov3_det")
def build_dinov3_det(backbone_name="dinov3_small", backbone_weight=None, **kwargs):
    """
    构建 DINOv3 DETR 检测模型。

    与其他模型统一，接受纯字典 kwargs，内部自行处理：
    - backbone 构建 + 预训练权重加载
    - position_embedding 枚举转换
    - kwargs → SimpleNamespace 转换（下游 build_backbone/build_transformer 需要）
    """
    from . import build_model as _build_model
    from .dinov3_backbone import PositionEncoding
    from utils.load_checkpoints import load_checkpoint

    # 1. 处理 PositionEncoding 枚举转换
    pos_str = kwargs.pop("position_embedding", "sine")
    _pos_map = {
        "sine": PositionEncoding.SINE,
        "learned": PositionEncoding.LEARNED,
        "sine_unnorm": PositionEncoding.SINE_UNNORM,
    }
    kwargs["position_embedding"] = _pos_map.get(pos_str, pos_str)

    # 2. 转换为 Namespace（build_backbone/build_transformer 需要 args.xxx 风格访问）
    args = _dict_to_namespace(kwargs)

    # 3. 构建 backbone ViT 并加载预训练权重
    backbone_model = _build_model(backbone_name)
    if backbone_weight is not None and os.path.exists(backbone_weight):
        backbone_model, _ = load_checkpoint(backbone_model, backbone_weight)
    else:
        logging.warning(f"⚠️ 预训练权重路径不存在: {backbone_weight}，跳过加载！")

    # 4. 组装 DETR
    backbone = build_backbone(backbone_model, args)
    transformer = build_transformer(args)
    model_class = PlainDETR if (not args.reparam) else PlainDETRReParam
    return model_class(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
    )