# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from utils.box_ops import box_xyxy_to_cxcywh, delta2bbox
from layers import GlobalDecoderLayer, GlobalDecoder, GlobalRpeDecoderLayer, GlobalRpeDecoder
from layers import TransformerEncoder, TransformerEncoderLayer
from layers import LayerNorm2D
from ._modelRegistry import register_model

@register_model("global_ape_decoder")
def build_global_ape_decoder(args):
    decoder_layer = GlobalDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        n_heads=args.nheads,
        norm_type=args.norm_type,
    )
    decoder = GlobalDecoder(
        decoder_layer,
        num_layers=args.dec_layers,
        return_intermediate=True,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.decoder_use_checkpoint,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
    )
    return decoder

@register_model("global_rpe_decoder")
def build_global_rpe_decomp_decoder(args):
    decoder_layer = GlobalRpeDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        n_heads=args.nheads,
        norm_type=args.norm_type,
        rpe_hidden_dim=args.decoder_rpe_hidden_dim,
        rpe_type=args.decoder_rpe_type,
        feature_stride=args.proposal_in_stride,
        reparam=args.reparam,
    )
    decoder = GlobalRpeDecoder(
        decoder_layer,
        num_layers=args.dec_layers,
        return_intermediate=True,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.decoder_use_checkpoint,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
        reparam=args.reparam,
    )
    return decoder

'''
Two-Stage Deformable DETR（两阶段可变形 DETR）的 Proposal 机制
它负责把图像特征转换成候选目标，并生成供盲人（Query）寻宝的初始 GPS 坐标。
1. 数据展平与层级编码：首先，它将多层特征图展平成一维序列，并为每个层级添加独特的位置信息。
2. 编码器(可选): 在搭配 ResNet 时，这里通常会过 6 层 Transformer Encoder。但因为我们的主干网络已经是强大的 DINOv3 (ViT) 了，
它的输出已经具备了全局自注意力，所以这里的 encoder 通常被设为 None，直接把特征（memory）送往下游。
3. Two-Stage 初始坐标生成: 早期的 DETR，那些 Query（盲人）是“天生的”，它们的初始搜索位置（reference_points）是随机初始化或者写死的。这导致它们在训练早期就像没头苍蝇一样乱撞，收敛极慢
Two-Stage 机制说：不要让盲人瞎找了，让图像特征自己告诉盲人应该去哪里找！具体是如何做的呢？分为一下步骤：
A: 密集的网格撒网 它在特征图的每一个像素位置，都生成了一个极小的边界框（grid 坐标 + 默认的 wh 大小）。如果特征图是 50x50，它瞬间生成了 2500 个初始坐标，覆盖全图。
B: MLP打分, 直接拿特征图过一个初步的分类头（class_embed），让模型给刚才生成的 2500 个框打分：“你觉得这里像不像有东西？”
然后通过 torch.topk，只筛选出得分最高的前 300 个位置（two_stage_num_proposals = 300）。
C: 那 300 个最有可能存在目标的初始坐标（reference_points） 被送进 get_proposal_pos_embed 生成高维正弦位置编码
将其分裂成两个，Query = tgt + query_embed
query_embed(位置嵌入) 防止模型在全图乱找，强迫注意力收敛在特定的参考点（Reference Point）附近。
tgt(内容嵌入) 在划定的范围内，看看有没有符合某种特征的物体 
在刚进入 Decoder 第 0 层时，如果是传统的单阶段 DETR，tgt 里面是一片空白（全填 0）；
而在现在的两阶段（Two-Stage）模式下，tgt 继承了第一阶段粗筛出来的那个区域原始的图像特征。
随着 Decoder 楼层一层层往上走，tgt 会通过 Cross-Attention 不断地从主干网络（Backbone）里吸取养分（图像特征）
穿过 6 层解码器后，tgt 的内容会变得极其丰富，它里面死死咬住了物体的语义。最后，它会被直接送进分类头（预测这是猫还是狗）和回归头（预测边界框的细微修正值）
D: 分类头和回归头只连接 tgt

预测的分数，生成最终的边界框和目标。

'''
class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_feature_levels=4,
        two_stage=False,
        two_stage_num_proposals=300,
        mixed_selection=False,
        norm_type="post_norm",
        decoder_type="deform",
        proposal_feature_levels=1,
        proposal_in_stride=16,
        proposal_tgt_strides=[8, 16, 32, 64],
        proposal_min_size=50,
        args=None,
        # transformer_encoder
        add_transformer_encoder=False,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_encoder_layers=6,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        assert norm_type in ["pre_norm", "post_norm"], f"expected norm type is pre_norm or post_norm, get {norm_type}"

        if decoder_type == "global_ape":
            self.decoder = build_global_ape_decoder(args)
        elif decoder_type == "global_rpe_decomp":
            self.decoder = build_global_rpe_decomp_decoder(args)
        else:
            raise NotImplementedError

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self.mixed_selection = mixed_selection
        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_min_size = proposal_min_size
        if two_stage and proposal_feature_levels > 1:
            assert len(proposal_tgt_strides) == proposal_feature_levels

            self.proposal_in_stride = proposal_in_stride
            self.enc_output_proj = nn.ModuleList([])
            for stride in proposal_tgt_strides:
                if stride == proposal_in_stride:
                    self.enc_output_proj.append(nn.Identity())
                elif stride > proposal_in_stride:
                    scale = int(math.log2(stride / proposal_in_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU(),
                        ]
                    layers.append(nn.Conv2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))
                else:
                    scale = int(math.log2(proposal_in_stride / stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU(),
                        ]
                    layers.append(nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        # ENCODER TRANSFORMER
        self.encoder = None
        if add_transformer_encoder:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

        if hasattr(self.decoder, "_reset_parameters"):
            self.decoder._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                indexing="ij"
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = None
        return output_memory, output_proposals, max_shape

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def expand_encoder_output(self, memory, memory_padding_mask, spatial_shapes):
        assert len(spatial_shapes) == 1, f"Get encoder output of shape {spatial_shapes}, not sure how to expand"

        bs, _, c = memory.shape
        h, w = spatial_shapes[0]

        _out_memory = memory.view(bs, h, w, c).permute(0, 3, 1, 2)
        _out_memory_padding_mask = memory_padding_mask.view(bs, h, w)

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(_out_memory_padding_mask[None].float(), size=mem.shape[-2:]).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-2:])

        out_memory = torch.cat([mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = None
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        return (
            reference_points,
            max_shape,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
        )

    def forward(self, srcs, masks, pos_embeds, query_embed=None, self_attn_mask=None):
        # TODO: we may remove this loop as we only have one feature level
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        level_start_index = None  # not used so far
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        if self.encoder is not None:
            memory = self.encoder(src_flatten, src_key_padding_mask=mask_flatten, pos=lvl_pos_embed_flatten)
        else:
            memory = src_flatten

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            (
                reference_points,
                max_shape,
                enc_outputs_class,
                enc_outputs_coord_unact,
                enc_outputs_delta,
                output_proposals,
            ) = self.get_reference_points(memory, mask_flatten, spatial_shapes)
            init_reference_out = reference_points
            pos_trans_out = torch.zeros((bs, self.two_stage_num_proposals, 2 * c), device=init_reference_out.device)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
            max_shape = None

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            lvl_pos_embed_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            self_attn_mask,
            max_shape,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
                enc_outputs_delta,
                output_proposals,
                max_shape,
            )
        return hs, init_reference_out, inter_references_out, None, None, None, None, None

'''
结合RPE（相对位置偏置） 而做出的架构升级
改进有三点：
1. 回归真实像素坐标：传统的 Deformable DETR 直接在特征图上生成参考点（reference points），它们的坐标是相对于特征图大小的归一化值（0-1）。
而 RPE 版本直接回归真实像素坐标，这样模型就不需要再去学习如何从特征图坐标映射到图像坐标了，简化了学习任务。
2. 抛弃Inverse Sigmoid：传统 Deformable DETR 在生成参考点时，先把坐标归一化到0-1范围内，然后再通过 inverse sigmoid 转换成实数空间。
这种做法虽然理论上可行，但在训练初期可能会导致数值不稳定，尤其是当参考点接近边界时。RPE 版本直接回归像素坐标，避免了这种不稳定性。
'''
class TransformerReParam(Transformer):
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            stride = self.proposal_tgt_strides[lvl]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                indexing="ij"
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) * stride
            wh = torch.ones_like(grid) * self.proposal_min_size * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)

        H_, W_ = spatial_shapes[0]
        stride = self.proposal_tgt_strides[0]
        mask_flatten_ = memory_padding_mask[:, : H_ * W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1, keepdim=True) * stride
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1, keepdim=True) * stride
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1)
        img_size = img_size.unsqueeze(1)  # [BS, 1, 4]

        output_proposals_valid = ((output_proposals > 0.01 * img_size) & (output_proposals < 0.99 * img_size)).all(
            -1, keepdim=True
        )
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1).repeat(1, 1, 1), max(H_, W_) * stride
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, max(H_, W_) * stride)

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = (valid_H[:, None, :], valid_W[:, None, :])
        return output_memory, output_proposals, max_shape

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = box_xyxy_to_cxcywh(delta2bbox(output_proposals, enc_outputs_delta, max_shape))

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact
        return (
            reference_points,
            max_shape,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
        )
    

@register_model("detr_head_transformer")
def build_transformer(args):
    model_class = Transformer if (not args.reparam) else TransformerReParam
    return model_class(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_feature_levels=args.num_feature_levels,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries_one2one + args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        norm_type=args.norm_type,
        decoder_type=args.decoder_type,
        proposal_feature_levels=args.proposal_feature_levels,
        proposal_in_stride=args.proposal_in_stride,
        proposal_tgt_strides=args.proposal_tgt_strides,
        args=args,
        proposal_min_size=args.proposal_min_size,
        # transformer_encoder
        add_transformer_encoder=args.add_transformer_encoder,
        num_encoder_layers=args.num_encoder_layers,
    )
