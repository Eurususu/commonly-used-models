import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ._lossRegistry import register_loss
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """DETR 的核心集合损失函数"""
    def __init__(self, num_classes, matcher, weight_dict, losses):
        """
        losses 列表通常包含: ['labels', 'boxes']
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """计算 Focal Loss (分类损失)"""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        
        # 提取真实标签，并将那些没匹配上的 Query 当作背景类 (用 0 填充)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 将目标类别转为 One-Hot 编码供 Focal Loss 计算
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1] # 丢弃最后一列的背景类

        # 计算 Focal Loss
        # 🌟 修复点 1：把 num_boxes 从参数列表里拿出来，并加上 reduction="none"
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=0.25, gamma=2.0, reduction="none")
        
        # 🌟 修复点 2：把整个 batch 的 Focal Loss 矩阵求和，然后在外面除以 num_boxes
        loss_ce = loss_ce.sum() / num_boxes
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2.0)
        # loss_ce = loss_ce.mean(1).sum() * src_logits.shape[1]
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算 L1 坐标损失与 GIoU 损失"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 距离
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        # GIoU (广义交并比)
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # 将匹配结果打平成批次索引，方便张量索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """引擎调用的入口"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 1. 调用匹配器拿到索引配对
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 计算当前批次总共有几个真实目标框 (用于归一化)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # 🌟 修复方案：先检查分布式环境是否已初始化
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1 # 如果没初始化，说明是单卡，直接设为 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # 3. 计算最终层的 Loss
        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss_type == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # 4. 🌟 DETR 的精髓：为解码器的每一层计算辅助损失 (Auxiliary Losses)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets) # 每一层重新匹配！
                for loss_type in self.losses:
                    if loss_type == 'labels':
                        l_dict = self.loss_labels(aux_outputs, targets, indices, num_boxes)
                    elif loss_type == 'boxes':
                        l_dict = self.loss_boxes(aux_outputs, targets, indices, num_boxes)
                    # 给字典加上层号后缀，比如 loss_ce_0, loss_bbox_3
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 5. 乘上我们自定义的权重系数 (Weight Dict)
        final_losses = {}
        for k, v in losses.items():
            # 🌟 修复：第一步，尝试精确匹配
            # 这样不仅可以保护像 loss_stage_1 这种本身自带数字的基础损失，
            # 还可以允许我们为特定的某一层单独设置独特的权重！
            if k in self.weight_dict:
                final_losses[k] = v * self.weight_dict[k]
            else:
                # 第二步：精确匹配失败，检查是否是 Aux 自动生成的后缀
                parts = k.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_key = parts[0]
                    # 第三步：用截断后的基础名去寻找权重
                    if base_key in self.weight_dict:
                        final_losses[k] = v * self.weight_dict[base_key]

        return final_losses # 这个字典就是被你 Trainer 里的 sum() 加起来的东西！
    

@register_loss("set_criterion")
def build_set_criterion(num_classes=80, matcher=None, losses=None):
    """
    负责拦截 YAML 的参数，组装对象后，再实例化真正的 SetCriterion。
    这里的形参 (num_classes, matcher, losses) 必须和 YAML kwargs 里的 Key 完全一致。
    """
    # 赋默认值，防止 YAML 没写
    if matcher is None:
        matcher = {'class': 2.0, 'bbox': 5.0, 'giou': 2.0}
    if losses is None:
        losses = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}

    # 1. 将 YAML 里的字典，实例化为真正的 Matcher 对象
    matcher_obj = HungarianMatcher(
        cost_class=matcher.get('class', 2.0),
        cost_bbox=matcher.get('bbox', 5.0),
        cost_giou=matcher.get('giou', 2.0)
    )

    # 2. 实例化并返回 SetCriterion
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher_obj,          # 传入实例化后的对象，而不是字典
        weight_dict=losses,           # YAML 里的 losses 其实对应这里的 weight_dict
        losses=['labels', 'boxes']    # 写死需要计算的具体 loss 分支
    )