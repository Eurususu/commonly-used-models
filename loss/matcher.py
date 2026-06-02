import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """计算预测集合与真实集合的最优二分图匹配"""
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad() # 匹配过程不需要计算梯度！
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 展平批次：把 [B, 300, C] 变成 [B*300, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # 使用 Sigmoid (Focal Loss 标配)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)            # [B*300, 4]

        # 2. 拼接所有的真实标签
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])


        # (1) 类别 Cost: 预测出的真实类别概率越高，Cost 越低
        # out_prob[:, tgt_ids] 会提取出所有预测框对应真实类别的概率
        # 把分类代价替换为官方 Deformable DETR 中严格推导的 Focal Cost 公式
        # 新版cost_class
        # 🌟 修复：严格对齐 Focal Loss 的类别 Cost
        alpha = 0.25
        gamma = 2.0
        # 计算将该 Query 预测为背景的代价 (对应 Focal Loss 的负样本项)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # 计算将该 Query 预测为目标真实类别的代价 (对应 Focal Loss 的正样本项)
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # 匹配该目标的净代价 = 预测为该目标的代价 - 预测为背景的代价
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # 旧版本的简单分类 Cost（不考虑 Focal Loss 的特殊形态，可能导致匹配质量不佳）
        # cost_class = -out_prob[:, tgt_ids]

        # (2) L1 距离 Cost: 算绝对值差距
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # (3) GIoU Cost: 算交并比 (越重合，GIoU越大，Cost越小)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 4. 融合最终的 Cost 矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # 🌟 终极护盾：强制把矩阵里残留的 NaN 或 Inf 替换成 100.0 (一个极大的惩罚代价)
        # 这样即使某个假框坏掉了，SciPy 也不会选它匹配，且程序绝对不会崩溃！
        C = torch.nan_to_num(C, nan=100.0, posinf=100.0, neginf=100.0)
        C = C.view(bs, num_queries, -1).cpu() # SciPy 只能在 CPU 上跑

        # 5. 根据每张图里的目标数量进行切片，并调用 SciPy 求最优解
        sizes = [len(v["boxes"]) for v in targets]
        # 🌟 加上 .numpy() 保证与 SciPy 完美兼容
        indices = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C.split(sizes, -1))]

        # 返回格式: [(预测框下标, 真实框下标), (预测框下标, 真实框下标), ...]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    

