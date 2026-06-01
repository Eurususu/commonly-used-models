import torch
import numpy as np
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def delta2bbox(
    proposals, deltas, max_shape=None, wh_ratio_clip=16 / 1000, clip_border=True, add_ctr_clamp=False, ctr_clamp=32
):
    dxy = deltas[..., :2]
    dwh = deltas[..., 2:]

    # Compute width/height of each roi
    pxy = proposals[..., :2]
    pwh = proposals[..., 2:]

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0).clamp_(max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0).clamp_(max=max_shape[0])
    return bboxes


def bbox2delta(proposals, gt, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    # hack for matcher
    if proposals.size() != gt.size():
        proposals = proposals[:, None]
        gt = gt[None]

    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph = proposals.unbind(-1)
    gx, gy, gw, gh = gt.unbind(-1)

    dx = (gx - px) / (pw + 0.1)
    dy = (gy - py) / (ph + 0.1)
    dw = torch.log(gw / (pw + 0.1))
    dh = torch.log(gh / (ph + 0.1))
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    # avoid unnecessary sync point if not needed
    if means != (0.0, 0.0, 0.0, 0.0) or stds != (1.0, 1.0, 1.0, 1.0):
        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

    return deltas

def box_iou(boxes1, boxes2):
    """
    计算普通的 IoU (交并比) 和 Union (并集面积)
    输入格式必须是 [左上角x, 左上角y, 右下角x, 右下角y]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 寻找交集的左上角和右下角坐标
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集的宽高 (clamp_min=0 防止没有交集时出现负数)
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # 并集面积 = 面积1 + 面积2 - 交集面积
    union = area1[:, None] + area2 - inter
    
    # 🌟 防御性编程：加一个极小的常数 1e-6，防止除以 0 产生 NaN
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    计算 GIoU (广义交并比)
    The Generalized Intersection over Union (GIoU) from: https://arxiv.org/abs/1902.09630
    输入格式必须是 [x0, y0, x1, y1]
    """
    # 确保坐标是合法的 (右下角必定大于左上角)
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # 1. 先计算普通的 IoU
    iou, union = box_iou(boxes1, boxes2)

    # 2. 寻找两个框的“最小闭包区域” (也就是能把这俩框完全包住的最小外接矩形)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算外接矩形的面积
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # 3. GIoU 公式 = IoU - (外接矩形面积 - 并集面积) / 外接矩形面积
    # 如果两个框完全重合，GIoU = 1；如果距离无限远，GIoU 趋近于 -1
    return iou - (area - union) / (area + 1e-6)