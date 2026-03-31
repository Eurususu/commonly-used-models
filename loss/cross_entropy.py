""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._lossRegistry import register_loss
import logging

__all__ = ['LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy']

@register_loss("LabelSmoothingCrossEntropy")
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, **kwargs):
        super(LabelSmoothingCrossEntropy, self).__init__()
        if kwargs:
            logging.warning(f"LabelSmoothingCrossEntropy 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

@register_loss("SoftTargetCrossEntropy")
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, **kwargs):
        super(SoftTargetCrossEntropy, self).__init__()
        if kwargs:
            logging.warning(f"SoftTargetCrossEntropy 收到了额外的参数 {kwargs}，但这些参数将被忽略！")

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
