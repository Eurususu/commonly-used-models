import torch.nn as nn
import torch
import logging

__all__ = ["DFL"]


class DFL(nn.Module):
    def __init__(self, c1: int = 16, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"DFL 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Paramter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    


