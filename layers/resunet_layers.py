"""UNet 残差块"""

import torch
from .resnet_layers import BasicBlock
from .unet_layers import DoubleConv
import logging

import torch.nn as nn

__all__ = [
    'ResDown',
    'ResUp',
]


class ResDown(nn.Module):
    """使用残差块的下采样模块"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"ResDown 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ResUp(nn.Module):
    """使用残差块的上采样模块"""

    def __init__(self, in_channels, out_channels, bilinear=True, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"ResUp 收到了额外的参数 {kwargs}，但这些参数将被忽略！")

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BasicBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
