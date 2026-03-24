""" Full assembly of the parts to form the complete network """

from .common import *
import torch
import torch.nn as nn
from .Registry import register_model
from .BaseModel import BaseModel

__all__ = ['UNet', 'ResUNet']


@register_model("unet")
class UNet(BaseModel):
    def __init__(self, num_channels=3, num_classes=1, bilinear=False, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(num_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, num_classes))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 现代 PyTorch 推荐的 Kaiming 初始化写法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @staticmethod
    def get_default_config():
        return {
            "num_channels": 3,
            "num_classes": 1,
            "bilinear": False
        }

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


@register_model("resunet")
class ResUNet(BaseModel):
    # 1. 增加默认值和 **kwargs 以完美兼容工厂函数的调用
    def __init__(self, num_channels=3, num_classes=1, bilinear=False, **kwargs):
        # 2. 正确初始化父类 BaseModel，将 num_classes 映射给父类的 num_classes
        super(ResUNet, self).__init__(num_classes=num_classes, **kwargs)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 最开始的第一层输入，通常不加残差，或者也用 BasicBlock
        self.inc = BasicBlock(num_channels, 64)
        
        # 下采样路径
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ResDown(512, 1024 // factor)
        
        # 上采样路径 (注意这里的 in_channels 是拼接后的通道数)
        self.up1 = ResUp(1024, 512 // factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128, 64, bilinear)
        
        self.outc = OutConv(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 现代 PyTorch 推荐的 Kaiming 初始化写法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    @staticmethod
    def get_default_config():
        return {
            "num_channels": 3,
            "num_classes": 1,
            "bilinear": False
        }
