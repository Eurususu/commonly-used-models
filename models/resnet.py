"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch.nn as nn
from layers import BasicBlock, BottleNeck
from ._modelRegistry import register_model
from .BaseModel import BaseModel
import logging

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class ResNet(BaseModel):

    def __init__(self, block, num_block, num_classes=100, **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"ResNet 收到了额外的参数 {kwargs}，但这些参数将被忽略！")

        self.in_channels = 64

        # # CIFAR 版本
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))


        # 标准 ImageNet 版本的 conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
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

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

@register_model("resnet18")
def resnet18(num_classes=100, **kwargs):
    """ return a ResNet 18 object
    """
    if kwargs:
        logging.warning(f"resnet18 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

@register_model("resnet34")
def resnet34(num_classes=100, **kwargs):
    """ return a ResNet 34 object
    """
    if kwargs:
        logging.warning(f"resnet34 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

@register_model("resnet50")
def resnet50(num_classes=100, **kwargs):
    """ return a ResNet 50 object
    """
    if kwargs:
        logging.warning(f"resnet50 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

@register_model("resnet101")
def resnet101(num_classes=100, **kwargs):
    """ return a ResNet 101 object
    """
    if kwargs:
        logging.warning(f"resnet101 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

@register_model("resnet152")
def resnet152(num_classes=100, **kwargs):
    """ return a ResNet 152 object
    """
    if kwargs:
        logging.warning(f"resnet152 收到了额外的参数 {kwargs}，但这些参数将被忽略！")
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)

