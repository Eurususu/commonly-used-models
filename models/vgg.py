'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
from .registry import register_model
from .BaseModel import BaseModel

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(BaseModel):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # # Initialize weights （原始 VGG 的初始化方式，现代 PyTorch 推荐使用 nn.init 模块）
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        
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
        x = self.features(x)
        x = self.avgpool(x) # 展平前先经过自适应池化
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def get_default_config():
        return {"num_classes": 10}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 如果有 BN，就不要 bias；如果没有 BN，保留默认的 bias=True
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


@register_model("vgg11")
def vgg11(num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes=num_classes)


@register_model("vgg11_bn")
def vgg11_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


@register_model("vgg13")
def vgg13(num_classes=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes=num_classes)


@register_model("vgg13_bn")
def vgg13_bn(num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)


@register_model("vgg16")
def vgg16(num_classes=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_classes)


@register_model("vgg16_bn")
def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)


@register_model("vgg19")
def vgg19(num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes)


@register_model("vgg19_bn")
def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)

