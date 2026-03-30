from ._transformsRegistry import register_transform
import torch

__all__ = ["AddGaussianNoise"]

@register_transform("AddGaussianNoise")
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
