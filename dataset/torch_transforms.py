import torchvision.transforms as T
from ._transformsRegistry import register_transform

__all__ = []

register_transform("resize")(T.Resize)
register_transform("center_crop")(T.CenterCrop)
register_transform("random_crop")(T.RandomCrop)
register_transform("random_horizontal_flip")(T.RandomHorizontalFlip)
register_transform("color_jitter")(T.ColorJitter)
register_transform("to_tensor")(T.ToTensor)
register_transform("normalize")(T.Normalize)