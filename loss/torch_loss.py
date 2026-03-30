import torch.nn as nn
from ._lossRegistry import register_loss

__all__ = []

# 常用损失函数注册
register_loss("L1Loss")(nn.L1Loss)
register_loss("MSELoss")(nn.MSELoss)
register_loss("CrossEntropyLoss")(nn.CrossEntropyLoss)
register_loss("BCELoss")(nn.BCELoss)
register_loss("BCEWithLogitsLoss")(nn.BCEWithLogitsLoss)
register_loss("NLLLoss")(nn.NLLLoss)
register_loss("PoissonNLLLoss")(nn.PoissonNLLLoss)
register_loss("KLDivLoss")(nn.KLDivLoss)
register_loss("MarginRankingLoss")(nn.MarginRankingLoss)
register_loss("HingeEmbeddingLoss")(nn.HingeEmbeddingLoss)
register_loss("CosineEmbeddingLoss")(nn.CosineEmbeddingLoss)
register_loss("MultiMarginLoss")(nn.MultiMarginLoss)
register_loss("MultiLabelSoftMarginLoss")(nn.MultiLabelSoftMarginLoss)
register_loss("SmoothL1Loss")(nn.SmoothL1Loss)
register_loss("SoftMarginLoss")(nn.SoftMarginLoss)
register_loss("HuberLoss")(nn.HuberLoss)