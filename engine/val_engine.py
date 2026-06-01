import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseValidator(ABC):
    """通用的基础验证引擎基类"""
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def _to_device(self, data):
        """递归设备适配器"""
        if isinstance(data, torch.Tensor): return data.to(self.device)
        elif isinstance(data, dict): return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list): return [self._to_device(v) for v in data]
        elif hasattr(data, 'to'): return data.to(self.device)
        return data

    @abstractmethod
    def evaluate(self):
        """留给子类实现的评估逻辑"""
        pass