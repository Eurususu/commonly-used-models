# engine/base_trainer.py
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """通用的基础训练引擎基类"""
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, save_dir="checkpoints", is_main_process=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.is_main_process = is_main_process
        
        self.best_metric = float('-inf') # 统一用一个指标来保存最优模型 (比如 mAP 或 Acc)，越高越好

        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))

    def _to_device(self, data):
        """递归设备适配器"""
        if isinstance(data, torch.Tensor): return data.to(self.device)
        elif isinstance(data, dict): return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list): return [self._to_device(v) for v in data]
        elif hasattr(data, 'to'): return data.to(self.device)
        return data

    @abstractmethod
    def train_step(self, batch):
        """
        【留给子类实现】单步训练逻辑
        需要返回: (总 Loss, 用于记录的日志字典)
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        【留给子类实现】验证集评估逻辑
        需要返回: 主要指标 (用于保存最优模型), 用于记录的日志字典
        """
        pass

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"🚀 Epoch {epoch}", disable=not self.is_main_process)

        for batch in pbar:
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 把核心计算丢给子类
            loss, log_dict = self.train_step(batch)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if self.is_main_process:
                pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in log_dict.items()})

        return total_loss / len(self.train_loader)

    def train(self, epochs):
        if self.is_main_process:
            print(f"🔥 开始训练！总 Epoch: {epochs}")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            
            if self.is_main_process:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('HyperParams/LR', self.optimizer.param_groups[0]['lr'], epoch)

            if self.val_loader:
                main_metric, val_logs = self.evaluate()
                
                if self.is_main_process:
                    print(f"[{epoch}/{epochs}] 📉 Train Loss: {train_loss:.4f} | Validation: {val_logs}")
                    for k, v in val_logs.items():
                        self.writer.add_scalar(f'Val/{k}', v, epoch)

                    if main_metric > self.best_metric:
                        self.best_metric = main_metric
                        save_path = os.path.join(self.save_dir, "best_model.pth")
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), save_path)
                        print(f"🌟 发现更优模型 (Metric: {main_metric:.4f})，已保存至: {save_path}")

            if self.scheduler:
                # 简化处理，统一 step
                self.scheduler.step()

        if self.is_main_process:     
            print("\n🏁 训练流程圆满结束！")
            self.writer.close()