import os
import torch
from tqdm import tqdm
from .val_engine import Evaluator

class Trainer:
    """核心训练引擎"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # 如果提供了验证集，则初始化验证引擎
        self.evaluator = Evaluator(self.model, self.val_loader, self.criterion, self.device) if val_loader else None

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_one_epoch(self, epoch):
        """执行单轮训练"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"🚀 Epoch {epoch} Training")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 核心五步曲
            self.optimizer.zero_grad()               # 1. 清空梯度
            outputs = self.model(inputs)             # 2. 前向传播
            loss = self.criterion(outputs, targets)  # 3. 计算 Loss
            loss.backward()                          # 4. 反向传播
            self.optimizer.step()                    # 5. 更新权重

            total_loss += loss.item()
            # 实时更新进度条上的 loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def train(self, epochs):
        """启动完整训练循环"""
        print(f"🔥 开始训练！总 Epoch: {epochs}, 使用设备: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # 1. 训练一个 Epoch
            train_loss = self.train_one_epoch(epoch)
            print(f"[{epoch}/{epochs}] 📈 Train Loss: {train_loss:.4f}")

            val_loss = None
            # 2. 执行验证
            if self.evaluator:
                val_loss, val_acc = self.evaluator.evaluate()
                print(f"[{epoch}/{epochs}] 📉 Val Loss: {val_loss:.4f} | 🎯 Val Acc: {val_acc:.4f}")

                # 3. 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_path = os.path.join(self.save_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"🌟 发现更低的验证集 Loss，已保存权重至: {save_path}")

            # 4. 学习率调度器更新
            if self.scheduler:
                # 处理依赖监控指标的特殊调度器 (如 ReduceLROnPlateau)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
        print("\n🏁 训练流程圆满结束！")