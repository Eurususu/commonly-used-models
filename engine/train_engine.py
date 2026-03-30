import os
import torch
from tqdm import tqdm
from .val_engine import Evaluator

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """核心训练引擎"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir="checkpoints", is_main_process=True):
        # 不要在这里 model.to(device) 了，因为在 train.py 里已经处理并包裹了 DDP
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.is_main_process = is_main_process
        
        # 如果提供了验证集，则初始化验证引擎
        self.evaluator = Evaluator(self.model, self.val_loader, self.criterion, self.device) if val_loader else None
        self.best_val_loss = float('inf')

        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            # 初始化 TensorBoard，日志会保存在 save_dir/logs 下
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))
    def train_one_epoch(self, epoch):
        """执行单轮训练"""
        self.model.train()
        total_loss = 0.0

        # 🌟 DDP 模式下，必须在每个 Epoch 告诉 Sampler 当前是第几轮，否则多卡数据不会打乱！
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        # 🌟 利用 tqdm 的 disable 参数，让其他非主进程全部闭嘴，不打印进度条！
        pbar = tqdm(self.train_loader, desc=f"🚀 Epoch {epoch} Training", disable=not self.is_main_process)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 核心五步曲
            self.optimizer.zero_grad()               # 1. 清空梯度
            outputs = self.model(inputs)             # 2. 前向传播
            loss = self.criterion(outputs, targets)  # 3. 计算 Loss
            loss.backward()                          # 4. 反向传播
            self.optimizer.step()                    # 5. 更新权重

            total_loss += loss.item()
            # # 只有主进程更新进度条的后缀
            if self.is_main_process:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def train(self, epochs):
        """启动完整训练循环"""
        if self.is_main_process:
            print(f"🔥 开始训练！总 Epoch: {epochs}, 使用设备: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # 1. 训练一个 Epoch
            train_loss = self.train_one_epoch(epoch)
            if self.is_main_process:
                print(f"[{epoch}/{epochs}] 📈 Train Loss: {train_loss:.4f}")
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('HyperParams/Learning_Rate', current_lr, epoch)

            val_loss = None
            # 2. 执行验证
            if self.evaluator:
                val_loss, val_acc = self.evaluator.evaluate()
                if self.is_main_process:
                    print(f"[{epoch}/{epochs}] 📉 Val Loss: {val_loss:.4f} | 🎯 Val Acc: {val_acc:.4f}")
                    self.writer.add_scalar('Loss/Val', val_loss, epoch)
                    self.writer.add_scalar('Metric/Accuracy', val_acc, epoch)

                # 3. 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_path = os.path.join(self.save_dir, "best_model.pth")
                    # 在 DDP 模式下，模型被包裹了一层 `module.`
                    # 为了以后单卡能正常读取，必须用 `model.module` 保存原模型权重
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(model_to_save.state_dict(), save_path)
                    print(f"🌟 发现更低的验证集 Loss，已保存权重至: {save_path}")

            # 4. 学习率调度器更新
            if self.scheduler:
                # 处理依赖监控指标的特殊调度器 (如 ReduceLROnPlateau)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        if self.is_main_process:     
            print("\n🏁 训练流程圆满结束！")
            self.writer.close() # 关闭 TensorBoard 写入器