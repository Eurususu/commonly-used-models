# engine/base_trainer.py
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import torch.distributed as dist
from utils.ema import ModelEMA
import copy

class BaseTrainer(ABC):
    """通用的基础训练引擎基类"""
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 device, save_dir="checkpoints", is_main_process=True,
                 clip_max_norm=None,
                 use_ema=True, ema_decay=0.9999):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.is_main_process = is_main_process
        self.clip_max_norm = clip_max_norm # 🌟 绑定到实例属性
        
        self.best_metric = float('-inf') # 统一用一个指标来保存最优模型 (比如 mAP 或 Acc)，越高越好

        # 初始化ema模型
        self.use_ema = use_ema
        if self.use_ema:
            if self.is_main_process: print(f"✨ 开启 EMA (衰减率: {ema_decay})，这将在验证和推理时提供更平滑、更高的精度。")
            self.ema_model = ModelEMA(self.model, decay=ema_decay, device=self.device)
        else:
            self.ema_model = None

        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))

    def _to_device(self, data):
        """递归设备适配器"""
        if isinstance(data, torch.Tensor): return data.to(self.device)
        elif isinstance(data, dict): return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list): return [self._to_device(v) for v in data]
        elif isinstance(data, tuple):return tuple(self._to_device(v) for v in data)
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

        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError(
                "🚨 致命错误：训练集 DataLoader 的批次数量为 0！\n"
                "可能的原因：\n"
                "1. 数据集目录为空，或读取逻辑导致 0 样本。\n"
                "2. 训练集总样本数小于 Batch Size，且 DataLoader 开启了 drop_last=True。\n"
                "请立即检查数据流配置！"
            )

        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"🚀 Epoch {epoch}", disable=not self.is_main_process)

        for batch in pbar:
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 把核心计算丢给子类
            loss, log_dict = self.train_step(batch)
            
            loss.backward()
            # 🌟 2. 核心魔法：在 step() 之前执行梯度裁剪！
            if self.clip_max_norm is not None and self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)

            self.optimizer.step()
            # 🌟 新增：在每次参数更新后，默默更新 EMA 影子权重
            if self.use_ema:
                self.ema_model.update(self.model)

            total_loss += loss.item()
            
            if self.is_main_process:
                pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in log_dict.items()})

        return total_loss / len(self.train_loader)

    def train(self, start_epoch, epochs):
        if self.is_main_process:
            print(f"🔥 开始训练！总 Epoch: {epochs}，当前从 Epoch {start_epoch} 启动")
        
        for epoch in range(start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            
            if self.is_main_process:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('HyperParams/LR', self.optimizer.param_groups[0]['lr'], epoch)

            if self.val_loader:
                model_without_ddp = self.model.module if hasattr(self.model, 'module') else self.model
                
                # 🌟 新增：验证前，用 EMA 权重“夺舍”主模型
                if self.use_ema:
                    current_state_dict = copy.deepcopy(model_without_ddp.state_dict()) # 备份原权重
                    model_without_ddp.load_state_dict(self.ema_model.module.state_dict()) # 注入 EMA 权重

                main_metric, val_logs = self.evaluate() # 此时评估的是 EMA 模型！

                # 🌟 新增：验证后，把主模型原本的权重还回去，准备下一轮的训练
                if self.use_ema:
                    model_without_ddp.load_state_dict(current_state_dict)
                    del current_state_dict # 及时释放内存

                # 🌟 验证结束后立即回收 CUDA 缓存，防止显存碎片化导致下一轮训练 OOM
                # 验证阶段 (no_grad) 的显存分配模式与训练不同，
                # 缓存中的碎片化小块无法满足训练时的大块连续显存需求
                torch.cuda.empty_cache()

                if self.is_main_process:
                    print(f"[{epoch}/{epochs}] 📉 Train Loss: {train_loss:.4f} | Validation: {val_logs}")
                    for k, v in val_logs.items():
                        self.writer.add_scalar(f'Val/{k}', v, epoch)

                    if main_metric > self.best_metric:
                        self.best_metric = main_metric
                        save_path = os.path.join(self.save_dir, "best_model.pth")
                        # 剥离 DDP 外壳，仅保存纯净模型权重给后续推理用
                        # 🌟 修复：既然评估的是 EMA，保存的 best_model 当然也必须是 EMA！
                        if self.use_ema:
                            torch.save(self.ema_model.module.state_dict(), save_path)
                        else:
                            torch.save(model_without_ddp.state_dict(), save_path)
                        print(f"🌟 发现更优模型 (Metric: {main_metric:.4f})，已保存至: {save_path}")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.val_loader:
                        self.scheduler.step(main_metric)
                    else:
                        raise ValueError("请传入验证集数据集，以便使用 ReduceLROnPlateau 调度器")
                else:
                    # 简化处理，统一 step
                    self.scheduler.step()

            # best_model.pth 只保存最优模型权重，
            # heckpoint_latest.pth 则保存最新的训练状态（包含优化器和调度器），方便断点续训。
            if self.is_main_process:
                model_without_ddp = self.model.module if hasattr(self.model, 'module') else self.model
                checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric  # 🌟 修复：记录历史最高分！
                    }
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

                # 🌟 新增：保存断点时，必须把 EMA 的状态也存进去！
                if self.use_ema:
                    checkpoint['model_ema_state_dict'] = self.ema_model.module.state_dict()

                latest_save_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
                tmp_save_path = os.path.join(self.save_dir, "checkpoint_latest.pth.tmp")
                torch.save(checkpoint, tmp_save_path)
                # 写入完毕后，瞬间替换 (原子操作)
                os.replace(tmp_save_path, latest_save_path)
            
            # 🌟 修复：等主进程安全地把几 GB 的权重写完硬盘，大家再一起跨入下一个 Epoch！
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if self.is_main_process:
            print("\n🏁 训练流程圆满结束！")
            self.writer.close()