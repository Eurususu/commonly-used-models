import torch.optim.lr_scheduler as lr_scheduler
from ._schedulerRegistry import register_scheduler

__all__ = []

# 1. 阶梯下降 (每隔 step_size 个 epoch，学习率乘以 gamma)
register_scheduler("step_lr")(lr_scheduler.StepLR)

# 2. 多阶梯下降 (在 milestones 指定的 epoch 处，学习率乘以 gamma)
register_scheduler("multistep_lr")(lr_scheduler.MultiStepLR)

# 3. 指数衰减 (每个 epoch 学习率乘以 gamma)
register_scheduler("exponential_lr")(lr_scheduler.ExponentialLR)

# 4. 余弦退火 (按照余弦曲线让学习率在 T_max 个 epoch 内降到 eta_min)
register_scheduler("cosine_annealing")(lr_scheduler.CosineAnnealingLR)

# 5. 基于指标的自适应下降 (当验证集 loss 不再下降时，才衰减学习率)
# ⚠️ 注意：这个调度器在 BaseTrainer 调用 .step() 时，需要把 val_loss 传给它！
register_scheduler("reduce_lr_on_plateau")(lr_scheduler.ReduceLROnPlateau)

# 6. 周期性余弦退火热重启 (带 WarmRestart 的高级版)
register_scheduler("cosine_annealing_warm_restarts")(lr_scheduler.CosineAnnealingWarmRestarts)