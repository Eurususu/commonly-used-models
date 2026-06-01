import torch.optim.lr_scheduler as lr_scheduler
from ._schedulerRegistry import register_scheduler
from typing import Literal
from torch.optim.optimizer import Optimizer
import math
from packaging.version import Version
import torch

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

register_scheduler("constant_lr")(lr_scheduler.ConstantLR)

register_scheduler("linear_lr")(lr_scheduler.LinearLR)

register_scheduler("polynomial_lr")(lr_scheduler.PolynomialLR)

register_scheduler("one_cycle_lr")(lr_scheduler.OneCycleLR)


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return (end - start) * pct + start

TORCH_VERSION = Version(torch.__version__)

@register_scheduler("warmup_one_cycle_lr")
class WarmupOneCycleLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | None = None,
        total_steps: int = 0,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.0,
        pct_start: float = 0.295,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1000.0,
        use_beta1: bool = True,
        update_momentum: bool = True,
        last_epoch: int = -1,
    ):
        """
        A variant of OneCycleLR with a warmup on top which potentially
        replaces the first phase of the original OneCycleLR.
        """
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.max_lr = max_lr
        self.min_point = float(pct_start * total_steps)
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.total_steps = total_steps
        self.use_beta1 = use_beta1
        self.anneal_strategy = anneal_strategy
        self.final_div_factor = final_div_factor
        self.update_momentum = update_momentum
        assert self.anneal_strategy in [
            "cos",
            "linear",
        ], f"Only cosine and linear-annealing strategy supported, got {self.anneal_strategy}"
        assert total_steps > 0

        # Initialize learning rate variables and momentum
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                assert last_epoch == -1
                ml = group["lr"]
                assert isinstance(ml, float)  # makes sure that the variable is well updated
                group["initial_lr"] = ml / div_factor
                group["max_lr"] = ml
                group["min_lr"] = group["initial_lr"] / final_div_factor
                # initialize learning rate
                group["lr"] = ml / final_div_factor if self.warmup_iters > 0 else group["initial_lr"]
            if self.use_beta1:
                group["betas"] = (self.max_momentum, *group["betas"][1:])
            elif self.update_momentum:
                group["momentum"] = self.max_momentum

        super().__init__(optimizer, last_epoch)

    def _anneal_func(self, *args, **kwargs):
        if self.anneal_strategy == "cos":
            return annealing_cos(*args, **kwargs)
        elif self.anneal_strategy == "linear":
            return annealing_linear(*args, **kwargs)

    def _compute_lr_momentum(self, optimizer_param_group):
        # torch.optim.lr_scheduler.LRScheduler does an initial
        # step that sets self._step_count = 1
        step_num = (self._step_count - 1) if self.last_epoch != -1 else 0
        momentum = 0
        if step_num < self.warmup_iters:
            if self.warmup_ratio:
                k = (1 - step_num / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = optimizer_param_group["max_lr"] * (1 - k)
                thelr = warmup_lr * (1 - step_num / self.total_steps)
            else:
                gmax = (
                    optimizer_param_group["max_lr"] * (1 + math.cos(math.pi * step_num / float(self.total_steps))) / 2
                )
                thelr = optimizer_param_group["max_lr"] / self.final_div_factor + gmax * step_num / float(
                    self.warmup_iters
                )
        else:
            pct = (step_num - self.warmup_iters) / float(self.total_steps - self.warmup_iters)
            step_num_to_use = step_num
            momentum = self._anneal_func(
                self.base_momentum,
                self.max_momentum,
                pct,
            )
            if self.anneal_strategy == "cos":
                step_num_to_use += 1
            thelr = self._anneal_func(
                optimizer_param_group["max_lr"],
                optimizer_param_group["min_lr"],
                step_num_to_use / float(self.total_steps),
            )
        return thelr, momentum

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        if TORCH_VERSION >= Version("2.4.0"):
            lr_scheduler._warn_get_lr_called_within_step(self)

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}"  # noqa: UP032
            )

        for group in self.optimizer.param_groups:
            computed_lr, computed_momentum = self._compute_lr_momentum(group)
            lrs.append(computed_lr)  # type: ignore[possibly-undefined]
            if self.use_beta1:
                group["betas"] = (computed_momentum, *group["betas"][1:])  # type: ignore[possibly-undefined]
            elif self.update_momentum:
                group["momentum"] = computed_momentum  # type: ignore[possibly-undefined]

        return lrs