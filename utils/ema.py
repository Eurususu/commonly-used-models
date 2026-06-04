import copy
import torch

class ModelEMA:
    """模型指数滑动平均 (EMA) 工具类"""
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        # 剥离 DDP 外壳深拷贝 (EMA 不需要算梯度，也不需要 DDP 同步)
        model_without_ddp = model.module if hasattr(model, 'module') else model
        self.module = copy.deepcopy(model_without_ddp)
        self.module.eval()
        self.device = device or next(model.parameters()).device
        self.module.to(self.device)
        
        # 彻底冻结 EMA 模型的梯度
        for param in self.module.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, model):
        """在每次 optimizer.step() 之后调用此方法"""
        model_without_ddp = model.module if hasattr(model, 'module') else model
        
        # zip 遍历两者的 state_dict (包含 params 和 BN 的 running_mean 等 buffers)
        for ema_v, model_v in zip(self.module.state_dict().values(), model_without_ddp.state_dict().values()):
            # EMA 更新公式：theta_ema = decay * theta_ema + (1 - decay) * theta_new
            ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v.to(ema_v.device))