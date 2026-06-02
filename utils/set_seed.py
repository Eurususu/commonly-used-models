import random
import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """
    锁死所有随机数种子，确保实验 100% 可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 兼容非 DDP 下的多卡模式
    
    # 🌟 极致复现模式 (可选)
    # 如果开启，会强迫 cuDNN 使用确定性算法，禁用自动寻优。
    # 优点：绝对的一模一样；缺点：训练速度可能会下降 5%~10%。
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 默认模式：允许 cuDNN 寻找最快卷积算法 (稍微牺牲一点点极端的数值一致性，换取速度)
        torch.backends.cudnn.benchmark = True