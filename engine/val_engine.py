import torch
from tqdm import tqdm

class Evaluator:
    """验证/评估引擎"""
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        """执行一整轮的验证集评估"""
        self.model.eval()
        total_loss = 0.0
        
        # 简单记录正确率 (这里以分类任务为例，后续可以重构接入专门的 metrics 模块)
        correct = 0
        total = 0

        # 关闭梯度计算，节省显存并加速
        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="📉 Validating", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 分类任务的通用 Accuracy 计算逻辑
                if outputs.ndim == 2: 
                    preds = torch.argmax(outputs, dim=1)
                    total += targets.size(0)
                    correct += (preds == targets).sum().item()

        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy