import torch
from PIL import Image

__all__ = ['Inferencer']

class Inferencer:
    """单张图像推理引擎"""
    def __init__(self, model, transforms, device, checkpoint_path=None):
        self.model = model.to(device)
        self.transforms = transforms
        self.device = device

        # 加载训练好的权重
        if checkpoint_path:
            self._load_weights(checkpoint_path)

        # 强制设为 eval 模式 (关闭 Dropout 和 BatchNorm 更新)
        self.model.eval()

    def _load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ 成功加载模型权重: {path}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载权重失败: {e}")

    def predict(self, image_path_or_tensor):
        """对单张图片进行推理"""
        
        # 1. 数据预处理
        if isinstance(image_path_or_tensor, str):
            # 如果传入的是路径，读取并执行 transform
            image = Image.open(image_path_or_tensor).convert('RGB')
            if self.transforms is None:
                raise ValueError("未提供 transforms，无法对原始图片进行处理！")
            # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        else:
            # 如果直接传入的是 Tensor，确保在正确的 device 上
            input_tensor = image_path_or_tensor.to(self.device)
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

        # 2. 前向传播
        with torch.no_grad():
            output = self.model(input_tensor)

        # 3. 后处理逻辑 (以分类任务为例)
        if output.ndim == 2:
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            return {"class_idx": pred_class, "confidence": confidence, "raw_logits": output}
            
        return output