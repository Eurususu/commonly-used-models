import torch
import torch.nn as nn

__all__ = ["ModelAnalyzer"]

class ModelAnalyzer:
    """
    深度学习模型结构与数据流剖析器
    基于 Forward Hook 动态捕获每一层的确切输入输出维度。
    """
    def __init__(self, model):
        self.model = model
        self.summary_data = []
        self.hooks = []

    def _get_shape(self, data):
        """递归解析复杂数据结构（Tensor, List, Tuple, Dict）的形状"""
        if isinstance(data, torch.Tensor):
            return str(list(data.shape))
        elif isinstance(data, (list, tuple)):
            # 如果元组里只有一个元素，直接返回该元素的形状，避免显示过于冗余
            if len(data) == 1:
                return self._get_shape(data[0])
            return "[" + ", ".join([self._get_shape(x) for x in data]) + "]"
        elif isinstance(data, dict):
            return "Dict{" + ", ".join([f"{k}: {self._get_shape(v)}" for k, v in data.items()]) + "}"
        elif hasattr(data, 'tensors') and hasattr(data, 'mask'): 
            # 💡 专门适配 DETR 的 NestedTensor
            return f"Nested(t={list(data.tensors.shape)}, m={list(data.mask.shape)})"
        else:
            return type(data).__name__

    def _hook_fn(self, module, inputs, outputs, name):
        """拦截器：捕获当前层的状态"""
        # 统计该层的参数量 (只算可训练参数)
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        self.summary_data.append({
            'name': name,
            'type': module.__class__.__name__,
            'input_shape': self._get_shape(inputs),
            'output_shape': self._get_shape(outputs),
            'params': params
        })

    def analyze(self, *dummy_inputs, **dummy_kwargs):
        """
        执行动态分析
        Args:
            dummy_inputs: 模拟输入的位置参数
            dummy_kwargs: 模拟输入的关键字参数
        """
        self.summary_data = []
        
        # 1. 注册 Hooks (只注册叶子节点/具体执行计算的层，避免容器层导致的重复统计)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0: # 判断是否为叶子节点
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self._hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

        # 2. 触发一次前向传播 (开启 eval 模式，且不计算梯度以节省显存)
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 将 dummy_inputs 转移到模型所在设备
        def _to_device(x):
            if isinstance(x, torch.Tensor): return x.to(device)
            return x
            
        dummy_inputs = [_to_device(x) for x in dummy_inputs]
        dummy_kwargs = {k: _to_device(v) for k, v in dummy_kwargs.items()}

        try:
            with torch.no_grad():
                self.model(*dummy_inputs, **dummy_kwargs)
        except Exception as e:
            print(f"❌ 分析过程中前向传播失败: {e}")
        finally:
            # 3. 无论成功与否，务必拆除 Hooks，否则会造成内存泄漏！
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

        self._print_summary()

    def _print_summary(self):
        """格式化打印出漂亮的表格"""
        if not self.summary_data:
            print("⚠️ 未收集到任何层的数据。")
            return

        # 计算自适应列宽
        max_name = max(20, max(len(d['name']) for d in self.summary_data))
        max_type = max(15, max(len(d['type']) for d in self.summary_data))
        max_in = max(25, max(len(d['input_shape']) for d in self.summary_data))
        max_out = max(25, max(len(d['output_shape']) for d in self.summary_data))
        
        total_width = max_name + max_type + max_in + max_out + 12 + 15
        
        print("=" * total_width)
        print(f"{'Layer Name':<{max_name}} | {'Type':<{max_type}} | {'Input Shape':<{max_in}} | {'Output Shape':<{max_out}} | {'Params':<10}")
        print("-" * total_width)
        
        total_params = 0
        for d in self.summary_data:
            print(f"{d['name']:<{max_name}} | {d['type']:<{max_type}} | {d['input_shape']:<{max_in}} | {d['output_shape']:<{max_out}} | {d['params']:<10,}")
            total_params += d['params']
            
        print("=" * total_width)
        print(f"🔥 总可训练参数量 (仅叶子层累加): {total_params:,}")
        print("=" * total_width)


# ==========================================
# 🚀 使用示例
# ==========================================
if __name__ == "__main__":
    from models import build_model
    model = build_model("dinov3_small")

    
    # 造一个 Dummy Input (Batch_size=2, Channel=3, H=224, W=224)
    dummy_tensor = torch.randn(2, 3, 224, 224)

    
    # 启动分析！
    analyzer = ModelAnalyzer(model)
    analyzer.analyze(dummy_tensor)