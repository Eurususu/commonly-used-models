import torch
import inspect
import torch.nn as nn
import layers
from ._modelRegistry import register_model

@register_model("dynamic_graph_model")
class DynamicSeqModel(nn.Module):
    '''
    基于配置文件的动态 DAG (有向无环图) 网络拼接器
    支持残差连接、特征融合等多分支结构
    '''

    def __init__(self, layers_cfg):
        super().__init__()
        # 使用 ModuleList 来容纳所有层，保持它们在网络中的顺序
        self.module_list = nn.ModuleList()
        # 记录每一层的路由来源
        self.from_indices = []
        
        print("🕸️  正在构建拓扑网络模型...")

        for i, layer_info in enumerate(layers_cfg):
            layer_name = layer_info.get("name")
            layer_args = layer_info.get("args", {})

            # 默认接收上一层 (-1)，如果是第一层 (0) 默认接收 "input"
            default_from = "input" if i == 0 else -1
            f_idx = layer_info.get("from", default_from)

            # --- 1. 获取类 ---
            if hasattr(layers, layer_name):
                layer_cls = getattr(layers, layer_name)
            elif hasattr(nn, layer_name):
                layer_cls = getattr(nn, layer_name)
            else:
                raise ValueError(f"❌ 构建失败：在 layers 目录中找不到组件 '{layer_name}'！"
                                 f"请检查拼写，或确认 scripts/update_init.py 是否已更新。")

            try:
                module_instance = layer_cls(**layer_args)
            except TypeError as e:
                raise TypeError(f"❌ 实例化 {layer_name} 失败，参数 {layer_args} 有误。详细报错: {e}")
            
            self.module_list.append(module_instance)
            self.from_indices.append(f_idx)
            


            print(f"  ├── 添加第{i}层:{layer_name} | From: {str(f_idx):<8} | 参数：{layer_args}")

        print("✅ 拓扑网络构建完毕！\n")
    
    def forward(self, x):
        # 记忆缓存
        saved_outputs = []
        
        for i, (layer, f_idx) in enumerate(zip(self.module_list, self.from_indices)):

            # --- 解析当前层的输入 ---
            if isinstance(f_idx, int) or isinstance(f_idx, str):
                # 单一输入
                if f_idx == "input":
                    current_input = x
                else:
                    current_input = saved_outputs[f_idx]
                
                # 单一输入直接传给 layer
                out = layer(current_input)

            elif isinstance(f_idx, list):
                # 多输入的情况 (传给 Concat 或 Add)，把取出的张量打包成列表
                current_input = []

                for f in f_idx:
                    if f == "input":
                        current_input.append(x)
                    else:
                        current_input.append(saved_outputs[f])
                
                sig = inspect.signature(layer.forward)
                valid_params = [
                    p for p in sig.parameters.values() 
                    if p.name != 'self' and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
                ]
                # 判断：如果 forward 需要接收的参数大于 1 个 (比如 def forward(self, x1, x2))
                if len(valid_params) > 1:
                    # 检查传入的张量数量是否和函数需要的参数数量一致
                    if len(current_input) != len(valid_params):
                        raise RuntimeError(f"❌ 拓扑错误：层 '{layer.__class__.__name__}' 需要 {len(valid_params)} 个输入，"
                                           f"但 YAML 中 from 字段只提供了 {len(current_input)} 个！")
                    # 使用星号 * 解包，等价于 layer(current_input[0], current_input[1])
                    out = layer(*current_input)
                else:
                    out = layer(current_input)


            else:
                raise ValueError(f"不支持的 from 参数类型: {f_idx}")
            

            # 把当前层的输出保存到记忆缓存中，供后面的层提取
            saved_outputs.append(out)

        return saved_outputs[-1]
        