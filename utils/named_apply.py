from typing import Callable
import torch.nn as nn

'''
在大模型（比如包含了几十个 block、成百上千个子模块的 DINOv3）中，
不同层的参数往往需要采用不同的初始化方差（比如 LayerScale 的初始值），
或者就像你之前发现的，针对名为 qkv 的层做特殊的掩码处理。如果没有 name 这个“导航坐标”，这种微操是根本无法实现的。
'''
def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 主干网络
        self.backbone = nn.Linear(5, 5)
        # 分类头
        self.head = nn.Linear(5, 2)

def official_init(module):
    # 🚨 痛点：我只知道传进来的是个 nn.Linear，但我根本不知道它是 backbone 还是 head！
    if isinstance(module, nn.Linear):
        # 没办法区分，只能一刀切，全部设为 1
        nn.init.constant_(module.weight, 1.0)
        print(f"官方 apply: 初始化了一个 {type(module).__name__}")

def custom_init(module, name):
    if isinstance(module, nn.Linear):
        if "head" in name:
            nn.init.constant_(module.weight, 0.0)
            print(f"✅ named_apply: 发现目标 '{name}'，权重已设为 0")
        else:
            nn.init.constant_(module.weight, 1.0)
            print(f"➡️ named_apply: 普通层 '{name}'，权重设为 1")


if __name__ == "__main__":
    # 假设我们现在有一个简单的任务：我们需要初始化一个模型，把主干网络（backbone）的权重设为 1，把分类头（head）的权重设为 0。
    model_A = ToyModel()
    print("=== 测试官方 apply ===")
    model_A.apply(official_init)

    model_B = ToyModel()
    print("\n=== 测试 named_apply ===")
    named_apply(fn=custom_init, module=model_B)
