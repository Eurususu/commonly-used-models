import torch
from debug_utils.view_struct import inspect_struct
from typing import Tuple, Dict, Any
import os
# def load_pretrained_weights(model: torch.nn.Module, weight_path: str) -> torch.nn.Module:
#     print(f"⏳ 正在加载预训练权重: {weight_path}")
    
#     # 1. 把权重加载到 CPU 内存中（防止直接加载到 GPU 导致显存峰值爆炸）
#     checkpoint = torch.load(weight_path, map_location="cpu")

#     # 分析权重的数据结构
#     # inspect_struct(checkpoint)
    
#     # 2. 剥离外壳：寻找真正的 state_dict
#     # 官方发布的 checkpoint 经常是一个大字典，包含 epoch、optimizer 等信息
#     if "model" in checkpoint:
#         state_dict = checkpoint["model"]
#     elif "student" in checkpoint:  # DINO 系列特有：有时会把权重存在 student 键下
#         state_dict = checkpoint["student"]
#     elif "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint  # 如果已经是纯净的权重字典，就直接用

#     # 3. 清洗键名 (Key Prefix)
#     # 如果官方是用多卡 (DDP) 训练的，键名前面往往会多出一个 "module." 或者 "backbone."
#     clean_state_dict = {}
#     for k, v in state_dict.items():
#         # 只要键名还以目标前缀开头，就一直进行剥离
#         while k.startswith("module.") or k.startswith("backbone."):
#             # 去除 DDP 带来的 'module.' 前缀
#             if k.startswith("module."):
#                 k = k[7:]
#             # 如果你只想要骨干网络，有时需要去除 'backbone.' 前缀
#             if k.startswith("backbone."):
#                 k = k[9:]
#         clean_state_dict[k] = v

#     # 4. 加载权重进模型
#     # strict=False 是灵魂！它允许模型和权重有少许不匹配（比如你多加了一个分类头）
#     load_msg = model.load_state_dict(clean_state_dict, strict=False)
    
#     # 5. 打印体检报告
#     print("✅ 权重加载完成！体检报告如下：")
#     if load_msg.missing_keys:
#         print(f"⚠️ 缺失的键 (模型有，但权重文件里没有):\n  {load_msg.missing_keys} , (共 {len(load_msg.missing_keys)} 个)")
#     if load_msg.unexpected_keys:
#         print(f"⚠️ 多余的键 (权重文件有，但模型里没有):\n  {load_msg.unexpected_keys} , (共 {len(load_msg.unexpected_keys)} 个)")
        
#     if not load_msg.missing_keys and not load_msg.unexpected_keys:
#         print("🎉 完美匹配！没有缺失或多余的键。")

#     return model


def load_checkpoint(
    model: torch.nn.Module, 
    weight_path: str, 
    strict: bool = True,                 # 推理/验证/续训必须 True，预训练加载设为 False
    prioritize_ema: bool = False,        # 推理/验证设为 True，续训和预训练设为 False
    strip_module_prefix: bool = True,    # 永远设为 True，兼容 DDP
    # strip_backbone_prefix: bool = False, # 只有在单加载 backbone 预训练时设为 True
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    大一统模型权重加载器，适配预训练、续训、验证、推理四大场景。
    
    返回:
        model: 加载好权重的模型
        checkpoint: 原始的字典对象 (方便后续提取 optimizer, epoch 等续训信息)
    """
    print(f"\n⏳ 正在加载权重: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"🚨 找不到权重文件: {weight_path}")

    # 1. 统一加载到 CPU 内存 (防显存爆炸)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    # 2. 智能提取真正的 state_dict
    state_dict = None
    if prioritize_ema and 'model_ema_state_dict' in checkpoint:
        state_dict = checkpoint['model_ema_state_dict']
    elif 'model_state_dict' in checkpoint: 
        state_dict = checkpoint['model_state_dict']
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "student" in checkpoint:  # 兼容 DINO 系列原版权重
        state_dict = checkpoint["student"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # 纯净的权重字典

    # 3. 清洗键名前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        if strip_module_prefix:
            while k.startswith("module."):
                k = k[7:]
        # if strip_backbone_prefix:
        #     while k.startswith("backbone."):
        #         k = k[9:]
        clean_state_dict[k] = v

    # 4. 注入模型并获取体检报告
    # 注意：如果 strict=False，就算不匹配也不会抛出异常，而是记录在 load_msg 里
    load_msg = model.load_state_dict(clean_state_dict, strict=strict)

    # 5. 打印体检报告
    if load_msg.missing_keys or load_msg.unexpected_keys:
        print("⚠️ 权重加载体检报告：发现不匹配的键！")
        if load_msg.missing_keys:
            print(f"  ❌ 缺失的键 (共 {len(load_msg.missing_keys)} 个): {load_msg.missing_keys[:5]} ...")
        if load_msg.unexpected_keys:
            print(f"  ❌ 多余的键 (共 {len(load_msg.unexpected_keys)} 个): {load_msg.unexpected_keys[:5]} ...")
        
        if strict:
            # 理论上 strict=True 时，PyTorch 底层已经报错了，这里做个防御性提示
            print("🛑 当前为严格模式 (strict=True)，程序将被中断！")
    else:
        print("🎉 完美匹配！没有缺失或多余的键。")

    # 返回模型本身，以及原始的 checkpoint 字典（方便外部提取优化器状态）
    return model, checkpoint



if __name__ == "__main__":
    # ================= 使用示例 =================
    from models import build_model

    # 1. 构建空壳模型
    dino_small = build_model("dinov3_small")

    # 2. 注入灵魂 (请替换为你实际的权重路径)
    weight_file = "/home/jia/anktechDrive/研发部/共享/算法模型/dinov3/vit_backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 
    dino_small, _ = load_checkpoint(dino_small, weight_file)

    # # 3. 如果有 GPU，别忘了把它放进显卡
    # dino_small = dino_small.cuda()
    # dino_small.eval() # 推理前务必设为 eval 模式