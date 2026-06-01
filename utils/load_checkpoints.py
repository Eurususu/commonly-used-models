import torch
from debug_utils.view_struct import inspect_struct
def load_pretrained_weights(model: torch.nn.Module, weight_path: str) -> torch.nn.Module:
    print(f"⏳ 正在加载预训练权重: {weight_path}")
    
    # 1. 把权重加载到 CPU 内存中（防止直接加载到 GPU 导致显存峰值爆炸）
    checkpoint = torch.load(weight_path, map_location="cpu")

    # 分析权重的数据结构
    # inspect_struct(checkpoint)
    
    # 2. 剥离外壳：寻找真正的 state_dict
    # 官方发布的 checkpoint 经常是一个大字典，包含 epoch、optimizer 等信息
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "student" in checkpoint:  # DINO 系列特有：有时会把权重存在 student 键下
        state_dict = checkpoint["student"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # 如果已经是纯净的权重字典，就直接用

    # 3. 清洗键名 (Key Prefix)
    # 如果官方是用多卡 (DDP) 训练的，键名前面往往会多出一个 "module." 或者 "backbone."
    clean_state_dict = {}
    for k, v in state_dict.items():
        # 去除 DDP 带来的 'module.' 前缀
        if k.startswith("module."):
            k = k[7:]
        # 如果你只想要骨干网络，有时需要去除 'backbone.' 前缀
        if k.startswith("backbone."):
            k = k[9:]
        clean_state_dict[k] = v

    # 4. 加载权重进模型
    # strict=False 是灵魂！它允许模型和权重有少许不匹配（比如你多加了一个分类头）
    load_msg = model.load_state_dict(clean_state_dict, strict=False)
    
    # 5. 打印体检报告
    print("✅ 权重加载完成！体检报告如下：")
    if load_msg.missing_keys:
        print(f"⚠️ 缺失的键 (模型有，但权重文件里没有):\n  {load_msg.missing_keys} , (共 {len(load_msg.missing_keys)} 个)")
    if load_msg.unexpected_keys:
        print(f"⚠️ 多余的键 (权重文件有，但模型里没有):\n  {load_msg.unexpected_keys} , (共 {len(load_msg.unexpected_keys)} 个)")
        
    if not load_msg.missing_keys and not load_msg.unexpected_keys:
        print("🎉 完美匹配！没有缺失或多余的键。")

    return model

if __name__ == "__main__":
    # ================= 使用示例 =================
    from models import build_model

    # 1. 构建空壳模型
    dino_small = build_model("dino_small")

    # 2. 注入灵魂 (请替换为你实际的权重路径)
    weight_file = "/home/jia/anktechDrive/研发部/共享/算法模型/dinov3/vit_backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 
    dino_small = load_pretrained_weights(dino_small, weight_file)

    # # 3. 如果有 GPU，别忘了把它放进显卡
    # dino_small = dino_small.cuda()
    # dino_small.eval() # 推理前务必设为 eval 模式