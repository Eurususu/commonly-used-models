import os
import sys
import yaml
import argparse
import torch

# 将项目根目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import build_model
from dataset import build_transforms
from engine.infer_engine import Inferencer

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 通用深度学习单图推理脚本")
    
    # 基础配置
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件的路径 (用于读取模型结构和预处理)")
    parser.add_argument('--checkpoint', type=str, required=True, help="训练好的模型权重 (.pth) 路径")
    
    # 待推理的图片路径
    parser.add_argument('--image', type=str, required=True, help="需要进行推理的图片路径")
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="运行设备")
    
    return parser.parse_args()

def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"❌ 找不到待推理的图片: {args.image}")
        
    cfg = load_yaml(args.config)
    
    print(f"{'='*50}")
    print(f"🕵️  初始化推理任务")
    print(f"🖼️  目标图片: {args.image}")
    print(f"📦 权重文件: {args.checkpoint}")
    print(f"🖥️  运行设备: {args.device}")
    print(f"{'='*50}\n")

    # ==========================================
    # 1. 组装预处理流水线 (复用 val_transforms)
    # ==========================================
    # 推理时，我们必须使用和验证集一模一样的 Transform（如确定的 Resize 和 Normalize），不能有随机裁剪或翻转
    if 'val_transforms' not in cfg['data']:
        raise ValueError("❌ 配置文件中缺少 'val_transforms'，不知道该如何预处理图片！")
        
    print("🔧 正在构建预处理流水线...")
    infer_transforms = build_transforms(cfg['data']['val_transforms'])

    # ==========================================
    # 2. 组装模型
    # ==========================================
    print("🧠 正在构建模型结构...")
    model = build_model(cfg['model']['name'], **cfg['model'].get('kwargs', {}))

    # ==========================================
    # 3. 拉起推理引擎 (内部会自动把模型推到 device 并加载权重)
    # ==========================================
    print("🚀 拉起推理引擎...")
    inferencer = Inferencer(
        model=model,
        transforms=infer_transforms,
        device=args.device,
        checkpoint_path=args.checkpoint
    )
    
    # ==========================================
    # 4. 执行预测！
    # ==========================================
    print("\n🎯 正在预测...")
    result = inferencer.predict(args.image)
    
    print(f"\n{'='*50}")
    print(f"🎉 推理结果出炉：")
    
    # 假设你的 Inferencer 返回的是包含类索引和置信度的字典 (符合我们之前写的分类逻辑)
    if isinstance(result, dict) and "class_idx" in result:
        print(f"👉 预测类别 ID : {result['class_idx']}")
        print(f"🔥 置信度 (Conf): {result['confidence'] * 100:.2f}%")
        
        # 可选：如果你有类别映射表，可以在这里打印具体的字符串名称
        # print(f"Raw Logits: {result['raw_logits']}")
    else:
        # 如果是分割任务，返回的可能是 [1, C, H, W] 的特征图
        print(f"👉 输出特征图形状: {result.shape}")
        
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()