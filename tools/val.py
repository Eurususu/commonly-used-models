import os
import sys
import yaml
import argparse
import torch

# 将项目根目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import build_model
from dataset import create_dataloader, build_transforms
from loss import build_loss
from engine.val_engine import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="📉 通用深度学习验证/评估脚本")
    
    # 核心配置文件 (必须和训练时用的一样，保证 Transform 和 Model 结构一致)
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件的路径")
    
    # 必须要传入训练好的权重文件
    parser.add_argument('--checkpoint', type=str, required=True, help="模型权重 (.pth) 的路径")
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="运行设备")
    
    return parser.parse_args()

def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    
    print(f"{'='*50}")
    print(f"📉 初始化验证任务 | 配置: {args.config}")
    print(f"📂 权重文件: {args.checkpoint}")
    print(f"🖥️  设备: {args.device}")
    print(f"{'='*50}\n")

    # ==========================================
    # 1. 组装验证数据流 (Val Data)
    # ==========================================
    if 'val_dataset' not in cfg['data']:
        raise ValueError("❌ 配置文件中缺少 'val_dataset' 的配置，无法进行验证！")

    print("📦 正在构建验证数据流...")
    val_transforms = build_transforms(cfg['data'].get('val_transforms', []))
    
    # ⚠️ 这里就严格使用 transforms 作为字典的 key 塞进去
    cfg['data']['val_dataset']['kwargs']['transforms'] = val_transforms
    
    val_loader = create_dataloader(
        dataset_name=cfg['data']['val_dataset']['name'],
        dataset_cfg=cfg['data']['val_dataset']['kwargs'],
        loader_cfg=cfg['data']['val_loader']
    )

    # ==========================================
    # 2. 组装模型与损失函数
    # ==========================================
    print("🧠 正在构建模型与损失函数...")
    model = build_model(cfg['model']['name'], **cfg['model'].get('kwargs', {}))
    criterion = build_loss(cfg['loss']['name'], **cfg['loss'].get('kwargs', {}))

    # ==========================================
    # 3. 加载预训练权重 (核心步骤)
    # ==========================================
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ 找不到权重文件: {args.checkpoint}")
        
    print(f"⏳ 正在加载权重: {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    print("✅ 权重加载成功！")

    model = model.to(args.device)

    # ==========================================
    # 4. 拉起验证引擎，开始评估！
    # ==========================================
    # 实例化我们在 engine/val_engine.py 里写的 Evaluator
    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=args.device
    )
    
    print("\n🚀 开始评估...")
    val_loss, val_acc = evaluator.evaluate()
    
    print(f"\n{'='*50}")
    print(f"🎉 评估完成！")
    print(f"📉 平均 Loss : {val_loss:.4f}")
    print(f"🎯 准确率 Acc : {val_acc * 100:.2f}%")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()