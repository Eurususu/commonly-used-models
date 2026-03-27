import os
import sys
import yaml
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models import build_model
from dataset import create_dataloader, build_transforms
from loss import build_loss
from optim import build_optimizer
from scheduler import build_scheduler
from engine.train_engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="🚀 通用深度学习训练脚本")
    
    # 核心配置文件路径 (必填项)
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件的路径")
    
    # 引擎运行参数 (从命令行传入)
    parser.add_argument('--epochs', type=int, default=50, help="训练的总 Epoch 数")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="运行设备 (cuda 或 cpu)")
    parser.add_argument('--save_dir', type=str, default='./checkpoints/exp_default', help="权重保存目录")
    
    # 可选项：是否从断点恢复训练
    parser.add_argument('--resume', type=str, default=None, help="恢复训练的 checkpoint 路径")
    
    return parser.parse_args()


def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    

def main():
    # 1. 解析命令行参数与 YAML 配置
    args = parse_args()
    cfg = load_yaml(args.config)
    
    print(f"{'='*50}")
    print(f"🔥 初始化训练任务 | 配置: {args.config}")
    print(f"🖥️  设备: {args.device} | Epochs: {args.epochs} | 保存至: {args.save_dir}")
    print(f"{'='*50}\n")

    # ==========================================
    # 2. 组装数据流 (Data)
    # ==========================================
    print("📦 正在构建数据流...")
    # 训练集
    train_transforms = build_transforms(cfg['data'].get('train_transforms', []))
    cfg['data']['train_dataset']['kwargs']['transforms'] = train_transforms
    train_loader = create_dataloader(
        dataset_name=cfg['data']['train_dataset']['name'],
        dataset_cfg=cfg['data']['train_dataset']['kwargs'],
        loader_cfg=cfg['data']['train_loader']
    )
    
    # 验证集 (可选配置)
    val_loader = None
    if 'val_dataset' in cfg['data']:
        val_transforms = build_transforms(cfg['data'].get('val_transforms', []))
        cfg['data']['val_dataset']['kwargs']['transforms'] = val_transforms
        val_loader = create_dataloader(
            dataset_name=cfg['data']['val_dataset']['name'],
            dataset_cfg=cfg['data']['val_dataset']['kwargs'],
            loader_cfg=cfg['data']['val_loader']
        )

    # ==========================================
    # 3. 组装算法核心 (Model & Loss)
    # ==========================================
    print("🧠 正在构建模型与损失函数...")
    model = build_model(cfg['model']['name'], **cfg['model'].get('kwargs', {}))
    criterion = build_loss(cfg['loss']['name'], **cfg['loss'].get('kwargs', {}))

    # ==========================================
    # 4. 组装动力系统 (Optim & Scheduler)
    # ==========================================
    print("⚙️  正在构建优化器与调度器...")
    optimizer = build_optimizer(
        model.parameters(), 
        cfg['optim']['name'], 
        **cfg['optim'].get('kwargs', {})
    )
    
    scheduler = None
    if 'scheduler' in cfg and cfg['scheduler'] is not None:
        scheduler = build_scheduler(
            optimizer, 
            cfg['scheduler']['name'], 
            **cfg['scheduler'].get('kwargs', {})
        )

    # ==========================================
    # 5. 断点恢复 (可选)
    # ==========================================
    if args.resume and os.path.exists(args.resume):
        print(f"⏳ 正在从 {args.resume} 恢复权重...")
        model.load_state_dict(torch.load(args.resume, map_location=args.device))

    # ==========================================
    # 6. 拉起引擎，开始训练！
    # ==========================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir
    )
    
    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()