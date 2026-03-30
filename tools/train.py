import os
import sys
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

    # ==========================================
    # 0. 🌐 DDP 分布式环境初始化
    # ==========================================
    # torchrun 会自动注入 LOCAL_RANK 环境变量。如果没有，说明是普通单卡运行 (-1)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_distributed = local_rank != -1

    if is_distributed:
        # 初始化进程组 (使用 nccl 后端，N卡专属最高效通信)
        dist.init_process_group(backend="nccl")
        # 绑定当前进程到指定GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # 获取全局进程数和当前进程的全局 ID
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        global_rank = 0

    # 为了避免多卡同时打印日志导致屏幕爆炸，我们只允许 Rank 0 (主进程) 打印
    is_main_process = global_rank == 0

    if is_main_process:
        print(f"{'='*50}")
        print(f"🔥 初始化训练任务 | 配置: {args.config}")
        print(f"🖥️  设备模式: {'DDP 多卡分布式' if is_distributed else '单卡/CPU'} | GPU数量: {world_size}")
        print(f"📦  Epochs: {args.epochs} | 保存至: {args.save_dir}")
        print(f"{'='*50}\n")

    # ==========================================
    # 1. 组装算法核心 (Model)
    # ==========================================
    if is_main_process: print("🧠 正在构建模型...")
    model = build_model(cfg['model']['name'], **cfg['model'].get('kwargs', {}))
    model = model.to(device)

    # 🌟 核心魔法：如果是多卡，使用 DDP 包裹模型
    if is_distributed:
        # 将普通的 BatchNorm 转换为跨卡同步的 SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 包裹 DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    


    # ==========================================
    # 2. 组装数据流 (Data) - 需要注入 DDP 状态
    # ==========================================
    if is_main_process: print("📦 正在构建数据流...")
    # 训练集
    train_transforms = build_transforms(cfg['data'].get('train_transforms', []))
    cfg['data']['train_dataset']['kwargs']['transforms'] = train_transforms
    # ⚠️ 注意：这里我们给 create_dataloader 多传了一个 is_distributed 标志
    train_loader = create_dataloader(
        dataset_name=cfg['data']['train_dataset']['name'],
        dataset_cfg=cfg['data']['train_dataset']['kwargs'],
        loader_cfg=cfg['data']['train_loader'],
        is_distributed=is_distributed # 新增参数
    )
    
    # 验证集 (可选配置)
    val_loader = None
    if 'val_dataset' in cfg['data']:
        val_transforms = build_transforms(cfg['data'].get('val_transforms', []))
        cfg['data']['val_dataset']['kwargs']['transforms'] = val_transforms
        val_loader = create_dataloader(
            dataset_name=cfg['data']['val_dataset']['name'],
            dataset_cfg=cfg['data']['val_dataset']['kwargs'],
            loader_cfg=cfg['data']['val_loader'],
            is_distributed=is_distributed # 新增参数
        )

    # ==========================================
    # 3. 组装动力系统 (Loss, Optim, Scheduler)
    # ==========================================
    if is_main_process: print("🧠 正在构建损失函数、优化器、调度器...")
    criterion = build_loss(cfg['loss']['name'], **cfg['loss'].get('kwargs', {}))
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
    # 4. 断点恢复 (可选)
    # ==========================================
    if args.resume and os.path.exists(args.resume):
        if is_main_process: print(f"⏳ 正在从 {args.resume} 恢复权重...")
        # 注意：多卡恢复时，需将权重映射到当前卡的 device
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # ==========================================
    # 5. 拉起引擎，开始训练！
    # ==========================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        is_main_process=is_main_process # 传入引擎，用来控制日志打印和权重保存
    )
    
    trainer.train(epochs=args.epochs)

    # 销毁分布式环境
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()