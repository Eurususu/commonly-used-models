import os
import sys
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_checkpoints import load_checkpoint


from models import build_model
from dataset import create_dataloader, build_transforms
from loss import build_loss
from optim import build_optimizer
from scheduler import build_scheduler
from engine import build_task
from utils.set_seed import set_seed


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

    # 从命令行参数或 YAML 中读取 base_seed，默认给个 42
    base_seed = cfg.get('seed', getattr(args, 'seed', 42))

    # 🔥 核心：让每个 GPU 拥有独立的种子，防止数据增强生成一模一样的随机变换
    seed = base_seed + global_rank

    # 建议 deterministic=False，保留 cuDNN 加速；发论文需严格复现时改为 True
    set_seed(seed, deterministic=False)

    if is_main_process:
        print(f"{'='*50}")
        print(f"🔥 初始化训练任务 | 配置: {args.config}")
        print(f"🖥️  设备模式: {'DDP 多卡分布式' if is_distributed else '单卡/CPU'} | GPU数量: {world_size}")
        print(f"🌱 基础随机种子: {base_seed} (当前进程种子: {seed})")
        print(f"📦  Epochs: {args.epochs} | 保存至: {args.save_dir}")
        print(f"{'='*50}\n")

    # ==========================================
    # 1. 组装算法核心 (Model)
    # ==========================================
    # 🌟 所有模型（包括 dinov3_det）统一通过 build_model(name, **kwargs) 构建
    # dinov3_det 的 backbone 构建、权重加载、枚举转换等逻辑已内聚到其工厂函数中
    if is_main_process: print("🧠 正在构建模型...")
    model_name = cfg['model']['name']
    model_kwargs = cfg['model'].get('kwargs', {})
    model = build_model(model_name, **model_kwargs)

    model = model.to(device)
    pretrained_weight = cfg['model'].get('pretrained_weight', None)
    if pretrained_weight is not None and os.path.exists(pretrained_weight):
        model, _ = load_checkpoint(model, pretrained_weight, strict=False)

    # 🌟 核心魔法：如果是多卡，使用 DDP 包裹模型
    if is_distributed:
        # 将普通的 BatchNorm 转换为跨卡同步的 SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # [🌟 修复] 动态判定：如果是 DETR 模型，强烈建议开启 find_unused_parameters
        find_unused = True if model_name == "dinov3_det" else False
        # 包裹 DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)



    # ==========================================
    # 2. 组装数据流 (Data) - 需要注入 DDP 状态 - [🌟 增加 DETR collate_fn 兼容]
    # ==========================================
    if is_main_process: print("📦 正在构建数据流...")
    # 获取特殊的 collate_fn (DETR 需要把图片拼成 NestedTensor)
    collate_fn = None
    if model_name == "dinov3_det":
        try:
            from utils.misc import collate_fn as detr_collate_fn
            collate_fn = detr_collate_fn
        except ImportError:
            if is_main_process: print("⚠️ 警告: 未找到 DETR 专用的 collate_fn，若 DataLoader 报错请检查。")

    # 把 collate_fn 塞进 loader_cfg 里传给下游
    if collate_fn is not None:
        cfg['data']['train_loader']['collate_fn'] = collate_fn
        if 'val_loader' in cfg['data']:
            cfg['data']['val_loader']['collate_fn'] = collate_fn

    # 训练集
    train_transforms = build_transforms(cfg['data'].get('train_transforms', []))
    cfg['data']['train_dataset']['kwargs']['transforms'] = train_transforms
    # ⚠️ 注意：这里我们给 create_dataloader 多传了一个 is_distributed 标志
    train_loader = create_dataloader(
        dataset_name=cfg['data']['train_dataset']['name'],
        dataset_cfg=cfg['data']['train_dataset']['kwargs'],
        loader_cfg=cfg['data']['train_loader'],
        is_distributed=is_distributed, # 新增参数
        seed=seed  # 👈 🌟 注入当前进程的种子
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
            is_distributed=is_distributed, # 新增参数
            seed=seed  # 👈 🌟 注入当前进程的种子
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
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if is_main_process: print(f"⏳ 正在从 {args.resume} 恢复权重...")

        # 🌟 修复 1：处理 DDP 包装器的 module. 前缀问题
        # 如果模型已经被 DDP 包装，我们需要提取底层的 model.module 来加载干净的权重
        model_without_ddp = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        _, checkpoint = load_checkpoint(model_without_ddp, args.resume)

        # 🌟 修复 2：全面恢复优化器、调度器和 Epoch 计数器
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 🌟 核心修复：遍历优化器里的每一个状态张量，强行搬运到当前 GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if is_main_process: print("✅ 优化器状态已恢复")

        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process: print("✅ 学习率调度器状态已恢复")

        if 'epoch' in checkpoint:
            # 恢复训练时，应该从保存的下一个 Epoch 开始
            start_epoch = checkpoint['epoch'] + 1
            if is_main_process: print(f"✅ 训练进度已恢复，将从 Epoch {start_epoch} 继续训练")

    # ==========================================
    # 5. 拉起引擎，开始训练！
    # ==========================================
    # 🌟 从 YAML 配置文件中读取具体的任务名称
    # 如果没写，默认兼容以前的分类任务
    task_name = cfg.get('task', 'train_classification')
    # 🌟 动态获取梯度裁剪阈值 (DETR 官方标配是 0.1)
    clip_max_norm = cfg.get('optim', {}).get('clip_max_norm', 0.1 if model_name == "dinov3_det" else None)
    trainer = build_task(
        name=task_name,
        criterion=criterion,       # 子类特有参数
        model=model,               # BaseTrainer 基础参数 ->
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        is_main_process=is_main_process,
        clip_max_norm=clip_max_norm  # 🌟 将参数传入基础引擎
    )
    if args.resume and os.path.exists(args.resume):
        trainer.best_metric = checkpoint.get('best_metric', 0.0)
    if is_main_process: print("🚀 开始训练...")
    trainer.train(start_epoch=start_epoch, epochs=args.epochs)

    # 销毁分布式环境
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
