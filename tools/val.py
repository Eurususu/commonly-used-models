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
from engine import build_task
from utils.set_seed import set_seed

'''
# 单卡验证
python tools/val.py --config config/yamls/dinov3_det.yaml --checkpoint checkpoints/exp_default/best_model.pth

# 4卡分布式验证 (极速跑完整个 COCO 验证集)
torchrun --nproc_per_node=4 tools/val.py --config config/yamls/dinov3_det.yaml --checkpoint checkpoints/exp_default/best_model.pth
'''


def parse_args():
    parser = argparse.ArgumentParser(description="🚀 通用深度学习独立验证/评估脚本")

    # 核心参数
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件的路径")
    parser.add_argument('--checkpoint', type=str, required=True, help="要评估的模型权重路径 (如 best_model.pth)")

    return parser.parse_args()


def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # ==========================================
    # 0. 🌐 DDP 分布式环境初始化 (必须保留)
    # ==========================================
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_distributed = local_rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        global_rank = 0

    is_main_process = global_rank == 0

    # 锁定随机种子 (保证数据预处理可复现)
    base_seed = cfg.get('seed', 42)
    seed = base_seed + global_rank
    set_seed(seed, deterministic=False)

    if is_main_process:
        print(f"{'='*50}")
        print(f"🔬 初始化独立验证任务 | 配置: {args.config}")
        print(f"🖥️  设备模式: {'DDP 多卡分布式' if is_distributed else '单卡/CPU'} | GPU数量: {world_size}")
        print(f"📦 目标权重: {args.checkpoint}")
        print(f"{'='*50}\n")

    # ==========================================
    # 1. 组装算法核心 (Model)
    # ==========================================
    if is_main_process: print("🧠 正在构建模型并加载权重...")
    model_name = cfg['model']['name']
    model_kwargs = cfg['model'].get('kwargs', {})
    model = build_model(model_name, **model_kwargs)
    
    # 🌟 核心：加载验证权重
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # # 兼容处理：支持 checkpoint_latest.pth (包含 optimizer 的大字典) 和 best_model.pth (纯权重)
    # state_dict = checkpoint.get('model_state_dict', checkpoint)
    # 兼容性处理：如果你保存的时候存的是纯权重，或者是包含了 'model_state_dict' 的完整字典
    if 'model_ema_state_dict' in checkpoint:
        state_dict = checkpoint['model_ema_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # 兼容只保存了模型权重的旧版本文件
    
    # 健壮性处理：防止有些旧权重保存时带了 'module.' 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v

    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)

    # 包装 DDP (验证阶段包装 DDP 不是为了算梯度，而是为了防止有些含有 SyncBatchNorm 的模型报错)
    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ==========================================
    # 2. 组装验证集数据流 (Data)
    # ==========================================
    if is_main_process: print("📦 正在构建验证集数据流...")
    
    # 获取 DETR 特殊的 collate_fn
    collate_fn = None
    if model_name == "dinov3_det":
        from utils.misc import collate_fn as detr_collate_fn
        collate_fn = detr_collate_fn
        cfg['data']['val_loader']['collate_fn'] = collate_fn

    if 'val_dataset' not in cfg['data']:
        raise ValueError("❌ YAML 配置文件中缺失 'val_dataset' 字段，无法执行验证！")

    val_transforms = build_transforms(cfg['data'].get('val_transforms', []))
    cfg['data']['val_dataset']['kwargs']['transforms'] = val_transforms
    
    val_loader = create_dataloader(
        dataset_name=cfg['data']['val_dataset']['name'],
        dataset_cfg=cfg['data']['val_dataset']['kwargs'],
        loader_cfg=cfg['data']['val_loader'],
        is_distributed=is_distributed,
        seed=seed 
    )

    # ==========================================
    # 3. 组装损失函数 (部分 Validator 的 __init__ 需要它)
    # ==========================================
    criterion = build_loss(cfg['loss']['name'], **cfg['loss'].get('kwargs', {}))

    # ==========================================
    # 4. 拉起验证器引擎，开始评估！
    # ==========================================
    # 这里直接获取验证任务的名称，如果没有指定，默认使用 val_detection
    val_task_name = cfg.get('val_task', 'val_detection')
    
    validator = build_task(
        name=val_task_name,
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        is_main_process=is_main_process
    )

    if is_main_process: print("\n🚀 开始执行评估...")
    
    # 调用评估函数 (根据你的 DetectionValidator 定义，这里传入 is_coco=True)
    main_metric, val_logs = validator.evaluate(is_coco=True)

    if is_main_process:
        print(f"\n{'='*50}")
        print(f"🎉 评估结束！最终核心指标: {main_metric:.4f}")
        for k, v in val_logs.items():
            print(f"   - {k}: {v:.4f}")
        print(f"{'='*50}\n")

    # 销毁分布式环境
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()