# dataset/dataloader.py
from torch.utils.data import DataLoader
from ._datasetRegistry import build_dataset
from torch.utils.data.distributed import DistributedSampler

def create_dataloader(dataset_name: str, dataset_cfg: dict, loader_cfg: dict, is_distributed=False):
    """
    统一的数据加载器构建工厂
    
    Args:
        dataset_name (str): 注册表中的 Dataset 名称 (如 "seg_dataset")
        dataset_cfg (dict): 传给 Dataset 初始化函数的参数 (如 root_dir)
        loader_cfg (dict): 传给 DataLoader 的参数 (如 batch_size, shuffle, num_workers)
    """
    # 1. 通过工厂模式造出 Dataset 实例
    dataset = build_dataset(dataset_name, **dataset_cfg)

    sampler = None
    shuffle = loader_cfg.get('shuffle', True)

    if is_distributed:
        # DDP 模式下，必须使用分布式采样器来切分数据
        sampler = DistributedSampler(dataset)
        # 有了 sampler，DataLoader 自己的 shuffle 必须是 False！
        shuffle = False
    
    # 2. 如果 dataset 为空，抛出异常避免后续训练报错
    if len(dataset) == 0:
        raise ValueError(f"数据集 {dataset_name} 为空！请检查 dataset_cfg 路径配置：{dataset_cfg}")
        
    # 3. 将 Dataset 包装成 DataLoader
    return DataLoader(
        dataset,
        batch_size=loader_cfg.get('batch_size', 16),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=loader_cfg.get('num_workers', 4),
        pin_memory=True
    )