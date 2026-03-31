# dataset/dataloader.py
from torch.utils.data import DataLoader
from ._datasetRegistry import build_dataset
from torch.utils.data.distributed import DistributedSampler

__all__ = ['create_dataloader']

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

    if len(dataset) == 0:
        raise ValueError(f"数据集 {dataset_name} 为空！请检查配置：{dataset_cfg}")

    dl_kwargs = loader_cfg.copy()
    # 设置合理的默认值 (如果 YAML 里没写，就用这些兜底)
    dl_kwargs.setdefault('batch_size', 16)
    dl_kwargs.setdefault('num_workers', 8)
    dl_kwargs.setdefault('pin_memory', True)

    sampler = None

    if is_distributed:
        # DDP 模式下，必须使用分布式采样器来切分数据
        sampler = DistributedSampler(dataset)
        dl_kwargs['sampler'] = sampler
        # 有了 sampler，DataLoader 自己的 shuffle 必须是 False！
        dl_kwargs['shuffle'] = False
    else:
        # 单卡模式下，如果 YAML 没写 shuffle，默认给 True
        dl_kwargs.setdefault('shuffle', True)
        
    # 3. 将 Dataset 包装成 DataLoader
    return DataLoader(
        dataset,
        **dl_kwargs
    )