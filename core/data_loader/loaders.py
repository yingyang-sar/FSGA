from torch.utils.data import DataLoader
from .dataset import SARDataset


def build_loaders(cfg):
    datasets = [SARDataset(mode, cfg=cfg) for mode in ("simulate", "train", "test")]
    if any(len(dataset) < cfg.BATCH_SIZE for dataset in datasets[:2]):
        raise ValueError("Both training splits must contain at least BATCH_SIZE images")
    loaders = []
    for i, dataset in enumerate(datasets):
        loaders.append(DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                                  shuffle=i < 2, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=True, drop_last=i < 2))
    return tuple(loaders)
