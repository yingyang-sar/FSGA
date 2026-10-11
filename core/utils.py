import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def resolve_seed(seed):
    """Resolve the automatic sentinel once, before seeding any random generator."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32 - 1]")
    return seed if seed != 0 else random.randint(1, 10000000)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def linear_schedule(start, end, epoch, total_epochs=None, warmup=None, *, cfg):
    if total_epochs is None:
        total_epochs = cfg.EPOCHS
    if warmup is None:
        warmup = cfg.ALIGN_WARMUP_EPOCHS
    if epoch <= warmup:
        return start
    ratio = float(epoch - warmup) / max(total_epochs - warmup, 1)
    ratio = min(max(ratio, 0.0), 1.0)
    return start + (end - start) * ratio


def align_ramp_weight(epoch, *, cfg):
    if epoch <= cfg.ALIGN_WARMUP_EPOCHS:
        return 0.0
    return min(1.0, float(epoch - cfg.ALIGN_WARMUP_EPOCHS) / max(cfg.ALIGN_RAMP_EPOCHS, 1))


def normalize_loss_weight(w, min_w=0.2, max_w=2.0):
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).detach()
    if w.numel() == 0:
        return w
    w = w / (w.mean() + 1e-06)
    return torch.clamp(w, min_w, max_w)


def freeze_bn2d(model):
    """Fix CNN BatchNorm2d statistics for small-batch SAR training."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
