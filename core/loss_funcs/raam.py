import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import linear_schedule


@torch.no_grad()
def build_confidence_fused_pseudo_labels(out_t, epoch, *, cfg):
    """
    Generate target pseudo labels by fused prediction and confidence filtering.

    The final fused logits are used to predict pseudo labels, and only samples
    with sufficiently high confidence are selected for target-domain adaptation.
    """
    logits_fuse = out_t['logits_fuse']
    prob_fuse = F.softmax(logits_fuse, dim=1)
    conf_fuse, pred_fuse = prob_fuse.max(dim=1)
    pseudo_label = pred_fuse
    pseudo_conf = conf_fuse
    conf_thr = linear_schedule(cfg.PSEUDO_CONF_START, cfg.PSEUDO_CONF_END, epoch, cfg=cfg)
    conf_mask = pseudo_conf > conf_thr
    if epoch <= cfg.ALIGN_WARMUP_EPOCHS:
        mask = torch.zeros_like(conf_mask, dtype=torch.bool)
    else:
        mask = conf_mask
    pseudo_weight = pseudo_conf.detach()
    return {'pseudo': pseudo_label.detach(), 'mask': mask.detach(), 'weight': pseudo_weight.detach(), 'conf_fuse': conf_fuse.detach(), 'pseudo_conf': pseudo_conf.detach(), 'conf_pass_num': int(conf_mask.sum().item()), 'conf_thr': conf_thr}


def build_scatter_consistency_margin(zmag_memory, z_mag, labels, *, cfg):
    """Build sample-wise angular margins from Z-radius reliability consistency."""
    if not cfg.USE_SCATTER_CONSISTENCY_MARGIN or z_mag.numel() == 0:
        margin = torch.full_like(z_mag.view(-1), float(cfg.M), device=z_mag.device)
        consistency = torch.ones_like(margin)
        return (margin.detach(), consistency.detach())
    mean, std = zmag_memory.get_mean_std()
    a = z_mag.detach().view(-1)
    y = labels.detach()
    mu = mean[y]
    sigma = torch.clamp(std[y] * cfg.ZMAG_SIGMA_SCALE, min=cfg.ZMAG_STD_MIN)
    valid = zmag_memory.valid_mask_for_labels(y)
    consistency = torch.exp(-0.5 * ((a - mu) / sigma) ** 2)
    consistency = torch.where(valid, consistency, torch.zeros_like(consistency))
    consistency = torch.nan_to_num(consistency, nan=0.0, posinf=1.0, neginf=0.0)
    consistency = torch.clamp(consistency, 0.0, 1.0)
    margin = cfg.M_BASE + (cfg.M_MAX - cfg.M_BASE) * consistency
    margin = torch.nan_to_num(margin, nan=cfg.M_BASE, posinf=cfg.M_MAX, neginf=cfg.M_BASE)
    margin = torch.clamp(margin, min=cfg.M_BASE, max=cfg.M_MAX)
    return (margin.detach(), consistency.detach())
