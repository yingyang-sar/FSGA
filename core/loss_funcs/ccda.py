import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import normalize_loss_weight


def source_direction_proto_loss(model, f_dir, labels, *, cfg):
    """Source-domain class mean direction to classifier prototype constraint."""
    proto = model.direction_classifier.prototypes()
    loss = torch.tensor(0.0, device=f_dir.device)
    cnt = 0
    for c in range(cfg.NUM_CLASSES):
        idx = labels == c
        if idx.any():
            mean_dir = F.normalize(f_dir[idx].mean(dim=0), dim=0, eps=1e-06)
            loss = loss + 1.0 - F.cosine_similarity(mean_dir, proto[c], dim=0)
            cnt += 1
    if cnt > 0:
        loss = loss / cnt
    return loss


def weighted_pseudo_cross_entropy(logits, pseudo, mask, weight):
    """Weighted target pseudo-label CE on reliable target samples only."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    ce = F.cross_entropy(logits[mask], pseudo[mask], reduction='none')
    w = weight[mask].detach()
    return (ce * w).sum() / (w.sum() + 1e-06)


def target_z_direction_align_loss(model, f_dir_t, pseudo, mask, weight):
    """Align reliable target z-directions to source classifier prototypes."""
    if mask.sum() == 0:
        return f_dir_t.new_tensor(0.0)
    proto = model.direction_classifier.prototypes()
    d_t = F.normalize(f_dir_t[mask], dim=1, eps=1e-06)
    y = pseudo[mask]
    p = proto[y].detach()
    w = normalize_loss_weight(weight[mask])
    loss_vec = 1.0 - (d_t * p).sum(dim=1)
    loss_vec = torch.nan_to_num(loss_vec, nan=0.0, posinf=2.0, neginf=0.0)
    return (loss_vec * w).mean()


def target_low_frequency_align_loss(low_memory, low_feat_t, pseudo, mask, weight):
    """Align reliable target low-frequency features to source low-frequency centers."""
    if mask.sum() == 0:
        return low_feat_t.new_tensor(0.0)
    final_mask = mask & low_memory.valid_mask_for_labels(pseudo)
    if final_mask.sum() == 0:
        return low_feat_t.new_tensor(0.0)
    centers = low_memory.get()
    feat_t = F.normalize(low_feat_t[final_mask], dim=1, eps=1e-06)
    y = pseudo[final_mask]
    center = centers[y].detach()
    w = normalize_loss_weight(weight[final_mask])
    loss_vec = 1.0 - (feat_t * center).sum(dim=1)
    loss_vec = torch.nan_to_num(loss_vec, nan=0.0, posinf=2.0, neginf=0.0)
    return (loss_vec * w).mean()


def target_low_frequency_center_compact_loss(target_low_memory, low_feat_t, pseudo, mask, weight):
    """Optional target-domain low-frequency center compactness loss."""
    if mask.sum() == 0:
        return low_feat_t.new_tensor(0.0)
    final_mask = mask & target_low_memory.valid_mask_for_labels(pseudo)
    if final_mask.sum() == 0:
        return low_feat_t.new_tensor(0.0)
    centers = target_low_memory.get()
    feat_t = F.normalize(low_feat_t[final_mask], dim=1, eps=1e-06)
    y = pseudo[final_mask]
    center = centers[y].detach()
    w = normalize_loss_weight(weight[final_mask])
    loss_vec = 1.0 - (feat_t * center).sum(dim=1)
    loss_vec = torch.nan_to_num(loss_vec, nan=0.0, posinf=2.0, neginf=0.0)
    return (loss_vec * w).mean()


def sliced_wasserstein_distance(source, target, *, cfg):
    """SWD between high-dimensional features via random one-dimensional projections."""
    if source.size(0) == 0 or target.size(0) == 0:
        return source.new_tensor(0.0)
    source = torch.nan_to_num(source, nan=0.0, posinf=10000.0, neginf=-10000.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=10000.0, neginf=-10000.0)
    if cfg.SWD_NORMALIZE_FEATURE:
        source = F.normalize(source, dim=1, eps=1e-06)
        target = F.normalize(target, dim=1, eps=1e-06)
    n = min(source.size(0), target.size(0))
    if n == 0:
        return source.new_tensor(0.0)
    source = source[:n]
    target = target[:n]
    feat_dim = source.size(1)
    projections = torch.randn(feat_dim, cfg.SWD_NUM_PROJECTIONS, device=source.device, dtype=source.dtype)
    projections = F.normalize(projections, dim=0, eps=1e-06)
    source_proj, _ = torch.sort(source @ projections, dim=0)
    target_proj, _ = torch.sort(target @ projections, dim=0)
    if cfg.SWD_P == 1:
        loss = torch.mean(torch.abs(source_proj - target_proj))
    else:
        loss = torch.mean((source_proj - target_proj) ** 2)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def high_frequency_feature_swd_align_loss(high_feat_s, high_feat_t, ys, pseudo, mask, *, cfg):
    """Class-conditional high-frequency feature distribution alignment with SWD."""
    if mask.sum() == 0:
        return high_feat_s.new_tensor(0.0)
    loss = high_feat_s.new_tensor(0.0)
    cnt = 0
    for c in range(cfg.NUM_CLASSES):
        s_idx = ys == c
        t_idx = (pseudo == c) & mask
        if s_idx.any() and t_idx.any():
            loss = loss + sliced_wasserstein_distance(high_feat_s[s_idx], high_feat_t[t_idx], cfg=cfg)
            cnt += 1
    if cnt > 0:
        loss = loss / cnt
    return loss
