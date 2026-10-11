import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import align_ramp_weight
from core.loss_funcs.raam import build_confidence_fused_pseudo_labels, build_scatter_consistency_margin
from core.loss_funcs.ccda import (source_direction_proto_loss, weighted_pseudo_cross_entropy, target_z_direction_align_loss, target_low_frequency_align_loss, target_low_frequency_center_compact_loss, high_frequency_feature_swd_align_loss)


def compute_loss(model, low_memory, zmag_memory, target_low_memory, xs_low, xs_high, ys, xt_low, xt_high, epoch, *, cfg):
    out_s = model(xs_low, xs_high, label=None)
    out_t = model(xt_low, xt_high, label=None)
    fd_s = out_s['f_dir']
    fd_t = out_t['f_dir']
    logits_high_s = out_s['logits_high']
    logits_high_t = out_t['logits_high']
    low_feat_s = out_s['low_feat']
    low_feat_t = out_t['low_feat']
    high_feat_s = out_s['high_feat']
    high_feat_t = out_t['high_feat']
    z_mag_s = out_s['z_mag']
    z_mag_t = out_t['z_mag']
    low_memory.update(low_feat_s, ys)
    zmag_memory.update(z_mag_s, ys)
    sample_margin_s, scatter_consistency_s = build_scatter_consistency_margin(zmag_memory, z_mag_s, ys, cfg=cfg)
    logits_dir_s = model.direction_classifier(fd_s, label=ys, sample_margin=sample_margin_s)
    logits_s = logits_dir_s + cfg.BETA_HIGH_LOGIT * logits_high_s
    loss_cls = F.cross_entropy(logits_s, ys)
    loss_proto = source_direction_proto_loss(model=model, f_dir=fd_s, labels=ys, cfg=cfg)
    pseudo_info = build_confidence_fused_pseudo_labels(out_t=out_t, epoch=epoch, cfg=cfg)
    pseudo = pseudo_info['pseudo']
    mask = pseudo_info['mask']
    weight = pseudo_info['weight']
    sample_margin_t, scatter_consistency_t = build_scatter_consistency_margin(zmag_memory, z_mag_t, pseudo, cfg=cfg)
    logits_dir_t = model.direction_classifier(fd_t, label=pseudo, sample_margin=sample_margin_t)
    logits_t = logits_dir_t + cfg.BETA_HIGH_LOGIT * logits_high_t
    loss_tgt_sca_cls = weighted_pseudo_cross_entropy(logits=logits_t, pseudo=pseudo, mask=mask, weight=weight)
    if mask.sum() > 0:
        target_low_memory.update(low_feat_t[mask], pseudo[mask])
    loss_zdir_align = target_z_direction_align_loss(model, fd_t, pseudo, mask, weight)
    loss_low_align = target_low_frequency_align_loss(low_memory, low_feat_t, pseudo, mask, weight)
    loss_tgt_center = target_low_frequency_center_compact_loss(target_low_memory, low_feat_t, pseudo, mask, weight)
    loss_high_align = high_frequency_feature_swd_align_loss(high_feat_s, high_feat_t, ys, pseudo, mask, cfg=cfg)
    aw = align_ramp_weight(epoch, cfg=cfg)
    total_loss = loss_cls + cfg.LAMBDA_PROTO * loss_proto + aw * (cfg.LAMBDA_TGT_SCA_CLS * loss_tgt_sca_cls + cfg.LAMBDA_ZDIR_ALIGN * loss_zdir_align + cfg.LAMBDA_LOW_ALIGN * loss_low_align + cfg.LAMBDA_TGT_CENTER * loss_tgt_center + cfg.LAMBDA_HIGH_ALIGN * loss_high_align)
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
    loss_dict = {'cls': float(loss_cls.detach().item()), 'proto': float(loss_proto.detach().item()), 'tgt_sca_cls': float(loss_tgt_sca_cls.detach().item()), 'zdir_align': float(loss_zdir_align.detach().item()), 'low_align': float(loss_low_align.detach().item()), 'tgt_center': float(loss_tgt_center.detach().item()), 'high_align': float(loss_high_align.detach().item()), 'zmag_margin': float(sample_margin_s.detach().mean().item()), 'scatter_consistency': float(scatter_consistency_s.detach().mean().item()), 'tgt_zmag_margin': float(sample_margin_t.detach().mean().item()), 'tgt_scatter_consistency': float(scatter_consistency_t.detach().mean().item()), 'zmag_s': float(z_mag_s.detach().mean().item()), 'zmag_t': float(z_mag_t.detach().mean().item()), 'pseudo_num': int(mask.sum().item()), 'conf_pass_num': pseudo_info['conf_pass_num'], 'conf_thr': round(pseudo_info['conf_thr'], 2), 'conf_fuse': float(pseudo_info['conf_fuse'].mean().item()), 'pseudo_conf': float(pseudo_info['pseudo_conf'].mean().item()), 'align_weight': float(aw)}
    return (total_loss, loss_dict)
