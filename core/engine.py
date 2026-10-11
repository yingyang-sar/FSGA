import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from core.utils import freeze_bn2d
from core.loss_funcs.objective import compute_loss


def train_one_epoch(model, low_memory, zmag_memory, target_low_memory, loader_s, loader_t, opt, epoch, *, cfg):
    model.train()
    freeze_bn2d(model)
    pbar = tqdm(zip(loader_s, loader_t), total=min(len(loader_s), len(loader_t)))
    total_loss = 0.0
    total_pseudo = 0
    total_target = 0
    steps = 0
    last_dic = None
    for (s_l, s_h, y), (t_l, t_h, _) in pbar:
        s_l = s_l.to(cfg.DEVICE, non_blocking=True)
        s_h = s_h.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)
        t_l = t_l.to(cfg.DEVICE, non_blocking=True)
        t_h = t_h.to(cfg.DEVICE, non_blocking=True)
        loss, dic = compute_loss(model=model, low_memory=low_memory, zmag_memory=zmag_memory, target_low_memory=target_low_memory, xs_low=s_l, xs_high=s_h, ys=y, xt_low=t_l, xt_high=t_h, epoch=epoch, cfg=cfg)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print('[Warning] non-finite loss detected, skip this batch.')
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.detach().item())
        total_pseudo += dic['pseudo_num']
        total_target += t_l.size(0)
        steps += 1
        last_dic = dic
        pbar.set_postfix({'loss': round(loss.item(), 4), 'pseudo': dic['pseudo_num'], 'conf': dic['conf_thr'], 'aw': round(dic['align_weight'], 2)})
    avg_loss = total_loss / max(steps, 1)
    pseudo_ratio = total_pseudo / max(total_target, 1)
    return (avg_loss, pseudo_ratio, last_dic)


@torch.no_grad()
def test(model, loader, *, cfg):
    model.eval()
    cor_fuse = 0
    cor_dir = 0
    total = 0
    for l, h, y in loader:
        l = l.to(cfg.DEVICE, non_blocking=True)
        h = h.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)
        out = model(l, h, label=None)
        pred_fuse = out['logits_fuse'].argmax(1)
        pred_dir = out['logits_dir'].argmax(1)
        cor_fuse += (pred_fuse == y).sum().item()
        cor_dir += (pred_dir == y).sum().item()
        total += y.size(0)
    return {'acc_fuse': cor_fuse / max(total, 1), 'acc_dir': cor_dir / max(total, 1)}


def print_epoch_log(epoch, train_loss, pseudo_ratio, eval_dic, best, last_dic):
    print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | PseudoRatio: {pseudo_ratio:.4f} | DirAcc: {eval_dic['acc_dir'] * 100:.2f}% | FuseAcc: {eval_dic['acc_fuse'] * 100:.2f}% | Best: {best * 100:.2f}%")
    if last_dic is None:
        return
    print(f"  LossDetail -> Cls={last_dic['cls']:.4f}, Proto={last_dic['proto']:.4f}, TgtScaCls={last_dic['tgt_sca_cls']:.4f}, ZDirAlign={last_dic['zdir_align']:.4f}, LowAlign={last_dic['low_align']:.4f}, TgtCenter={last_dic['tgt_center']:.4f}, HighAlign={last_dic['high_align']:.4f}, ZMagMargin={last_dic['zmag_margin']:.4f}, ScatterCons={last_dic['scatter_consistency']:.4f}, TgtZMagMargin={last_dic['tgt_zmag_margin']:.4f}, TgtScatterCons={last_dic['tgt_scatter_consistency']:.4f}, ZMagS={last_dic['zmag_s']:.4f}, ZMagT={last_dic['zmag_t']:.4f}, Pseudo={last_dic['pseudo_num']}, ConfPass={last_dic['conf_pass_num']}, ConfThr={last_dic['conf_thr']:.2f}, ConfFuse={last_dic['conf_fuse']:.4f}, PseudoConf={last_dic['pseudo_conf']:.4f}, AlignW={last_dic['align_weight']:.2f}")
