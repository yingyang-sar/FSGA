"""Inference checkpoints, retaining the original state-dict and memory keys."""
from pathlib import Path
import torch


def save_checkpoint(model, low_memory, zmag_memory, target_low_memory, epoch, best, cfg):
    Path(cfg.SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "low_centers": low_memory.centers.detach().cpu(),
        "low_valid": low_memory.valid.detach().cpu(),
        "zmag_mean": zmag_memory.mean.detach().cpu(),
        "zmag_mean_sq": zmag_memory.mean_sq.detach().cpu(),
        "zmag_valid": zmag_memory.valid.detach().cpu(),
        "target_low_centers": target_low_memory.centers.detach().cpu(),
        "target_low_valid": target_low_memory.valid.detach().cpu(),
        "epoch": epoch, "best": best, "class_names": cfg.CLASS_NAMES,
        "config": cfg.as_dict(),
        "selection_split": "test", "selection_metric": "acc_fuse",
    }, cfg.SAVE_PATH)


def load_checkpoint(path, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a FSGA checkpoint containing a 'model' state_dict")
    if checkpoint.get("class_names") != cfg.CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from configuration")
    saved = checkpoint.get("config", {})
    # Old checkpoints stored only a subset; validate every relevant key present.
    for key in ("BACKBONE", "FEAT_DIM", "PROJ_HIDDEN", "IMG_SIZE", "WAVELET",
                "LOG_STABILIZE", "LOG_SCALE", "BETA_HIGH_FEAT", "BETA_HIGH_LOGIT",
                "HIGH_INPUT_ATT_RATIO", "HIGH_STRUCT_GATE_RATIO", "S", "M"):
        if key in saved and saved[key] != getattr(cfg, key):
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    return checkpoint
