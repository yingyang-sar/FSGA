import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pywt
from PIL import Image
from torch.utils.data import Dataset


def log_stabilize_image(x, *, cfg):
    """Log-domain compression for SAR intensity images."""
    x = x.astype(np.float32)
    if not cfg.LOG_STABILIZE:
        return x
    return np.log1p(cfg.LOG_SCALE * x) / np.log1p(cfg.LOG_SCALE)


def normalize_low_band(x):
    """Min-max normalization for the LL band."""
    x = x.astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-06:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min + 1e-06)


def normalize_high_bands_simple(LH, HL, HH):
    """
    Standardize high-frequency coefficients for numerical stability.

    No handcrafted quantile threshold, no quantile-guided gate, and no
    percentile clipping are used. High-frequency stabilization is learned by
    the network.
    """
    high = np.stack([LH.astype(np.float32), HL.astype(np.float32), HH.astype(np.float32)], axis=0)
    mean = high.mean()
    std = high.std() + 1e-06
    high = (high - mean) / std
    high = np.clip(high, -3.0, 3.0)
    high = (high + 3.0) / 6.0
    return high.astype(np.float32)


def resize_np_to_tensor(x, size=224):
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze(0)


def wavelet_decompose_dual(img: Image.Image, *, cfg):
    """Return low-frequency and high-frequency inputs after DWT."""
    img = img.convert('L').resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = log_stabilize_image(img_np, cfg=cfg)
    LL, (LH, HL, HH) = pywt.dwt2(img_np, cfg.WAVELET)
    LL = normalize_low_band(LL)
    high = normalize_high_bands_simple(LH, HL, HH)
    LL = resize_np_to_tensor(LL, cfg.IMG_SIZE)
    LH = resize_np_to_tensor(high[0], cfg.IMG_SIZE)
    HL = resize_np_to_tensor(high[1], cfg.IMG_SIZE)
    HH = resize_np_to_tensor(high[2], cfg.IMG_SIZE)
    low_img = (LL - 0.5) / 0.5
    high_img = torch.cat([LH, HL, HH], dim=0)
    high_img = (high_img - 0.5) / 0.5
    low_img = torch.nan_to_num(low_img, nan=0.0, posinf=1.0, neginf=-1.0)
    high_img = torch.nan_to_num(high_img, nan=0.0, posinf=1.0, neginf=-1.0)
    return (low_img, high_img)


class SARDataset(Dataset):

    def __init__(self, mode, *, cfg):
        self.cfg = cfg
        self.root = os.path.join(self.cfg.ROOT, mode)
        self.images = []
        self.labels = []
        for cls in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, cls)
            if not os.path.isdir(d) or cls not in self.cfg.CLASS_TO_IDX:
                continue
            lab = self.cfg.CLASS_TO_IDX[cls]
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp')):
                    self.images.append(os.path.join(d, f))
                    self.labels.append(lab)
        print(f'[Dataset] {mode}: {len(self.images)} images')
        if len(self.images) == 0:
            raise RuntimeError(f'No images found in {self.root}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as img:
            low_img, high_img = wavelet_decompose_dual(img, cfg=self.cfg)
        return (low_img, high_img, self.labels[idx])
