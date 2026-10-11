import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCenterMemory:
    """Class-wise feature center memory for low-frequency structure alignment."""

    def __init__(self, num_classes, feat_dim, momentum=0.9, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.device = device
        self.centers = torch.zeros(num_classes, feat_dim, device=device)
        self.valid = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, feat, labels):
        feat = F.normalize(feat.detach(), dim=1, eps=1e-06)
        labels = labels.detach()
        for c in range(self.num_classes):
            idx = labels == c
            if idx.sum() == 0:
                continue
            mean_f = F.normalize(feat[idx].mean(dim=0), dim=0, eps=1e-06)
            if not self.valid[c]:
                self.centers[c] = mean_f
                self.valid[c] = True
            else:
                self.centers[c] = self.momentum * self.centers[c] + (1.0 - self.momentum) * mean_f
                self.centers[c] = F.normalize(self.centers[c], dim=0, eps=1e-06)

    def get(self):
        return F.normalize(self.centers, dim=1, eps=1e-06)

    def valid_mask_for_labels(self, labels):
        return self.valid[labels]


class ZMagnitudeMemory:
    """
    Source-domain class-wise Z-radius statistics.

    The Z radius is ||z|| / sqrt(D). It is not interpreted as calibrated
    physical scattering strength. Instead, it is treated as a learned proxy of
    class-conditioned representation reliability. The source-domain class-wise
    radius statistics provide the reference distribution for adaptive margins.
    """

    def __init__(self, num_classes, momentum=0.9, device='cuda', *, cfg):
        self.cfg = cfg
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device
        self.mean = torch.zeros(num_classes, device=device)
        self.mean_sq = torch.zeros(num_classes, device=device)
        self.valid = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, z_mag, labels):
        z_mag = z_mag.detach().view(-1)
        labels = labels.detach()
        for c in range(self.num_classes):
            idx = labels == c
            if idx.sum() == 0:
                continue
            vals = z_mag[idx]
            mean = vals.mean()
            mean_sq = (vals ** 2).mean()
            if not self.valid[c]:
                self.mean[c] = mean
                self.mean_sq[c] = mean_sq
                self.valid[c] = True
            else:
                self.mean[c] = self.momentum * self.mean[c] + (1.0 - self.momentum) * mean
                self.mean_sq[c] = self.momentum * self.mean_sq[c] + (1.0 - self.momentum) * mean_sq

    def get_mean_std(self):
        var = torch.clamp(self.mean_sq - self.mean ** 2, min=self.cfg.ZMAG_STD_MIN ** 2)
        return (self.mean, torch.sqrt(var))

    def valid_mask_for_labels(self, labels):
        return self.valid[labels]
