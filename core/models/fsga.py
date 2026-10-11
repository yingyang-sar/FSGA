import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


class SectorFace(nn.Module):
    """Hyperspherical angular classifier with optional sample-wise adaptive margin."""

    def __init__(self, feat_dim, *, cfg):
        self.cfg = cfg
        super().__init__()
        self.W = nn.Parameter(torch.randn(self.cfg.NUM_CLASSES, feat_dim))
        nn.init.xavier_uniform_(self.W)
        self.cos_m = math.cos(self.cfg.M)
        self.sin_m = math.sin(self.cfg.M)

    def prototypes(self):
        return F.normalize(self.W, dim=1, eps=1e-06)

    def forward(self, f_dir, label=None, sample_margin=None):
        f_dir = F.normalize(f_dir, dim=1, eps=1e-06)
        W = self.prototypes()
        cos = F.linear(f_dir, W)
        cos = torch.clamp(cos, -0.999, 0.999)
        if label is None:
            return cos * self.cfg.S
        sin = torch.sqrt(torch.clamp(1.0 - cos ** 2, min=1e-06))
        if sample_margin is None:
            cos_theta_m = cos * self.cos_m - sin * self.sin_m
        else:
            m = sample_margin.view(-1, 1).to(f_dir.device).to(f_dir.dtype)
            m = torch.clamp(m, min=0.0, max=0.5)
            cos_theta_m = cos * torch.cos(m) - sin * torch.sin(m)
        onehot = F.one_hot(label, self.cfg.NUM_CLASSES).float().to(f_dir.device)
        logits = torch.where(onehot == 1, cos_theta_m, cos)
        return logits * self.cfg.S


class NormalizedLinearClassifier(nn.Module):
    """
    High-frequency evidence classifier.

    BETA_HIGH_LOGIT is currently 0, so this branch does not affect logits, but it
    is kept to preserve the original training trajectory and checkpoint structure.
    """

    def __init__(self, feat_dim, *, cfg):
        self.cfg = cfg
        super().__init__()
        self.W = nn.Parameter(torch.randn(self.cfg.NUM_CLASSES, feat_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        x = F.normalize(x, dim=1, eps=1e-06)
        W = F.normalize(self.W, dim=1, eps=1e-06)
        return F.linear(x, W) * self.cfg.S


def _build_resnet(backbone: str, weights='IMAGENET1K_V1'):
    """Create a torchvision ResNet without the final FC."""
    name = backbone.lower()
    weight_enum = {'resnet18': models.ResNet18_Weights, 'resnet34': models.ResNet34_Weights, 'resnet50': models.ResNet50_Weights}
    ctor = {'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50}
    if name not in ctor:
        raise ValueError(f'Unsupported backbone: {backbone}')
    w = getattr(weight_enum[name], weights) if weights else None
    return ctor[name](weights=w)


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, in_channels=3, backbone=None, *, cfg):
        self.cfg = cfg
        if backbone is None:
            backbone = self.cfg.BACKBONE
        super().__init__()
        base = _build_resnet(backbone, weights='IMAGENET1K_V1' if self.cfg.PRETRAINED else None)
        old_conv = base.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            elif in_channels == 3:
                new_conv.weight.copy_(old_conv.weight)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        base.conv1 = new_conv
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = self.cfg.FEAT_DIM

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.nan_to_num(x, nan=0.0, posinf=10000.0, neginf=-10000.0)


def slim_mlp(in_dim, out_dim, hidden=None, dropout=0.2):
    """Compact projection used by FSGA-Lite (replaces heavy 2048-d stacks)."""
    h = hidden if hidden is not None else max(128, min(in_dim, out_dim) // 2)
    return nn.Sequential(nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(h, out_dim))


class LearnableStableHighEncoder(nn.Module):
    """Structure-guided learnable high-frequency stabilization encoder (lite)."""

    def __init__(self, feat_dim=None, *, cfg):
        self.cfg = cfg
        if feat_dim is None:
            feat_dim = self.cfg.FEAT_DIM
        super().__init__()
        self.high_branch = ResNetFeatureExtractor(in_channels=3, backbone=self.cfg.BACKBONE, cfg=self.cfg)
        self.subband_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3, 8, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(8, 3, kernel_size=1, bias=True), nn.Sigmoid())
        gate_hidden = max(128, feat_dim // 2)
        self.structure_gate = nn.Sequential(nn.Linear(feat_dim * 2, gate_hidden), nn.LayerNorm(gate_hidden), nn.ReLU(inplace=True), nn.Linear(gate_hidden, feat_dim), nn.Sigmoid())
        self.high_proj = slim_mlp(feat_dim, feat_dim, hidden=self.cfg.PROJ_HIDDEN, dropout=0.2)

    def forward(self, x_high, low_feat):
        band_gate = self.subband_gate(x_high)
        x_high = x_high * (1.0 + self.cfg.HIGH_INPUT_ATT_RATIO * (2.0 * band_gate - 1.0))
        high_raw = self.high_branch(x_high)
        high_raw = torch.clamp(high_raw, -10.0, 10.0)
        gate = self.structure_gate(torch.cat([high_raw, low_feat.detach()], dim=1))
        stable_high_raw = high_raw * (1.0 + self.cfg.HIGH_STRUCT_GATE_RATIO * (2.0 * gate - 1.0))
        stable_high_raw = torch.clamp(stable_high_raw, -10.0, 10.0)
        high_feat = self.high_proj(stable_high_raw)
        high_feat = torch.nan_to_num(high_feat, nan=0.0, posinf=10000.0, neginf=-10000.0)
        high_sem = F.normalize(high_feat, dim=1, eps=1e-06)
        return {'high_raw': high_raw, 'stable_high_raw': stable_high_raw, 'high_feat': high_feat, 'high_sem': high_sem}


class FBSRLearnableHypersphericalModel(nn.Module):
    """FSGA-Lite: same geometry/adaptation logic, lighter ResNet + slim MLPs."""

    def __init__(self, *, cfg):
        self.cfg = cfg
        super().__init__()
        feat_dim = self.cfg.FEAT_DIM
        self.feat_dim = feat_dim
        self.backbone_name = self.cfg.BACKBONE
        self.low_branch = ResNetFeatureExtractor(in_channels=1, backbone=self.cfg.BACKBONE, cfg=self.cfg)
        self.low_proj = slim_mlp(feat_dim, feat_dim, hidden=self.cfg.PROJ_HIDDEN, dropout=0.2)
        self.high_encoder = LearnableStableHighEncoder(feat_dim=feat_dim, cfg=self.cfg)
        self.fusion_proj = slim_mlp(feat_dim * 2, feat_dim, hidden=self.cfg.PROJ_HIDDEN, dropout=0.2)
        self.direction_classifier = SectorFace(feat_dim, cfg=self.cfg)
        self.high_evidence_classifier = NormalizedLinearClassifier(feat_dim, cfg=self.cfg)

    def forward(self, x_low, x_high, label=None):
        feat_low = self.low_branch(x_low)
        feat_low = torch.clamp(feat_low, -10.0, 10.0)
        low_feat = self.low_proj(feat_low)
        low_feat = torch.nan_to_num(low_feat, nan=0.0, posinf=10000.0, neginf=-10000.0)
        high_out = self.high_encoder(x_high=x_high, low_feat=feat_low)
        high_feat = high_out['high_feat']
        high_sem = high_out['high_sem']
        z = self.fusion_proj(torch.cat([low_feat, self.cfg.BETA_HIGH_FEAT * high_feat], dim=1))
        z = torch.nan_to_num(z, nan=0.0, posinf=10000.0, neginf=-10000.0)
        z = torch.clamp(z, -10.0, 10.0)
        z_mag = torch.norm(z, p=2, dim=1, keepdim=True).clamp_min(1e-06) / math.sqrt(self.feat_dim)
        f_dir = F.normalize(z, dim=1, eps=1e-06)
        logits_dir = self.direction_classifier(f_dir, label=label)
        logits_high = self.high_evidence_classifier(high_sem)
        logits_fuse = logits_dir + self.cfg.BETA_HIGH_LOGIT * logits_high
        return {'logits_fuse': logits_fuse, 'logits_dir': logits_dir, 'logits_high': logits_high, 'f_dir': f_dir, 'z_mag': z_mag, 'low_feat': low_feat, 'high_feat': high_feat, 'high_sem': high_sem, 'stable_high_raw': high_out['stable_high_raw']}

# Public alias; original parameter names remain unchanged for checkpoint loading.
FSGA = FBSRLearnableHypersphericalModel
