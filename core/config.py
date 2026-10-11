"""Explicit settings; training resolves the automatic seed at startup."""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import yaml

@dataclass
class Settings:
    ROOT: str = 'datasets2_sample_qpm/scenario_I'
    CLASS_NAMES: list = field(default_factory=lambda: ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23'])
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LR: float = 0.0001
    WEIGHT_DECAY: float = 1e-05
    IMG_SIZE: int = 128
    WAVELET: str = 'haar'
    BACKBONE: str = 'resnet18'
    LOG_STABILIZE: bool = True
    LOG_SCALE: float = 10.0
    BETA_HIGH_FEAT: float = 0.5
    BETA_HIGH_LOGIT: int = 0
    HIGH_INPUT_ATT_RATIO: float = 0.3
    HIGH_STRUCT_GATE_RATIO: float = 0.5
    S: int = 10
    M: float = 0.3
    USE_SCATTER_CONSISTENCY_MARGIN: bool = True
    M_BASE: float = 0.3
    M_MAX: float = 0.45
    ZMAG_MEMORY_MOMENTUM: float = 0.9
    ZMAG_SIGMA_SCALE: float = 2.0
    ZMAG_STD_MIN: float = 0.02
    PSEUDO_CONF_START: float = 0.85
    PSEUDO_CONF_END: float = 0.95
    ALIGN_WARMUP_EPOCHS: int = 5
    ALIGN_RAMP_EPOCHS: int = 15
    LAMBDA_PROTO: float = 0.2
    LAMBDA_TGT_SCA_CLS: float = 0.05
    LAMBDA_ZDIR_ALIGN: float = 0.1
    LAMBDA_LOW_ALIGN: float = 0.15
    LAMBDA_TGT_CENTER: float = 0.0
    LAMBDA_HIGH_ALIGN: float = 0.15
    SWD_NUM_PROJECTIONS: int = 32
    SWD_P: int = 2
    SWD_NORMALIZE_FEATURE: bool = True
    DEVICE: str = 'cpu'
    SEED: int = 0  # Automatic seed for training; resolved value is saved per run.
    NUM_WORKERS: int = 4
    PRETRAINED: bool = True
    SAVE_PATH: str = 'outputs/scenario_I/best.pth'

    @property
    def NUM_CLASSES(self):
        return len(self.CLASS_NAMES)

    @property
    def CLASS_TO_IDX(self):
        return {name: idx for idx, name in enumerate(self.CLASS_NAMES)}

    @property
    def FEAT_DIM(self):
        return {"resnet18": 512, "resnet34": 512, "resnet50": 2048}[self.BACKBONE]

    @property
    def PROJ_HIDDEN(self):
        return max(128, self.FEAT_DIM // 4)

    def as_dict(self):
        return asdict(self)

    def validate(self):
        if type(self.SEED) is not int or not 0 <= self.SEED < 2**32:
            raise ValueError("SEED must be an integer in [0, 2**32 - 1]; 0 means automatic training seed")
        if self.BACKBONE not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError("Unsupported BACKBONE")
        if self.BATCH_SIZE < 2 or self.EPOCHS < 1 or self.IMG_SIZE < 32:
            raise ValueError("Require BATCH_SIZE >= 2, EPOCHS >= 1, IMG_SIZE >= 32")
        if self.NUM_WORKERS < 0 or self.LR <= 0 or self.WEIGHT_DECAY < 0:
            raise ValueError("Invalid worker count or optimizer settings")
        if not 0 <= self.PSEUDO_CONF_START <= self.PSEUDO_CONF_END <= 1:
            raise ValueError("Pseudo-label confidence thresholds must increase within [0, 1]")
        if not 0 <= self.M_BASE <= self.M_MAX <= 0.5:
            raise ValueError("Require 0 <= M_BASE <= M_MAX <= 0.5")
        if self.SWD_NUM_PROJECTIONS < 1 or self.SWD_P not in (1, 2):
            raise ValueError("Invalid SWD settings")
        if self.CLASS_NAMES != Settings().CLASS_NAMES:
            raise ValueError("Preserve the original SAMPLE class order")
        return self


def load_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed = {f.name for f in fields(Settings)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Settings(**data).validate()
