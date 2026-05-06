from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "D:/Spring 2026/CV"
    psmnet_dir: str = "/content/PSMNet"
    crop_h: int = 256
    crop_w: int = 512
    batch_size: int = 1
    epochs: int = 50
    lr: float = 1e-4
    max_disp: int = 192
    n_disp_bins: int = 64
    ce_sigma: float = 3.0
    use_feature_gate: bool = True
    feat_channels: int = 32
    degrade_prob_train: float = 0.7
    degrade_prob_eval: float = 1.0
    degrade_base_seed: int = 12345
    degrade_camera: str = "left"
    degrade_type: str = "noise"
    degrade_severity: int = 2
    raw_sup_weight: float = 0.25
    self_sup_weight: float = 0.50
    gate_loss_weight: float = 0.1
