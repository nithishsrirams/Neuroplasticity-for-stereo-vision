import torch
import torch.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels >= 8 and channels % 8 == 0 else 4 if channels % 4 == 0 else 1
    return nn.GroupNorm(groups, channels)


class BranchQualityGate(nn.Module):
    """
    Predicts a scalar confidence score for a stereo branch from low-level image cues.
    The score is used as a global quality prior, not a per-pixel mask.
    """
    def __init__(self, in_channels: int = 3, hidden: int = 16):
        super().__init__()
        mid_channels = hidden * 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            _group_norm(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mid_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
        feat = self.stem(x)
        pooled = self.pool(feat)
        score = self.score_head(pooled)
        return score

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class FeatureGate(nn.Module):
    """
    Produces gated feature maps using a learned confidence estimate combined with
    a global branch weight.
    """
    def __init__(self, in_channels: int = 3, feat_channels: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(feat_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.confidence = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, global_weight: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")

        feat = self.backbone(x)
        local_conf = self.activation(self.confidence(feat))

        if global_weight.ndim == 0:
            global_weight = global_weight.view(1, 1, 1, 1)
        elif global_weight.ndim == 1:
            global_weight = global_weight[:, None, None, None]

        gate_map = local_conf * global_weight
        gated_feat = feat * gate_map
        return gated_feat, gate_map

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
