import torch
import torch.nn as nn


class BranchQualityGate(nn.Module):
    def __init__(self, in_channels=3, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class FeatureGate(nn.Module):
    def __init__(self, in_channels=3, feat_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, feat_channels),
            nn.ReLU(inplace=True),
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(feat_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, global_weight):
        feat = self.encoder(x)
        spa_conf = self.conf_head(feat)
        gate_map = spa_conf * global_weight
        return feat * gate_map, gate_map
