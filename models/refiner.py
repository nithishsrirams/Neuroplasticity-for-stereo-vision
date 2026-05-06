import torch
import torch.nn as nn


class RefinementNetCE(nn.Module):
    def __init__(self, in_channels, hidden=64, n_bins=64, max_disp=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_bins, 3, padding=1),
        )
        self.register_buffer("bin_centers", torch.linspace(0.0, max_disp - 1.0, n_bins))

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        disp = (probs * self.bin_centers.view(1, -1, 1, 1)).sum(1)
        return logits, disp
