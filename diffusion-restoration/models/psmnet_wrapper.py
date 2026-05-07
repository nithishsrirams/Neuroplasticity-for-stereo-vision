import importlib.util
import os
import subprocess
import sys

import torch
import torch.nn as nn

from configs.default import PSMNET_DIR
from models.gates import BranchQualityGate


def ensure_psmnet():
    if not os.path.exists(PSMNET_DIR):
        subprocess.run(
            ["git", "clone", "https://github.com/JiaRenChang/PSMNet.git", PSMNET_DIR],
            check=True,
        )
    if PSMNET_DIR not in sys.path:
        sys.path.append(PSMNET_DIR)


def get_pred(output):
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if isinstance(output, dict):
        output = output.get("corrected", output.get("raw", next(iter(output.values()))))
    if output.dim() == 4 and output.size(1) == 1:
        output = output[:, 0]
    return output


class GatedPSMNet(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        ensure_psmnet()
        stackhourglass_path = os.path.join(PSMNET_DIR, "models", "stackhourglass.py")
        spec = importlib.util.spec_from_file_location("psmnet_stackhourglass", stackhourglass_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load PSMNet stackhourglass from {stackhourglass_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PSMNet = module.PSMNet
        self.backbone = PSMNet(maxdisp)
        self.gate = BranchQualityGate()

    def forward(self, left, right):
        q_left = self.gate(left)
        q_right = self.gate(right)
        weights = torch.softmax(torch.cat([q_left, q_right], dim=1), dim=1)
        w_left = weights[:, 0].view(-1, 1, 1, 1)
        w_right = weights[:, 1].view(-1, 1, 1, 1)
        pred = self.backbone(left, right)
        return pred, {"q_left": q_left, "q_right": q_right, "w_left": w_left, "w_right": w_right}
