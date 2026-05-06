import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn

from models.gates import BranchQualityGate


def get_pred(output):
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if isinstance(output, dict):
        output = output.get("corrected", output.get("raw", next(iter(output.values()))))
    if output.dim() == 4 and output.size(1) == 1:
        output = output[:, 0]
    return output


class GatedPSMNet(nn.Module):
    def __init__(self, maxdisp, psmnet_dir):
        super().__init__()
        models_dir = os.path.join(psmnet_dir, "models")
        mod_path = os.path.join(models_dir, "stackhourglass.py")
        if not os.path.exists(mod_path):
            raise FileNotFoundError(f"PSMNet file not found: {mod_path}")
        pkg_name = "psmnet_models"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [models_dir]
            sys.modules[pkg_name] = pkg

        spec = importlib.util.spec_from_file_location(f"{pkg_name}.stackhourglass", mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from: {mod_path}")
        psm_mod = importlib.util.module_from_spec(spec)
        psm_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.stackhourglass"] = psm_mod
        spec.loader.exec_module(psm_mod)
        PSMNet = psm_mod.PSMNet

        self.backbone = PSMNet(maxdisp)
        self.gate = BranchQualityGate()

    def forward(self, left, right):
        q_left = self.gate(left)
        q_right = self.gate(right)
        weights = torch.softmax(torch.cat([q_left, q_right], dim=1), dim=1)
        w_left = weights[:, 0].view(-1, 1, 1, 1)
        w_right = weights[:, 1].view(-1, 1, 1, 1)
        pred = self.backbone(left, right)
        return pred, {"w_left": w_left, "w_right": w_right, "q_left": q_left, "q_right": q_right}
