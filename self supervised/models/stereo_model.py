import torch
import torch.nn as nn

from models.gates import FeatureGate
from models.psmnet_wrapper import GatedPSMNet, get_pred
from models.refiner import RefinementNetCE


class StereoWithCorrection(nn.Module):
    """Stereo model with optional feature gating and disparity refinement."""

    def __init__(self, cfg):
        super().__init__()
        self.use_feature_gate = bool(cfg.use_feature_gate)
        self.stereo = GatedPSMNet(cfg.max_disp, cfg.psmnet_dir)

        if self.use_feature_gate:
            self.feat_gate_left = FeatureGate(3, cfg.feat_channels)
            self.feat_gate_right = FeatureGate(3, cfg.feat_channels)
            refiner_in = cfg.feat_channels * 2 + 1
        else:
            refiner_in = 9

        self.refiner = RefinementNetCE(
            refiner_in,
            n_bins=cfg.n_disp_bins,
            max_disp=cfg.max_disp,
        )

    def _build_refiner_input(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disp_in: torch.Tensor,
        w_left: torch.Tensor,
        w_right: torch.Tensor,
        gate_info: dict,
    ) -> torch.Tensor:
        if self.use_feature_gate:
            feat_l, gate_map_l = self.feat_gate_left(left, w_left)
            feat_r, gate_map_r = self.feat_gate_right(right, w_right)
            gate_info["gate_map_left"] = gate_map_l
            gate_info["gate_map_right"] = gate_map_r
            return torch.cat([feat_l, feat_r, disp_in], dim=1)

        b, _, h, w = left.shape
        l_g = left * w_left
        r_g = right * w_right
        return torch.cat(
            [
                l_g,
                r_g,
                disp_in,
                w_left.expand(b, 1, h, w),
                w_right.expand(b, 1, h, w),
            ],
            dim=1,
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        if left.ndim != 4 or right.ndim != 4:
            raise ValueError(f"Expected NCHW tensors, got {left.shape=} and {right.shape=}")
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes must match, got {left.shape=} and {right.shape=}")

        pred_raw, gate_info = self.stereo(left, right)
        disp_raw = get_pred(pred_raw)
        disp_in = disp_raw.unsqueeze(1)

        w_left = gate_info["w_left"]
        w_right = gate_info["w_right"]

        refine_in = self._build_refiner_input(
            left=left,
            right=right,
            disp_in=disp_in,
            w_left=w_left,
            w_right=w_right,
            gate_info=gate_info,
        )

        logits, corrected = self.refiner(refine_in)
        return {
            "raw": disp_raw,
            "corrected": corrected,
            "logits": logits,
            "gate_info": gate_info,
        }
