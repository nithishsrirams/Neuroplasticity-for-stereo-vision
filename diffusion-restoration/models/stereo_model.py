import torch
import torch.nn as nn

from models.gates import FeatureGate
from models.psmnet_wrapper import GatedPSMNet, get_pred
from models.refiner import RefinementNetCE, SimpleEnhancer


def unpack_batch_meta(meta_batch):
    if meta_batch is None:
        return []
    if isinstance(meta_batch, list):
        return meta_batch
    batch_size = len(meta_batch["degraded"])
    return [
        {
            "degraded": bool(meta_batch["degraded"][i]),
            "type":     meta_batch["type"][i],
            "level":    int(meta_batch["level"][i]),
            "side":     meta_batch["side"][i],
        }
        for i in range(batch_size)
    ]


class StereoWithCorrection(nn.Module):
    def __init__(self, maxdisp, use_feature_gate, feat_channels, n_bins):
        super().__init__()
        self.use_feature_gate = use_feature_gate
        self.stereo = GatedPSMNet(maxdisp)
        self.enhancer = SimpleEnhancer()

        if use_feature_gate:
            self.feat_gate_left = FeatureGate(3, feat_channels)
            self.feat_gate_right = FeatureGate(3, feat_channels)
            refiner_in = feat_channels * 2 + 1
        else:
            refiner_in = 9

        self.refiner = RefineNet(refiner_in, hidden=64, n_bins=n_bins, max_disp=maxdisp)

    def enhance(self, img, meta):
        if meta is None:
            return img
        meta_list = unpack_batch_meta(meta)
        mask = torch.tensor([m.get("degraded", False) for m in meta_list], device=img.device).bool()
        if mask.any():
            enhanced = img.clone()
            idx = mask.nonzero(as_tuple=True)[0]
            enhanced[idx] = self.enhancer(img[idx])
            return enhanced
        return img

    def forward(self, left, right, meta=None):
        left = self.enhance(left, meta)
        right = self.enhance(right, meta)

        pred_raw, gate_info = self.stereo(left, right)
        disp_raw = get_pred(pred_raw)
        disp_in = disp_raw.unsqueeze(1)

        w_left = gate_info["w_left"]
        w_right = gate_info["w_right"]

        if self.use_feature_gate:
            gated_feat_L, gate_map_L = self.feat_gate_left(left, w_left)
            gated_feat_R, gate_map_R = self.feat_gate_right(right, w_right)
            refine_in = torch.cat([gated_feat_L, gated_feat_R, disp_in], dim=1)
            gate_info["gate_map_left"] = gate_map_L
            gate_info["gate_map_right"] = gate_map_R
        else:
            B = left.size(0)
            L_gated = left * w_left
            R_gated = right * w_right
            w_L_map = w_left.expand(B, 1, left.size(2), left.size(3))
            w_R_map = w_right.expand(B, 1, left.size(2), left.size(3))
            refine_in = torch.cat([L_gated, R_gated, disp_in, w_L_map, w_R_map], dim=1)

        logits, disp_corrected = self.refiner(refine_in)

        return {"raw": disp_raw, "corrected": disp_corrected, "logits": logits, "gate_info": gate_info}
