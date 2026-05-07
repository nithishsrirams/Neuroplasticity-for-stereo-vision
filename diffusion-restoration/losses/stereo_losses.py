import torch
import torch.nn.functional as F


def disparity_to_soft_target(disp, bin_centers, sigma):
    diff = (disp.unsqueeze(1) - bin_centers.view(1, -1, 1, 1)) ** 2 / (2.0 * sigma**2)
    return torch.softmax(-diff, dim=1)


def ce_self_supervised_loss(deg_logits, clean_disp, bin_centers, valid_mask, sigma):
    soft_target = disparity_to_soft_target(clean_disp.detach(), bin_centers, sigma)
    ce_map = -(soft_target * F.log_softmax(deg_logits, dim=1)).sum(dim=1)
    return ce_map[valid_mask].mean() if valid_mask.sum() > 0 else ce_map.mean()


def gate_reg_loss(gate_info, meta):
    if gate_info is None or meta is None:
        return torch.tensor(0.0, device=device)

    meta_list = unpack_batch_meta(meta)
    if not meta_list:
        return torch.tensor(0.0, device=device)

    wl = gate_info["w_left"].view(gate_info["w_left"].size(0), -1).mean(dim=1)
    wr = gate_info["w_right"].view(gate_info["w_right"].size(0), -1).mean(dim=1)
    pred_w = torch.stack([wl, wr], dim=1)

    targets = []
    for m in meta_list:
        side = m.get("side", "none")
        if not m.get("degraded", False):
            targets.append([0.50, 0.50])
        elif side == "left":
            targets.append([0.20, 0.80])
        elif side == "right":
            targets.append([0.80, 0.20])
        else:
            targets.append([0.50, 0.50])

    target = torch.tensor(targets, dtype=pred_w.dtype, device=pred_w.device)
    return F.mse_loss(pred_w, target)
