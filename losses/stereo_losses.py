import torch
import torch.nn.functional as F


def disparity_to_soft_target(disp, bin_centers, sigma):
    d = disp.unsqueeze(1)
    bins = bin_centers.view(1, -1, 1, 1)
    diff = (d - bins) ** 2 / (2.0 * sigma**2)
    return torch.softmax(-diff, dim=1)


def ce_self_supervised_loss(logits, target_disp, bin_centers, valid_mask, sigma):
    soft_target = disparity_to_soft_target(target_disp.detach(), bin_centers, sigma)
    log_probs = F.log_softmax(logits, dim=1)
    ce_map = -(soft_target * log_probs).sum(dim=1)
    if valid_mask.sum() == 0:
        return ce_map.mean()
    return ce_map[valid_mask].mean()


def collated_meta_to_list(meta):
    if meta is None:
        return []
    if isinstance(meta, list):
        return meta
    if isinstance(meta, dict):
        batch_size = None
        for v in meta.values():
            if isinstance(v, (list, tuple)):
                batch_size = len(v)
                break
            if torch.is_tensor(v) and v.ndim >= 1:
                batch_size = v.shape[0]
                break
        if batch_size is None:
            return [meta]
        out = []
        for i in range(batch_size):
            item = {}
            for k, v in meta.items():
                if isinstance(v, (list, tuple)):
                    item[k] = v[i]
                elif torch.is_tensor(v) and v.ndim >= 1:
                    item[k] = v[i].item() if v[i].numel() == 1 else v[i]
                else:
                    item[k] = v
            out.append(item)
        return out
    return [meta]


def gate_alignment_loss(gate_info, meta):
    if gate_info is None or meta is None:
        return torch.tensor(0.0)

    meta_list = collated_meta_to_list(meta)
    if not meta_list:
        return torch.tensor(0.0, device=gate_info["w_left"].device)

    wl = gate_info["w_left"].view(gate_info["w_left"].size(0), -1).mean(dim=1)
    wr = gate_info["w_right"].view(gate_info["w_right"].size(0), -1).mean(dim=1)
    pred_w = torch.stack([wl, wr], dim=1)

    targets = []
    for m in meta_list:
        degraded = bool(m.get("degraded", False))
        side = m.get("side", "none")
        if not degraded:
            targets.append([0.50, 0.50])
        elif side == "left":
            targets.append([0.20, 0.80])
        elif side == "right":
            targets.append([0.80, 0.20])
        else:
            targets.append([0.50, 0.50])

    target = torch.tensor(targets, dtype=pred_w.dtype, device=pred_w.device)
    return F.mse_loss(pred_w, target)
