import torch


@torch.no_grad()
def validate_epe(model, loader, device):
    """Per-batch averages over samples with valid disparity; includes branch gate weights."""
    model.eval()
    total_epe, total_d1, total_w_left, total_w_right, n = 0.0, 0.0, 0.0, 0.0, 0
    for left, right, gt, _ in loader:
        left, right, gt = left.to(device), right.to(device), gt.to(device)
        out = model(left, right)
        pred = out["corrected"]
        gi = out.get("gate_info") or {}
        w_left, w_right = gi.get("w_left"), gi.get("w_right")
        valid = (gt > 0) & torch.isfinite(gt)
        if valid.sum() == 0:
            continue
        err = (pred - gt).abs()[valid]
        gt_valid = gt[valid]
        total_epe += err.mean().item()
        total_d1 += ((err > 3.0) & (err / torch.clamp(gt_valid, min=1.0) > 0.05)).float().mean().item()
        if w_left is not None:
            total_w_left += w_left.mean().item()
        if w_right is not None:
            total_w_right += w_right.mean().item()
        n += 1
    denom = max(1, n)
    return {
        "EPE": total_epe / denom,
        "D1-all": total_d1 / denom,
        "w_left": total_w_left / denom,
        "w_right": total_w_right / denom,
    }
