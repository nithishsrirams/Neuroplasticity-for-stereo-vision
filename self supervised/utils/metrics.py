import torch


@torch.no_grad()
def validate_epe(model, loader, device):
    model.eval()
    total_epe, n = 0.0, 0
    for left, right, gt, _ in loader:
        left, right, gt = left.to(device), right.to(device), gt.to(device)
        pred = model(left, right)["corrected"]
        valid = (gt > 0) & torch.isfinite(gt)
        if valid.sum() == 0:
            continue
        total_epe += (pred[valid] - gt[valid]).abs().mean().item()
        n += 1
    return {"EPE": total_epe / max(1, n)}
