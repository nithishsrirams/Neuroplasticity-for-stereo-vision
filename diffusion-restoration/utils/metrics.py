import numpy as np
import torch


@torch.no_grad()
def validate(model, loader, device, title="Validation"):
    model.eval()
    epe_scores = []
    d1_scores = []
    left_weights = []
    right_weights = []

    for batch in loader:
        left = batch[0].to(device)
        right = batch[1].to(device)
        gt = batch[2].to(device)

        out = model(left, right)
        pred = out["corrected"]
        gate_info = out.get("gate_info")

        valid = (gt > 0) & torch.isfinite(gt)
        if valid.sum() == 0:
            continue

        err = (pred - gt).abs()[valid]
        gt_valid = gt[valid]

        epe_scores.append(err.mean().item())
        d1_scores.append(((err > 3.0) & (err / torch.clamp(gt_valid, min=1.0) > 0.05)).float().mean().item())

        if gate_info is not None:
            left_weights.append(gate_info["w_left"].mean().item())
            right_weights.append(gate_info["w_right"].mean().item())

    metrics = {
        "EPE": np.mean(epe_scores) if epe_scores else 0.0,
        "D1-all": np.mean(d1_scores) if d1_scores else 0.0,
        "avg_w_left": np.mean(left_weights) if left_weights else 0.0,
        "avg_w_right": np.mean(right_weights) if right_weights else 0.0,
    }

    print(f"\n{title}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics
