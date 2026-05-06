from pathlib import Path

import torch


def save_checkpoint(path, epoch, model, optimizer, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, model, device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
