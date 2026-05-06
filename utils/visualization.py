from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def disparity_to_vis(disp: np.ndarray) -> np.ndarray:
    m = max(1e-6, float(disp.max()))
    return (disp / m * 255.0).clip(0, 255).astype(np.uint8)


def save_disparity_images(disp_tensor: torch.Tensor, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    disp = disp_tensor.detach().cpu().numpy()
    depth_u16 = (disp * 256.0).astype(np.uint16)
    vis = disparity_to_vis(disp)
    Image.fromarray(depth_u16).save(out_dir / f"{stem}_depth_u16.png")
    Image.fromarray(vis).save(out_dir / f"{stem}_depth_vis.png")


def show_disparity(title: str, disp_tensor: torch.Tensor):
    disp = disp_tensor.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(disp, cmap="plasma")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
