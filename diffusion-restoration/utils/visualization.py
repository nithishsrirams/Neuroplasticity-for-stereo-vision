import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.kitti_dataset import tensor_to_uint8_img


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = np.arange(1, len(history["clean_epe"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["clean_epe"], label="Clean EPE")
    plt.plot(epochs, history["degraded_epe"], label="Degraded EPE")
    plt.xlabel("Epoch")
    plt.ylabel("EPE")
    plt.title("diffusion_restoration_cleaned EPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diffusion_restoration_epe_plot.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["clean_d1"], label="Clean D1-all")
    plt.plot(epochs, history["degraded_d1"], label="Degraded D1-all")
    plt.xlabel("Epoch")
    plt.ylabel("D1-all")
    plt.title("diffusion_restoration_cleaned D1-all")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diffusion_restoration_d1_all_plot.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["ce_self_loss"], label="CE self-sup")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("diffusion_restoration_cleaned self-supervised CE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diffusion_restoration_ce_plot.png"), dpi=150)
    plt.close()


@torch.no_grad()
def visualize_clean_and_degraded_example(dataset):
    sample = dataset[0]
    if len(sample) != 8:
        raise ValueError("expected paired dataset (8-tuple)")
    clean_left, clean_right, deg_left, deg_right, _, _name, _, deg_meta = sample

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    imgs = [clean_left, clean_right, deg_left, deg_right]
    titles = [
        "clean left",
        "clean right",
        f"deg left  — {deg_meta['type']} L{deg_meta['level']}",
        f"deg right — {deg_meta['type']} L{deg_meta['level']}",
    ]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(tensor_to_uint8_img(img))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
