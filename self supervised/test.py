import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.default import Config
from data.kitti_dataset import DegradedKITTIDisparityDataset, KITTIDisparityDataset, find_kitti_training
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import load_checkpoint
from utils.metrics import validate_epe
from utils.visualization import save_disparity_images, show_disparity


def _annotate_bars(ax, heights, fmt: str) -> None:
    for rect, h in zip(ax.patches, heights):
        y = rect.get_height()
        ax.annotate(
            fmt.format(y),
            xy=(rect.get_x() + rect.get_width() / 2, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_epe_d1_plots(clean_metrics: dict, degraded_metrics: dict, path: Path, suptitle: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Clean", "Degraded"]
    x = range(len(labels))
    epe_vals = [clean_metrics["EPE"], degraded_metrics["EPE"]]
    d1_frac = [clean_metrics["D1-all"], degraded_metrics["D1-all"]]
    d1_pct = [v * 100.0 for v in d1_frac]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    colors = ("#2ca02c", "#d62728")
    axes[0].bar(x, epe_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("EPE (pixels)")
    axes[0].set_title("End-point error")
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_axisbelow(True)
    _annotate_bars(axes[0], epe_vals, "{:.3f}")

    axes[1].bar(x, d1_pct, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("D1-all (%)")
    axes[1].set_title("D1-all (>3px & >5%)")
    axes[1].set_ylim(bottom=0)
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.35)
    axes[1].set_axisbelow(True)
    _annotate_bars(axes[1], d1_pct, "{:.2f}%")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def maybe_display_saved_plot(plot_path: Path) -> None:
    """Show the PNG in Jupyter/Colab when the script runs in-notebook (not via subprocess)."""
    try:
        ip = __import__("IPython").get_ipython()
    except ImportError:
        return
    if ip is None:
        return
    from IPython.display import Image, display

    display(Image(filename=str(plot_path)))


def save_loader_preds(model, loader, device, out_dir, show=False, title_prefix=""):
    model.eval()
    out_dir = Path(out_dir)
    for left, right, _, names in loader:
        with torch.no_grad():
            pred = model(left.to(device), right.to(device))["corrected"]
        for i, name in enumerate(names):
            stem = Path(name).stem
            save_disparity_images(pred[i], out_dir, stem)
            if show:
                show_disparity(f"{title_prefix} {stem}", pred[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/Spring 2026/CV")
    parser.add_argument("--psmnet-dir", default="/content/PSMNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/test_disparity")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--degrade-type", default="blur", choices=["blur", "noise", "occlusion"])
    parser.add_argument("--degrade-severity", type=int, default=4)
    parser.add_argument("--degrade-camera", default="left", choices=["left", "right"])
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        psmnet_dir=args.psmnet_dir,
        degrade_type=args.degrade_type,
        degrade_severity=args.degrade_severity,
        degrade_camera=args.degrade_camera,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_root = find_kitti_training(cfg.data_root)
    if train_root is None:
        raise FileNotFoundError(f"Could not locate KITTI training folder under {cfg.data_root}")

    files = sorted(os.listdir(os.path.join(train_root, "image_2")))
    split = int(0.9 * len(files))
    # using last 10% as eval
    train_files = files[:split]   # not used but kept just in case
    eval_files = files[split:]
    if len(eval_files) == 0:
        raise ValueError("No eval files found, check your data split")

    clean_loader = DataLoader(KITTIDisparityDataset(train_root, eval_files, cfg.crop_h, cfg.crop_w, training=False), batch_size=1, shuffle=False)
    degraded_loader = DataLoader(
        DegradedKITTIDisparityDataset(
            train_root, eval_files, cfg.crop_h, cfg.crop_w,
            training=False, degrade_prob=1.0,
            base_seed=cfg.degrade_base_seed,
            degrade_camera=cfg.degrade_camera,
            degrade_type=cfg.degrade_type, degrade_severity=cfg.degrade_severity,
        ),
        batch_size=1, shuffle=False,
    )

    model = StereoWithCorrection(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    clean_metrics = validate_epe(model, clean_loader, device)
    degraded_metrics = validate_epe(model, degraded_loader, device)

    def _print_split(title: str, m: dict) -> None:
        print(f"\n{title}")
        print(f"  Average EPE:        {m['EPE']:.4f}")
        print(f"  Average D1-all:     {m['D1-all']:.4f}")
        print(f"  Average left w.:    {m['w_left']:.4f}")
        print(f"  Average right w.:   {m['w_right']:.4f}")

    _print_split("Clean", clean_metrics)
    _print_split("Degraded", degraded_metrics)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"clean": clean_metrics, "degraded": degraded_metrics}, f, indent=2)
    print("\nWrote", metrics_path)

    plot_path = out / "epe_d1_comparison.png"
    plot_title = f"{args.degrade_type} sev{args.degrade_severity} {args.degrade_camera}"
    save_epe_d1_plots(clean_metrics, degraded_metrics, plot_path, suptitle=plot_title)
    print("Wrote", plot_path)
    maybe_display_saved_plot(plot_path)

    save_loader_preds(model, clean_loader, device, out / "clean", args.show)
    save_loader_preds(model, degraded_loader, device, out / "degraded", args.show, "degraded")
    print("Saved disparity/depth images to:", out)


if __name__ == "__main__":
    main()
