import argparse
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
    print("Clean EPE:", round(clean_metrics['EPE'], 4))
    print(f"Degraded EPE: {degraded_metrics['EPE']:.4f}")

    out = Path(args.output_dir)
    save_loader_preds(model, clean_loader, device, out / "clean", args.show)
    save_loader_preds(model, degraded_loader, device, out / "degraded", args.show, "degraded")
    print("Saved disparity/depth images to:", out)


if __name__ == "__main__":
    main()
