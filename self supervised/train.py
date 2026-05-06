import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.default import Config
from data.kitti_dataset import DegradedKITTIDisparityDataset, KITTIDisparityDataset, PairedKITTIDisparityDataset, find_kitti_training
from losses.stereo_losses import ce_self_supervised_loss, gate_alignment_loss
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import save_checkpoint
from utils.metrics import validate_epe
from utils.visualization import save_disparity_images, show_disparity


def forward_pass(model, left, right):
    return model(left, right)


def log_epoch_stats(stats):
    print(
        "epoch {epoch}/{epochs} | train={train_loss:.4f} raw={raw:.4f} corr={corr:.4f} "
        "ce={ce:.4f} gate={gate:.4f} clean_epe={clean_epe:.4f} degraded_epe={degraded_epe:.4f}".format(**stats)
    )


def train_one_epoch(model, loader, optimizer, device, cfg, epoch, debug=False):
    model.train()
    total = 0.0
    raw_total = 0.0
    corr_total = 0.0
    ce_total = 0.0
    gate_total = 0.0
    n_batches = 0

    for clean_left, clean_right, deg_left, deg_right, gt, _, clean_meta, deg_meta in loader:
        clean_left, clean_right = clean_left.to(device), clean_right.to(device)
        deg_left, deg_right = deg_left.to(device), deg_right.to(device)
        gt = gt.to(device)
        clean_out = forward_pass(model, clean_left, clean_right)
        deg_out = forward_pass(model, deg_left, deg_right)
        valid = (gt > 0) & torch.isfinite(gt)

        # KITTI occasionally gives sparse invalid/border-heavy labels after crops.
        # Skip these to avoid unstable gradients from tiny valid supports.
        if valid.sum() < 500:
            continue

        raw_clean_loss = F.smooth_l1_loss(clean_out["raw"][valid], gt[valid])
        corr_clean_loss = F.smooth_l1_loss(clean_out["corrected"][valid], gt[valid])
        raw_deg_loss = F.smooth_l1_loss(deg_out["raw"][valid], gt[valid])
        corr_deg_loss = F.smooth_l1_loss(deg_out["corrected"][valid], gt[valid])

        # The degraded branch is supervised against clean corrected disparity
        # so corruption robustness does not drift away from stereo geometry.
        ce_self_loss = ce_self_supervised_loss(
            deg_out["logits"], clean_out["corrected"].detach(), model.refiner.bin_centers, valid, cfg.ce_sigma
        )
        ce_clean_loss = ce_self_supervised_loss(
            clean_out["logits"], gt, model.refiner.bin_centers, valid, cfg.ce_sigma
        )
        gate_loss = (
            gate_alignment_loss(clean_out["gate_info"], clean_meta).to(device)
            + gate_alignment_loss(deg_out["gate_info"], deg_meta).to(device)
        )

        loss_terms = {
            "raw_clean": raw_clean_loss,
            "raw_deg": raw_deg_loss,
            "corr_clean": corr_clean_loss,
            "corr_deg": corr_deg_loss,
            "ce_self": ce_self_loss,
            "ce_clean": ce_clean_loss,
            "gate": gate_loss,
        }

        raw_loss = loss_terms["raw_clean"]
        raw_loss += loss_terms["raw_deg"]
        raw_loss *= 0.5

        corr_loss = loss_terms["corr_clean"]
        corr_loss += loss_terms["corr_deg"]
        corr_loss *= 0.5

        ce_loss = loss_terms["ce_self"] + 0.5 * loss_terms["ce_clean"]
        loss = (
            loss_terms["corr_clean"]
            + loss_terms["corr_deg"]
            + cfg.raw_sup_weight * (loss_terms["raw_clean"] + loss_terms["raw_deg"])
            + cfg.self_sup_weight * ce_loss
            + cfg.gate_loss_weight * loss_terms["gate"]
        )
        if torch.isnan(loss):
            print("NaN detected in loss, skipping batch")
            continue
        if debug and epoch == 0 and n_batches == 0:
            print(
                "Initial corrected range:",
                clean_out["corrected"].min().item(),
                clean_out["corrected"].max().item(),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item()
        raw_total += raw_loss.item()
        corr_total += corr_loss.item()
        ce_total += ce_loss.item()
        gate_total += loss_terms["gate"].item()
        n_batches += 1

    return {
        "train_loss": total / max(1, n_batches),
        "raw": raw_total / max(1, n_batches),
        "corr": corr_total / max(1, n_batches),
        "ce": ce_total / max(1, n_batches),
        "gate": gate_total / max(1, n_batches),
        "n_batches": n_batches,
    }


def evaluate(model, val_clean_loader, val_degraded_loader, device):
    clean_metrics = validate_epe(model, val_clean_loader, device)
    degraded_metrics = validate_epe(model, val_degraded_loader, device)
    return clean_metrics, degraded_metrics


def save_visualizations(model, val_clean_loader, val_degraded_loader, device, viz_dir, epoch, show=False):
    # Quick qualitative check for correction behavior under synthetic corruption.
    # TODO: move this into a small evaluator module once metrics stabilize.
    with torch.no_grad():
        model.eval()
        c_left, c_right, _, c_name = next(iter(val_clean_loader))
        d_left, d_right, _, d_name = next(iter(val_degraded_loader))
        c_pred = forward_pass(model, c_left.to(device), c_right.to(device))["corrected"][0]
        d_pred = forward_pass(model, d_left.to(device), d_right.to(device))["corrected"][0]
        ep_dir = viz_dir / f"epoch_{epoch + 1:03d}"
        save_disparity_images(c_pred, ep_dir / "clean", Path(c_name[0]).stem)
        save_disparity_images(d_pred, ep_dir / "degraded", Path(d_name[0]).stem)
        if show:
            show_disparity(f"Train epoch {epoch + 1} clean", c_pred)
            show_disparity(f"Train epoch {epoch + 1} degraded", d_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/Spring 2026/CV")
    parser.add_argument("--psmnet-dir", default="/content/PSMNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = Config(data_root=args.data_root, psmnet_dir=args.psmnet_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    epochs = cfg.epochs
    lr = cfg.lr

    train_root = find_kitti_training(cfg.data_root)
    if train_root is None:
        raise FileNotFoundError(f"Could not locate KITTI training folder under {cfg.data_root}")

    files = sorted(os.listdir(os.path.join(train_root, "image_2")))
    random.seed(42)
    random.shuffle(files)
    split = int(0.9 * len(files))
    train_files, val_files = files[:split], files[split:]

    train_ds = PairedKITTIDisparityDataset(
        train_root,
        train_files,
        cfg.crop_h,
        cfg.crop_w,
        training=True,
        degrade_prob=cfg.degrade_prob_train,
        base_seed=cfg.degrade_base_seed,
        degrade_camera=cfg.degrade_camera,
        degrade_type=cfg.degrade_type,
        degrade_severity=cfg.degrade_severity,
    )
    val_clean_ds = KITTIDisparityDataset(train_root, val_files, cfg.crop_h, cfg.crop_w, training=False)
    val_degraded_ds = DegradedKITTIDisparityDataset(
        train_root,
        val_files,
        cfg.crop_h,
        cfg.crop_w,
        training=False,
        degrade_prob=cfg.degrade_prob_eval,
        base_seed=cfg.degrade_base_seed,
        degrade_camera=cfg.degrade_camera,
        degrade_type=cfg.degrade_type,
        degrade_severity=cfg.degrade_severity,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=1, shuffle=False, num_workers=0)
    val_degraded_loader = DataLoader(val_degraded_ds, batch_size=1, shuffle=False, num_workers=0)

    model = StereoWithCorrection(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    viz_dir = out_dir / "train_disparity"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    best_epe = float("inf")

    for epoch in range(epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch, debug=args.debug)
        clean_metrics, degraded_metrics = evaluate(model, val_clean_loader, val_degraded_loader, device)
        stats = {
            "epoch": epoch + 1,
            "epochs": epochs,
            "train_loss": train_stats["train_loss"],
            "raw": train_stats["raw"],
            "corr": train_stats["corr"],
            "ce": train_stats["ce"],
            "gate": train_stats["gate"],
            "clean_epe": clean_metrics["EPE"],
            "degraded_epe": degraded_metrics["EPE"],
        }
        log_epoch_stats(stats)

        save_visualizations(model, val_clean_loader, val_degraded_loader, device, viz_dir, epoch, show=args.show)

        metrics = {"clean_epe": clean_metrics["EPE"], "degraded_epe": degraded_metrics["EPE"], "train_loss": train_stats["train_loss"]}
        save_checkpoint(ckpt_dir / "last.pth", epoch, model, optimizer, metrics)
        if clean_metrics["EPE"] < best_epe:
            best_epe = clean_metrics["EPE"]
            save_checkpoint(ckpt_dir / "best.pth", epoch, model, optimizer, metrics)


if __name__ == "__main__":
    main()
