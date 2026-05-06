
import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.default import Config
from data.kitti_dataset import DegradedKITTIDisparityDataset, KITTIDisparityDataset, find_kitti_training
from losses.stereo_losses import ce_self_supervised_loss
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import save_checkpoint
from utils.visualization import save_disparity_images, show_disparity


def validate_metrics(model, loader, device):
    model.eval()
    total_epe, total_d1, n = 0.0, 0.0, 0
    with torch.no_grad():
        for left, right, gt, _ in loader:
            left, right, gt = left.to(device), right.to(device), gt.to(device)
            pred = model(left, right)["corrected"]
            valid = (gt > 0) & torch.isfinite(gt)
            if valid.sum() == 0:
                continue
            err = (pred - gt).abs()[valid]
            gt_valid = gt[valid]
            epe = err.mean().item()
            d1 = ((err > 3.0) & (err / torch.clamp(gt_valid, min=1.0) > 0.05)).float().mean().item()
            total_epe += epe
            total_d1 += d1
            n += 1
    return {"EPE": total_epe / max(1, n), "D1-all": total_d1 / max(1, n)}


def save_history_plots(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("loss per epoch")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["clean_epe"], label="Clean EPE")
    plt.plot(epochs, history["degraded_epe"], label="Degraded EPE")
    plt.xlabel("Epoch"); plt.ylabel("EPE"); plt.title("EPE vs Epoch")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "epe_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["clean_d1"], label="Clean D1-all")
    plt.plot(epochs, history["degraded_d1"], label="Degraded D1-all")
    plt.xlabel("Epoch"); plt.ylabel("D1-all"); plt.title("D1-all vs Epoch")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "d1_curve.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--psmnet-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = Config(data_root=args.data_root, psmnet_dir=args.psmnet_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_root = find_kitti_training(cfg.data_root)
    if train_root is None:
        raise FileNotFoundError(f"Could not locate KITTI training folder under {cfg.data_root}")

    files = sorted(os.listdir(os.path.join(train_root, "image_2")))
    random.seed(42); random.shuffle(files)
    split = int(0.9 * len(files))
    train_files, val_files = files[:split], files[split:]

    train_ds = KITTIDisparityDataset(train_root, train_files, cfg.crop_h, cfg.crop_w, training=True)
    val_clean_ds = KITTIDisparityDataset(train_root, val_files, cfg.crop_h, cfg.crop_w, training=False)
    val_deg_ds = DegradedKITTIDisparityDataset(
        train_root, val_files, cfg.crop_h, cfg.crop_w, training=False,
        degrade_prob=cfg.degrade_prob_eval,
        base_seed=cfg.degrade_base_seed,
        degrade_camera=cfg.degrade_camera,
        degrade_type=cfg.degrade_type,
        degrade_severity=cfg.degrade_severity,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    clean_loader = DataLoader(val_clean_ds, batch_size=1, shuffle=False, num_workers=0)
    deg_loader   = DataLoader(val_deg_ds,   batch_size=1, shuffle=False, num_workers=0)

    model = StereoWithCorrection(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    sample_dir = out_dir / "train_disparity"
    for d in [ckpt_dir, plot_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "clean_epe": [],
        "degraded_epe": [],
        "clean_d1": [],
        "degraded_d1": []
    }

    best_epe = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0  # renamed inconsistently from steps

        for left, right, gt, _ in train_loader:
            left, right, gt = left.to(device), right.to(device), gt.to(device)
            out = model(left, right)
            valid = (gt > 0) & torch.isfinite(gt)
            if valid.sum() == 0:
                continue

            raw_loss = F.smooth_l1_loss(out["raw"][valid], gt[valid])
            corr_loss = F.smooth_l1_loss(out["corrected"][valid], gt[valid])
            ce_loss = ce_self_supervised_loss(out["logits"], gt, model.refiner.bin_centers, valid, cfg.ce_sigma)
            loss = corr_loss + cfg.raw_sup_weight * raw_loss + cfg.self_sup_weight * ce_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  step {n_batches} loss {loss.item():.4f}")

        train_loss = total_loss / max(1, n_batches)
        clean_m = validate_metrics(model, clean_loader, device)
        deg_m = validate_metrics(model, deg_loader, device)

        history["train_loss"].append(train_loss)
        history["clean_epe"].append(clean_m["EPE"])
        history["degraded_epe"].append(deg_m["EPE"])
        history["clean_d1"].append(clean_m["D1-all"])
        history["degraded_d1"].append(deg_m["D1-all"])

        print(f"Epoch {epoch+1}/{cfg.epochs} | loss={train_loss:.4f} | clean_epe={clean_m['EPE']:.4f} deg_epe={deg_m['EPE']:.4f}")

        with torch.no_grad():
            model.eval()
            c_left, c_right, _, c_name = next(iter(clean_loader))
            d_left, d_right, _, d_name = next(iter(deg_loader))
            c_pred = model(c_left.to(device), c_right.to(device))["corrected"][0]
            d_pred = model(d_left.to(device), d_right.to(device))["corrected"][0]
            ep_dir = sample_dir / f"epoch_{epoch+1:03d}"
            save_disparity_images(c_pred, ep_dir / "clean", Path(c_name[0]).stem)
            save_disparity_images(d_pred, ep_dir / "degraded", Path(d_name[0]).stem)
            if args.show:
                show_disparity(f"Epoch {epoch+1} clean", c_pred)
                show_disparity(f"Epoch {epoch+1} degraded", d_pred)

        save_checkpoint(ckpt_dir / "last.pth", epoch, model, optimizer, {
            "train_loss": train_loss, "clean_epe": clean_m["EPE"],
            "clean_d1": clean_m["D1-all"], "degraded_epe": deg_m["EPE"], "degraded_d1": deg_m["D1-all"]
        })
        if clean_m["EPE"] < best_epe:
            best_epe = clean_m["EPE"]
            save_checkpoint(ckpt_dir / "best.pth", epoch, model, optimizer, {
                "train_loss": train_loss, "clean_epe": clean_m["EPE"],
                "clean_d1": clean_m["D1-all"], "degraded_epe": deg_m["EPE"], "degraded_d1": deg_m["D1-all"]
            })

        save_history_plots(history, plot_dir)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print("Saved checkpoints:", ckpt_dir)
    print("Saved plots:", plot_dir)
    print("Saved history:", out_dir / "history.json")


if __name__ == "__main__":
    main()
