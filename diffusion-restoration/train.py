import json
import os
import random
import zipfile

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import default as cfg
from data.kitti_dataset import (
    DegradedKITTIDisparityDataset,
    KITTIDisparityDataset,
    KITTITestDataset,
    PairedKITTIDisparityDataset,
    find_kitti_testing_root,
    find_kitti_training,
)
from losses.stereo_losses import ce_self_supervised_loss, gate_alignment_loss
from models.stereo_model import StereoWithCorrection, collated_meta_to_list
from utils.checkpoint import save_checkpoint
from utils.metrics import validate
from utils.visualization import plot_history, save_history, visualize_clean_and_degraded_example


def ensure_kitti_extracted():
    os.makedirs(cfg.KITTI_EXTRACT_PATH, exist_ok=True)
    if len(os.listdir(cfg.KITTI_EXTRACT_PATH)) == 0:
        if not os.path.exists(cfg.KITTI_ZIP_PATH):
            raise FileNotFoundError(f"KITTI zip not found at {cfg.KITTI_ZIP_PATH}")
        print("Extracting KITTI to", cfg.KITTI_EXTRACT_PATH)
        with zipfile.ZipFile(cfg.KITTI_ZIP_PATH, "r") as z:
            z.extractall(cfg.KITTI_EXTRACT_PATH)


def build_or_load_split(train_img_dir):
    all_files = sorted(os.listdir(train_img_dir))
    random.seed(cfg.SEED)
    random.shuffle(all_files)
    if os.path.exists(cfg.SPLIT_PATH):
        with open(cfg.SPLIT_PATH, "r", encoding="utf-8") as f:
            split = json.load(f)
        return split["train_files"], split["val_files"]
    split_idx = int(cfg.TRAIN_SPLIT * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    with open(cfg.SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seed": cfg.SEED, "train_files": train_files, "val_files": val_files}, f, indent=2)
    return train_files, val_files


def load_pretrained_backbone(model, path):
    if not os.path.exists(path):
        print("No pretrained backbone at", path, "— training from scratch")
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.stereo.backbone.load_state_dict(state, strict=False)
    print("Loaded pretrained KITTI2015 backbone from", path)


def train_one_epoch(model, loader, optimizer):
    model.train()
    totals = {k: 0.0 for k in ["loss", "raw_loss", "corr_loss", "ce_self_loss", "gate_loss", "diff_loss"]}
    n = 0
    bin_centers = model.refiner.bin_centers
    for batch in loader:
        clean_left, clean_right = batch[0].to(cfg.DEVICE), batch[1].to(cfg.DEVICE)
        deg_left, deg_right = batch[2].to(cfg.DEVICE), batch[3].to(cfg.DEVICE)
        gt = batch[4].to(cfg.DEVICE)
        clean_meta = collated_meta_to_list(batch[6])
        deg_meta = collated_meta_to_list(batch[7])
        optimizer.zero_grad(set_to_none=True)
        clean_out = model(clean_left, clean_right, batch[6])
        deg_out = model(deg_left, deg_right, batch[7])
        valid = (gt > 0) & torch.isfinite(gt)
        if valid.sum() == 0:
            continue
        raw_clean_loss = F.smooth_l1_loss(clean_out["raw"][valid], gt[valid])
        corr_clean_loss = F.smooth_l1_loss(clean_out["corrected"][valid], gt[valid])
        raw_deg_loss = F.smooth_l1_loss(deg_out["raw"][valid], gt[valid])
        corr_deg_loss = F.smooth_l1_loss(deg_out["corrected"][valid], gt[valid])
        ce_self_loss = ce_self_supervised_loss(
            deg_out["logits"], clean_out["corrected"], bin_centers, valid, cfg.CE_SIGMA
        )
        ce_clean_loss = ce_self_supervised_loss(clean_out["logits"], gt, bin_centers, valid, cfg.CE_SIGMA)
        gate_loss = gate_reg_loss(clean_out["gate_info"], clean_meta, cfg.DEVICE) + gate_reg_loss(
            deg_out["gate_info"], deg_meta, cfg.DEVICE
        )
        diff_loss = F.mse_loss(deg_out["corrected"], clean_out["corrected"].detach())
        loss = (
            corr_clean_loss
            + corr_deg_loss
            + cfg.RAW_SUP_WEIGHT * (raw_clean_loss + raw_deg_loss)
            + cfg.DIFF_LOSS_WEIGHT * diff_loss
            + cfg.SELF_SUP_WEIGHT * (ce_self_loss + 0.5 * ce_clean_loss)
            + cfg.GATE_LOSS_WEIGHT * gate_loss
        )
        loss.backward()
        optimizer.step()
        totals["loss"] += loss.item()
        totals["raw_loss"] += (raw_clean_loss.item() + raw_deg_loss.item()) / 2
        totals["corr_loss"] += (corr_clean_loss.item() + corr_deg_loss.item()) / 2
        totals["ce_self_loss"] += ce_self_loss.item()
        totals["gate_loss"] += gate_loss.item()
        totals["diff_loss"] += diff_loss.item()
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def run_test_inference(model, loader, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for left, right, name in loader:
        out = model(left.to(cfg.DEVICE), right.to(cfg.DEVICE))
        pred = torch.clamp(out["corrected"], min=0)
        pred_np = (pred[0].cpu().numpy() * 256.0).astype(np.uint16)
        Image.fromarray(pred_np).save(os.path.join(out_dir, name[0]))
    print("Test predictions saved to:", out_dir)


def main():
    print("device:", cfg.DEVICE)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    ensure_kitti_extracted()
    kitti_train_path = find_kitti_training(cfg.KITTI_EXTRACT_PATH)
    if not kitti_train_path:
        raise FileNotFoundError(f"KITTI training folder not found under {cfg.KITTI_EXTRACT_PATH}")

    train_files, val_files = build_or_load_split(os.path.join(kitti_train_path, "image_2"))
    print(f"train={len(train_files)}  val={len(val_files)}")

    train_ds = PairedKITTIDisparityDataset(
        kitti_train_path,
        train_files,
        cfg.CROP_H,
        cfg.CROP_W,
        True,
        cfg.TRAIN_DEGRADE_PROB,
        cfg.DEGRADE_BASE_SEED,
        cfg.DEGRADE_CAMERA,
        cfg.DEGRADE_TYPE,
        cfg.DEGRADE_SEVERITY,
    )
    val_clean_ds = KITTIDisparityDataset(kitti_train_path, val_files, cfg.CROP_H, cfg.CROP_W, False)
    val_deg_ds = DegradedKITTIDisparityDataset(
        kitti_train_path,
        val_files,
        cfg.CROP_H,
        cfg.CROP_W,
        False,
        cfg.DEGRADE_PROB,
        cfg.DEGRADE_BASE_SEED,
        cfg.DEGRADE_CAMERA,
        cfg.DEGRADE_TYPE,
        cfg.DEGRADE_SEVERITY,
    )
    test_root = find_kitti_testing_root(cfg.KITTI_EXTRACT_PATH)
    test_ds = KITTITestDataset(cfg.KITTI_EXTRACT_PATH, cfg.CROP_H, cfg.CROP_W) if test_root else None

    train_loader = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_clean_loader = DataLoader(val_clean_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    val_deg_loader = DataLoader(val_deg_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0) if test_ds else None

    print(
        f"gate: {'feature-level' if cfg.USE_FEATURE_GATE else 'image-level'}, "
        f"{cfg.N_DISP_BINS} disp bins, CE sigma={cfg.CE_SIGMA}"
    )
    print("KITTI training path:", kitti_train_path)

    model = StereoWithCorrection(cfg.MAX_DISP, cfg.USE_FEATURE_GATE, cfg.FEAT_CHANNELS, cfg.N_DISP_BINS).to(cfg.DEVICE)
    load_pretrained_backbone(model, cfg.PRETRAINED_BACKBONE_PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    history = {k: [] for k in ["train_loss", "ce_self_loss", "clean_epe", "clean_d1", "degraded_epe", "degraded_d1"]}
    best_d1 = float("inf")

    if cfg.RUN_DEGRADED_VISUALIZATION:
        visualize_clean_and_degraded_example(train_ds)

    for epoch in range(cfg.EPOCHS):
        train_stats = train_one_epoch(model, train_loader, optimizer)
        clean_m = validate(model, val_clean_loader, cfg.DEVICE, f"clean val (epoch {epoch + 1})")
        deg_m = validate(model, val_deg_loader, cfg.DEVICE, f"deg val   (epoch {epoch + 1})")

        history["train_loss"].append(train_stats["loss"])
        history["ce_self_loss"].append(train_stats["ce_self_loss"])
        history["clean_epe"].append(clean_m["EPE"])
        history["clean_d1"].append(clean_m["D1-all"])
        history["degraded_epe"].append(deg_m["EPE"])
        history["degraded_d1"].append(deg_m["D1-all"])

        print(f"\nepoch {epoch + 1}")
        print(
            f"  loss={train_stats['loss']:.4f}  raw={train_stats['raw_loss']:.4f}  "
            f"corr={train_stats['corr_loss']:.4f}  ce={train_stats['ce_self_loss']:.4f}  "
            f"gate={train_stats['gate_loss']:.4f}"
        )
        print(f"  clean   EPE={clean_m['EPE']:.4f}  D1={clean_m['D1-all']:.4f}")
        print(
            f"  degraded EPE={deg_m['EPE']:.4f}  D1={deg_m['D1-all']:.4f}  "
            f"w_L={deg_m['avg_w_left']:.3f}  w_R={deg_m['avg_w_right']:.3f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "clean_metrics": clean_m,
            "deg_metrics": deg_m,
            "train_stats": train_stats,
        }
        save_checkpoint(state, os.path.join(cfg.CHECKPOINT_DIR, f"last_epoch_{epoch}.pth"))
        if clean_m["D1-all"] < best_d1:
            best_d1 = clean_m["D1-all"]
            save_checkpoint(state, os.path.join(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_BEST))
            print("  saved best checkpoint")

    print(f"\ndone — best clean D1-all: {best_d1:.4f}")
    save_history(history, cfg.HISTORY_PATH)
    plot_history(history, cfg.OUTPUT_DIR)

    if cfg.EVAL_ON_CLEAN_AND_DEGRADED:
        validate(model, val_clean_loader, cfg.DEVICE, "final clean")
        validate(model, val_deg_loader, cfg.DEVICE, "final degraded")

    if test_loader:
        run_test_inference(model, test_loader, cfg.TEST_PRED_DIR)


if __name__ == "__main__":
    main()
