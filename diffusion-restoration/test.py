import json
import os

import torch
from torch.utils.data import DataLoader

from configs import default as cfg
from data.kitti_dataset import DegradedKITTIDisparityDataset, KITTIDisparityDataset, find_kitti_training
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import load_checkpoint
from utils.metrics import validate


def main():
    kitti_train_path = find_kitti_training(cfg.KITTI_EXTRACT_PATH)
    if not kitti_train_path:
        raise FileNotFoundError("KITTI training root not found.")
    if not os.path.exists(cfg.SPLIT_PATH):
        raise FileNotFoundError(f"Split file missing: {cfg.SPLIT_PATH}")
    with open(cfg.SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_files = split["val_files"]
    val_clean_ds = KITTIDisparityDataset(kitti_train_path, val_files, cfg.CROP_H, cfg.CROP_W, training=False)
    val_deg_ds = DegradedKITTIDisparityDataset(
        kitti_train_path,
        val_files,
        cfg.CROP_H,
        cfg.CROP_W,
        training=False,
        degrade_prob=cfg.DEGRADE_PROB,
        base_seed=cfg.DEGRADE_BASE_SEED,
        degrade_camera=cfg.DEGRADE_CAMERA,
        degrade_type=cfg.DEGRADE_TYPE,
        degrade_severity=cfg.DEGRADE_SEVERITY,
    )
    val_clean_loader = DataLoader(val_clean_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    val_deg_loader = DataLoader(val_deg_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    model = StereoWithCorrection(cfg.MAX_DISP, cfg.USE_FEATURE_GATE, cfg.FEAT_CHANNELS, cfg.N_DISP_BINS).to(cfg.DEVICE)
    ckpt = load_checkpoint(os.path.join(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_BEST), cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    clean_metrics = validate(model, val_clean_loader, cfg.DEVICE, "diffusion_restoration_final_clean")
    deg_metrics = validate(model, val_deg_loader, cfg.DEVICE, "diffusion_restoration_final_degraded")
    print("Clean:", clean_metrics)
    print("Degraded:", deg_metrics)


if __name__ == "__main__":
    main()
