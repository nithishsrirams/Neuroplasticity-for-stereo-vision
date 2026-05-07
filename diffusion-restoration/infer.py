import os

from PIL import Image
import torch
from torch.utils.data import DataLoader

from configs import default as cfg
from data.kitti_dataset import KITTITestDataset, find_kitti_testing_root
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import load_checkpoint


@torch.no_grad()
def run_test_inference(model, loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for left, right, name in loader:
        out = model(left.to(cfg.DEVICE), right.to(cfg.DEVICE))
        pred = torch.clamp(out["corrected"], min=0)
        pred_np = (pred[0].cpu().numpy() * 256.0).astype("uint16")
        Image.fromarray(pred_np).save(os.path.join(out_dir, f"diffusion_restoration_{name[0]}"))
    print("Saved predictions to:", out_dir)


def main():
    test_root = find_kitti_testing_root(cfg.KITTI_EXTRACT_PATH)
    if not test_root:
        raise FileNotFoundError("KITTI testing root not found.")
    test_ds = KITTITestDataset(cfg.KITTI_EXTRACT_PATH, cfg.CROP_H, cfg.CROP_W)
    test_loader = DataLoader(test_ds, cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    model = StereoWithCorrection(cfg.MAX_DISP, cfg.USE_FEATURE_GATE, cfg.FEAT_CHANNELS, cfg.N_DISP_BINS).to(cfg.DEVICE)
    ckpt = load_checkpoint(os.path.join(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_BEST), cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    run_test_inference(model, test_loader, cfg.TEST_PRED_DIR)


if __name__ == "__main__":
    main()
