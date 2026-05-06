import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from configs.default import Config
from data.degradation import degrade
from models.stereo_model import StereoWithCorrection
from utils.checkpoint import load_checkpoint
from utils.visualization import save_disparity_images, show_disparity


def load_rgb_tensor(path, h, w):
    arr = np.array(Image.open(path).convert("RGB").resize((w, h))).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/infer_disparity")  # shorter flag
    parser.add_argument("--degrade", action="store_true")
    parser.add_argument("--degrade-type", default="blur", choices=["blur", "noise", "occlusion"])
    parser.add_argument("--degrade-severity", type=int, default=4)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--psmnet-dir", default="/content/PSMNet")
    args = parser.parse_args()

    cfg = Config(psmnet_dir=args.psmnet_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = StereoWithCorrection(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    if not Path(args.left).exists():
        raise FileNotFoundError(f"Left image not found: {args.left}")

    left_img = load_rgb_tensor(args.left, cfg.crop_h, cfg.crop_w)
    right = load_rgb_tensor(args.right, cfg.crop_h, cfg.crop_w)

    out_dir = Path(args.out)
    print("loading model...")  # remove later
    # pred_clean = model(left, right)["disparity"]  # old key name
    with torch.no_grad():
        pred_clean = model(left_img.to(device), right.to(device))["corrected"][0]
    save_disparity_images(pred_clean, out_dir / "clean", Path(args.left).stem)
    if args.show:
        show_disparity("clean disparity/depth", pred_clean)

    if args.degrade:
        rng = np.random.default_rng(123)
        left_np = left_img[0].permute(1, 2, 0).numpy()
        left_np = (left_np * 255).astype(np.uint8)
        left_deg_np, _ = degrade(left_np, args.degrade_type, args.degrade_severity, rng)
        left_deg_np = left_deg_np / 255.0
        left_deg = torch.tensor(left_deg_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pred_deg = model(left_deg.to(device), right.to(device))["corrected"][0]
        save_disparity_images(pred_deg, out_dir / "degraded", Path(args.left).stem)
        if args.show:
            show_disparity("degraded disparity/depth", pred_deg)

    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
