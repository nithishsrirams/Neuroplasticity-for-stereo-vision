import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.degradation import degrade, tensor_to_uint8_img, uint8_img_to_tensor


def find_kitti_training(base):
    candidates = [
        os.path.join(base, "training"),
        os.path.join(base, "data_scene_flow", "training"),
        os.path.join(base, "data_stereo_flow", "training"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, dirs, _ in os.walk(base):
        if "image_2" in dirs and "image_3" in dirs and "disp_occ_0" in dirs:
            return root
    return None


class KITTIDisparityDataset(Dataset):
    def __init__(self, root, files, crop_h=256, crop_w=512, training=True):
        self.left_dir = os.path.join(root, "image_2")
        self.right_dir = os.path.join(root, "image_3")
        self.disp_dir = os.path.join(root, "disp_occ_0")
        self.files = [f for f in files if f.endswith(".png") and "_10.png" in f]
        self.crop_h, self.crop_w, self.training = crop_h, crop_w, training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        left = np.array(Image.open(os.path.join(self.left_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        right = np.array(Image.open(os.path.join(self.right_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        disp = np.array(Image.open(os.path.join(self.disp_dir, name))).astype(np.float32) / 256.0

        h, w = disp.shape
        if h >= self.crop_h and w >= self.crop_w:
            y = random.randint(0, h - self.crop_h) if self.training else (h - self.crop_h) // 2
            x = random.randint(0, w - self.crop_w) if self.training else (w - self.crop_w) // 2
            left = left[y : y + self.crop_h, x : x + self.crop_w]
            right = right[y : y + self.crop_h, x : x + self.crop_w]
            disp = disp[y : y + self.crop_h, x : x + self.crop_w]

        return (
            torch.from_numpy(left).permute(2, 0, 1).float(),
            torch.from_numpy(right).permute(2, 0, 1).float(),
            torch.from_numpy(disp).float(),
            name,
        )


class DegradedKITTIDisparityDataset(KITTIDisparityDataset):
    def __init__(
        self,
        root,
        files,
        crop_h=256,
        crop_w=512,
        training=False,
        degrade_prob=1.0,
        base_seed=12345,
        degrade_camera="left",
        degrade_type="blur",
        degrade_severity=4,
    ):
        super().__init__(root, files, crop_h, crop_w, training)
        self.degrade_prob = degrade_prob
        self.base_seed = base_seed
        self.degrade_camera = degrade_camera
        self.degrade_type = degrade_type
        self.degrade_severity = degrade_severity

    def __getitem__(self, idx):
        left, right, disp, name = super().__getitem__(idx)
        py_rng = random.Random(self.base_seed + idx)
        np_rng = np.random.default_rng(self.base_seed + idx)

        if py_rng.random() < self.degrade_prob:
            left_u8 = tensor_to_uint8_img(left)
            right_u8 = tensor_to_uint8_img(right)
            side = self.degrade_camera if self.degrade_camera in ["left", "right"] else "left"
            if side == "left":
                left_u8, mask = degrade(left_u8, self.degrade_type, int(self.degrade_severity), np_rng)
                if mask is not None:
                    disp = disp * torch.from_numpy(mask).float()
            else:
                right_u8, _ = degrade(right_u8, self.degrade_type, int(self.degrade_severity), np_rng)
            left = uint8_img_to_tensor(left_u8).float()
            right = uint8_img_to_tensor(right_u8).float()

        return left, right, disp, name


def choose_setting(value, options, rng):
    if value == "random":
        return options[int(rng.integers(0, len(options)))]
    return value


def random_degrade(img, rng, fixed_type=None, fixed_level=None):
    types = ["blur", "noise", "occlusion"]
    dtype = fixed_type if fixed_type not in (None, "random") else types[int(rng.integers(0, 3))]
    level = fixed_level if fixed_level not in (None, "random") else int(rng.integers(1, 5))
    degraded, mask = degrade(img, dtype, int(level), rng)
    return degraded, {"type": dtype, "level": int(level), "mask": mask}


class PairedKITTIDisparityDataset(KITTIDisparityDataset):
    def __init__(
        self,
        root,
        files,
        crop_h=256,
        crop_w=512,
        training=True,
        degrade_prob=0.7,
        base_seed=12345,
        degrade_camera="random",
        degrade_type="random",
        degrade_severity="random",
    ):
        super().__init__(root, files, crop_h, crop_w, training)
        self.degrade_prob = degrade_prob
        self.base_seed = base_seed
        self.degrade_camera = degrade_camera
        self.degrade_type = degrade_type
        self.degrade_severity = degrade_severity

    def __getitem__(self, idx):
        clean_left, clean_right, clean_disp, name = super().__getitem__(idx)
        py_rng = random.Random(self.base_seed + idx)
        np_rng = np.random.default_rng(self.base_seed + idx)

        deg_left_u8 = tensor_to_uint8_img(clean_left)
        deg_right_u8 = tensor_to_uint8_img(clean_right)
        deg_meta = {"degraded": False, "type": "clean", "level": 0, "side": "none"}
        clean_meta = {"degraded": False, "type": "clean", "level": 0, "side": "none"}

        if py_rng.random() < self.degrade_prob:
            side = choose_setting(self.degrade_camera, ["left", "right"], np_rng)
            dtype = choose_setting(self.degrade_type, ["blur", "noise", "occlusion"], np_rng)
            level = choose_setting(self.degrade_severity, [1, 2, 3, 4], np_rng)
            if side == "left":
                deg_left_u8, info = random_degrade(deg_left_u8, np_rng, fixed_type=dtype, fixed_level=level)
                if info["mask"] is not None:
                    clean_disp = clean_disp * torch.from_numpy(info["mask"]).float()
            else:
                deg_right_u8, info = random_degrade(deg_right_u8, np_rng, fixed_type=dtype, fixed_level=level)
            deg_meta = {"degraded": True, "type": info["type"], "level": info["level"], "side": side}

        deg_left = uint8_img_to_tensor(deg_left_u8).float()
        deg_right = uint8_img_to_tensor(deg_right_u8).float()
        return clean_left, clean_right, deg_left, deg_right, clean_disp.float(), name, clean_meta, deg_meta
