import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.degradation import choose_setting, random_degrade


def find_kitti_training(base):
    for parts in (("training",), ("data_scene_flow", "training"), ("data_stereo_flow", "training")):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    for root, dirs, _ in os.walk(base):
        if {"image_2", "image_3", "disp_occ_0"}.issubset(dirs):
            return root
    return None


def find_kitti_testing_root(base):
    for parts in (("testing",), ("data_scene_flow", "testing")):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    for root, dirs, _ in os.walk(base):
        if "image_2" in dirs and "image_3" in dirs:
            return root
    return None


def tensor_to_uint8_img(t):
    return (t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


def uint8_img_to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


class KITTIDisparityDataset(Dataset):
    def __init__(self, root, files, crop_h=256, crop_w=512, training=True):
        self.left_dir = os.path.join(root, "image_2")
        self.right_dir = os.path.join(root, "image_3")
        self.disp_dir = os.path.join(root, "disp_occ_0")
        self.files = [f for f in files if "_10.png" in f]
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.training = training

    def __len__(self):
        return len(self.files)

    def _load(self, name):
        left = np.array(Image.open(os.path.join(self.left_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        right = np.array(Image.open(os.path.join(self.right_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        disp = np.array(Image.open(os.path.join(self.disp_dir, name))).astype(np.float32) / 256.0
        return left, right, disp

    def _crop(self, left, right, disp, rng):
        h, w = disp.shape
        if h < self.crop_h or w < self.crop_w:
            return left, right, disp
        y = rng.randint(0, h - self.crop_h) if self.training else (h - self.crop_h) // 2
        x = rng.randint(0, w - self.crop_w) if self.training else (w - self.crop_w) // 2
        return (
            left[y : y + self.crop_h, x : x + self.crop_w],
            right[y : y + self.crop_h, x : x + self.crop_w],
            disp[y : y + self.crop_h, x : x + self.crop_w],
        )

    def __getitem__(self, idx):
        name = self.files[idx]
        rng = random.Random(idx) if self.training else random
        left, right, disp = self._load(name)
        left, right, disp = self._crop(left, right, disp, rng)
        return (
            torch.from_numpy(left).permute(2, 0, 1).float(),
            torch.from_numpy(right).permute(2, 0, 1).float(),
            torch.from_numpy(disp).float(),
            name,
        )


class PairedKITTIDisparityDataset(Dataset):
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
        self.left_dir = os.path.join(root, "image_2")
        self.right_dir = os.path.join(root, "image_3")
        self.disp_dir = os.path.join(root, "disp_occ_0")
        self.files = [f for f in files if "_10.png" in f]
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.training = training
        self.degrade_prob = degrade_prob
        self.base_seed = base_seed
        self.degrade_camera = degrade_camera
        self.degrade_type = degrade_type
        self.degrade_severity = degrade_severity

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        py_rng = random.Random(self.base_seed + idx)
        np_rng = np.random.default_rng(self.base_seed + idx)

        left = np.array(Image.open(os.path.join(self.left_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        right = np.array(Image.open(os.path.join(self.right_dir, name)).convert("RGB")).astype(np.float32) / 255.0
        disp = np.array(Image.open(os.path.join(self.disp_dir, name))).astype(np.float32) / 256.0

        h, w = disp.shape
        if h >= self.crop_h and w >= self.crop_w:
            y = py_rng.randint(0, h - self.crop_h) if self.training else (h - self.crop_h) // 2
            x = py_rng.randint(0, w - self.crop_w) if self.training else (w - self.crop_w) // 2
            left = left[y : y + self.crop_h, x : x + self.crop_w]
            right = right[y : y + self.crop_h, x : x + self.crop_w]
            disp = disp[y : y + self.crop_h, x : x + self.crop_w]

        clean_left = torch.from_numpy(left).permute(2, 0, 1).float()
        clean_right = torch.from_numpy(right).permute(2, 0, 1).float()
        clean_disp = torch.from_numpy(disp).float()

        deg_left_u8 = tensor_to_uint8_img(clean_left)
        deg_right_u8 = tensor_to_uint8_img(clean_right)
        deg_meta = {"degraded": False, "type": "clean", "level": 0, "side": "none"}

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

        clean_meta = {"degraded": False, "type": "clean", "level": 0, "side": "none"}
        return clean_left, clean_right, deg_left, deg_right, clean_disp, name, clean_meta, deg_meta


class DegradedKITTIDisparityDataset(Dataset):
    def __init__(
        self,
        root,
        files,
        crop_h=256,
        crop_w=512,
        training=False,
        degrade_prob=1.0,
        base_seed=12345,
        degrade_camera="random",
        degrade_type="random",
        degrade_severity="random",
    ):
        self.base = KITTIDisparityDataset(root, files, crop_h, crop_w, training)
        self.degrade_prob = degrade_prob
        self.base_seed = base_seed
        self.degrade_camera = degrade_camera
        self.degrade_type = degrade_type
        self.degrade_severity = degrade_severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        left, right, disp, name = self.base[idx]
        py_rng = random.Random(self.base_seed + idx)
        np_rng = np.random.default_rng(self.base_seed + idx)

        meta = {"degraded": False, "type": "clean", "level": 0, "side": "none"}

        if py_rng.random() < self.degrade_prob:
            left_u8 = tensor_to_uint8_img(left)
            right_u8 = tensor_to_uint8_img(right)

            side = choose_setting(self.degrade_camera, ["left", "right"], np_rng)
            dtype = choose_setting(self.degrade_type, ["blur", "noise", "occlusion"], np_rng)
            level = choose_setting(self.degrade_severity, [1, 2, 3, 4], np_rng)

            if side == "left":
                left_u8, info = random_degrade(left_u8, np_rng, fixed_type=dtype, fixed_level=level)
                if info["mask"] is not None:
                    disp = disp * torch.from_numpy(info["mask"]).float()
            else:
                right_u8, info = random_degrade(right_u8, np_rng, fixed_type=dtype, fixed_level=level)

            left = uint8_img_to_tensor(left_u8)
            right = uint8_img_to_tensor(right_u8)
            meta = {"degraded": True, "type": info["type"], "level": info["level"], "side": side}

        return left.float(), right.float(), disp.float(), name, meta


class KITTITestDataset(Dataset):
    def __init__(self, root, crop_h=256, crop_w=512):
        self.left_dir = os.path.join(root, "testing", "image_2")
        self.right_dir = os.path.join(root, "testing", "image_3")
        self.files = sorted(os.listdir(self.left_dir))
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        left = Image.open(os.path.join(self.left_dir, name)).convert("RGB").resize((self.crop_w, self.crop_h), Image.BILINEAR)
        right = Image.open(os.path.join(self.right_dir, name)).convert("RGB").resize((self.crop_w, self.crop_h), Image.BILINEAR)
        left = torch.from_numpy(np.array(left).astype(np.float32) / 255.0).permute(2, 0, 1).float()
        right = torch.from_numpy(np.array(right).astype(np.float32) / 255.0).permute(2, 0, 1).float()
        return left, right, name
