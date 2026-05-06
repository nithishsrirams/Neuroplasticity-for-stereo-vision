import cv2
import numpy as np


def tensor_to_uint8_img(t):
    return (t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


def uint8_img_to_tensor(img):
    import torch

    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def apply_blur(img, level):
    ks = [3, 5, 9, 15][max(1, min(level, 4)) - 1]
    return cv2.GaussianBlur(img, (ks, ks), 0)


def apply_noise(img, level, rng):
    sigma = [10, 20, 40, 80][max(1, min(level, 4)) - 1]
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def apply_occlusion(img, level, rng):
    if level >= 4:
        return np.zeros_like(img)
    h, w, _ = img.shape
    size = max(1, int(min(h, w) * [0.1, 0.2, 0.4][max(1, min(level, 3)) - 1]))
    x = int(rng.integers(0, max(1, w - size + 1)))
    y = int(rng.integers(0, max(1, h - size + 1)))
    out = img.copy()
    out[y : y + size, x : x + size] = 0
    return out


def degrade(img, dtype, level, rng):
    if dtype == "blur":
        return apply_blur(img, level), None
    if dtype == "noise":
        return apply_noise(img, level, rng), None
    if dtype == "occlusion":
        deg = apply_occlusion(img, level, rng)
        mask = (deg.sum(axis=2) > 0).astype(np.float32)
        return deg, mask
    return img, None
