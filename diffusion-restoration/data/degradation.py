import cv2
import numpy as np


def apply_blur(img, level):
    ks = [3, 5, 9, 15][max(1, min(level, 4)) - 1]
    return cv2.GaussianBlur(img, (ks, ks), 0)


def apply_noise(img, level, rng):
    sigma = [5, 10, 20, 30][max(1, min(level, 4)) - 1]
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_occlusion(img, level, rng):
    h, w, _ = img.shape
    frac = [0.1, 0.2, 0.3, 0.4][max(1, min(level, 4)) - 1]
    size = max(1, int(min(h, w) * frac))
    x = int(rng.integers(0, max(1, w - size + 1)))
    y = int(rng.integers(0, max(1, h - size + 1)))
    out = img.copy()
    out[y : y + size, x : x + size] = 0
    return out


def choose_setting(value, options, rng):
    if value == "random":
        return options[int(rng.integers(0, len(options)))]
    return value


def degrade(img, dtype="blur", level=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if dtype == "blur":
        return apply_blur(img, level), None
    if dtype == "noise":
        return apply_noise(img, level, rng), None
    if dtype == "occlusion":
        deg = apply_occlusion(img, level, rng)
        mask = (deg.sum(axis=2) > 0).astype(np.float32)
        return deg, mask
    return img, None


def random_degrade(img, rng, fixed_type=None, fixed_level=None):
    dtype = fixed_type if fixed_type not in (None, "random") else ["blur", "noise", "occlusion"][int(rng.integers(0, 3))]
    level = fixed_level if fixed_level not in (None, "random") else int(rng.integers(1, 5))
    degraded, mask = degrade(img, dtype, level, rng=rng)
    return degraded, {"type": dtype, "level": level, "mask": mask}
