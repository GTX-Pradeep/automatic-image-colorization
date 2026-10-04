"""Step 1-2: loading, resizing, RGB <-> YUV, and train/test split."""
import os
import random
import numpy as np
from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder):
    """Recursively collect all image paths under `folder` (sorted, reproducible)."""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def load_rgb(path, width=256):
    """Load any image as float RGB in [0,1], resized to a constant width (aspect kept)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_h = max(1, int(round(h * width / w)))
    img = img.resize((width, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def is_grayscale(rgb, tol=0.02):
    """True if the picture is (almost) already black & white -> paper discards these."""
    return np.abs(rgb[..., 0] - rgb[..., 1]).mean() < tol and np.abs(rgb[..., 1] - rgb[..., 2]).mean() < tol


def to_yuv(rgb):
    """Returns Y (H,W), U (H,W), V (H,W)."""
    yuv = rgb2yuv(rgb)
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]


def to_rgb(y, u, v):
    """Combine Y,U,V back into RGB in [0,1]."""
    return np.clip(yuv2rgb(np.stack([y, u, v], axis=-1)), 0, 1)


def to_gray_rgb(rgb):
    """What a 'black & white photo' looks like: Y only, U=V=0."""
    y, _, _ = to_yuv(rgb)
    return to_rgb(y, np.zeros_like(y), np.zeros_like(y))


def split_paths(paths, n_train, n_test, seed=42):
    """Reproducible random split. Document the seed in the report!"""
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    return paths[:n_train], paths[n_train:n_train + n_test]