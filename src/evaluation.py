"""Evaluation metrics for comparing colorized images with ground truth."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rgb_distance(pred, true):
    """Compute the mean per-pixel Euclidean distance in RGB space."""
    return float(np.mean(np.linalg.norm(pred - true, axis=-1)))


def psnr(pred, true):
    """Compute Peak Signal-to-Noise Ratio between two RGB images."""
    return float(peak_signal_noise_ratio(true, pred, data_range=1.0))


def ssim(pred, true):
    """Compute Structural Similarity Index between two RGB images."""
    return float(
        structural_similarity(
            true,
            pred,
            channel_axis=-1,
            data_range=1.0
        )
    )


def all_metrics(pred, true):
    """Compute all evaluation metrics and return them as a dictionary."""
    return {
        "rgb_dist": rgb_distance(pred, true),
        "psnr": psnr(pred, true),
        "ssim": ssim(pred, true)
    }