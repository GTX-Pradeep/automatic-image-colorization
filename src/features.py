import numpy as np
from scipy import ndimage

PATCH = 10


def centroids(labels):
    """Compute the centroid of each superpixel."""
    k = labels.max() + 1
    cs = ndimage.center_of_mass(
        np.ones_like(labels),
        labels,
        index=np.arange(k)
    )
    return np.array(cs)


def extract_features(y, labels, patch=PATCH, extra=False):
    """Extract FFT-based features from patches centered on superpixels."""
    H, W = y.shape
    half = patch // 2

    # Reflect padding prevents boundary patches from becoming incomplete.
    ypad = np.pad(y, half, mode="reflect")

    cs = centroids(labels)
    feats = []

    for r, c in cs:
        r, c = int(round(r)), int(round(c))
        p = ypad[r:r + patch, c:c + patch]

        # FFT magnitude captures the local frequency and texture information.
        feats.append(np.abs(np.fft.fft2(p)).ravel())

    X = np.array(feats)

    if extra:
        idx = np.arange(labels.max() + 1)
        mean = ndimage.mean(y, labels, idx)
        std = ndimage.standard_deviation(y, labels, idx)
        pos = np.column_stack([cs[:, 0] / H, cs[:, 1] / W])

        X = np.hstack([
            X,
            mean[:, None],
            std[:, None],
            pos
        ])

    return X


def region_means(channel, labels):
    """Compute the mean channel value within each superpixel."""
    return np.asarray(
        ndimage.mean(
            channel,
            labels,
            np.arange(labels.max() + 1)
        )
    )