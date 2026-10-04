"""Color reconstruction using baseline, SVR, and SVR + ICM modes."""

import numpy as np
from tqdm import tqdm

from . import preprocessing as pp
from .segmentation import segment, region_adjacency
from .features import extract_features, region_means
from .smoothing import build_neighbors, icm


def build_dataset(paths, width=256, n_segments=400, extra=False):
    """Build FFT features and corresponding U/V chrominance targets."""
    Xs, Us, Vs = [], [], []

    for p in tqdm(paths, desc="building training set"):
        rgb = pp.load_rgb(p, width)

        if pp.is_grayscale(rgb):
            continue

        y, u, v = pp.to_yuv(rgb)
        lab = segment(y, n_segments)

        Xs.append(extract_features(y, lab, extra=extra))
        Us.append(region_means(u, lab))
        Vs.append(region_means(v, lab))

    return np.vstack(Xs), np.concatenate(Us), np.concatenate(Vs)


def mean_chroma_baseline(U, V):
    """Compute the average U and V values used by the baseline colorizer."""
    return float(np.mean(U)), float(np.mean(V))


def colorize_y(
    y,
    model,
    mode="svr_icm",
    n_segments=400,
    extra=False,
    sigma=0.1,
    gamma=2.0,
    percentile=70,
    baseline_uv=(0.0, 0.0)
):
    """Convert a luminance image into an RGB colorized image."""
    if mode == "baseline":
        u = np.full_like(y, baseline_uv[0])
        v = np.full_like(y, baseline_uv[1])

        return pp.to_rgb(y, u, v), {}

    lab = segment(y, n_segments)
    X = extract_features(y, lab, extra=extra)

    mu_u, mu_v = model.predict(X)

    if mode == "svr_icm":
        Xs = model.scaler.transform(X)

        nb = build_neighbors(
            region_adjacency(lab),
            Xs,
            lab.max() + 1,
            percentile
        )

        mu_u = icm(mu_u, nb, sigma, gamma)
        mu_v = icm(mu_v, nb, sigma, gamma)

    u, v = mu_u[lab], mu_v[lab]

    return pp.to_rgb(y, u, v), {"labels": lab}