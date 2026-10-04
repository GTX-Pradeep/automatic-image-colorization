"""Train and evaluate the colorization pipeline on a selected train/test split."""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from skimage.io import imsave

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src import preprocessing as pp
from src.colorization import (
    build_dataset,
    colorize_y,
    mean_chroma_baseline
)
from src.model import ChromaSVR
from src.evaluation import all_metrics


ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data/raw")
ap.add_argument("--n_train", type=int, default=150)
ap.add_argument("--n_test", type=int, default=40)
ap.add_argument("--width", type=int, default=256)
ap.add_argument("--n_segments", type=int, default=400)
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--epsilon", type=float, default=0.01)
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument(
    "--extra",
    action="store_true",
    help="Use additional brightness and position features."
)
ap.add_argument("--tag", default="")

a = ap.parse_args()


paths = pp.list_images(a.data)
train_p, test_p = pp.split_paths(
    paths,
    a.n_train,
    a.n_test
)

print(
    f"{len(paths)} images found -> "
    f"{len(train_p)} train / {len(test_p)} test"
)


t0 = time.time()

X, U, V = build_dataset(
    train_p,
    a.width,
    a.n_segments,
    a.extra
)

print("training matrix:", X.shape)

model = ChromaSVR(
    C=a.C,
    epsilon=a.epsilon
).fit(X, U, V)

print(f"trained in {time.time() - t0:.0f}s")


base_uv = mean_chroma_baseline(U, V)

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "extra": a.extra,
        "n_segments": a.n_segments,
        "baseline_uv": base_uv,
        "sigma": a.sigma,
        "gamma": a.gamma
    },
    f"models/colorizer{a.tag}.joblib"
)


rows = []

out_dir = "outputs/improved" if a.extra else "outputs/svr"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("outputs/baseline", exist_ok=True)


for k, p in enumerate(test_p):
    rgb = pp.load_rgb(p, a.width)

    if pp.is_grayscale(rgb):
        continue

    y, _, _ = pp.to_yuv(rgb)

    res = {}

    for mode in ["baseline", "svr", "svr_icm"]:
        pred, _ = colorize_y(
            y,
            model,
            mode,
            a.n_segments,
            a.extra,
            a.sigma,
            a.gamma,
            baseline_uv=base_uv
        )

        res[mode] = pred

        m = all_metrics(pred, rgb)
        m.update(
            image=os.path.basename(p),
            mode=mode
        )
        rows.append(m)

    gray = pp.to_gray_rgb(rgb)

    m = all_metrics(gray, rgb)
    m.update(
        image=os.path.basename(p),
        mode="grayscale"
    )
    rows.append(m)

    if k < 12:
        fig, ax = plt.subplots(
            1,
            5,
            figsize=(17, 3.6)
        )

        comparisons = [
            ("Original", rgb),
            ("Grayscale input", gray),
            ("Baseline (mean colour)", res["baseline"]),
            ("SVR only", res["svr"]),
            ("SVR + MRF/ICM", res["svr_icm"])
        ]

        for axi, (title, image) in zip(ax, comparisons):
            axi.imshow(image)
            axi.set_title(title, fontsize=9)
            axi.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{out_dir}/compare_{k:02d}{a.tag}.png",
            dpi=110
        )

        plt.close()

        imsave(
            f"outputs/baseline/baseline_{k:02d}.png",
            (res["baseline"] * 255).astype(np.uint8),
            check_contrast=False
        )


df = pd.DataFrame(rows)

df.to_csv(
    f"outputs/metrics_per_image{a.tag}.csv",
    index=False
)

summary = (
    df.groupby("mode")[["rgb_dist", "psnr", "ssim"]]
    .mean()
    .reindex([
        "grayscale",
        "baseline",
        "svr",
        "svr_icm"
    ])
)

summary.to_csv(
    f"outputs/metrics_summary{a.tag}.csv"
)

print(
    "\n=== Mean metrics over test set "
    "(rgb_dist: lower is better) ===\n",
    summary.round(4)
)