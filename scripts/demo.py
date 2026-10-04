"""Generate a visual comparison of the colorization pipeline."""

import os
import sys
import joblib

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src import preprocessing as pp
from src.colorization import colorize_y


img = sys.argv[1]

ck = joblib.load(
    sys.argv[2]
    if len(sys.argv) > 2
    else "models/colorizer.joblib"
)

rgb = pp.load_rgb(img, 256)

y, _, _ = pp.to_yuv(rgb)

kw = {
    "n_segments": ck["n_segments"],
    "extra": ck["extra"],
    "sigma": ck["sigma"],
    "gamma": ck["gamma"]
}

svr, info = colorize_y(
    y,
    ck["model"],
    "svr",
    **kw
)

icm, _ = colorize_y(
    y,
    ck["model"],
    "svr_icm",
    **kw
)

gray = pp.to_gray_rgb(rgb)

fig, ax = plt.subplots(
    1,
    4,
    figsize=(16, 4)
)

comparisons = [
    ("Grayscale input", gray),
    (
        "SLIC superpixels",
        mark_boundaries(
            gray,
            info["labels"],
            color=(1, 1, 0)
        )
    ),
    ("SVR only", svr),
    ("SVR + MRF/ICM", icm)
]

for a_, title, image in zip(ax, comparisons):
    a_.imshow(image)
    a_.set_title(title)
    a_.axis("off")

plt.tight_layout()
plt.savefig(
    "demo_output.png",
    dpi=120
)

print("saved demo_output.png")

plt.show()