"""Step 4: SLIC superpixels computed ONLY from the Y (grayscale) channel."""
import numpy as np
from skimage.segmentation import slic


def segment(y, n_segments=400, compactness=0.1):
    """Return an integer label map (H,W) with labels 0..K-1.

    We segment on Y alone because at test time Y is all we have.
    """
    labels = slic(
        y,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        start_label=0
    )

    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(y.shape)


def region_adjacency(labels):
    """Sorted list of (i,j) pairs, i<j, of superpixels that touch (4-connectivity)."""
    pairs = set()

    a, b = labels[:, :-1], labels[:, 1:]
    m = a != b
    pairs.update(
        zip(
            np.minimum(a[m], b[m]).tolist(),
            np.maximum(a[m], b[m]).tolist()
        )
    )

    a, b = labels[:-1, :], labels[1:, :]
    m = a != b
    pairs.update(
        zip(
            np.minimum(a[m], b[m]).tolist(),
            np.maximum(a[m], b[m]).tolist()
        )
    )

    return sorted(pairs)