"""MRF-based chrominance smoothing using Iterated Conditional Modes."""

import numpy as np


def build_neighbors(pairs, X_scaled, k, percentile=70):
    """Build superpixel neighborhoods using feature-distance filtering."""
    nb = [[] for _ in range(k)]

    if len(pairs) == 0:
        return nb

    pairs = np.array(pairs)

    d = np.linalg.norm(
        X_scaled[pairs[:, 0]] - X_scaled[pairs[:, 1]],
        axis=1
    )

    threshold = np.percentile(d, percentile)

    for i, j in pairs[d <= threshold]:
        nb[i].append(j)
        nb[j].append(i)

    return nb


def icm(mu, neighbors, sigma=0.1, gamma=2.0, max_iter=50, tol=1e-5):
    """Refine chrominance predictions using Iterated Conditional Modes."""
    c = mu.copy()
    a = 1.0 / sigma**2

    for _ in range(max_iter):
        delta = 0.0

        for i in range(len(c)):
            nb = neighbors[i]

            if not nb:
                continue

            new = (
                a * mu[i] +
                2 * gamma * c[nb].sum()
            ) / (
                a + 2 * gamma * len(nb)
            )

            delta = max(delta, abs(new - c[i]))
            c[i] = new

        if delta < tol:
            break

    return c