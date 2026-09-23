"""Cell-by-cell reference for ``dalponte2016``, written straight from the spec.

Slow and loop-based on purpose: it is the oracle the vectorized
implementation is checked against.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio as rio


def dalponte2016_reference(
    chm: np.ndarray,
    seeds: list[tuple[int, int]],
    transform: rio.Affine,
    min_height: float,
    max_height: float | None,
    min_relative_height: float,
    min_relative_crown_height: float,
    max_crown_radius: float,
) -> np.ndarray:
    nrows, ncols = chm.shape
    labels = np.zeros((nrows, ncols), dtype=np.int32)
    top = {}
    cells = {}
    for i, (r, c) in enumerate(seeds):
        k = i + 1
        labels[r, c] = k
        top[k] = float(chm[r, c])
        cells[k] = [float(chm[r, c])]

    def in_range(h: float) -> bool:
        if not math.isfinite(h) or h < min_height:
            return False
        return max_height is None or h <= max_height

    grows = {k: in_range(float(chm[r, c])) for k, (r, c) in enumerate(seeds, 1)}

    while True:
        means = {k: sum(v) / len(v) for k, v in cells.items()}
        added = {}
        for r in range(nrows):
            for c in range(ncols):
                if labels[r, c] != 0:
                    continue
                h = float(chm[r, c])
                if not in_range(h):
                    continue
                best = None
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < nrows and 0 <= cc < ncols):
                        continue
                    k = int(labels[rr, cc])
                    if k == 0 or not grows[k]:
                        continue
                    sr, sc = seeds[k - 1]
                    dx = (c - sc) * transform.a + (r - sr) * transform.b
                    dy = (c - sc) * transform.d + (r - sr) * transform.e
                    if not (
                        h >= min_relative_height * top[k]
                        and h >= min_relative_crown_height * means[k]
                        and h <= 1.05 * top[k]
                        and dx * dx + dy * dy <= max_crown_radius**2
                    ):
                        continue
                    if (
                        best is None
                        or top[k] > top[best]
                        or (top[k] == top[best] and k < best)
                    ):
                        best = k
                if best is not None:
                    added[(r, c)] = best
        if not added:
            return labels
        for (r, c), k in added.items():
            labels[r, c] = k
            cells[k].append(float(chm[r, c]))
