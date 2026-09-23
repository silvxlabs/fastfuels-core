"""The pre-frontier ``_segment``, kept verbatim as a test oracle.

It rescans the whole grid every growth step. The frontier-based ``_segment``
in ``fastfuels_core.itd.crown_segmentation`` must match it exactly.
"""

from __future__ import annotations

import numpy as np
import rasterio as rio

from fastfuels_core.itd.crown_segmentation import MAX_RELATIVE_TO_TREETOP

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def segment_full_scan(
    chm: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_labels: np.ndarray,
    *,
    min_height: float,
    max_height: float,
    min_relative_height: float,
    min_relative_crown_height: float,
    max_crown_radius: float,
    transform: rio.Affine,
) -> np.ndarray:
    """Region-grow crowns by rescanning the whole grid every step.

    ``seed_rows`` / ``seed_cols`` index into ``chm``; ``seed_labels`` are the
    labels written for each seed and decide ties.
    """
    nrows, ncols = chm.shape
    labels = np.zeros((nrows, ncols), dtype=np.int32)
    if len(seed_labels) == 0 or chm.size == 0:
        return labels

    chm = chm.astype(np.float64, copy=False)
    labels[seed_rows, seed_cols] = seed_labels

    # Per-label crown state, indexed by label.
    n = int(seed_labels.max()) + 1
    top = np.full(n, np.nan)
    top[seed_labels] = chm[seed_rows, seed_cols]
    seed_row_of = np.zeros(n, dtype=np.int64)
    seed_col_of = np.zeros(n, dtype=np.int64)
    seed_row_of[seed_labels] = seed_rows
    seed_col_of[seed_labels] = seed_cols
    total = np.zeros(n)
    total[seed_labels] = top[seed_labels]
    count = np.zeros(n)
    count[seed_labels] = 1.0

    with np.errstate(invalid="ignore"):
        in_range = np.isfinite(chm) & (chm >= min_height) & (chm <= max_height)
    # A crown whose treetop cell fails the height range stays one cell.
    grows = np.zeros(n, dtype=bool)
    grows[seed_labels] = in_range[seed_rows, seed_cols]
    if not grows.any():
        return labels

    a, b, _, d, e, _ = transform[:6]
    radius_sq = max_crown_radius * max_crown_radius
    flat_chm = chm.ravel()

    while True:
        mean = total / np.where(count > 0, count, 1.0)
        free = (labels == 0) & in_range

        best_label = np.zeros(nrows * ncols, dtype=np.int32)
        best_top = np.full(nrows * ncols, -np.inf)

        for dr, dc in _NEIGHBOURS:
            neighbour = _shift(labels, dr, dc)
            cand = np.flatnonzero(free & (neighbour > 0))
            if cand.size == 0:
                continue
            k = neighbour.ravel()[cand]
            h = flat_chm[cand]
            r, c = np.divmod(cand, ncols)
            drow = (r - seed_row_of[k]).astype(np.float64)
            dcol = (c - seed_col_of[k]).astype(np.float64)
            dx = dcol * a + drow * b
            dy = dcol * d + drow * e
            ok = (
                grows[k]
                & (h >= min_relative_height * top[k])
                & (h >= min_relative_crown_height * mean[k])
                & (h <= MAX_RELATIVE_TO_TREETOP * top[k])
                & (dx * dx + dy * dy <= radius_sq)
            )
            cand, k = cand[ok], k[ok]
            better = (top[k] > best_top[cand]) | (
                (top[k] == best_top[cand]) & (k < best_label[cand])
            )
            cand, k = cand[better], k[better]
            best_label[cand] = k
            best_top[cand] = top[k]

        added = np.flatnonzero(best_label)
        if added.size == 0:
            return labels

        k = best_label[added]
        labels.ravel()[added] = k
        total += np.bincount(k, weights=flat_chm[added], minlength=n)
        count += np.bincount(k, minlength=n)


def _shift(labels: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return ``out`` with ``out[r, c] = labels[r + dr, c + dc]`` (0 off-grid)."""
    out = np.zeros_like(labels)
    nrows, ncols = labels.shape
    dst_r = slice(max(0, -dr), nrows - max(0, dr))
    src_r = slice(max(0, dr), nrows - max(0, -dr))
    dst_c = slice(max(0, -dc), ncols - max(0, dc))
    src_c = slice(max(0, dc), ncols - max(0, -dc))
    out[dst_r, dst_c] = labels[src_r, src_c]
    return out
