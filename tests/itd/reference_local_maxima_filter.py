"""Reference (eager) implementation of local maxima filters for regression testing.

A plain, unchunked scipy implementation of the same detection rules as the
chunked filters.  It is used exclusively in tests to verify that the chunked
implementation produces identical results.

Each treetop is the pixel of its 8-connected component nearest the component's
centroid, ties going to the smallest row, then column.  Distances are compared
exactly in integers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from scipy.ndimage import label, maximum_filter


def _maximum_filter(chm: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Maximum over the in-array pixels of the footprint.

    For a disc this equals scipy's reflect boundary; constant -inf padding
    also stays correct when the footprint is much larger than the array.
    """
    return maximum_filter(chm, footprint=footprint, mode="constant", cval=-np.inf)


def circular_footprint_reference(w: int) -> np.ndarray:
    """Disc of diameter ``w`` pixels: offsets within ``w / 2``, ``w`` x ``w``."""
    offsets = np.arange(w) - w // 2
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= (w / 2) ** 2


def _extract_treetops_reference(
    chm: np.ndarray,
    chm_max_filtered: np.ndarray,
    transform: rio.Affine,
    min_height: float,
) -> pd.DataFrame:
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)
    labeled_maxima, num_labels = label(local_maxima_mask, structure=np.ones((3, 3)))

    if num_labels == 0:
        return pd.DataFrame(columns=["x", "y", "height"])

    rows_out = []
    cols_out = []
    heights = []
    for lbl in range(1, num_labels + 1):
        r_arr, c_arr = np.where(labeled_maxima == lbl)
        n = len(r_arr)
        r_sum, c_sum = int(r_arr.sum()), int(c_arr.sum())
        best = min(
            zip(r_arr.tolist(), c_arr.tolist()),
            key=lambda rc: ((rc[0] * n - r_sum) ** 2 + (rc[1] * n - c_sum) ** 2, rc),
        )
        rows_out.append(best[0])
        cols_out.append(best[1])
        heights.append(float(chm[best]))

    xs, ys = rio.transform.xy(transform, rows_out, cols_out)

    return pd.DataFrame({"x": xs, "y": ys, "height": heights})


def fixed_window_filter_reference(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    window_size_meters: float = 3.0,
) -> pd.DataFrame:
    chm = chm_da.values
    transform = chm_da.rio.transform()

    window_size_pixels = int(window_size_meters / spatial_resolution)
    if window_size_pixels % 2 == 0:
        window_size_pixels += 1
    if window_size_pixels < 3:
        window_size_pixels = 3

    footprint = circular_footprint_reference(window_size_pixels)
    chm_max_filtered = _maximum_filter(chm, footprint)

    return _extract_treetops_reference(chm, chm_max_filtered, transform, min_height)


def variable_window_filter_reference(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    crown_ratio: float = 0.05,
    crown_offset: float = 3.0,
) -> pd.DataFrame:
    chm = chm_da.values
    transform = chm_da.rio.transform()

    crown_width_meters = (chm * crown_ratio) + crown_offset
    required_windows = (crown_width_meters / spatial_resolution).astype(int)
    required_windows = np.where(
        required_windows % 2 == 0, required_windows + 1, required_windows
    )
    required_windows = np.maximum(required_windows, 3)

    vw_max = np.zeros_like(chm)
    unique_windows = np.unique(required_windows)

    for w in unique_windows:
        footprint = circular_footprint_reference(int(w))
        chm_max_filtered = _maximum_filter(chm, footprint)
        mask = required_windows == w
        vw_max[mask] = chm_max_filtered[mask]

    return _extract_treetops_reference(chm, vw_max, transform, min_height)
