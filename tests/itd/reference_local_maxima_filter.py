"""Reference (eager) implementation of local maxima filters for regression testing.

This module is a frozen copy of the original scipy-based implementation prior to
the dask-image refactor. It is used exclusively in tests to verify that the new
chunked implementation produces identical results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from scipy.ndimage import label, maximum_filter, maximum_position


def _extract_treetops_reference(
    chm: np.ndarray,
    chm_max_filtered: np.ndarray,
    transform: rio.Affine,
    min_height: float,
) -> pd.DataFrame:
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)
    labeled_maxima, num_labels = label(local_maxima_mask)

    if num_labels == 0:
        return pd.DataFrame(columns=["x", "y", "height"])

    indices = np.arange(1, num_labels + 1)
    positions = maximum_position(chm, labels=labeled_maxima, index=indices)

    if num_labels == 1 and isinstance(positions[0], (int, np.integer)):
        positions = [positions]

    rows = [int(p[0]) for p in positions]
    cols = [int(p[1]) for p in positions]
    heights = chm[rows, cols]
    xs, ys = rio.transform.xy(transform, rows, cols)

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

    y, x = np.ogrid[
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
    ]
    footprint = x * x + y * y <= (window_size_pixels // 2) ** 2
    chm_max_filtered = maximum_filter(chm, footprint=footprint)

    return _extract_treetops_reference(chm, chm_max_filtered, transform, min_height)


def variable_window_filter_reference(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    crown_ratio: float = 0.10,
    crown_offset: float = 1.0,
) -> pd.DataFrame:
    chm = chm_da.values
    transform = chm_da.rio.transform()

    crown_width_meters = (chm * crown_ratio) + crown_offset
    required_windows = (crown_width_meters / spatial_resolution).astype(int)
    required_windows = np.where(
        required_windows % 2 == 0, required_windows + 1, required_windows
    )

    vw_max = np.zeros_like(chm)
    unique_windows = np.unique(required_windows)

    for w in unique_windows:
        if w <= 1:
            mask = required_windows == w
            vw_max[mask] = chm[mask]
            continue

        y, x = np.ogrid[-w // 2 : w // 2 + 1, -w // 2 : w // 2 + 1]
        footprint = x * x + y * y <= (w // 2) ** 2
        chm_max_filtered = maximum_filter(chm, footprint=footprint)
        mask = required_windows == w
        vw_max[mask] = chm_max_filtered[mask]

    return _extract_treetops_reference(chm, vw_max, transform, min_height)
