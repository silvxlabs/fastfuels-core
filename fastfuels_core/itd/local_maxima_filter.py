from __future__ import annotations

import dask
import dask.array as da
import dask.dataframe as dd
import dask_image.ndfilters
import dask_image.ndmeasure
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr

DEFAULT_CHUNK_SIZE = 2048

_CANDIDATE_COLUMNS = ["label", "row", "col", "height"]

_OUTPUT_META = pd.DataFrame(
    {
        "x": pd.Series(dtype="float64"),
        "y": pd.Series(dtype="float64"),
        "height": pd.Series(dtype="float64"),
    }
)

_CANDIDATE_META = pd.DataFrame(
    {
        "label": pd.Series(dtype="int64"),
        "row": pd.Series(dtype="int64"),
        "col": pd.Series(dtype="int64"),
        "height": pd.Series(dtype="float64"),
    }
)


def _prepare_chm(chm_da: xr.DataArray) -> tuple[da.Array, rio.Affine]:
    """Return the CHM as a dask array and its affine transform."""
    if chm_da.ndim != 2:
        raise ValueError("CHM must be a 2D DataArray")

    transform = chm_da.rio.transform()

    if isinstance(chm_da.data, da.Array):
        return chm_da.data, transform

    return da.from_array(chm_da.values, chunks=chm_da.shape), transform


def _build_circular_footprint(window_size_pixels: int) -> np.ndarray:
    y, x = np.ogrid[
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
    ]
    return x * x + y * y <= (window_size_pixels // 2) ** 2


def _extract_block_candidates(
    chm_block: np.ndarray,
    labeled_block: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> pd.DataFrame:
    """Per-chunk extraction: find the max-value position per label within this chunk.

    Uses np.argmax for position selection, which deterministically picks the
    first occurrence in raster-scan order for tied values. This matches
    dask_image.ndmeasure.maximum_position's behavior and guarantees identical
    results regardless of chunk layout.
    """
    unique_labels = np.unique(labeled_block)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return _CANDIDATE_META.copy()

    records = []
    for lbl in unique_labels:
        mask = labeled_block == lbl
        flat_indices = np.flatnonzero(mask)
        values = chm_block.flat[flat_indices]
        best = np.argmax(values)
        local_row, local_col = np.unravel_index(flat_indices[best], chm_block.shape)

        records.append(
            {
                "label": int(lbl),
                "row": int(local_row) + row_offset,
                "col": int(local_col) + col_offset,
                "height": float(chm_block[int(local_row), int(local_col)]),
            }
        )

    return pd.DataFrame(records, columns=_CANDIDATE_COLUMNS)


def _deduplicate_and_convert(
    candidates: pd.DataFrame,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Deduplicate cross-chunk labels and convert pixel coords to spatial coords.

    For labels that span chunk boundaries, keeps the candidate with the highest
    height value (the true maximum position). Then converts row/col pixel
    coordinates to x/y spatial coordinates via the rasterio transform.
    """
    if candidates.empty:
        return _OUTPUT_META.copy()

    best = candidates.sort_values(
        ["height", "row", "col"], ascending=[False, True, True]
    ).drop_duplicates(subset=["label"], keep="first")

    rows = best["row"].values
    cols = best["col"].values
    xs, ys = rio.transform.xy(transform, rows.tolist(), cols.tolist())
    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=np.float64),
            "y": np.asarray(ys, dtype=np.float64),
            "height": best["height"].values.astype(np.float64),
        }
    )


def _extract_treetops(
    chm: da.Array,
    chm_max_filtered: da.Array,
    transform: rio.Affine,
    min_height: float,
) -> dd.DataFrame:
    """Extract treetop spatial points from a filtered CHM. Fully lazy."""
    # Step 1: lazy boolean mask
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)

    # Step 2: globally-consistent connected-component labeling (lazy)
    labeled, _num_labels = dask_image.ndmeasure.label(local_maxima_mask)

    # Step 3: per-chunk candidate extraction (lazy via delayed)
    row_starts = np.cumsum((0,) + chm.chunks[0][:-1])
    col_starts = np.cumsum((0,) + chm.chunks[1][:-1])
    offsets = [
        (int(row_starts[i]), int(col_starts[j]))
        for i in range(len(chm.chunks[0]))
        for j in range(len(chm.chunks[1]))
    ]

    chm_blocks = chm.to_delayed().ravel()
    labeled_blocks = labeled.to_delayed().ravel()

    partitions = [
        dd.from_delayed(
            dask.delayed(_extract_block_candidates)(cb, lb, ro, co),
            meta=_CANDIDATE_META,
        )
        for cb, lb, (ro, co) in zip(chm_blocks, labeled_blocks, offsets)
    ]

    candidates = dd.concat(partitions)

    # Step 4: deduplicate cross-chunk labels and convert to spatial coords (lazy).
    # The candidate DataFrame is small (one row per label per chunk), so
    # collecting it into a single delayed function is safe for memory.
    delayed_result = dask.delayed(_deduplicate_and_convert)(candidates, transform)
    return dd.from_delayed(delayed_result, meta=_OUTPUT_META)


def variable_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    crown_ratio: float = 0.10,
    crown_offset: float = 1.0,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Variable Window Filter (VWF).

    Calculates the search window size dynamically using a linear allometric
    relationship: Crown_Width_m = (Height_m * crown_ratio) + crown_offset.

    Algorithm Validation & Scientific Context:
    - Popescu & Wynne (2004): Validated dynamic window sizing based on allometry.
      https://doi.org/10.14358/PERS.70.5.589
    - Chen et al. (2006): Validated VWF applied specifically to continuous CHMs.
      https://doi.org/10.14358/PERS.72.8.923

    Args:
        chm_da (xr.DataArray): The Canopy Height Model data array.
        min_height (float): Minimum height threshold in CHM units (meters).
        spatial_resolution (float): Pixel size of the CHM in meters (e.g., 0.5, 1.0).
        crown_ratio (float): The multiplier for tree height to estimate crown width.
            Defaults to 0.10 (10%).
        crown_offset (float): The base crown width in meters. Defaults to 1.0m.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    if min_height < 0:
        raise ValueError("min_height cannot be negative")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution must be positive")

    chm, transform = _prepare_chm(chm_da)

    # Derive the set of unique window sizes from CHM value range.
    # Two scalar reductions (chunk-by-chunk) avoid materializing the full array.
    min_h, max_h = da.compute(chm.min(), chm.max())
    min_w = int((float(min_h) * crown_ratio + crown_offset) / spatial_resolution)
    max_w = int((float(max_h) * crown_ratio + crown_offset) / spatial_resolution)
    if min_w % 2 == 0:
        min_w -= 1
    if max_w % 2 == 0:
        max_w += 1
    min_w = max(min_w, 1)
    unique_windows = range(min_w, max_w + 2, 2)

    # Per-pixel window size (lazy)
    crown_width_meters = (chm * crown_ratio) + crown_offset
    required_windows = (crown_width_meters / spatial_resolution).astype(int)
    required_windows = da.where(
        required_windows % 2 == 0, required_windows + 1, required_windows
    )

    vw_max = da.zeros_like(chm)
    for w in unique_windows:
        w = int(w)
        if w <= 1:
            vw_max = da.where(required_windows == w, chm, vw_max)
            continue
        footprint = _build_circular_footprint(w)
        filtered = dask_image.ndfilters.maximum_filter(chm, footprint=footprint)
        vw_max = da.where(required_windows == w, filtered, vw_max)

    return _extract_treetops(chm, vw_max, transform, min_height)


def fixed_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    window_size_meters: float = 3.0,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Fixed Window Local Maxima (FW-LM) filter.

    Applies a static, circular search window across the entire Canopy Height Model
    to identify local maxima.

    Algorithm Validation & Scientific Context:
    - Wulder et al. (2000): The foundational paper validating the use of fixed-size
      optical/spatial windows for extracting tree locations from high-resolution data.
      https://doi.org/10.1016/S0034-4257(00)00103-6
    - Chen et al. (2006): Used this exact fixed-window methodology as the baseline
      to compare against Variable Window Filters, demonstrating that fixed windows
      are prone to high omission errors in mixed stands.
      https://doi.org/10.14358/PERS.72.8.923

    Args:
        chm_da (xr.DataArray): The Canopy Height Model data array.
        min_height (float): Minimum height threshold in CHM units (meters).
        spatial_resolution (float): Pixel size of the CHM in meters (e.g., 0.5, 1.0).
        window_size_meters (float): The fixed diameter of the search window in meters.
            Defaults to 3.0m.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    if min_height < 0:
        raise ValueError("min_height cannot be negative")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution must be positive")

    chm, transform = _prepare_chm(chm_da)

    window_size_pixels = int(window_size_meters / spatial_resolution)
    if window_size_pixels % 2 == 0:
        window_size_pixels += 1
    if window_size_pixels < 3:
        window_size_pixels = 3

    footprint = _build_circular_footprint(window_size_pixels)
    chm_max_filtered = dask_image.ndfilters.maximum_filter(chm, footprint=footprint)

    return _extract_treetops(chm, chm_max_filtered, transform, min_height)
