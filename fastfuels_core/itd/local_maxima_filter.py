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

_CANDIDATE_COLUMNS = [
    "label",
    "height",
    "centroid_row_sum",
    "centroid_col_sum",
    "centroid_count",
    "is_boundary",
]

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
        "height": pd.Series(dtype="float64"),
        "centroid_row_sum": pd.Series(dtype="float64"),
        "centroid_col_sum": pd.Series(dtype="float64"),
        "centroid_count": pd.Series(dtype="int64"),
        "is_boundary": pd.Series(dtype="bool"),
    }
)


def _prepare_chm(chm_da: xr.DataArray) -> tuple[da.Array, rio.Affine]:
    """Return the CHM as a chunked dask array and its affine transform.

    If the input is already dask-backed it is returned as-is. Otherwise
    the numpy array is wrapped and chunked to ``DEFAULT_CHUNK_SIZE`` so
    that downstream operations run in parallel.
    """
    if chm_da.ndim != 2:
        raise ValueError("CHM must be a 2D DataArray")

    transform = chm_da.rio.transform()

    if isinstance(chm_da.data, da.Array):
        return chm_da.data, transform

    return da.from_array(chm_da.values, chunks=DEFAULT_CHUNK_SIZE), transform


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
    """Per-chunk extraction: compute centroid position per label within this chunk.

    Within the local-maxima mask every pixel of a connected component has the
    same CHM value (a pixel whose neighbor is higher cannot satisfy
    ``chm == max_filter``).  The centroid of all component pixels is therefore
    a natural, chunk-layout-invariant representative position.

    Centroid components (row_sum, col_sum, count) are additive across chunks,
    enabling exact global-centroid computation during boundary deduplication
    without materializing all pixel coordinates.

    Labels touching the block edge are flagged ``is_boundary=True`` because
    they may span adjacent chunks and require cross-chunk deduplication.
    Interior labels are guaranteed complete within this single block.
    """
    unique_labels = np.unique(labeled_block)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return _CANDIDATE_META.copy()

    nrows, ncols = labeled_block.shape
    edge_mask = np.zeros((nrows, ncols), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    records = []
    for lbl in unique_labels:
        mask = labeled_block == lbl
        rows_arr, cols_arr = np.where(mask)
        n_pixels = len(rows_arr)

        records.append(
            {
                "label": int(lbl),
                "height": float(chm_block[rows_arr[0], cols_arr[0]]),
                "centroid_row_sum": float(np.sum(rows_arr)) + row_offset * n_pixels,
                "centroid_col_sum": float(np.sum(cols_arr)) + col_offset * n_pixels,
                "centroid_count": n_pixels,
                "is_boundary": bool(np.any(mask & edge_mask)),
            }
        )

    return pd.DataFrame(records, columns=_CANDIDATE_COLUMNS)


def _candidates_to_spatial(
    candidates: pd.DataFrame,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Convert centroid pixel coordinates to x/y spatial coordinates."""
    if candidates.empty:
        return _OUTPUT_META.copy()

    rows = candidates["centroid_row_sum"].values / candidates["centroid_count"].values
    cols = candidates["centroid_col_sum"].values / candidates["centroid_count"].values
    xs, ys = rio.transform.xy(transform, rows.tolist(), cols.tolist())
    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=np.float64),
            "y": np.asarray(ys, dtype=np.float64),
            "height": candidates["height"].values.astype(np.float64),
        }
    )


def _process_interior_candidates(
    partition: pd.DataFrame,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Convert interior (non-boundary) candidates directly to spatial coords.

    Interior labels exist in exactly one chunk, so no cross-chunk
    deduplication is needed — convert and emit immediately.
    """
    interior = partition[~partition["is_boundary"]]
    return _candidates_to_spatial(interior, transform)


def _deduplicate_boundary_candidates(
    candidates: pd.DataFrame,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Deduplicate boundary labels and convert to spatial coords.

    For labels that span chunk boundaries, combines centroid components
    (row_sum, col_sum, count) across chunks to compute the exact global
    centroid.  This is chunk-layout invariant because addition is
    associative and commutative.
    """
    if candidates.empty:
        return _OUTPUT_META.copy()

    combined = (
        candidates.groupby("label")
        .agg(
            {
                "height": "first",
                "centroid_row_sum": "sum",
                "centroid_col_sum": "sum",
                "centroid_count": "sum",
            }
        )
        .reset_index()
    )

    return _candidates_to_spatial(combined, transform)


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

    # Step 4a: interior labels — convert coords per-partition (fully chunked).
    # Interior labels exist in exactly one chunk so no dedup is needed.
    interior_output = candidates.map_partitions(
        _process_interior_candidates, transform, meta=_OUTPUT_META
    )

    # Step 4b: boundary labels — collect and deduplicate.
    # Only labels touching a chunk edge may span multiple chunks. Their count
    # is proportional to chunk *perimeter*, not area, keeping the
    # materialized set small even for very large CHMs.
    boundary_candidates = candidates.map_partitions(
        lambda part: part[part["is_boundary"]], meta=_CANDIDATE_META
    )
    boundary_output = dd.from_delayed(
        dask.delayed(_deduplicate_boundary_candidates)(
            boundary_candidates, transform
        ),
        meta=_OUTPUT_META,
    )

    return dd.concat([interior_output, boundary_output])


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
