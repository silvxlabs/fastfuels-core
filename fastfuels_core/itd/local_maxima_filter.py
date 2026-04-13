from __future__ import annotations

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter as scipy_maximum_filter

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


def _chunked_maximum_filter(chm: da.Array, footprint: np.ndarray) -> da.Array:
    """Apply scipy maximum_filter chunk-wise via map_overlap."""
    depth = {i: s // 2 for i, s in enumerate(footprint.shape)}
    return da.map_overlap(
        scipy_maximum_filter,
        chm,
        depth=depth,
        boundary="reflect",
        dtype=chm.dtype,
        footprint=footprint,
    )


def _extract_block_candidates(
    chm_block: np.ndarray,
    mask_block: np.ndarray,
    row_offset: int,
    col_offset: int,
    label_offset: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-chunk: label locally, extract candidates, and return boundary edges.

    Runs ``scipy.ndimage.label`` on the chunk's local-maxima mask and extracts
    one candidate per connected component.  Labels are offset by
    ``label_offset`` to be globally unique across chunks.

    Within the local-maxima mask every pixel of a connected component has the
    same CHM value (a pixel whose neighbor is higher cannot satisfy
    ``chm == max_filter``).  The centroid of all component pixels is therefore
    a natural, chunk-layout-invariant representative position.

    Returns a tuple of:
    - candidates DataFrame
    - bottom_edge_labels: 1-D array of labels along the last row
    - right_edge_labels: 1-D array of labels along the last column
    - top_edge_labels: 1-D array of labels along the first row
    - left_edge_labels: 1-D array of labels along the first column

    Edge label arrays are used by the boundary adjacency step to detect
    cross-chunk connections without materializing the full labeled array.
    """
    labeled_block, _ = scipy_label(mask_block)

    # Offset labels to be globally unique
    labeled_block[labeled_block > 0] += label_offset

    # Extract boundary edge labels for adjacency detection
    bottom_edge = labeled_block[-1, :].copy()
    right_edge = labeled_block[:, -1].copy()
    top_edge = labeled_block[0, :].copy()
    left_edge = labeled_block[:, 0].copy()

    unique_labels = np.unique(labeled_block)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        empty = _CANDIDATE_META.copy()
        return empty, bottom_edge, right_edge, top_edge, left_edge

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

    df = pd.DataFrame(records, columns=_CANDIDATE_COLUMNS)
    return df, bottom_edge, right_edge, top_edge, left_edge


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


def _find_edge_merge_pairs(
    bottom_edge: np.ndarray,
    top_edge: np.ndarray,
) -> list[tuple[int, int]]:
    """Find label pairs that should merge across a horizontal chunk boundary.

    ``bottom_edge`` is the last row of labels from the upper chunk;
    ``top_edge`` is the first row of labels from the lower chunk.
    Where both have a non-zero label at the same column, those two labels
    are connected (part of the same plateau straddling the boundary).
    """
    pairs = []
    connected = (bottom_edge > 0) & (top_edge > 0)
    for idx in np.flatnonzero(connected):
        a, b = int(bottom_edge[idx]), int(top_edge[idx])
        if a != b:
            pairs.append((a, b))
    return pairs


def _union_find_merge(
    all_candidates: pd.DataFrame,
    merge_pairs: list[tuple[int, int]],
    transform: rio.Affine,
) -> pd.DataFrame:
    """Merge boundary candidates whose labels are transitively connected.

    Uses union-find to group labels, then combines centroid components
    within each group.  Returns spatial (x, y, height) output.
    """
    if all_candidates.empty:
        return _OUTPUT_META.copy()

    # Union-find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in merge_pairs:
        union(a, b)

    # Map every boundary label to its root
    all_candidates = all_candidates.copy()
    all_candidates["group"] = all_candidates["label"].map(lambda lbl: find(lbl))

    combined = (
        all_candidates.groupby("group")
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
    """Extract treetop spatial points from a filtered CHM.

    Fully lazy and memory-bounded: each chunk is labeled independently
    with ``scipy.ndimage.label``, avoiding any global synchronization.
    Cross-chunk plateaus are detected via 1-D boundary-edge comparison
    and merged with a lightweight union-find.
    """
    # Step 1: lazy boolean mask
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)

    # Step 2: chunk grid metadata
    n_row_chunks = len(chm.chunks[0])
    n_col_chunks = len(chm.chunks[1])
    row_starts = np.cumsum((0,) + chm.chunks[0][:-1])
    col_starts = np.cumsum((0,) + chm.chunks[1][:-1])

    # Label offsets ensure globally unique labels without a global pass.
    # Upper bound: each chunk can have at most (chunk_rows * chunk_cols) // 2
    # labels.  We use the chunk area as a safe offset per chunk.
    chunk_areas = [
        int(chm.chunks[0][i]) * int(chm.chunks[1][j])
        for i in range(n_row_chunks)
        for j in range(n_col_chunks)
    ]
    label_offsets = np.cumsum([0] + chunk_areas[:-1])

    # Step 3: per-chunk extraction (lazy via delayed)
    chm_blocks = chm.to_delayed().ravel()
    mask_blocks = local_maxima_mask.to_delayed().ravel()

    chunk_results = []
    for k, (cb, mb) in enumerate(zip(chm_blocks, mask_blocks)):
        i, j = divmod(k, n_col_chunks)
        ro, co = int(row_starts[i]), int(col_starts[j])
        lo = int(label_offsets[k])
        chunk_results.append(
            dask.delayed(_extract_block_candidates)(cb, mb, ro, co, lo)
        )

    # Step 4: boundary adjacency detection (lazy, operates on 1-D edge slices)
    # Each chunk_result is (candidates_df, bottom, right, top, left).
    # Compare adjacent chunks' shared edges to find merge pairs.
    edge_merge_delayed = []
    for i in range(n_row_chunks):
        for j in range(n_col_chunks):
            k = i * n_col_chunks + j
            # Vertical adjacency: this chunk's bottom ↔ chunk below's top
            if i + 1 < n_row_chunks:
                k_below = (i + 1) * n_col_chunks + j
                edge_merge_delayed.append(
                    dask.delayed(_find_edge_merge_pairs)(
                        chunk_results[k][1],       # bottom edge
                        chunk_results[k_below][3],  # top edge
                    )
                )
            # Horizontal adjacency: this chunk's right ↔ chunk right's left
            if j + 1 < n_col_chunks:
                k_right = k + 1
                edge_merge_delayed.append(
                    dask.delayed(_find_edge_merge_pairs)(
                        chunk_results[k][2],       # right edge
                        chunk_results[k_right][4],  # left edge
                    )
                )

    # Step 5: build candidate partitions from chunk results
    partitions = [
        dd.from_delayed(
            dask.delayed(lambda cr: cr[0])(cr),
            meta=_CANDIDATE_META,
        )
        for cr in chunk_results
    ]
    candidates = dd.concat(partitions)

    # Step 6a: interior labels — emit directly per-partition (no dedup needed)
    interior_output = candidates.map_partitions(
        _process_interior_candidates, transform, meta=_OUTPUT_META
    )

    # Step 6b: boundary labels — collect, merge via union-find, convert
    boundary_candidates = candidates.map_partitions(
        lambda part: part[part["is_boundary"]], meta=_CANDIDATE_META
    )

    def _collect_merge_pairs(*pair_lists: list[tuple[int, int]]) -> list:
        result = []
        for pl in pair_lists:
            result.extend(pl)
        return result

    all_merge_pairs = dask.delayed(_collect_merge_pairs)(*edge_merge_delayed)

    boundary_output = dd.from_delayed(
        dask.delayed(_union_find_merge)(
            boundary_candidates, all_merge_pairs, transform
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
    max_height: float | None = None,
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
        max_height (float | None): Maximum canopy height in meters. Determines
            the largest search window. If None, computed from the CHM.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    if min_height < 0:
        raise ValueError("min_height cannot be negative")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution must be positive")

    chm, transform = _prepare_chm(chm_da)

    if max_height is None:
        max_height = float(chm.max().compute())

    # Window sizes must be odd (centered kernel). Derive the range of odd
    # sizes from min_height and max_height.
    min_w = int((min_height * crown_ratio + crown_offset) / spatial_resolution)
    max_w = int((max_height * crown_ratio + crown_offset) / spatial_resolution)
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
        filtered = _chunked_maximum_filter(chm, footprint)
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
    chm_max_filtered = _chunked_maximum_filter(chm, footprint)

    return _extract_treetops(chm, chm_max_filtered, transform, min_height)
