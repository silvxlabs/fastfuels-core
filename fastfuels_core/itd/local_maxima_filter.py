from __future__ import annotations

from collections.abc import Sequence

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
_MIN_WINDOW_PIXELS = 3

_CANDIDATE_COLUMNS = [
    "label",
    "height",
    "centroid_row_sum",
    "centroid_col_sum",
    "centroid_count",
    "row",
    "col",
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
        "row": pd.Series(dtype="int64"),
        "col": pd.Series(dtype="int64"),
        "is_boundary": pd.Series(dtype="bool"),
    }
)

_BOUNDARY_PIXEL_COLUMNS = ["label", "row", "col"]

# Local maxima are labelled 8-connected: pixels touching at a corner join.
_CONNECTIVITY = np.ones((3, 3), dtype=bool)


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
    """Return a ``w`` x ``w`` disc of diameter ``w`` pixels (``w`` odd).

    Keeps every offset within ``w / 2`` of the centre, so ``w = 3`` is the
    full 3 x 3 neighbourhood.
    """
    half = window_size_pixels // 2
    y, x = np.ogrid[-half : half + 1, -half : half + 1]
    return x * x + y * y <= (window_size_pixels / 2) ** 2


def _chunked_maximum_filter(chm: da.Array, footprint: np.ndarray) -> da.Array:
    """Apply scipy maximum_filter chunk-wise via map_overlap.

    The footprint is first cropped to offsets shorter than the array along
    each axis.  A reflected neighbour is never farther from the pixel than
    the offset that reached it, so the cropped disc has the same maximum, and
    the halo never exceeds the array (``map_overlap`` cannot pad further, and
    scipy's reflect mode misreads memory when a footprint is much larger
    than the array).

    ``map_overlap`` rechunks when a chunk is thinner than the overlap depth;
    the result is rechunked back so its blocks line up with ``chm``'s.
    """
    half = [min(s // 2, n - 1) for s, n in zip(footprint.shape, chm.shape)]
    centre = [s // 2 for s in footprint.shape]
    footprint = footprint[tuple(slice(c - h, c + h + 1) for c, h in zip(centre, half))]
    filtered = da.map_overlap(
        scipy_maximum_filter,
        chm,
        depth=dict(enumerate(half)),
        boundary="reflect",
        dtype=chm.dtype,
        footprint=footprint,
    )
    return filtered.rechunk(chm.chunks)


def _nearest_to_centroid(
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    row_sum: np.ndarray,
    col_sum: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    """Return the index of each label's pixel nearest its component centroid.

    ``labels``, ``rows`` and ``cols`` describe one pixel each, in global pixel
    indices; ``row_sum``, ``col_sum`` and ``count`` are the whole component's
    totals, aligned with the pixels.  Ties go to the smallest row, then the
    smallest column.  The result depends only on the component's pixels, so it
    is the same for every chunk layout.

    Returns indices into the pixel arrays, one per distinct label, in label
    order.
    """
    # Distances are scaled by ``count`` so the centroid is never divided out.
    dr = rows * count.astype(np.float64) - row_sum
    dc = cols * count.astype(np.float64) - col_sum
    d2 = dr * dr + dc * dc
    order = np.lexsort((cols, rows, d2, labels))
    sorted_labels = labels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_labels[1:] != sorted_labels[:-1]
    return order[first]


def _extract_block_candidates(
    chm_block: np.ndarray,
    mask_block: np.ndarray,
    row_offset: int,
    col_offset: int,
    label_offset: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-chunk: label locally, extract candidates, and return boundary edges.

    Runs ``scipy.ndimage.label`` on the chunk's local-maxima mask and extracts
    one candidate per 8-connected component.  Labels are offset by
    ``label_offset`` to be globally unique across chunks.

    Every search window is at least 3 pixels, so it contains a pixel's eight
    neighbours, and two mask pixels touching at an edge or a corner must have
    the same CHM value: a component is one plateau.
    A component is still not necessarily convex: on a quantised CHM a plateau
    can be ring-shaped, and its centroid can fall outside it.  The treetop is
    therefore placed on the component pixel nearest the centroid (see
    ``_nearest_to_centroid``).  Interior components resolve that pixel here;
    components touching the chunk edge also return their pixels so it can be
    resolved after the cross-chunk merge.

    Returns a tuple of:
    - candidates DataFrame
    - bottom_edge_labels: 1-D array of labels along the last row
    - right_edge_labels: 1-D array of labels along the last column
    - top_edge_labels: 1-D array of labels along the first row
    - left_edge_labels: 1-D array of labels along the first column
    - boundary pixels DataFrame: ``label``, ``row``, ``col`` (global indices)
      of every pixel in a component touching the chunk edge

    Edge label arrays are used by the boundary adjacency step to detect
    cross-chunk connections without materializing the full labeled array.
    """
    labeled_block, n_labels = scipy_label(mask_block, structure=_CONNECTIVITY)

    def global_labels(local: np.ndarray) -> np.ndarray:
        out = local.astype(np.int64)
        out[out > 0] += label_offset
        return out

    # Extract boundary edge labels for adjacency detection
    bottom_edge = global_labels(labeled_block[-1, :])
    right_edge = global_labels(labeled_block[:, -1])
    top_edge = global_labels(labeled_block[0, :])
    left_edge = global_labels(labeled_block[:, 0])

    if n_labels == 0:
        empty_pixels = pd.DataFrame(
            {c: pd.Series(dtype="int64") for c in _BOUNDARY_PIXEL_COLUMNS}
        )
        return (
            _CANDIDATE_META.copy(),
            bottom_edge,
            right_edge,
            top_edge,
            left_edge,
            empty_pixels,
        )

    flat = np.flatnonzero(labeled_block)
    local_ids = labeled_block.ravel()[flat].astype(np.int64)
    pixel_labels = local_ids + label_offset
    local_rows, local_cols = np.divmod(flat, labeled_block.shape[1])
    rows = local_rows + row_offset
    cols = local_cols + col_offset

    # Per-label totals, indexed by local label.
    count = np.bincount(local_ids, minlength=n_labels + 1)
    row_sum = np.bincount(local_ids, weights=rows, minlength=n_labels + 1)
    col_sum = np.bincount(local_ids, weights=cols, minlength=n_labels + 1)

    edges = np.concatenate([bottom_edge, right_edge, top_edge, left_edge])
    is_boundary = np.zeros(n_labels + 1, dtype=bool)
    is_boundary[edges[edges > 0] - label_offset] = True

    nearest = _nearest_to_centroid(
        pixel_labels,
        rows,
        cols,
        row_sum[local_ids],
        col_sum[local_ids],
        count[local_ids],
    )
    ids = local_ids[nearest]
    candidates = pd.DataFrame(
        {
            "label": pixel_labels[nearest],
            "height": chm_block[local_rows[nearest], local_cols[nearest]].astype(
                np.float64
            ),
            "centroid_row_sum": row_sum[ids],
            "centroid_col_sum": col_sum[ids],
            "centroid_count": count[ids],
            "row": rows[nearest],
            "col": cols[nearest],
            "is_boundary": is_boundary[ids],
        },
        columns=_CANDIDATE_COLUMNS,
    )

    on_boundary = is_boundary[local_ids]
    boundary_pixels = pd.DataFrame(
        {
            "label": pixel_labels[on_boundary],
            "row": rows[on_boundary],
            "col": cols[on_boundary],
        }
    )
    return candidates, bottom_edge, right_edge, top_edge, left_edge, boundary_pixels


def _pixels_to_spatial(
    rows: np.ndarray,
    cols: np.ndarray,
    heights: np.ndarray,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Convert treetop pixel indices to x/y pixel-centre coordinates."""
    if len(rows) == 0:
        return _OUTPUT_META.copy()

    xs, ys = rio.transform.xy(transform, rows.tolist(), cols.tolist())
    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=np.float64),
            "y": np.asarray(ys, dtype=np.float64),
            "height": np.asarray(heights, dtype=np.float64),
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
    return _pixels_to_spatial(
        interior["row"].values,
        interior["col"].values,
        interior["height"].values,
        transform,
    )


def _find_edge_merge_pairs(
    bottom_edge: np.ndarray,
    top_edge: np.ndarray,
) -> list[tuple[int, int]]:
    """Find label pairs that should merge across a shared chunk edge.

    ``bottom_edge`` is the last row (or column) of labels from one chunk;
    ``top_edge`` is the first row (or column) of labels from the chunk across
    the edge.  Two non-zero labels in the same or neighbouring positions are
    8-connected (part of the same plateau straddling the edge).
    """
    pairs = []
    for shift in (-1, 0, 1):
        a = bottom_edge[max(0, -shift) : len(bottom_edge) - max(0, shift)]
        b = top_edge[max(0, shift) : len(top_edge) - max(0, -shift)]
        connected = (a > 0) & (b > 0) & (a != b)
        pairs.extend(zip(a[connected].tolist(), b[connected].tolist()))
    return pairs


def _find_corner_merge_pairs(a: int, b: int) -> list[tuple[int, int]]:
    """Merge two corner pixels of diagonally adjacent chunks if both are set."""
    return [(int(a), int(b))] if a > 0 and b > 0 else []


_BOUNDARY_TREETOP_META = pd.DataFrame(
    {
        "row": pd.Series(dtype="int64"),
        "col": pd.Series(dtype="int64"),
        "height": pd.Series(dtype="float64"),
    }
)


def _union_find_merge_to_pixels(
    all_candidates: pd.DataFrame,
    merge_pairs: list[tuple[int, int]],
    boundary_pixels: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Merge boundary candidates and return each treetop's pixel indices.

    Labels joined by ``merge_pairs`` form one component.  Its treetop is the
    component pixel nearest the merged centroid, chosen from
    ``boundary_pixels`` by the same rule as for interior components.  The
    affine transform is deferred so callers can first route each merged
    treetop to the chunk that owns its pixel.
    """
    if all_candidates.empty:
        return _BOUNDARY_TREETOP_META.copy()

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

    labels = all_candidates["label"].to_numpy(dtype=np.int64)
    groups = np.fromiter((find(int(lbl)) for lbl in labels), np.int64, len(labels))
    totals = (
        pd.DataFrame(
            {
                "group": groups,
                "height": all_candidates["height"].to_numpy(dtype=np.float64),
                "row_sum": all_candidates["centroid_row_sum"].to_numpy(),
                "col_sum": all_candidates["centroid_col_sum"].to_numpy(),
                "count": all_candidates["centroid_count"].to_numpy(),
            }
        )
        .groupby("group")
        .agg({"height": "first", "row_sum": "sum", "col_sum": "sum", "count": "sum"})
    )

    pixels = pd.concat([p for p in boundary_pixels if not p.empty], ignore_index=True)
    group_of_label = pd.Series(groups, index=labels)
    pixel_groups = group_of_label.reindex(pixels["label"].to_numpy()).to_numpy()
    pixel_totals = totals.reindex(pixel_groups)
    rows = pixels["row"].to_numpy(dtype=np.int64)
    cols = pixels["col"].to_numpy(dtype=np.int64)

    nearest = _nearest_to_centroid(
        pixel_groups,
        rows,
        cols,
        pixel_totals["row_sum"].to_numpy(),
        pixel_totals["col_sum"].to_numpy(),
        pixel_totals["count"].to_numpy(),
    )
    return pd.DataFrame(
        {
            "row": rows[nearest],
            "col": cols[nearest],
            "height": pixel_totals["height"].to_numpy()[nearest],
        }
    )


def _union_find_merge(
    all_candidates: pd.DataFrame,
    merge_pairs: list[tuple[int, int]],
    boundary_pixels: Sequence[pd.DataFrame],
    transform: rio.Affine,
) -> pd.DataFrame:
    """Merge boundary candidates and return spatial (x, y, height) output.

    Thin wrapper around ``_union_find_merge_to_pixels`` that applies the affine
    transform.  Kept for direct callers/tests that want the merged spatial
    output in one step.
    """
    merged = _union_find_merge_to_pixels(all_candidates, merge_pairs, boundary_pixels)
    return _pixels_to_spatial(
        merged["row"].values, merged["col"].values, merged["height"].values, transform
    )


def _slice_boundary_to_chunk_and_transform(
    merged_boundary_pixels: pd.DataFrame,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    transform: rio.Affine,
) -> pd.DataFrame:
    """Filter merged boundary treetops to a single chunk's pixel bounds.

    A merged treetop belongs to chunk ``[row_lo, row_hi) × [col_lo, col_hi)``
    iff its pixel lies in that range.  Each merged treetop therefore lands in
    exactly one chunk's partition.
    """
    rows = merged_boundary_pixels["row"].values
    cols = merged_boundary_pixels["col"].values
    mask = (rows >= row_lo) & (rows < row_hi) & (cols >= col_lo) & (cols < col_hi)
    return _pixels_to_spatial(
        rows[mask],
        cols[mask],
        merged_boundary_pixels["height"].values[mask],
        transform,
    )


def _concat_interior_boundary(
    interior: pd.DataFrame, boundary: pd.DataFrame
) -> pd.DataFrame:
    """Concat the interior treetops with the boundary slice belonging to the
    same chunk.  Returns an empty meta frame when both inputs are empty."""
    if interior.empty and boundary.empty:
        return _OUTPUT_META.copy()
    if interior.empty:
        return boundary.reset_index(drop=True)
    if boundary.empty:
        return interior.reset_index(drop=True)
    return pd.concat([interior, boundary], ignore_index=True)


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
    # Each chunk_result is (candidates_df, bottom, right, top, left, pixels).
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
                        chunk_results[k][1],  # bottom edge
                        chunk_results[k_below][3],  # top edge
                    )
                )
            # Horizontal adjacency: this chunk's right ↔ chunk right's left
            if j + 1 < n_col_chunks:
                k_right = k + 1
                edge_merge_delayed.append(
                    dask.delayed(_find_edge_merge_pairs)(
                        chunk_results[k][2],  # right edge
                        chunk_results[k_right][4],  # left edge
                    )
                )
            # Diagonal adjacency across the corner shared by four chunks:
            # bottom-right pixel ↔ the lower-right chunk's top-left pixel, and
            # the right chunk's bottom-left pixel ↔ the lower chunk's top-right.
            if i + 1 < n_row_chunks and j + 1 < n_col_chunks:
                k_below = (i + 1) * n_col_chunks + j
                edge_merge_delayed.append(
                    dask.delayed(_find_corner_merge_pairs)(
                        chunk_results[k][1][-1], chunk_results[k_below + 1][3][0]
                    )
                )
                edge_merge_delayed.append(
                    dask.delayed(_find_corner_merge_pairs)(
                        chunk_results[k + 1][1][0], chunk_results[k_below][3][-1]
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

    # Step 6b: boundary labels — collect and merge via union-find, then place
    # each merged treetop on its component pixel nearest the merged centroid.
    # Defer the transform so we can first route each merged treetop to the
    # chunk that owns it.
    boundary_candidates = candidates.map_partitions(
        lambda part: part[part["is_boundary"]], meta=_CANDIDATE_META
    )

    def _collect_merge_pairs(*pair_lists: list[tuple[int, int]]) -> list:
        result = []
        for pl in pair_lists:
            result.extend(pl)
        return result

    all_merge_pairs = dask.delayed(_collect_merge_pairs)(*edge_merge_delayed)

    merged_boundary_pixels = dask.delayed(_union_find_merge_to_pixels)(
        boundary_candidates, all_merge_pairs, [cr[5] for cr in chunk_results]
    )

    # Step 7: route each merged boundary treetop to the chunk that contains it,
    # then concat with that chunk's interior partition. Output has exactly
    # ``n_chunks`` partitions, each spatially aligned with its CHM chunk.
    interior_delayeds = interior_output.to_delayed()

    combined_delayeds = []
    for i in range(n_row_chunks):
        for j in range(n_col_chunks):
            k = i * n_col_chunks + j
            row_lo = int(row_starts[i])
            row_hi = row_lo + int(chm.chunks[0][i])
            col_lo = int(col_starts[j])
            col_hi = col_lo + int(chm.chunks[1][j])
            boundary_slice = dask.delayed(_slice_boundary_to_chunk_and_transform)(
                merged_boundary_pixels, row_lo, row_hi, col_lo, col_hi, transform
            )
            combined_delayeds.append(
                dask.delayed(_concat_interior_boundary)(
                    interior_delayeds[k], boundary_slice
                )
            )

    return dd.from_delayed(combined_delayeds, meta=_OUTPUT_META)


def variable_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    crown_ratio: float = 0.05,
    crown_offset: float = 3.0,
    unique_windows: Sequence[int] | None = None,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Variable Window Filter (VWF).

    Calculates the search window size dynamically using a linear allometric
    relationship: Crown_Width_m = (Height_m * crown_ratio) + crown_offset.
    The window is rounded up to an odd number of pixels, and is at least 3.
    Of the linear forms and Popescu & Wynne's quadratics tried on the
    NeonTreeEvaluation benchmark (Weinstein et al. 2021,
    https://doi.org/10.1371/journal.pcbi.1009180), the defaults scored best
    at 0.5 m and tied for best at 1 m.

    A local maximum is an 8-connected set of equal-height pixels, each at
    least as tall as every pixel in its window.  Each treetop is placed at the
    centre of a pixel of its local maximum: the pixel nearest the maximum's
    centroid, with ties going to the smallest row, then column.  No two
    treetops share a CHM cell.

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
            Defaults to 0.05 (5%).
        crown_offset (float): The base crown width in meters. Defaults to 3.0m.
        unique_windows: Optional caller-supplied list of odd, positive window
            sizes (in pixels) to iterate.  When provided, skips the internal
            ``da.unique`` scan over the CHM — this is the only synchronous
            reduction in VWF, so passing this argument makes graph
            construction fully lazy.  The caller is responsible for supplying
            a superset of the window sizes that actually appear in the data;
            sizes missing from the list will silently produce no treetops at
            the corresponding pixels.  Extra sizes are safe but cost one
            ``map_overlap`` pass each at compute time.  Sizes below 3 are
            raised to 3, like the computed windows.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    if min_height < 0:
        raise ValueError("min_height cannot be negative")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution must be positive")

    if unique_windows is not None:
        windows_to_use = _validate_unique_windows(unique_windows)
    else:
        windows_to_use = None

    chm, transform = _prepare_chm(chm_da)

    # Per-pixel window size (lazy).
    crown_width_meters = (chm * crown_ratio) + crown_offset
    required_windows = (crown_width_meters / spatial_resolution).astype(int)
    required_windows = da.where(
        required_windows % 2 == 0, required_windows + 1, required_windows
    )
    # A 1-pixel window equals its own maximum, so every such pixel would enter
    # the mask; 3 is the smallest window that compares a pixel to its neighbours.
    required_windows = da.maximum(required_windows, _MIN_WINDOW_PIXELS)

    if windows_to_use is None:
        # Iterate only the window sizes that actually occur in the data. `da.unique`
        # is a tree reduction (per-chunk np.unique, merged pairwise up the tree) —
        # bounded memory, never materializes the full required_windows array.
        # This is the one synchronous step in VWF; pass `unique_windows` to skip it.
        windows_to_use = np.asarray(da.unique(required_windows).compute())

    vw_max = da.zeros_like(chm)
    for w in windows_to_use:
        w = int(w)
        footprint = _build_circular_footprint(w)
        filtered = _chunked_maximum_filter(chm, footprint)
        vw_max = da.where(required_windows == w, filtered, vw_max)

    return _extract_treetops(chm, vw_max, transform, min_height)


def _validate_unique_windows(unique_windows: Sequence[int]) -> np.ndarray:
    """Validate a caller-supplied ``unique_windows`` argument and return a
    sorted, deduplicated ``np.ndarray[int]``.

    Each entry must be a positive odd integer (the VWF kernel is a centered
    circular footprint).
    """
    materialized = list(unique_windows)
    if not materialized:
        raise ValueError("unique_windows must not be empty")
    validated: list[int] = []
    for raw in materialized:
        w = int(raw)
        if w < 1:
            raise ValueError(f"unique_windows entries must be >= 1, got {w}")
        if w % 2 == 0:
            raise ValueError(f"unique_windows entries must be odd, got {w}")
        validated.append(max(w, _MIN_WINDOW_PIXELS))
    return np.asarray(sorted(set(validated)), dtype=int)


def fixed_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    window_size_meters: float = 3.0,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Fixed Window Local Maxima (FW-LM) filter.

    Applies a static, circular search window across the entire Canopy Height Model
    to identify local maxima.  Treetops are placed as in ``variable_window_filter``.

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
    if window_size_pixels < _MIN_WINDOW_PIXELS:
        window_size_pixels = _MIN_WINDOW_PIXELS

    footprint = _build_circular_footprint(window_size_pixels)
    chm_max_filtered = _chunked_maximum_filter(chm, footprint)

    return _extract_treetops(chm, chm_max_filtered, transform, min_height)
