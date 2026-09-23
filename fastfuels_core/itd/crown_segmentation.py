from __future__ import annotations

import math

import dask
import dask.array as da
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr

# A cell may be at most this factor above its crown's treetop cell.
MAX_RELATIVE_TO_TREETOP = 1.05

# (row, col) offsets of the four neighbours a crown grows through.
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def dalponte2016(
    chm_da: xr.DataArray,
    treetops: pd.DataFrame,
    min_height: float,
    max_height: float | None,
    min_relative_height: float,
    min_relative_crown_height: float,
    max_crown_radius: float,
) -> xr.DataArray:
    """Segments tree crowns on a CHM by region growing from known treetops.

    Each treetop seeds a crown with the CHM cell that contains its ``(x, y)``.
    Crowns then grow one ring of 4-connected cells per step. An unlabelled
    cell next to crown ``k`` joins it when:

    - its CHM value is finite and within ``[min_height, max_height]``;
    - it is at least ``min_relative_height`` times ``k``'s treetop cell;
    - it is at least ``min_relative_crown_height`` times ``k``'s mean height;
    - it is at most 1.05 times ``k``'s treetop cell;
    - its center is within ``max_crown_radius`` of ``k``'s treetop cell center.

    Every cell in a step is tested against crown means as they were at the
    start of that step, so the result does not depend on scan order. A cell
    that qualifies for several crowns goes to the crown with the taller
    treetop cell; ties go to the lower label. Growth stops when a step adds
    no cells. A crown whose treetop cell fails the height range stays a
    one-cell crown. The CHM is used as given, without smoothing.

    When ``chm_da`` is dask-backed, each chunk is segmented with a halo of
    ``2 * max_crown_radius`` using every treetop inside the extended block,
    and the result has the same chunks. The result is deterministic for a
    given chunk layout; it is not proven identical to an unchunked run.

    Algorithm:
    - Dalponte & Coomes (2016): Tree-centric mapping of forest carbon density
      from airborne laser scanning and hyperspectral data.
      https://doi.org/10.1111/2041-210X.12575

    Differences from lidR's ``dalponte2016``: crown means are updated once per
    step rather than per cell, ``max_crown_radius`` is a circle in metres
    rather than a square pixel window, and growth reaches the image edges.

    Args:
        chm_da (xr.DataArray): 2D Canopy Height Model in metres.
        treetops (pd.DataFrame): Treetops with ``x`` and ``y`` columns in the
            CHM's CRS, e.g. the output of ``fixed_window_filter`` or
            ``variable_window_filter`` after ``.compute()``.
        min_height (float): Minimum CHM height (m) of a crown cell. ``>= 0``.
        max_height (float | None): Maximum CHM height (m) of a crown cell, or
            ``None`` for no ceiling.
        min_relative_height (float): Fraction of the treetop cell height a
            crown cell must reach. In ``[0, 1)``.
        min_relative_crown_height (float): Fraction of the crown's mean height
            a crown cell must reach. In ``[0, 1)``.
        max_crown_radius (float): Maximum distance (m) from the treetop cell
            center to a crown cell center. ``> 0``.

    Returns:
        xr.DataArray: int32 labels on ``chm_da``'s grid. Label ``k`` is the
        treetop in row ``k - 1`` of ``treetops``; ``0`` is no crown.

    Raises:
        ValueError: If the CHM is not 2D, a parameter is out of range, a
            treetop lies outside the CHM, or two treetops share a cell.
    """
    _validate_parameters(
        min_height,
        max_height,
        min_relative_height,
        min_relative_crown_height,
        max_crown_radius,
    )
    if chm_da.ndim != 2:
        raise ValueError("CHM must be a 2D DataArray")
    if not isinstance(treetops, pd.DataFrame):
        raise TypeError("treetops must be a pandas DataFrame")
    missing = {"x", "y"} - set(treetops.columns)
    if missing:
        raise ValueError(f"treetops is missing columns: {sorted(missing)}")

    transform = chm_da.rio.transform()
    seed_rows, seed_cols = _treetop_cells(treetops, transform, chm_da.shape)

    params = dict(
        min_height=float(min_height),
        max_height=math.inf if max_height is None else float(max_height),
        min_relative_height=float(min_relative_height),
        min_relative_crown_height=float(min_relative_crown_height),
        max_crown_radius=float(max_crown_radius),
        transform=transform,
    )

    if isinstance(chm_da.data, da.Array):
        labels = _segment_chunked(chm_da.data, seed_rows, seed_cols, **params)
    else:
        labels = _segment(
            np.asarray(chm_da.values),
            seed_rows,
            seed_cols,
            np.arange(1, len(seed_rows) + 1, dtype=np.int32),
            **params,
        )

    return xr.DataArray(
        labels, coords=chm_da.coords, dims=chm_da.dims, name="crown_label"
    )


def _validate_parameters(
    min_height: float,
    max_height: float | None,
    min_relative_height: float,
    min_relative_crown_height: float,
    max_crown_radius: float,
) -> None:
    if not (math.isfinite(min_height) and min_height >= 0):
        raise ValueError("min_height must be finite and >= 0")
    if max_height is not None and not (
        math.isfinite(max_height) and max_height >= min_height
    ):
        raise ValueError("max_height must be None or finite and >= min_height")
    if not 0 <= min_relative_height < 1:
        raise ValueError("min_relative_height must be in [0, 1)")
    if not 0 <= min_relative_crown_height < 1:
        raise ValueError("min_relative_crown_height must be in [0, 1)")
    if not (math.isfinite(max_crown_radius) and max_crown_radius > 0):
        raise ValueError("max_crown_radius must be finite and > 0")


def _treetop_cells(
    treetops: pd.DataFrame,
    transform: rio.Affine,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (row, col) of the CHM cell containing each treetop."""
    xs = treetops["x"].to_numpy(dtype=np.float64)
    ys = treetops["y"].to_numpy(dtype=np.float64)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("treetop coordinates must be finite")

    inv = ~transform
    cols = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(np.int64)
    rows = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(np.int64)

    outside = (rows < 0) | (rows >= shape[0]) | (cols < 0) | (cols >= shape[1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} treetop(s) lie outside the CHM, "
            f"e.g. row {int(np.flatnonzero(outside)[0])} of treetops"
        )

    flat = rows * shape[1] + cols
    if len(np.unique(flat)) != len(flat):
        raise ValueError("two or more treetops fall in the same CHM cell")

    return rows, cols


def _segment(
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
    """Region-grow crowns on an in-memory CHM.

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


def _halo_cells(max_crown_radius: float, transform: rio.Affine) -> int:
    """Number of cells covering ``2 * max_crown_radius`` along either axis."""
    a, b, _, d, e, _ = transform[:6]
    cell_size = min(math.hypot(a, d), math.hypot(b, e))
    return math.ceil(2 * max_crown_radius / cell_size)


def _segment_chunked(
    chm: da.Array,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    *,
    max_crown_radius: float,
    transform: rio.Affine,
    **params,
) -> da.Array:
    """Segment each chunk with a ``2 * max_crown_radius`` halo."""
    halo = _halo_cells(max_crown_radius, transform)
    nrows, ncols = chm.shape
    row_starts = np.cumsum((0,) + chm.chunks[0][:-1])
    col_starts = np.cumsum((0,) + chm.chunks[1][:-1])
    seed_labels = np.arange(1, len(seed_rows) + 1, dtype=np.int32)

    blocks = []
    for i, row_len in enumerate(chm.chunks[0]):
        row = []
        for j, col_len in enumerate(chm.chunks[1]):
            r0, c0 = int(row_starts[i]), int(col_starts[j])
            r1, c1 = r0 + row_len, c0 + col_len
            er0, er1 = max(0, r0 - halo), min(nrows, r1 + halo)
            ec0, ec1 = max(0, c0 - halo), min(ncols, c1 + halo)

            inside = (
                (seed_rows >= er0)
                & (seed_rows < er1)
                & (seed_cols >= ec0)
                & (seed_cols < ec1)
            )
            block = dask.delayed(_segment_block)(
                chm[er0:er1, ec0:ec1],
                seed_rows[inside] - er0,
                seed_cols[inside] - ec0,
                seed_labels[inside],
                (r0 - er0, r1 - er0, c0 - ec0, c1 - ec0),
                max_crown_radius=max_crown_radius,
                transform=transform,
                **params,
            )
            row.append(da.from_delayed(block, shape=(row_len, col_len), dtype=np.int32))
        blocks.append(row)

    return da.block(blocks)


def _segment_block(
    chm: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_labels: np.ndarray,
    core: tuple[int, int, int, int],
    **params,
) -> np.ndarray:
    """Segment an extended block and return the labels of its core cells."""
    labels = _segment(np.asarray(chm), seed_rows, seed_cols, seed_labels, **params)
    r0, r1, c0, c1 = core
    return labels[r0:r1, c0:c1]
