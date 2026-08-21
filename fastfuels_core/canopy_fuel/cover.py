"""Per-cell projected canopy cover (%).

Crowns are flat disks at the tree top. What varies between methods is
how crowns that overlap each other are counted, and which trees are
counted at all; every method clips crowns to the cell the same way, so
the methods differ only in that treatment and compare directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.features import rasterize

from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.canopy_fuel.geometry import disk_cell_overlaps

VALID_METHODS = ("crown_union", "crown_overlap", "cover_fraction")

# Cap on each supersampled canopy-cover strip mask (uint8 bytes).
_COVER_STRIP_BYTES = 64 * 2**20


def _crown_overlap_cover(
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Crookston & Stage (1999) cover, per cell.

    ``NC_CA.C CA_Overlap`` divides accumulated crown area by an acre
    because a FuelCalc plot is one; the ground area here is the cell, so
    that is the denominator. Both are the same Poisson argument: with
    stems placed at random, the expected uncovered fraction of a patch
    of ground is ``exp(-crown area / patch area)``.
    """
    a, _, _, _, e, _ = transform
    area = np.zeros(shape, dtype=np.float64)
    if len(x) == 0:
        return area

    radius = np.maximum(radius, 0.0)
    flat = area.reshape(-1)
    for cell, overlap in disk_cell_overlaps(x, y, radius, transform, shape):
        flat += np.bincount(cell, weights=overlap, minlength=flat.size)
    cell_area = abs(a * e)
    return 100.0 * (1.0 - np.exp(-area / cell_area))


def _crown_union_cover(
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    supersample: int,
) -> np.ndarray:
    """Geometric union of the crown disks, as a covered fraction per cell.

    Crown disks are rasterized as a binary union on a grid of
    ``supersample x supersample`` fine pixels per output cell, then
    block-reduced. Fine masks are built in horizontal strips so peak
    memory stays bounded regardless of lattice size and supersample
    factor.
    """
    a, _, c, _, e, f = transform
    ny, nx = shape
    crowns = shapely.buffer(shapely.points(x, y), radius, quad_segs=16)
    max_radius = float(radius.max())

    rows_per_strip = max(1, int(_COVER_STRIP_BYTES // (nx * supersample**2)))
    cover = np.empty(shape, dtype=np.float64)
    for row_start in range(0, ny, rows_per_strip):
        row_stop = min(row_start + rows_per_strip, ny)
        y_north = f + row_start * e  # e < 0
        y_south = f + row_stop * e
        in_strip = (y >= y_south - max_radius) & (y <= y_north + max_radius)
        strip_shape = ((row_stop - row_start) * supersample, nx * supersample)
        if in_strip.any():
            mask = rasterize(
                crowns[in_strip],
                out_shape=strip_shape,
                transform=Affine(
                    a / supersample, 0.0, c, 0.0, e / supersample, y_north
                ),
                fill=0,
                default_value=1,
                all_touched=False,
                dtype="uint8",
            )
            block = mask.reshape(
                row_stop - row_start, supersample, nx, supersample
            ).mean(axis=(1, 3))
        else:
            block = 0.0
        cover[row_start:row_stop] = block
    return cover


def canopy_cover(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    crown_radius_column: str | None = None,
    crown_radius_equations: str = "crookston_stage",
    method: str = "crown_overlap",
    height_threshold: float = 2.0,
    supersample: int | None = None,
) -> np.ndarray:
    """Per-cell projected canopy cover (%).

    ``method`` decides how crowns that overlap each other are counted.
    Every method clips crowns to the cell the same way, so they differ
    only in that treatment and can be compared directly.

    ``"crown_overlap"`` (default)
        Crookston & Stage (1999) random-overlap correction,
        ``100 * (1 - exp(-crown area in cell / cell area))``: the cover
        expected if stems were placed at random. This is FuelCalc's
        estimator, and it is the one to use when reproducing FuelCalc
        or LANDFIRE. It answers the question a stand table forces; with
        known stem positions the union below resolves overlap exactly
        rather than in expectation. The crown area in a cell is the
        exact disk/cell intersection, so a crown straddling a boundary
        contributes to both cells in proportion.
    ``"crown_union"``
        The geometric union of the crown disks: overlapping crowns count
        once. Crown disks are rasterized as a binary union on a fine
        grid of ``supersample x supersample`` pixels per output cell,
        then block-reduced to covered fractions. ``supersample``
        defaults to a fine pixel of at most 0.5 m so crown edges are
        resolved at any output resolution; only the covered fractions
        are contract.

    ``"cover_fraction"``
        The union again, but only over trees taller than
        ``height_threshold`` (m, default 2.0). A CHM-style measure: it
        reports where the canopy *surface* clears a height rather than
        the projection of all suspended canopy, so it is the cover
        variable to compare against one derived from a canopy height
        model. ``height_threshold=0`` makes it identical to
        ``crown_union``.

        Crowns are flat disks at the tree top here, as they are for
        every method in this function, so a tree either clears the
        threshold everywhere under its crown or nowhere. A real canopy
        surface tapers, and a crown near the threshold would cover part
        of its footprint; resolving that needs a crown profile and
        would make this method the only one not built on disks, so it
        is left as a separate refinement rather than folded in here.

    Crown radii come from ``crown_radius_column`` or
    ``crown_radius_equations`` (see :mod:`crown_radius`), which is
    independent of ``method``.

    Returns
    -------
    numpy.ndarray
        Cover (%), shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        For an unknown ``method``, or a rotated transform.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown canopy cover method {method!r}; expected "
            f"'crown_union', 'crown_overlap' or 'cover_fraction'."
        )
    if height_threshold < 0.0:
        raise ValueError(
            f"height_threshold must be non-negative, got {height_threshold}."
        )
    a, b_rot, _, d_rot, _, _ = transform
    if b_rot != 0.0 or d_rot != 0.0:
        raise ValueError("Rotated transforms are not supported.")

    if method == "cover_fraction":
        trees = trees[trees["height"].to_numpy(dtype=np.float64) > height_threshold]
    if len(trees) == 0:
        return np.zeros(shape)

    x = trees["x"].to_numpy(dtype=np.float64)
    y = trees["y"].to_numpy(dtype=np.float64)
    radius = max_crown_radius(
        trees,
        crown_radius_column=crown_radius_column,
        equations=crown_radius_equations,
    )
    if method == "crown_overlap":
        return _crown_overlap_cover(x, y, radius, transform, shape)
    if supersample is None:
        supersample = max(2, int(np.ceil(abs(a) / 0.5)))
    return 100.0 * _crown_union_cover(x, y, radius, transform, shape, supersample)
