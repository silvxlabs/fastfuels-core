"""Exact disk / axis-aligned-rectangle intersection area.

Both the vertical profile and canopy cover attribute a circular crown
to the cells it covers, and both need the intersection area exactly
rather than by sampling: a crown straddling a cell boundary must give
each cell its true share, and the shares must sum to the crown area.
:func:`disk_cell_overlaps` walks the cells each disk reaches and yields
those areas per tree, so the two stages share one traversal.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _unit_disk_corner_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Area of the unit disk within the corner region {X <= x, Y <= y}.

    Closed form: integrate the clipped chord length from -1 to x. The
    integrand is ``clip(y, -sqrt(1-t^2), sqrt(1-t^2)) + sqrt(1-t^2)``,
    which is the full chord ``2*sqrt(1-t^2)`` where the chord lies
    entirely below y (for y >= 0), zero where it lies entirely above
    (y < 0), and ``y + sqrt(1-t^2)`` in between; the pieces meet at
    ``t = +/-sqrt(1-y^2)``.
    """
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    t_star = np.sqrt(np.clip(1.0 - y * y, 0.0, None))

    def antiderivative(t):
        # Integral of sqrt(1-t^2).
        return 0.5 * (
            t * np.sqrt(np.clip(1.0 - t * t, 0.0, None))
            + np.arcsin(np.clip(t, -1.0, 1.0))
        )

    outer = np.where(y >= 0.0, 2.0, 0.0)
    left_hi = np.minimum(x, -t_star)
    left = outer * (antiderivative(left_hi) - antiderivative(-1.0))
    mid_hi = np.clip(x, -t_star, t_star)
    middle = (y * mid_hi + antiderivative(mid_hi)) - (
        y * -t_star + antiderivative(-t_star)
    )
    right_hi = np.maximum(x, t_star)
    right = outer * (antiderivative(right_hi) - antiderivative(t_star))
    return left + middle + right


def disk_rect_overlap_area(
    cx: np.ndarray,
    cy: np.ndarray,
    radius: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Exact intersection area of disks and axis-aligned rectangles.

    All arguments broadcast; rectangles are ``[x0, x1] x [y0, y1]`` with
    ``x0 < x1`` and ``y0 < y1``. Evaluated by inclusion-exclusion over
    the four corner integrals of the unit disk.
    """
    r = np.asarray(radius, dtype=np.float64)
    u0, u1 = (x0 - cx) / r, (x1 - cx) / r
    v0, v1 = (y0 - cy) / r, (y1 - cy) / r
    area_unit = (
        _unit_disk_corner_area(u1, v1)
        - _unit_disk_corner_area(u0, v1)
        - _unit_disk_corner_area(u1, v0)
        + _unit_disk_corner_area(u0, v0)
    )
    return np.clip(area_unit, 0.0, np.pi) * r * r


def disk_cell_overlaps(
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(flat cell index, overlap area)`` per tree for every cell a disk reaches.

    One pair per offset in the neighbourhood the largest disk spans;
    each is aligned with the trees. Cells outside the lattice carry
    area 0 and index 0, so a disk overhanging the boundary simply loses
    the overhanging slice. Offsets that touch no in-bounds cell for any
    tree are skipped.
    """
    a, _, c, _, e, f = transform
    ny, nx = shape
    col_lo = np.floor((x - radius - c) / a).astype(np.int64)
    col_hi = np.floor((x + radius - c) / a).astype(np.int64)
    row_lo = np.floor((y + radius - f) / e).astype(np.int64)  # e < 0
    row_hi = np.floor((y - radius - f) / e).astype(np.int64)
    for row_offset in range(int((row_hi - row_lo).max()) + 1):
        rows = row_lo + row_offset
        y_hi = f + rows * e  # north edge; e < 0 makes y_hi > y_lo
        y_lo = y_hi + e
        for col_offset in range(int((col_hi - col_lo).max()) + 1):
            cols = col_lo + col_offset
            x_lo = c + cols * a
            area = disk_rect_overlap_area(x, y, radius, x_lo, x_lo + a, y_lo, y_hi)
            in_bounds = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
            area = np.where(in_bounds, area, 0.0)
            if not area.any():
                continue
            yield np.where(in_bounds, rows * nx + cols, 0), area
