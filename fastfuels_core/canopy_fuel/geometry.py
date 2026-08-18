"""Exact disk / axis-aligned-rectangle intersection area.

Both the vertical profile and canopy cover attribute a circular crown
to the cells it covers, and both need the intersection area exactly
rather than by sampling: a crown straddling a cell boundary must give
each cell its true share, and the shares must sum to the crown area.
"""

from __future__ import annotations

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
