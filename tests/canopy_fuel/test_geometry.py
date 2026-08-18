"""Tests for :mod:`fastfuels_core.canopy_fuel.geometry`.

The disk/rectangle intersection is the primitive both the crown-projected
profile and canopy cover split crowns with, so it is checked against an
independent brute-force integration and against configurations whose
area is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area


def overlap(cx, cy, r, x0, x1, y0, y1):
    """Scalar wrapper: the function itself takes broadcastable arrays."""
    return disk_rect_overlap_area(
        *(np.array([float(v)]) for v in (cx, cy, r, x0, x1, y0, y1))
    )[0]


def brute_force_overlap(cx, cy, r, x0, x1, y0, y1, n=2000):
    """Supersampled point-in-disk integration over the rectangle."""
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    inside = (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
    return inside.mean() * (x1 - x0) * (y1 - y0)


@pytest.mark.parametrize("case", range(50))
def test_matches_brute_force_integration(case):
    """Random configurations agree with supersampled integration."""
    rng = np.random.default_rng(7 + case)
    cx, cy = rng.uniform(-5, 5, 2)
    r = rng.uniform(0.3, 6.0)
    x0, y0 = rng.uniform(-6, 4, 2)
    x1, y1 = x0 + rng.uniform(0.5, 6.0), y0 + rng.uniform(0.5, 6.0)
    analytic = overlap(cx, cy, r, x0, x1, y0, y1)
    brute = brute_force_overlap(cx, cy, r, x0, x1, y0, y1)
    assert abs(analytic - brute) < max(0.01 * np.pi * r * r, 1e-3)


def test_disk_inside_rectangle_is_the_whole_disk():
    assert overlap(0, 0, 1, -5, 5, -5, 5) == pytest.approx(np.pi, rel=1e-12)


def test_rectangle_inside_disk_is_the_whole_rectangle():
    assert overlap(0, 0, 10, -1, 1, -1, 1) == pytest.approx(4.0, rel=1e-12)


def test_disjoint_shapes_do_not_overlap():
    assert overlap(0, 0, 1, 2, 3, 2, 3) == pytest.approx(0.0, abs=1e-12)


def test_first_quadrant_is_a_quarter_disk():
    assert overlap(0, 0, 2, 0, 5, 0, 5) == pytest.approx(np.pi, rel=1e-12)


def test_quadrants_partition_the_disk():
    """Four abutting rectangles must recover exactly the disk area.

    Off-center so each quadrant takes a different, unequal share; if the
    corner integrals did not compose by inclusion-exclusion the total
    would not close.
    """
    r = 3.0
    quadrants = [(-9, 0, -9, 0), (0, 9, -9, 0), (-9, 0, 0, 9), (0, 9, 0, 9)]
    total = sum(overlap(0.5, -0.25, r, *q) for q in quadrants)
    assert total == pytest.approx(np.pi * r * r, rel=1e-10)
