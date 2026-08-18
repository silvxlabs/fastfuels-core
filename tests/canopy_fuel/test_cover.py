"""Tests for :mod:`fastfuels_core.canopy_fuel.cover`, ``crown_union``.

The union counts overlapping crowns once, which is the property these
tests are built around: adding a crown that covers no new ground must
not change the answer. The other two methods, ``crown_overlap`` and
``cover_fraction``, are covered in
:mod:`tests.canopy_fuel.test_fuelcalc_parity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.canopy_fuel.cover import canopy_cover
from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area
from tests.canopy_fuel.builders import (
    CELL_AREA,
    SHAPE,
    TRANSFORM,
    random_stand,
    single_tree,
    stand_on_lattice,
)

# 0.15 m pixels: fine enough that discretization noise sits well under
# the tolerances below, so the analytic area is what is being tested.
FINE = 200


def cover_of(trees, **kwargs):
    kwargs.setdefault("crown_radius_column", "crad")
    return canopy_cover(trees, TRANSFORM, SHAPE, **kwargs)


class TestSingleCrown:
    def test_cover_is_the_crown_area_over_the_cell_area(self):
        cover = cover_of(single_tree(crad=4.0), supersample=FINE)
        np.testing.assert_allclose(
            cover[2, 1], 100.0 * np.pi * 16.0 / CELL_AREA, rtol=0.01
        )

    def test_no_other_cell_is_touched(self):
        cover = cover_of(single_tree(crad=4.0), supersample=FINE)
        assert cover.sum() == cover[2, 1]

    @pytest.mark.parametrize("cell", [(1, 0), (1, 1), (2, 0), (2, 1)])
    def test_a_straddling_crown_matches_the_analytic_overlap(self, cell):
        """Rasterized per-cell cover against the closed-form disk/cell area."""
        row, col = cell
        cover = cover_of(single_tree(x=1031.0, y=4941.0, crad=5.0))
        x_lo = 1000.0 + col * 30.0
        y_hi = 5000.0 - row * 30.0
        analytic = disk_rect_overlap_area(
            *(
                np.array([v])
                for v in (1031.0, 4941.0, 5.0, x_lo, x_lo + 30.0, y_hi - 30.0, y_hi)
            )
        )[0]
        np.testing.assert_allclose(
            cover[row, col], 100.0 * analytic / CELL_AREA, atol=0.35
        )


class TestOverlapIsCountedOnce:
    def test_two_identical_crowns_cover_what_one_covers(self):
        one = single_tree(crad=5.0)
        two = pd.concat([one, one], ignore_index=True)
        np.testing.assert_array_equal(cover_of(one), cover_of(two))

    def test_a_crown_nested_inside_another_adds_nothing(self):
        big = single_tree(crad=6.0)
        nested = pd.concat([big, single_tree(crad=2.0)], ignore_index=True)
        np.testing.assert_array_equal(cover_of(big), cover_of(nested))

    def test_disjoint_crowns_add_up(self):
        """12 m apart with 3 m radii, so the two disks cannot touch."""
        trees = pd.concat(
            [single_tree(x=1040.0, crad=3.0), single_tree(x=1052.0, crad=3.0)],
            ignore_index=True,
        )
        np.testing.assert_allclose(
            cover_of(trees, supersample=FINE)[2, 1],
            100.0 * 2 * np.pi * 9.0 / CELL_AREA,
            rtol=0.01,
        )


class TestCoverEdgeCases:
    def test_cover_is_a_percentage(self):
        cover = canopy_cover(stand_on_lattice(100, seed=22), TRANSFORM, SHAPE)
        assert (cover >= 0.0).all() and (cover <= 100.0).all()

    def test_an_empty_stand_covers_nothing(self):
        assert canopy_cover(random_stand(0), TRANSFORM, SHAPE).sum() == 0.0

    def test_strip_chunking_does_not_change_the_result(self, monkeypatch):
        """Masks are built in strips to bound memory; that must be invisible."""
        import fastfuels_core.canopy_fuel.cover as m

        trees = stand_on_lattice(60, seed=21)
        whole = canopy_cover(trees, TRANSFORM, SHAPE)
        monkeypatch.setattr(m, "_COVER_STRIP_BYTES", SHAPE[1] * 40**2)
        np.testing.assert_array_equal(canopy_cover(trees, TRANSFORM, SHAPE), whole)

    def test_a_rotated_transform_raises(self):
        with pytest.raises(ValueError, match="Rotated"):
            canopy_cover(
                stand_on_lattice(5), (30.0, 1.0, 1000.0, 0.0, -30.0, 5000.0), SHAPE
            )
