"""Tests for :mod:`fastfuels_core.canopy_fuel.cover`.

Three methods over the same crown disks, so what separates them is how
they treat overlap and which trees they count:

``crown_union``
    Counts overlapping crowns once. Adding a crown that covers no new
    ground must not change the answer.
``crown_overlap``
    Reads only total crown area, so it cannot see arrangement. Its
    values are pinned against FuelCalc's ``CA_Overlap`` in
    :mod:`tests.canopy_fuel.test_fuelcalc_parity`; what is here is the
    behaviour that distinguishes it from the union.
``cover_fraction``
    The union restricted by tree height. It has no FuelCalc
    counterpart, so what is pinned is its relation to ``crown_union``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.canopy_fuel.cover import canopy_cover
from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species
from tests.canopy_fuel.builders import (
    CELL_AREA,
    ONE_CELL_SHAPE,
    ONE_CELL_TRANSFORM,
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
        cover = cover_of(single_tree(crad=4.0), supersample=FINE, method="crown_union")
        np.testing.assert_allclose(
            cover[2, 1], 100.0 * np.pi * 16.0 / CELL_AREA, rtol=0.01
        )

    def test_no_other_cell_is_touched(self):
        cover = cover_of(single_tree(crad=4.0), supersample=FINE, method="crown_union")
        assert cover.sum() == cover[2, 1]

    @pytest.mark.parametrize("cell", [(1, 0), (1, 1), (2, 0), (2, 1)])
    def test_a_straddling_crown_matches_the_analytic_overlap(self, cell):
        """Rasterized per-cell cover against the closed-form disk/cell area."""
        row, col = cell
        cover = cover_of(
            single_tree(x=1031.0, y=4941.0, crad=5.0), method="crown_union"
        )
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
        np.testing.assert_array_equal(
            cover_of(one, method="crown_union"), cover_of(two, method="crown_union")
        )

    def test_a_crown_nested_inside_another_adds_nothing(self):
        big = single_tree(crad=6.0)
        nested = pd.concat([big, single_tree(crad=2.0)], ignore_index=True)
        np.testing.assert_array_equal(
            cover_of(big, method="crown_union"), cover_of(nested, method="crown_union")
        )

    def test_disjoint_crowns_add_up(self):
        """12 m apart with 3 m radii, so the two disks cannot touch."""
        trees = pd.concat(
            [single_tree(x=1040.0, crad=3.0), single_tree(x=1052.0, crad=3.0)],
            ignore_index=True,
        )
        np.testing.assert_allclose(
            cover_of(trees, supersample=FINE, method="crown_union")[2, 1],
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


def one_cell_stand(heights, x=None, y=None, dbh=None, species_code=122):
    """A stand inside the single 30 m cell, sized by Purves allometry."""
    n = len(heights)
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "x": rng.uniform(4.0, 26.0, n) if x is None else x,
            "y": -rng.uniform(4.0, 26.0, n) if y is None else y,
            "fia_species_code": species_code,
            "dbh": np.linspace(5.0, 40.0, n) if dbh is None else dbh,
            "height": np.asarray(heights, dtype=float),
            "crown_ratio": np.full(n, 0.6),
        }
    )


def one_cell_cover(trees, **kwargs):
    return canopy_cover(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE, **kwargs)[0, 0]


class TestCrownOverlap:
    """Crookston & Stage's random-overlap correction."""

    def test_it_cannot_see_arrangement(self):
        """Its defining property, and its limitation.

        The estimator reads only total crown area, so translating stems
        within a cell cannot move it. The union sees the difference,
        which is the whole reason both methods exist.
        """
        rng = np.random.default_rng(11)
        n = 25
        layouts = [
            (rng.uniform(4, 26, n), -rng.uniform(4, 26, n)),
            (
                np.tile(np.linspace(4, 26, 5), 5),
                -np.repeat(np.linspace(4, 26, 5), 5),
            ),
            (rng.normal(15, 1.5, n), -rng.normal(15, 1.5, n)),
        ]
        overlap, union = [], []
        for x, y in layouts:
            trees = one_cell_stand(
                np.full(n, 15.0),
                x=np.clip(x, 3, 27),
                y=-np.clip(-y, 3, 27),
                dbh=np.full(n, 25.0),
            )
            overlap.append(one_cell_cover(trees, method="crown_overlap"))
            union.append(one_cell_cover(trees, method="crown_union"))
        assert max(overlap) - min(overlap) < 1e-9
        assert max(union) - min(union) > 20.0

    def test_it_agrees_with_the_union_when_nothing_can_overlap(self):
        """One crown in a cell: no overlap to resolve, so both are exact.

        1 - exp(-p) != p, so they agree only in the limit; a crown
        covering under 2% of the cell puts the two within a tenth of a
        point.
        """
        trees = one_cell_stand([7.0], x=[15.0], y=[-15.0], dbh=[8.0])
        union = one_cell_cover(trees)
        assert union < 2.0
        assert one_cell_cover(trees, method="crown_overlap") == pytest.approx(
            union, abs=0.1
        )


class TestCoverFraction:
    """The union restricted by tree height, the CHM-comparable variable."""

    def test_a_zero_threshold_is_the_plain_union(self):
        trees = one_cell_stand(np.linspace(0.5, 25.0, 30))
        assert one_cell_cover(
            trees, method="cover_fraction", height_threshold=0.0
        ) == pytest.approx(one_cell_cover(trees, method="crown_union"), abs=1e-12)

    def test_raising_the_threshold_can_only_lower_cover(self):
        trees = one_cell_stand(np.linspace(0.5, 25.0, 30))
        covers = [
            one_cell_cover(trees, method="cover_fraction", height_threshold=t)
            for t in np.arange(0.0, 30.0, 0.5)
        ]
        assert all(b <= a + 1e-12 for a, b in zip(covers, covers[1:]))
        assert covers[0] > 0.0
        assert covers[-1] == 0.0

    def test_the_threshold_is_strict(self):
        """A tree exactly at the threshold does not clear it."""
        trees = one_cell_stand([2.0, 2.0])
        assert one_cell_cover(trees, method="cover_fraction", height_threshold=2.0) == 0
        assert one_cell_cover(trees, method="cover_fraction", height_threshold=1.99) > 0

    def test_understorey_is_what_separates_it_from_the_union(self):
        trees = one_cell_stand([1.0, 1.2, 1.5, 18.0])
        fraction = one_cell_cover(trees, method="cover_fraction", height_threshold=2.0)
        assert fraction < one_cell_cover(trees) - 1.0

    def test_a_stand_entirely_below_the_threshold_covers_nothing(self):
        assert (
            one_cell_cover(one_cell_stand([0.5, 1.0, 1.9]), method="cover_fraction")
            == 0.0
        )

    def test_a_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="height_threshold"):
            one_cell_cover(
                one_cell_stand([10.0]), method="cover_fraction", height_threshold=-1.0
            )


@pytest.mark.parametrize("method", ["crown_union", "crown_overlap"])
def test_cover_counts_species_excluded_from_bulk_density(method):
    """Cover is not gated by the species inclusion flag.

    Broadleaf canopy occupies ground whether or not it is treated as
    crown-fire fuel, so ``exclude_hardwoods`` gates the bulk-density
    bands and leaves cover alone. FuelCalc does the same: ``PTL_CanCov``
    (``NC_PTL2.C:44``) loops over every live record, and the inclusion
    flag is read in exactly one place in the whole source,
    ``NC_PTL.C:731``, inside the bulk-density loop.
    """
    excluded = fuelcalc_species()
    excluded = excluded[excluded["INCL_CBD"] == "No"]
    assert not excluded.empty
    trees = one_cell_stand(
        [18.0], x=[15.0], y=[-15.0], dbh=[30.0], species_code=int(excluded.index[0])
    )
    assert one_cell_cover(trees, method=method) > 0.0


def test_an_unknown_method_raises():
    with pytest.raises(ValueError, match="canopy cover method"):
        one_cell_cover(one_cell_stand([15.0]), method="crookston")
