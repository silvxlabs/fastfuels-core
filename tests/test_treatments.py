"""Tests for :class:`fastfuels_core.treatments.DirectionalThinToTreeDensity`.

The three older treatments in that module are untested; this covers the
one added for tree-density thinning. Its parity against FuelCalc's
thinning of the tutorial stands is in
:mod:`tests.canopy_fuel.test_fuelcalc_comparison`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.treatments import (
    DirectionalThinToTreeDensity,
    ThinningDirection,
)


def stand(**diameters: int) -> pd.DataFrame:
    """A stand from ``{diameter_cm: count}``, given as ``d10=5``."""
    rows = []
    for name, count in diameters.items():
        rows += [float(name.lstrip("d"))] * count
    return pd.DataFrame({"DIA": rows, "SPECIES": "PSME"})


class TestThinningDirection:
    def test_from_below_removes_the_smallest(self):
        out = DirectionalThinToTreeDensity(target=2).apply(stand(d5=2, d20=2))
        assert sorted(out["DIA"]) == [20.0, 20.0]

    def test_from_above_removes_the_largest(self):
        out = DirectionalThinToTreeDensity(
            target=2, direction=ThinningDirection.ABOVE
        ).apply(stand(d5=2, d20=2))
        assert sorted(out["DIA"]) == [5.0, 5.0]


class TestDiameterBounds:
    """Bounds are exclusive, and trees outside them are never cut."""

    def test_a_tree_over_the_maximum_survives_the_target(self):
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(
                target=1, direction=ThinningDirection.ABOVE, max_diameter=15.0
            ).apply(stand(d5=2, d20=2))
        assert sorted(out["DIA"]) == [20.0, 20.0]

    def test_a_tree_under_the_minimum_survives_the_target(self):
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(target=1, min_diameter=10.0).apply(
                stand(d5=2, d20=2)
            )
        assert sorted(out["DIA"]) == [5.0, 5.0]

    def test_a_tree_exactly_on_the_bound_is_not_eligible(self):
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(
                target=0, max_diameter=10.0, min_diameter=10.0
            ).apply(stand(d10=3))
        assert len(out) == 3


class TestCutEfficiency:
    def test_it_caps_what_one_record_gives_up(self):
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(
                target=1, direction=ThinningDirection.ABOVE, cut_efficiency=0.5
            ).apply(stand(d20=10))
        assert len(out) == 5

    def test_records_are_reached_in_turn_rather_than_cleared(self):
        """The second record is only touched once the first has given
        up its share, and the target can stop the walk part way."""
        out = DirectionalThinToTreeDensity(
            target=12, direction=ThinningDirection.ABOVE, cut_efficiency=0.9
        ).apply(stand(d20=10, d10=10))
        assert sorted(out["DIA"]) == [10.0] * 10 + [20.0] * 2

    def test_identical_trees_of_different_species_are_separate_records(self):
        trees = pd.DataFrame(
            {"DIA": [20.0] * 20, "SPECIES": ["PSME"] * 10 + ["PIPO"] * 10}
        )
        out = DirectionalThinToTreeDensity(target=2, cut_efficiency=0.9).apply(trees)
        assert sorted(out["SPECIES"]) == ["PIPO", "PSME"]

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_it_must_be_a_fraction(self, bad):
        with pytest.raises(ValueError, match="cut_efficiency"):
            DirectionalThinToTreeDensity(target=1, cut_efficiency=bad)


class TestTargetIsUnreachable:
    def test_it_warns_and_leaves_what_it_cannot_cut(self):
        with pytest.warns(RuntimeWarning, match="above the target"):
            out = DirectionalThinToTreeDensity(target=1, max_diameter=10.0).apply(
                stand(d20=5)
            )
        assert len(out) == 5

    def test_a_stand_already_under_target_is_untouched(self):
        trees = stand(d20=3)
        out = DirectionalThinToTreeDensity(target=10).apply(trees)
        pd.testing.assert_frame_equal(out, trees)


class TestFrameHandling:
    def test_the_input_is_not_modified(self):
        trees = stand(d5=5, d20=5)
        before = trees.copy()
        DirectionalThinToTreeDensity(target=2).apply(trees)
        pd.testing.assert_frame_equal(trees, before)

    def test_duplicate_index_labels_do_not_take_extra_trees(self):
        """A stand replicated from expansion factors repeats labels."""
        trees = stand(d5=4, d20=4).set_axis(np.repeat([0, 1], 4))
        out = DirectionalThinToTreeDensity(target=6).apply(trees)
        assert len(out) == 6

    def test_survivors_keep_their_original_order(self):
        trees = stand(d20=2, d5=2, d30=2)
        out = DirectionalThinToTreeDensity(target=4).apply(trees)
        assert list(out["DIA"]) == [20.0, 20.0, 30.0, 30.0]

    def test_no_helper_columns_are_left_behind(self):
        trees = stand(d5=5, d20=5)
        out = DirectionalThinToTreeDensity(target=2).apply(trees)
        assert list(out.columns) == list(trees.columns)

    def test_an_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="thinning direction"):
            DirectionalThinToTreeDensity(target=1, direction="sideways")
