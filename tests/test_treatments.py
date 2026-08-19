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
    """The cut runs at a steady rate along the eligible trees.

    Every fixture above builds rows that are identical apart from
    diameter, which is the shape a stand table expands to. A stem list
    from a real inventory is the opposite -- every tree distinct -- and
    :class:`TestDistinctStems` covers that, because a rate defined
    per-group of identical trees silently removes nothing there.
    """

    def test_it_caps_the_cut(self):
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(
                target=1, direction=ThinningDirection.ABOVE, cut_efficiency=0.5
            ).apply(stand(d20=10))
        assert len(out) == 5

    def test_a_run_of_equal_trees_keeps_its_share(self):
        """A steady rate takes cut_efficiency of each contiguous run.

        The twenties are reached first and give up nine of ten; the
        target stops the walk before the tens are touched at all.
        """
        out = DirectionalThinToTreeDensity(
            target=12, direction=ThinningDirection.ABOVE, cut_efficiency=0.9
        ).apply(stand(d20=10, d10=10))
        assert sorted(out["DIA"]) == [10.0] * 10 + [20.0] * 2

    def test_the_rate_carries_across_a_tie_in_diameter(self):
        """Two species at one diameter each keep their tenth."""
        trees = pd.DataFrame(
            {"DIA": [20.0] * 20, "SPECIES": ["PSME"] * 10 + ["PIPO"] * 10}
        )
        out = DirectionalThinToTreeDensity(target=2, cut_efficiency=0.9).apply(trees)
        assert sorted(out["SPECIES"]) == ["PIPO", "PSME"]

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_it_must_be_a_fraction(self, bad):
        with pytest.raises(ValueError, match="cut_efficiency"):
            DirectionalThinToTreeDensity(target=1, cut_efficiency=bad)


class TestDistinctStems:
    """A stem list where no two trees are alike.

    This is what a FastFuels inventory looks like -- one row per tree,
    each with its own coordinates -- and it is the shape that breaks a
    cut rate defined as a fraction of each group of identical trees.
    """

    @staticmethod
    def stem_list(n=100):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "DIA": np.linspace(5.0, 25.0, n),
                "X": rng.uniform(0, 100, n),
                "Y": rng.uniform(0, 100, n),
            }
        )

    def test_it_reaches_the_target(self):
        out = DirectionalThinToTreeDensity(target=50, cut_efficiency=0.9).apply(
            self.stem_list()
        )
        assert len(out) == 50

    def test_cut_efficiency_is_a_fraction_of_the_trees(self):
        """With no target to stop it, the rate is what remains."""
        with pytest.warns(RuntimeWarning):
            out = DirectionalThinToTreeDensity(target=0, cut_efficiency=0.4).apply(
                self.stem_list()
            )
        assert len(out) == 60

    def test_it_still_thins_from_below(self):
        out = DirectionalThinToTreeDensity(target=50).apply(self.stem_list())
        assert out["DIA"].min() > self.stem_list()["DIA"].median()

    def test_a_column_of_nulls_does_not_protect_a_tree(self):
        """Grouping on all columns would drop these from eligibility."""
        trees = self.stem_list().assign(CROWN_CLASS=[None] * 80 + ["C"] * 20)
        out = DirectionalThinToTreeDensity(target=50).apply(trees)
        assert len(out) == 50


class TestTargetIsUnreachable:
    def test_cut_efficiency_can_hold_the_target_out_of_reach(self):
        with pytest.warns(RuntimeWarning, match="above the target"):
            out = DirectionalThinToTreeDensity(target=0, cut_efficiency=0.5).apply(
                stand(d20=10)
            )
        assert len(out) == 5

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
