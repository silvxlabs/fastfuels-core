"""Tests for :mod:`fastfuels_core.canopy_fuel.canopy_height`.

CBH and canopy height come from one scan of the profile against a
threshold, so they are asserted as the pair the function returns. The
span they bracket is the union of the qualifying layers: CBH is the
bottom of the lowest and canopy height the top of the highest, which
keeps ``chm - cbh`` the true depth of qualifying canopy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.canopy_fuel.bulk_density import VALID_EDGES
from fastfuels_core.canopy_fuel.canopy_height import (
    crown_base_statistic,
    height_percentile,
    height_percentile_depth,
    mean_crown_length,
    profile_threshold_heights,
    validate_cbh_method,
    validate_crown_base_percentile,
    validate_chm_method,
)
from tests.canopy_fuel.builders import (
    ONE_CELL_SHAPE,
    ONE_CELL_TRANSFORM,
    column_profile,
)


def sparse_profile(seed, occupancy):
    """A noisy profile with most cells and layers empty."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 0.05, (40, 6, 7)) * (
        rng.uniform(0, 1, (40, 6, 7)) > occupancy
    )


class TestThresholdRule:
    def test_the_span_brackets_the_qualifying_layers(self):
        """Layers 2-4 clear 0.012, so the span is their union, 2 m to 5 m."""
        profile = column_profile([0, 0.005, 0.02, 0.05, 0.02, 0.005])
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=None
        )
        np.testing.assert_allclose(cbh, [[2.0]])
        np.testing.assert_allclose(chm, [[5.0]])

    def test_a_layer_exactly_at_threshold_qualifies(self):
        """The guide's rule is "greater than or equal to"; pin the equality."""
        profile = column_profile([0, 0.012, 0.05, 0.012, 0])
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=None
        )
        np.testing.assert_allclose(cbh, [[1.0]])
        np.testing.assert_allclose(chm, [[4.0]])

    def test_the_relative_rule_lowers_the_threshold_in_a_sparse_cell(self):
        """max = 0.05 -> min(0.005, 0.012) = 0.005, so 0.006 now qualifies."""
        profile = column_profile([0, 0.006, 0.02, 0.05, 0.02, 0.006])
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=0.1
        )
        np.testing.assert_allclose(cbh, [[1.0]])
        np.testing.assert_allclose(chm, [[6.0]])

    def test_a_single_qualifying_layer_still_has_positive_depth(self):
        """chm - cbh must never be zero; a load-over-depth CBD divides by it."""
        cbh, chm = profile_threshold_heights(
            column_profile([0, 0.05, 0]), layer_depth=1.0, relative_fraction=None
        )
        assert chm[0, 0] - cbh[0, 0] == pytest.approx(1.0)

    def test_a_cell_with_no_canopy_is_nan(self):
        profile = np.zeros((4, 2, 2))
        profile[1, 0, 0] = 0.5
        cbh, chm = profile_threshold_heights(profile, layer_depth=1.0)
        assert np.isnan(cbh[1, 1]) and np.isnan(chm[1, 1])

    def test_a_cell_with_canopy_is_not_nan(self):
        profile = np.zeros((4, 2, 2))
        profile[1, 0, 0] = 0.5
        cbh, chm = profile_threshold_heights(profile, layer_depth=1.0)
        np.testing.assert_allclose(cbh[0, 0], 1.0)
        np.testing.assert_allclose(chm[0, 0], 2.0)

    def test_the_base_is_never_above_the_top(self):
        cbh, chm = profile_threshold_heights(sparse_profile(4, 0.7), layer_depth=0.3048)
        defined = ~np.isnan(cbh)
        assert (cbh[defined] < chm[defined]).all()


class TestSmoothing:
    """Smoothing spreads density past the canopy; the extent bounds undo it."""

    def test_the_span_is_pulled_back_to_the_layers_holding_fuel(self):
        """The smoothed scan alone would claim layers 1 and 5, which are empty."""
        profile = column_profile([0, 0, 0.03, 0.03, 0, 0, 0])
        cbh, chm = profile_threshold_heights(
            profile,
            layer_depth=1.0,
            threshold=0.012,
            relative_fraction=None,
            smoothing_window=3.0,
        )
        np.testing.assert_allclose(cbh, [[2.0]])
        np.testing.assert_allclose(chm, [[4.0]])

    def test_a_thin_canopy_qualifies_before_smoothing(self):
        cbh, chm = profile_threshold_heights(
            column_profile([0, 0.03, 0, 0]),
            layer_depth=1.0,
            threshold=0.012,
            relative_fraction=None,
            smoothing_window=None,
        )
        np.testing.assert_allclose(cbh, [[1.0]])
        np.testing.assert_allclose(chm, [[2.0]])

    def test_smoothing_can_dilute_a_thin_canopy_to_nothing(self):
        """[0, 0.03, 0, 0] smooths to [0.015, 0.01, 0.01, 0].

        Truncating the window at the profile floor averages layer 0 over
        two layers instead of three, so the only layer clearing 0.012
        holds no fuel while the one that holds fuel falls short. The
        qualifying span and the fuel span are disjoint, which is no
        canopy.
        """
        cbh, chm = profile_threshold_heights(
            column_profile([0, 0.03, 0, 0]),
            layer_depth=1.0,
            threshold=0.012,
            relative_fraction=None,
            smoothing_window=3.0,
        )
        assert np.isnan(cbh).all() and np.isnan(chm).all()

    def test_the_extent_bounds_are_inert_without_smoothing(self):
        """Unsmoothed, every qualifying layer holds fuel by construction."""
        profile = sparse_profile(11, 0.6)
        cbh, chm = profile_threshold_heights(profile, layer_depth=0.3048)
        occupied = profile > 0
        lowest = occupied.argmax(axis=0) * 0.3048
        highest = (40 - occupied[::-1].argmax(axis=0)) * 0.3048
        defined = ~np.isnan(cbh)
        assert (cbh[defined] >= lowest[defined] - 1e-12).all()
        assert (chm[defined] <= highest[defined] + 1e-12).all()


class TestSmoothingEdge:
    """Which convention the scan smooths with moves the reported top.

    ``smoothing_edge`` only matters when ``smoothing_window`` is set,
    and only near the ends of the profile, but that is exactly where
    both heights are read off.
    """

    # A crown occupying layers 3-6 of a 10 m column, over a threshold
    # low enough that a diluted layer still clears it.
    CANOPY = [0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0, 0, 0]

    def heights(self, values, **kwargs):
        cbh, chm = profile_threshold_heights(
            column_profile(values), layer_depth=1.0, **kwargs
        )
        return float(cbh[0, 0]), float(chm[0, 0])

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_it_is_inert_without_a_smoothing_window(self, edge):
        """With no window there is nothing for the convention to decide."""
        assert self.heights(self.CANOPY, smoothing_edge=edge) == self.heights(
            self.CANOPY
        )

    def test_truncate_carries_the_top_a_layer_higher(self):
        """A sparse crown tip clears the threshold once inflated.

        Layers 5-7 are dense and 8-9 are the thin top of the crown. The
        topmost window runs past the profile, so truncating it divides
        by three where the other two divide by five, and the tip clears
        a threshold it otherwise misses. The extent bound cannot undo
        this: those layers do hold fuel.
        """
        crown = [0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.002, 0.002]
        tops = {
            edge: self.heights(
                crown,
                threshold=0.015,
                relative_fraction=None,
                smoothing_window=5.0,
                smoothing_edge=edge,
            )[1]
            for edge in VALID_EDGES
        }
        assert tops["truncate"] == pytest.approx(10.0)
        assert tops["fuelcalc"] == pytest.approx(9.0)
        assert tops["slab"] == pytest.approx(9.0)

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_a_canopy_clear_of_both_ends_reads_the_same(self, edge):
        """The conventions differ only where the window runs off."""
        assert self.heights(
            self.CANOPY, smoothing_window=3.0, smoothing_edge=edge
        ) == self.heights(self.CANOPY, smoothing_window=3.0)

    def test_the_default_is_fuelcalcs(self):
        """Pinned, because the three disagree on where the canopy ends."""
        crown = [0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.002, 0.002]
        settings = dict(threshold=0.015, relative_fraction=None, smoothing_window=5.0)
        assert self.heights(crown, **settings) == self.heights(
            crown, smoothing_edge="fuelcalc", **settings
        )

    def test_an_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            self.heights(self.CANOPY, smoothing_window=5.0, smoothing_edge="bogus")


class TestCrownBaseStatistic:
    """CBH as a plain summary statistic of the per-tree crown bases in a cell.

    The crown base of a tree is ``height * (1 - crown_ratio)``; a cell's
    value reduces its trees' crown bases by ``statistic``. The statistics
    form a conservative-to-liberal spectrum (minimum <= lower percentile
    <= median <= mean for a right-tailed stand), which is the point of
    offering them: the mean is Van Wagner's stand definition, the lower
    tail captures low ladder fuel a mean would hide.
    """

    @staticmethod
    def four_trees_in_one_cell():
        # Crown bases 2, 4, 6, 12 m, all at the cell centre so they share
        # one group. The tallest carries the most fuel.
        trees = pd.DataFrame(
            {
                "x": [15.0, 15.0, 15.0, 15.0],
                "y": [-15.0, -15.0, -15.0, -15.0],
                "height": [10.0, 10.0, 10.0, 20.0],
                "crown_ratio": [0.8, 0.6, 0.4, 0.4],
            }
        )
        fuel = np.array([1.0, 1.0, 1.0, 10.0])
        return trees, fuel

    def value(self, **kwargs):
        trees, fuel = self.four_trees_in_one_cell()
        return crown_base_statistic(
            trees, fuel, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE, **kwargs
        )

    def test_minimum_is_the_lowest_crown_base(self):
        np.testing.assert_allclose(self.value(statistic="minimum"), [[2.0]])

    def test_the_median_is_the_middle_crown_base(self):
        np.testing.assert_allclose(
            self.value(statistic="percentile", percentile=50), [[5.0]]
        )

    def test_a_lower_percentile_is_conservative(self):
        np.testing.assert_allclose(
            self.value(statistic="percentile", percentile=20), [[3.2]]
        )

    def test_the_mean_is_one_tree_one_vote_by_default(self):
        # (2 + 4 + 6 + 12) / 4 = 6.0, ignoring fuel weight.
        np.testing.assert_allclose(self.value(statistic="mean"), [[6.0]])

    def test_the_weighted_mean_lets_heavy_crowns_pull_it(self):
        # (2 + 4 + 6 + 10*12) / (1 + 1 + 1 + 10) = 132 / 13.
        np.testing.assert_allclose(
            self.value(statistic="mean", weight_by_available_fuel=True),
            [[132.0 / 13.0]],
        )

    def test_the_statistics_form_a_conservative_spectrum(self):
        minimum = float(self.value(statistic="minimum")[0, 0])
        p20 = float(self.value(statistic="percentile", percentile=20)[0, 0])
        median = float(self.value(statistic="percentile", percentile=50)[0, 0])
        mean = float(self.value(statistic="mean")[0, 0])
        assert minimum <= p20 <= median <= mean

    def test_an_empty_stand_is_all_nan(self):
        trees = pd.DataFrame({"x": [], "y": [], "height": [], "crown_ratio": []})
        out = crown_base_statistic(
            trees, np.array([]), ONE_CELL_TRANSFORM, ONE_CELL_SHAPE
        )
        assert np.isnan(out).all()


class TestValidateCrownBasePercentile:
    @pytest.mark.parametrize("method", ["minimum", "mean", "bulk_density_threshold"])
    def test_methods_without_a_percentile_pass(self, method):
        validate_crown_base_percentile(method, None)

    def test_percentile_with_a_value_passes(self):
        validate_crown_base_percentile("percentile", 20.0)

    def test_percentile_without_a_value_raises(self):
        with pytest.raises(ValueError, match="requires a percentile"):
            validate_crown_base_percentile("percentile", None)


class TestValidateCbhMethod:
    @pytest.mark.parametrize(
        "method", ["bulk_density_threshold", "mean", "percentile", "minimum"]
    )
    def test_the_methods_pass(self, method):
        validate_cbh_method(method)

    def test_an_unknown_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            validate_cbh_method("bogus")


class TestHeightPercentile:
    """A per-cell percentile of tree heights, an alternative canopy height.

    Unlike the threshold canopy height, this reads the tree heights
    directly, so it is the measure to compare against a lidar canopy
    height model.
    """

    @staticmethod
    def three_trees(percentile):
        # Heights 10, 20, 30 m, all in the single cell.
        trees = pd.DataFrame(
            {
                "x": [15.0, 15.0, 15.0],
                "y": [-15.0, -15.0, -15.0],
                "height": [10.0, 20.0, 30.0],
            }
        )
        return height_percentile(
            trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE, percentile=percentile
        )

    def test_the_hundredth_percentile_is_the_tallest(self):
        np.testing.assert_allclose(self.three_trees(100.0), [[30.0]])

    def test_the_zeroth_percentile_is_the_shortest(self):
        np.testing.assert_allclose(self.three_trees(0.0), [[10.0]])

    def test_the_fiftieth_percentile_is_the_median(self):
        np.testing.assert_allclose(self.three_trees(50.0), [[20.0]])

    def test_it_interpolates_between_heights(self):
        # 99th of [10, 20, 30] linearly interpolates to 29.8 m.
        np.testing.assert_allclose(self.three_trees(99.0), [[29.8]])

    def test_the_default_percentile_is_the_ninety_ninth(self):
        trees = pd.DataFrame(
            {"x": [15.0, 15.0], "y": [-15.0, -15.0], "height": [10.0, 30.0]}
        )
        out = height_percentile(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        np.testing.assert_allclose(out, [[29.8]])

    def test_an_empty_stand_is_all_nan(self):
        trees = pd.DataFrame({"x": [], "y": [], "height": []})
        out = height_percentile(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        assert np.isnan(out).all()


class TestValidateChmMethod:
    def test_the_two_methods_pass(self):
        validate_chm_method("bulk_density_threshold")
        validate_chm_method("height_percentile")

    def test_an_unknown_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            validate_chm_method("bogus")


class TestMeanCrownLength:
    """The mean per-tree crown length, a load-over-depth canopy depth."""

    def test_it_averages_the_crown_lengths_in_the_cell(self):
        # Crown lengths height*crown_ratio: 5 m and 15 m, mean 10 m.
        trees = pd.DataFrame(
            {
                "x": [15.0, 15.0],
                "y": [-15.0, -15.0],
                "height": [10.0, 30.0],
                "crown_ratio": [0.5, 0.5],
            }
        )
        out = mean_crown_length(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        np.testing.assert_allclose(out, [[10.0]])

    def test_it_is_unweighted(self):
        # One tree one vote, whatever the crown sizes; a 2 m and a 10 m
        # crown length average to 6 m.
        trees = pd.DataFrame(
            {
                "x": [15.0, 15.0],
                "y": [-15.0, -15.0],
                "height": [4.0, 20.0],
                "crown_ratio": [0.5, 0.5],
            }
        )
        out = mean_crown_length(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        np.testing.assert_allclose(out, [[6.0]])

    def test_an_empty_stand_is_all_nan(self):
        trees = pd.DataFrame({"x": [], "y": [], "height": [], "crown_ratio": []})
        out = mean_crown_length(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        assert np.isnan(out).all()


class TestHeightPercentileDepth:
    """90th-percentile tree height minus median crown base, a canopy depth."""

    def test_it_is_the_top_height_over_the_median_base(self):
        # Heights 10/20/30 -> 90th percentile 28; crown bases (cr=0.5)
        # 5/10/15 -> median 10. Depth 28 - 10 = 18 m.
        trees = pd.DataFrame(
            {
                "x": [15.0, 15.0, 15.0],
                "y": [-15.0, -15.0, -15.0],
                "height": [10.0, 20.0, 30.0],
                "crown_ratio": [0.5, 0.5, 0.5],
            }
        )
        out = height_percentile_depth(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        np.testing.assert_allclose(out, [[18.0]])

    def test_an_empty_stand_is_all_nan(self):
        trees = pd.DataFrame({"x": [], "y": [], "height": [], "crown_ratio": []})
        out = height_percentile_depth(trees, ONE_CELL_TRANSFORM, ONE_CELL_SHAPE)
        assert np.isnan(out).all()
