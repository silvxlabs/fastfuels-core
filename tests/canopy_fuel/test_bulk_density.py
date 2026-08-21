"""Tests for :mod:`fastfuels_core.canopy_fuel.bulk_density`.

CBD is the maximum running mean of the profile. What the mean takes to
lie past the ends of the profile is the ``edge`` choice; the default is
FuelCalc's ground-clamped rule, and :mod:`tests.canopy_fuel.test_fuelcalc_parity`
pins it against the C. The fixed-depth ``slab`` rule is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.bulk_density import (
    VALID_EDGES,
    biomass_percentile_depth,
    cbd_load_over_depth,
    cbd_running_mean,
    profile_running_mean,
    validate_cbd_depth,
    validate_cbd_method,
    window_in_layers,
)
from tests.canopy_fuel.builders import column_profile


class TestRunningMean:
    def test_takes_the_maximum_window_mean(self):
        """3 m windows over [0,0,2,4,6,0] are 2/3, 2, 4, 10/3 -> 4."""
        profile = column_profile([0, 0, 2, 4, 6, 0])
        np.testing.assert_allclose(
            cbd_running_mean(profile, layer_depth=1.0, window=3.0), [[4.0]]
        )

    def test_no_window_is_the_densest_single_layer(self):
        profile = column_profile([0, 0, 2, 4, 6, 0])
        np.testing.assert_allclose(
            cbd_running_mean(profile, layer_depth=1.0, window=None), [[6.0]]
        )

    def test_smoothing_can_only_lower_the_answer(self):
        rng = np.random.default_rng(3)
        profile = rng.uniform(0, 1, (30, 4, 5))
        smoothed = cbd_running_mean(profile, layer_depth=0.3048, window=3.0)
        peak = cbd_running_mean(profile, layer_depth=0.3048, window=None)
        assert (smoothed <= peak + 1e-12).all()


class TestWindowInLayers:
    """A depth in metres resolves to one odd layer count for every edge."""

    @pytest.mark.parametrize(
        "window, layer_depth, expected",
        [
            (5 * 0.3048, 0.3048, 5),  # FuelCalc's five 1-ft layers
            (3.0, 0.3048, 11),  # 9.84 layers -> 10 -> widened to 11
            (3.0, 1.0, 3),
            (0.1, 1.0, 1),  # never below one layer
        ],
    )
    def test_resolves_to_the_nearest_odd_count(self, window, layer_depth, expected):
        assert window_in_layers(window, layer_depth) == expected

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_an_even_window_spans_the_same_layers_under_every_edge(self, edge):
        """Ten unit layers mid-profile under a 10-layer window.

        Resolved to 11 layers, no window contains the whole canopy, so
        every edge reports 10/11 -- not 1.0 under one rule and 10/11
        under the others.
        """
        profile = column_profile([0.0] * 20 + [1.0] * 10 + [0.0] * 20)
        assert cbd_running_mean(
            profile, layer_depth=1.0, window=10.0, edge=edge
        ) == pytest.approx(10.0 / 11.0)


class TestFixedDepthDenominator:
    def test_a_profile_shallower_than_the_window_is_diluted(self):
        """Two 3 kg/m3 layers under a 3 m window average to 2, not 3.

        The window is a slab of fixed depth, so it is not shortened at
        the ends of the profile; a canopy thinner than the window really
        does have a lower bulk density over any 3 m slab containing it.
        """
        np.testing.assert_allclose(
            cbd_running_mean(
                column_profile([3.0, 3.0]),
                layer_depth=1.0,
                window=3.0,
                edge="slab",
            ),
            [[2.0]],
        )


class TestEdgeConventions:
    """A running mean needs a rule for what lies past the profile ends.

    The three disagree only near the ends, and by enough to move a
    reported height a layer, which is why the choice is named rather
    than assumed.
    """

    # Windows of 5 over [1..7]. Interior layers agree; the ends do not.
    RAMP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    @pytest.mark.parametrize(
        "edge, expected",
        [
            # Zero-padded both ends, denominator always 5.
            ("slab", [1.2, 2.0, 3.0, 4.0, 5.0, 4.4, 3.6]),
            # Clamped at the ground and counted short there; zero-padded
            # above against the full denominator.
            ("fuelcalc", [2.0, 2.5, 3.0, 4.0, 5.0, 4.4, 3.6]),
            # Counted short at both ends.
            ("truncate", [2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0]),
        ],
    )
    def test_each_convention_treats_the_ends_its_own_way(self, edge, expected):
        got = profile_running_mean(column_profile(self.RAMP), 5, edge=edge)
        np.testing.assert_allclose(got.ravel(), expected)

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_the_interior_is_the_same_under_all_three(self, edge):
        got = profile_running_mean(column_profile(self.RAMP), 5, edge=edge)
        np.testing.assert_allclose(got.ravel()[2:5], [3.0, 4.0, 5.0])

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_a_window_of_one_layer_is_the_profile(self, edge):
        profile = column_profile(self.RAMP)
        np.testing.assert_allclose(profile_running_mean(profile, 1, edge=edge), profile)

    def test_truncate_reports_more_at_the_top_than_the_other_two(self):
        """Dividing the topmost window by a short denominator inflates it.

        This is what carries a height threshold past the canopy: the
        layers above the crown are averaged over however few layers
        remain instead of against the full window.
        """
        top = {
            edge: profile_running_mean(column_profile(self.RAMP), 5, edge=edge).ravel()[
                -1
            ]
            for edge in VALID_EDGES
        }
        assert top["truncate"] > top["fuelcalc"]
        assert top["fuelcalc"] == pytest.approx(top["slab"])

    def test_an_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            profile_running_mean(column_profile(self.RAMP), 5, edge="bogus")


class TestCbdEdgeSelection:
    def test_fuelcalc_concentrates_a_canopy_resting_on_the_ground(self):
        """The shrinking ground denominator reports a higher density.

        Two 3 kg/m3 layers on the ground average to 2 over a fixed 3 m
        slab and to 3 under FuelCalc's rule, which shortens the window
        at the ground rather than padding it.
        """
        profile = column_profile([3.0, 3.0])
        assert cbd_running_mean(
            profile, layer_depth=1.0, window=3.0, edge="slab"
        ) == pytest.approx(2.0)
        assert cbd_running_mean(
            profile, layer_depth=1.0, window=3.0, edge="fuelcalc"
        ) == pytest.approx(3.0)

    def test_the_default_is_fuelcalcs(self):
        """Pinned, because the three disagree against the ground."""
        profile = column_profile([3.0, 3.0])
        assert cbd_running_mean(profile, layer_depth=1.0, window=3.0) == pytest.approx(
            cbd_running_mean(profile, layer_depth=1.0, window=3.0, edge="fuelcalc")
        )
        assert cbd_running_mean(profile, layer_depth=1.0, window=3.0) != pytest.approx(
            cbd_running_mean(profile, layer_depth=1.0, window=3.0, edge="slab")
        )

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_a_canopy_clear_of_both_ends_is_the_same_under_all_three(self, edge):
        profile = column_profile([0, 0, 0, 2, 4, 6, 0, 0, 0])
        assert cbd_running_mean(
            profile, layer_depth=1.0, window=3.0, edge=edge
        ) == pytest.approx(4.0)


class TestLoadOverDepth:
    """CBD as canopy fuel load divided by a canopy depth."""

    def test_it_divides_load_by_depth(self):
        cbd = cbd_load_over_depth(np.array([3.0, 6.0]), np.array([2.0, 3.0]))
        np.testing.assert_allclose(cbd, [1.5, 2.0])

    def test_a_nan_depth_is_zero_density(self):
        # Empty cells carry no depth; CBD is 0 there, as elsewhere.
        cbd = cbd_load_over_depth(np.array([0.0]), np.array([np.nan]))
        np.testing.assert_array_equal(cbd, [0.0])

    def test_a_non_positive_depth_is_zero_density(self):
        cbd = cbd_load_over_depth(np.array([3.0, 3.0]), np.array([0.0, -1.0]))
        np.testing.assert_array_equal(cbd, [0.0, 0.0])


class TestBiomassPercentileDepth:
    """The height span holding the central 80% of canopy biomass."""

    def test_it_trims_a_tenth_of_the_biomass_off_each_end(self):
        # Uniform density over layers 2-4 (3 m of crown); trimming the
        # bottom and top 10% of biomass leaves the central 2.4 m.
        profile = column_profile([0, 0, 0.05, 0.05, 0.05, 0])
        depth = biomass_percentile_depth(profile, layer_depth=1.0)
        np.testing.assert_allclose(depth, [[2.4]])

    def test_it_interpolates_within_the_crossing_layer(self):
        # All biomass in one 2 m layer: the 10-90 span is 80% of it, 1.6 m.
        profile = column_profile([0, 0.1, 0])
        depth = biomass_percentile_depth(profile, layer_depth=2.0)
        np.testing.assert_allclose(depth, [[1.6]])

    def test_an_empty_cell_is_nan(self):
        profile = np.zeros((5, 1, 1))
        assert np.isnan(biomass_percentile_depth(profile, layer_depth=1.0)).all()


class TestValidateCbdMethod:
    def test_the_two_methods_pass(self):
        validate_cbd_method("maximum_running_mean")
        validate_cbd_method("load_over_depth")

    def test_an_unknown_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            validate_cbd_method("bogus")


class TestValidateCbdDepth:
    @pytest.mark.parametrize(
        "depth",
        [
            "canopy_depth",
            "mean_crown_length",
            "biomass_percentile",
            "height_percentile",
        ],
    )
    def test_the_four_depths_pass(self, depth):
        validate_cbd_depth(depth)

    def test_an_unknown_depth_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            validate_cbd_depth("bogus")
