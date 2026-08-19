"""Tests for :mod:`fastfuels_core.canopy_fuel.bulk_density`.

CBD is the maximum running mean of the profile over a fixed-depth slab.
The denominator is the window depth at every height, which is what makes
the same slab of fuel report the same density wherever it sits;
:mod:`tests.canopy_fuel.test_fuelcalc_parity` pins that invariance
against the implementation it diverges from.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.bulk_density import (
    VALID_EDGES,
    cbd_running_mean,
    profile_running_mean,
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


class TestFixedDepthDenominator:
    def test_a_profile_shallower_than_the_window_is_diluted(self):
        """Two 3 kg/m3 layers under a 3 m window average to 2, not 3.

        The window is a slab of fixed depth, so it is not shortened at
        the ends of the profile; a canopy thinner than the window really
        does have a lower bulk density over any 3 m slab containing it.
        """
        np.testing.assert_allclose(
            cbd_running_mean(column_profile([3.0, 3.0]), layer_depth=1.0, window=3.0),
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

    def test_the_default_is_the_fixed_depth_slab(self):
        """Pinned: CBD stays invariant to how high the canopy sits."""
        profile = column_profile([3.0, 3.0])
        assert cbd_running_mean(profile, layer_depth=1.0, window=3.0) == pytest.approx(
            cbd_running_mean(profile, layer_depth=1.0, window=3.0, edge="slab")
        )
        assert cbd_running_mean(profile, layer_depth=1.0, window=3.0) != pytest.approx(
            cbd_running_mean(profile, layer_depth=1.0, window=3.0, edge="fuelcalc")
        )

    @pytest.mark.parametrize("edge", VALID_EDGES)
    def test_a_canopy_clear_of_both_ends_is_the_same_under_all_three(self, edge):
        profile = column_profile([0, 0, 0, 2, 4, 6, 0, 0, 0])
        assert cbd_running_mean(
            profile, layer_depth=1.0, window=3.0, edge=edge
        ) == pytest.approx(4.0)
