"""Tests for :mod:`fastfuels_core.canopy_fuel.bulk_density`.

CBD is the maximum running mean of the profile over a fixed-depth slab.
The denominator is the window depth at every height, which is what makes
the same slab of fuel report the same density wherever it sits;
:mod:`tests.canopy_fuel.test_fuelcalc_parity` pins that invariance
against the implementation it diverges from.
"""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.bulk_density import cbd_running_mean
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
