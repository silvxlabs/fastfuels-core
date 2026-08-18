"""Tests for :mod:`fastfuels_core.canopy_fuel.canopy_height`.

CBH and canopy height come from one scan of the profile against a
threshold, so they are asserted as the pair the function returns. The
span they bracket is the union of the qualifying layers: CBH is the
bottom of the lowest and canopy height the top of the highest, which
keeps ``chm - cbh`` the true depth of qualifying canopy.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.canopy_height import profile_threshold_heights
from tests.canopy_fuel.builders import column_profile


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
