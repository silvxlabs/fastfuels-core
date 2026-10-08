"""Tests for :mod:`fastfuels_core.fuel_models.disturbed_fuel_update`.

The FDist and zone rasters are passed in directly, and rules are built by
hand, so each expected fuel model is known. How FDist is built is tested in
:mod:`tests.fuel_models.test_fdist_builder`; how rules are matched in
:mod:`tests.fuel_models.test_ruleset_lookup`.

What is pinned: only pixels with an FDist code above 0 are updated, and
every pixel without a new value -- undisturbed, nodata, unmatched, or
matched to a row with no value -- keeps last year's. ``updated`` marks
exactly the pixels that took a new value, and bad input fails before any
work is done.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.fuel_models.disturbed_fuel_update import update_fuel_models

_RULE_DEFAULTS = {
    "Zone": 1,
    "EVT": 7000,
    "Cover_Low": 0,
    "Cover_High": 999,
    "Height_Low": 0,
    "Height_High": 999,
    "BPSRF": "any",
    "OnOff": "On",
    "Wildcard": "any",
}


def _rules(*rows: dict) -> pd.DataFrame:
    """A Master_Rulesets table; fields a row leaves out get permissive defaults."""
    return pd.DataFrame([{**_RULE_DEFAULTS, **row} for row in rows])


BOTH_FIRES = _rules({"DIST": 121, "FBFM40": 165}, {"DIST": 111, "FBFM40": 142})


def _update(codes, previous=None, rules=BOTH_FIRES, **overrides):
    """update_fuel_models with zone 1 and FVT/FVC/FVH/BPS the default rules accept."""
    codes = np.asarray(codes)
    shape = codes.shape
    if previous is None:
        previous = np.full(shape, 100, dtype=np.int16)
    kwargs = {
        "fuel_model": "FBFM40",
        "dist": codes,
        "zone": np.full(shape, 1),
        "fvt": np.full(shape, 7000),
        "fvc": np.full(shape, 150),
        "fvh": np.full(shape, 110),
        "bps": np.full(shape, 11),
        "rules": rules,
    }
    kwargs.update(overrides)
    return update_fuel_models(previous, **kwargs)


class TestUpdate:
    def test_updates_disturbed_pixels_and_keeps_the_rest(self):
        dist = [[0, 121, 0], [111, 0, -9999]]
        previous = np.array([[102, 102, 102], [183, 183, 183]], dtype=np.int16)
        output, updated = _update(dist, previous)
        np.testing.assert_array_equal(output, [[102, 165, 102], [142, 183, 183]])
        np.testing.assert_array_equal(
            updated, [[False, True, False], [True, False, False]]
        )

    def test_keeps_last_years_dtype_and_leaves_it_unchanged(self):
        previous = np.array([[100, 100]], dtype=np.int16)
        output, _ = _update([[121, 0]], previous)
        assert output.dtype == np.int16
        np.testing.assert_array_equal(previous, [[100, 100]])

    def test_disturbed_pixel_without_a_rule_keeps_last_year(self):
        rules = _rules({"DIST": 121, "FBFM40": 165})  # no rule for FDist 111
        output, updated = _update([[121, 111]], rules=rules)
        np.testing.assert_array_equal(output, [[165, 100]])
        np.testing.assert_array_equal(updated, [[True, False]])

    def test_matched_row_without_a_value_keeps_last_year(self):
        rules = _rules({"DIST": 121, "FBFM40": 165.0}, {"DIST": 111, "FBFM40": np.nan})
        output, updated = _update([[121, 111]], rules=rules)
        np.testing.assert_array_equal(output, [[165, 100]])
        np.testing.assert_array_equal(updated, [[True, False]])

    def test_zone_selects_the_rules(self):
        # Only zone 1 has rules, so the zone-2 pixel keeps last year's value.
        output, updated = _update([[121, 121]], zone=np.array([[1, 2]]))
        np.testing.assert_array_equal(output, [[165, 100]])
        np.testing.assert_array_equal(updated, [[True, False]])

    def test_nothing_disturbed_returns_last_year(self):
        previous = np.array([[102, 183, 188]], dtype=np.int16)
        output, updated = _update([[0, 0, -9999]], previous)
        np.testing.assert_array_equal(output, previous)
        assert not updated.any()

    def test_disturbed_but_not_updated_count(self):
        # The count the docstring describes: disturbed pixels that kept last
        # year's value.
        rules = _rules({"DIST": 121, "FBFM40": 165})
        dist = np.array([[121, 111, 111, 0]])
        _, updated = _update(dist, rules=rules)
        assert ((dist > 0) & ~updated).sum() == 2


class TestValidation:
    @pytest.mark.parametrize("dist", [[[121]], [[0]]])
    def test_unknown_fuel_model_raises_even_when_nothing_is_disturbed(self, dist):
        with pytest.raises(ValueError, match="FBFM99"):
            _update(dist, fuel_model="FBFM99")

    @pytest.mark.parametrize("name", ["dist", "zone", "fvt", "fvc", "fvh", "bps"])
    def test_shape_mismatch_raises(self, name):
        with pytest.raises(ValueError, match=f"{name} has shape"):
            _update([[0, 0]], **{name: np.zeros((1, 3), dtype=int)})

    def test_non_numeric_fuel_model_raises(self):
        rules = _rules({"DIST": 0, "FBFM40": "GS1"})
        with pytest.raises(ValueError, match="isn't numeric"):
            _update([[0]], rules=rules)
