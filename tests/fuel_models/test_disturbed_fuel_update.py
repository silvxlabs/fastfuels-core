"""Tests for :mod:`fastfuels_core.fuel_models.disturbed_fuel_update`.

The FDist and zone rasters are passed in directly, and rules are built by
hand, so each expected fuel model is known. How FDist is built is tested in
:mod:`tests.fuel_models.test_fdist_builder`; how rules are matched in
:mod:`tests.fuel_models.test_ruleset_lookup`.

What is pinned: every pixel, disturbed or not, takes its matched rule's
value, and only a pixel with no matching rule keeps last year's.
``matched`` marks exactly the pixels whose value came from a rule, last
year's grid and its type are left as they were, and bad input fails
before any work is done.
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


# A rule for undisturbed pixels (DIST 0) and one for each of two fires.
ALL_DIST = _rules(
    {"DIST": 0, "FBFM40": 101},
    {"DIST": 121, "FBFM40": 165},
    {"DIST": 111, "FBFM40": 142},
)


def _update(codes, previous=None, rules=ALL_DIST, **overrides):
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
    def test_every_pixel_takes_its_rule(self):
        # Undisturbed pixels (DIST 0) are matched like disturbed ones.
        output, matched = _update([[0, 121], [111, 0]])
        np.testing.assert_array_equal(output, [[101, 165], [142, 101]])
        assert matched.all()

    def test_pixel_without_a_rule_keeps_last_year(self):
        rules = _rules({"DIST": 121, "FBFM40": 165})  # no rule for 0 or 111
        previous = np.array([[102, 102, 183]], dtype=np.int16)
        output, matched = _update([[0, 121, 111]], previous, rules=rules)
        np.testing.assert_array_equal(output, [[102, 165, 183]])
        np.testing.assert_array_equal(matched, [[False, True, False]])

    def test_zone_selects_the_rules(self):
        # Only zone 1 has rules, so the zone-2 pixel keeps last year's value.
        output, matched = _update([[121, 121]], zone=np.array([[1, 2]]))
        np.testing.assert_array_equal(output, [[165, 100]])
        np.testing.assert_array_equal(matched, [[True, False]])

    def test_keeps_last_years_dtype_and_leaves_it_unchanged(self):
        previous = np.array([[100, 100]], dtype=np.int16)
        output, _ = _update([[121, 0]], previous)
        assert output.dtype == np.int16
        np.testing.assert_array_equal(previous, [[100, 100]])

    def test_fallback_count(self):
        # Pixels that kept last year's value, as griddle would count them.
        rules = _rules({"DIST": 121, "FBFM40": 165})
        _, matched = _update([[121, 111, 111, 0]], rules=rules)
        assert (~matched).sum() == 3


class TestValidation:
    def test_unknown_fuel_model_raises(self):
        with pytest.raises(ValueError, match="FBFM99"):
            _update([[0]], fuel_model="FBFM99")

    @pytest.mark.parametrize("name", ["dist", "zone", "fvt", "fvc", "fvh", "bps"])
    def test_shape_mismatch_raises(self, name):
        with pytest.raises(ValueError, match=f"{name} has shape"):
            _update([[0, 0]], **{name: np.zeros((1, 3), dtype=int)})
