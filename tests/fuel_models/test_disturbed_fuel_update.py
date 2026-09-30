"""Tests for :mod:`fastfuels_core.fuel_models.disturbed_fuel_update`.

The real FDist builder runs; the map zone lookup is replaced with one that
puts every cell in zone 1, since the geometry has its own tests. Rules are
built by hand, so each expected fuel model is known.

What is pinned: only disturbed pixels are updated, and every pixel without
a new value -- undisturbed, unrepresentable, unmapped, unmatched, or
matched to a row with no value -- keeps last year's. The counts report
each case, the zone lookup is skipped when nothing is disturbed, and bad
input fails before any work is done. Each disturbance mode reaches the
rules; how each builds its FDist is tested in
:mod:`tests.fuel_models.test_fdist_builder`.

LF2025 codes used below: 2882 is Wildfire/Moderate (FDist 121), 2011 is
Fire/Unburned-Low (FDist 111), 2801 is Development and 2971 is Herbicide
(both FDist 0), and 1 isn't in the table (unmapped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine

import fastfuels_core.fuel_models.disturbed_fuel_update as disturbed_fuel_update
from fastfuels_core.fuel_models.disturbed_fuel_update import (
    FuelModelUpdate,
    update_fuel_models,
)
from fastfuels_core.fuel_models.ruleset_lookup import build_ruleset_index

WILDFIRE_MODERATE = 2882  # FDist 121
FIRE_UNBURNED_LOW = 2011  # FDist 111
DEVELOPMENT = 2801
HERBICIDE = 2971
NOT_IN_TABLE = 1

TRANSFORM = Affine(30, 0, 0, 0, -30, 0)
CRS = "EPSG:5070"

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


def _index(*rows: dict):
    """A RulesetIndex from rules; fields a row leaves out get permissive defaults."""
    return build_ruleset_index(pd.DataFrame([{**_RULE_DEFAULTS, **r} for r in rows]))


BOTH_FIRES = _index({"DIST": 121, "FBFM40": 165}, {"DIST": 111, "FBFM40": 142})


@pytest.fixture(autouse=True)
def zone_1(monkeypatch):
    """Every cell is in map zone 1."""
    monkeypatch.setattr(
        disturbed_fuel_update,
        "lookup_lf_zones",
        lambda transform, shape, crs: np.full(shape, 1, dtype=np.int32),
    )


def _update(ldist=None, previous=None, index=BOTH_FIRES, **overrides):
    """update_fuel_models with uniform FVT/FVC/FVH/BPS the default rules accept.

    Defaults to ``disturbance="ldist"``. The grid's shape comes from
    ``ldist``, or from ``fdist`` when no LDist is given.
    """
    if ldist is not None:
        ldist = np.asarray(ldist)
    shape = np.shape(ldist if ldist is not None else overrides["fdist"])
    if previous is None:
        previous = np.full(shape, 100, dtype=np.int16)
    kwargs = {
        "fuel_model": "FBFM40",
        "disturbance": "ldist",
        "ldist": ldist,
        "fvt": np.full(shape, 7000),
        "fvc": np.full(shape, 150),
        "fvh": np.full(shape, 110),
        "bps": np.full(shape, 11),
        "transform": TRANSFORM,
        "crs": CRS,
        "index": index,
    }
    kwargs.update(overrides)
    return update_fuel_models(previous, **kwargs)


class TestUpdate:
    def test_updates_disturbed_pixels_and_keeps_the_rest(self):
        ldist = [
            [0, WILDFIRE_MODERATE, DEVELOPMENT],
            [FIRE_UNBURNED_LOW, HERBICIDE, NOT_IN_TABLE],
        ]
        previous = np.array([[102, 102, 102], [183, 183, 183]], dtype=np.int16)
        update = _update(ldist, previous)

        assert isinstance(update, FuelModelUpdate)
        np.testing.assert_array_equal(update.output, [[102, 165, 102], [142, 183, 183]])
        assert update.n_disturbed == 2
        assert update.n_unmatched == 0
        assert update.n_unmapped == 1
        assert list(update.warnings) == [HERBICIDE]

    def test_keeps_last_years_dtype_and_leaves_it_unchanged(self):
        previous = np.array([[100, 100]], dtype=np.int16)
        update = _update([[WILDFIRE_MODERATE, 0]], previous)
        assert update.output.dtype == np.int16
        np.testing.assert_array_equal(previous, [[100, 100]])

    def test_disturbed_pixel_without_a_rule_keeps_last_year(self):
        index = _index({"DIST": 121, "FBFM40": 165})  # no rule for FDist 111
        update = _update([[WILDFIRE_MODERATE, FIRE_UNBURNED_LOW]], index=index)
        np.testing.assert_array_equal(update.output, [[165, 100]])
        assert update.n_disturbed == 2
        assert update.n_unmatched == 1

    def test_matched_row_without_a_value_keeps_last_year(self):
        index = _index({"DIST": 121, "FBFM40": 165.0}, {"DIST": 111, "FBFM40": np.nan})
        update = _update([[WILDFIRE_MODERATE, FIRE_UNBURNED_LOW]], index=index)
        np.testing.assert_array_equal(update.output, [[165, 100]])
        assert update.n_unmatched == 0

    def test_zone_lookup_gets_the_grid(self, monkeypatch):
        calls = []

        def record(transform, shape, crs):
            calls.append((transform, shape, crs))
            return np.full(shape, 1, dtype=np.int32)

        monkeypatch.setattr(disturbed_fuel_update, "lookup_lf_zones", record)
        _update([[WILDFIRE_MODERATE, 0], [0, 0]])
        assert calls == [(TRANSFORM, (2, 2), CRS)]


class TestDisturbanceModes:
    def test_ldist_and_last_year_fdist_mode(self):
        # Last year's 121 ages to 122; this year's LDist gives 121.
        index = _index({"DIST": 121, "FBFM40": 165}, {"DIST": 122, "FBFM40": 188})
        update = _update(
            [[0, WILDFIRE_MODERATE]],
            index=index,
            disturbance="ldist_and_last_year_fdist",
            fdist=np.array([[121, 0]]),
        )
        np.testing.assert_array_equal(update.output, [[188, 165]])
        assert update.n_disturbed == 2

    def test_fdist_mode_uses_only_one_year_old_codes(self):
        update = _update(disturbance="fdist", fdist=np.array([[121, 122, 0, -9999]]))
        np.testing.assert_array_equal(update.output, [[165, 100, 100, 100]])
        assert update.n_disturbed == 1
        assert update.n_unmapped == 0
        assert update.warnings == {}


class TestNothingDisturbed:
    def test_returns_last_year_without_a_zone_lookup(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("zone lookup should be skipped")

        monkeypatch.setattr(disturbed_fuel_update, "lookup_lf_zones", fail)
        previous = np.array([[102, 183, 188]], dtype=np.int16)
        update = _update([[0, DEVELOPMENT, HERBICIDE]], previous)
        np.testing.assert_array_equal(update.output, previous)
        assert update.n_disturbed == 0
        assert update.n_unmatched == 0


class TestValidation:
    @pytest.mark.parametrize("ldist", [[[WILDFIRE_MODERATE]], [[0]]])
    def test_unknown_fuel_model_raises_even_when_nothing_is_disturbed(self, ldist):
        with pytest.raises(ValueError, match="FBFM99"):
            _update(ldist, fuel_model="FBFM99")

    @pytest.mark.parametrize("name", ["fvt", "fvc", "fvh", "bps"])
    def test_raster_shape_mismatch_raises(self, name):
        with pytest.raises(ValueError, match=name):
            _update([[0, 0]], **{name: np.zeros((1, 3), dtype=int)})

    def test_shape_is_checked_against_previous(self):
        with pytest.raises(ValueError, match="to match previous"):
            _update([[0, 0]], previous=np.zeros((2, 1), dtype=np.int16))

    def test_fdist_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="fdist"):
            _update(
                [[0, 0]],
                disturbance="ldist_and_last_year_fdist",
                fdist=np.zeros((1, 3), dtype=int),
            )

    def test_mode_missing_its_input_raises(self):
        with pytest.raises(ValueError, match="needs fdist"):
            _update([[0, 0]], disturbance="fdist")
