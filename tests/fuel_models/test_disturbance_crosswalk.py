"""Tests for :mod:`fastfuels_core.fuel_models.disturbance_crosswalk`.

The FDist code is ``100*D_TYPE + 10*D_SEVERITY + D_TIME`` with D_TIME
fixed at 1. Three kinds of row produce a code of 0, and the tests keep
them apart: no disturbance (no warning), a real disturbance FDist can't
encode (warning), and a real disturbance with a missing severity (no
warning). A code missing from the attribute table entirely is a fourth
case: it maps to -9999, not 0, and is counted as unmapped.

LF2025 codes used below: 2882 is Wildfire/Moderate, 2801 is Development,
2971 is Herbicide, 0 is background and -9999 is Fill-NoData.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.fuel_models.disturbance_crosswalk import (
    LDIST_TYPE_UNREPRESENTABLE,
    FDistRaster,
    FDistResult,
    UnknownLdistSeverityError,
    UnknownLdistTypeError,
    _apply_fdist_crosswalk,  # noqa
    _build_fdist_crosswalk,  # noqa
    _fdist_crosswalk,  # noqa
    _lf2025_ldist_attributes,  # noqa
    ldist_raster_to_fdist_raster,
    ldist_row_to_fdist,
)

WILDFIRE_MODERATE = 2882
DEVELOPMENT = 2801
HERBICIDE = 2971


def _decode_table(rows):
    """A minimal attribute table from (VALUE, DIST_TYPE, SEVERITY) tuples."""
    return pd.DataFrame(rows, columns=["VALUE", "DIST_TYPE", "SEVERITY"])


class TestLdistRowToFdist:
    @pytest.mark.parametrize(
        "dist_type, severity, expected",
        [
            ("Wildfire", "Moderate", 121),
            ("Prescribed Fire", "Low", 111),
            ("Fire", "High", 131),
            ("Mechanical Add", "Moderate", 221),
            ("Clearcut", "High", 331),
            ("Weather", "Low", 411),
            ("Insects/Disease", "Moderate", 521),
            ("Mechanical Unknown", "Low", 611),
            ("Mastication", "High", 731),
        ],
    )
    def test_encodes_type_severity_and_time(self, dist_type, severity, expected):
        assert ldist_row_to_fdist(dist_type, severity) == FDistResult(expected)

    @pytest.mark.parametrize("severity", ["Unburned/Low", "Increased Green"])
    def test_fire_only_severities_map_to_low(self, severity):
        assert ldist_row_to_fdist("Fire", severity).fdist_code == 111

    @pytest.mark.parametrize(
        "dist_type, severity",
        [
            (None, None),
            ("Water", None),
            ("Development", "Low"),
            ("Fill-NoData", "Fill-NoData"),
        ],
    )
    def test_no_disturbance_is_zero_without_warning(self, dist_type, severity):
        assert ldist_row_to_fdist(dist_type, severity) == FDistResult(0)

    @pytest.mark.parametrize("dist_type", sorted(LDIST_TYPE_UNREPRESENTABLE))
    def test_unrepresentable_is_zero_with_warning(self, dist_type):
        result = ldist_row_to_fdist(dist_type, "Low")
        assert result.fdist_code == 0
        assert dist_type in result.warning

    @pytest.mark.parametrize("severity", [None, "Fill-NoData"])
    def test_missing_severity_is_zero_without_warning(self, severity):
        # Deliberate: no LF2025 row has this, so it isn't flagged.
        assert ldist_row_to_fdist("Fire", severity) == FDistResult(0)

    def test_unknown_type_raises(self):
        with pytest.raises(UnknownLdistTypeError, match="Volcano"):
            ldist_row_to_fdist("Volcano", "Low")

    def test_unknown_severity_raises(self):
        with pytest.raises(UnknownLdistSeverityError, match="Extreme"):
            ldist_row_to_fdist("Fire", "Extreme")

    def test_unknown_errors_are_value_errors(self):
        assert issubclass(UnknownLdistTypeError, ValueError)
        assert issubclass(UnknownLdistSeverityError, ValueError)


class TestBuildFdistCrosswalk:
    def test_sorts_by_value_and_aligns_codes(self):
        table = _decode_table(
            [(30, "Fire", "High"), (10, "Water", None), (20, "Thinning", "Low")]
        )
        codes, fdist, warnings = _build_fdist_crosswalk(table)
        np.testing.assert_array_equal(codes, [10, 20, 30])
        np.testing.assert_array_equal(fdist, [0, 311, 131])
        assert warnings == {}

    def test_nan_labels_are_treated_as_missing(self):
        table = _decode_table([(0, np.nan, np.nan), (1, "Fire", np.nan)])
        _, fdist, warnings = _build_fdist_crosswalk(table)
        np.testing.assert_array_equal(fdist, [0, 0])
        assert warnings == {}

    def test_warnings_keyed_by_python_int_code(self):
        table = _decode_table([(5, "Herbicide", "Low"), (6, "Fire", "Low")])
        _, _, warnings = _build_fdist_crosswalk(table)
        assert list(warnings) == [5]
        assert type(next(iter(warnings))) is int

    def test_unknown_type_in_table_raises(self):
        with pytest.raises(UnknownLdistTypeError):
            _build_fdist_crosswalk(_decode_table([(1, "Volcano", "Low")]))


class TestApplyFdistCrosswalk:
    codes = np.array([10, 20, 30])
    fdist = np.array([0, 311, 131], dtype=np.int32)

    def test_remaps_and_keeps_shape(self):
        raster = np.array([[10, 20], [30, 20]])
        out, n_unmapped = _apply_fdist_crosswalk(raster, self.codes, self.fdist)
        np.testing.assert_array_equal(out, [[0, 311], [131, 311]])
        assert out.dtype == np.int32
        assert n_unmapped == 0

    @pytest.mark.parametrize("missing", [5, 15, 35])
    def test_missing_code_is_nodata_and_counted(self, missing):
        # Below the smallest code, between codes, and above the largest.
        raster = np.array([10, missing, missing])
        out, n_unmapped = _apply_fdist_crosswalk(raster, self.codes, self.fdist)
        np.testing.assert_array_equal(out, [0, -9999, -9999])
        assert n_unmapped == 2


class TestPackagedLf2025Table:
    def test_every_row_crosswalks(self):
        # Building raises if any DIST_TYPE or SEVERITY isn't catalogued.
        codes, fdist, warnings = _fdist_crosswalk()
        assert len(codes) == len(fdist) == 133
        assert np.all(np.diff(codes) > 0)
        assert len(warnings) == 8

    def test_loaders_are_cached(self):
        assert _lf2025_ldist_attributes() is _lf2025_ldist_attributes()
        assert _fdist_crosswalk() is _fdist_crosswalk()


class TestLdistRasterToFdistRaster:
    def test_remaps_known_codes(self):
        raster = np.array([[WILDFIRE_MODERATE, DEVELOPMENT], [0, WILDFIRE_MODERATE]])
        result = ldist_raster_to_fdist_raster(raster)
        assert isinstance(result, FDistRaster)
        np.testing.assert_array_equal(result.fdist, [[121, 0], [0, 121]])
        assert result.fdist.dtype == np.int32
        assert result.n_unmapped == 0
        assert result.warnings == {}

    def test_fill_and_background_are_no_disturbance_not_unmapped(self):
        result = ldist_raster_to_fdist_raster(np.array([-9999, 0]))
        np.testing.assert_array_equal(result.fdist, [0, 0])
        assert result.n_unmapped == 0

    def test_code_missing_from_table_is_nodata_and_counted(self):
        result = ldist_raster_to_fdist_raster(np.array([WILDFIRE_MODERATE, 1, 1]))
        np.testing.assert_array_equal(result.fdist, [121, -9999, -9999])
        assert result.n_unmapped == 2

    def test_warns_only_for_codes_present(self):
        raster = np.array([[HERBICIDE, WILDFIRE_MODERATE], [0, HERBICIDE]])
        result = ldist_raster_to_fdist_raster(raster)
        assert list(result.warnings) == [HERBICIDE]
        assert "Herbicide" in result.warnings[HERBICIDE]
        np.testing.assert_array_equal(result.fdist, [[0, 121], [0, 0]])

    def test_declared_nodata_is_no_disturbance_not_unmapped(self):
        raster = np.array([WILDFIRE_MODERATE, 32767, 32767])
        result = ldist_raster_to_fdist_raster(raster, nodata=32767)
        np.testing.assert_array_equal(result.fdist, [121, 0, 0])
        assert result.n_unmapped == 0

    def test_without_declared_nodata_its_code_is_unmapped(self):
        result = ldist_raster_to_fdist_raster(np.array([32767]))
        np.testing.assert_array_equal(result.fdist, [-9999])
        assert result.n_unmapped == 1
