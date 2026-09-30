"""Tests for :mod:`fastfuels_core.fuel_models.fdist_builder`.

FDist codes are ``100*D_TYPE + 10*D_SEVERITY + D_TIME``, and FDist nodata
is -9999.

Converting LDist: three kinds of row produce a code of 0, and the tests
keep them apart -- no disturbance (no warning), a real disturbance FDist
can't encode (warning), and a real disturbance with a missing severity
(no warning). A code missing from the attribute table entirely maps to
-9999 and is counted as unmapped.

Building the FDist the rules use: only one-year-old codes (ending in 1,
above 0) are kept or aged; everything else, including -9999 nodata, becomes
0. In ``ldist_and_last_year_fdist"`` mode this year's disturbance wins
wherever there is one.

LANDFIRE LDist codes used below: 2882 is Wildfire/Moderate (FDist 121), 2801
is Development, 2971 is Herbicide, 0 is background and -9999 is
Fill-NoData.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.fuel_models.fdist_builder import (
    DISTURBANCE_MODES,
    LDIST_TYPE_UNREPRESENTABLE,
    FDistRaster,
    FDistResult,
    UnknownLdistSeverityError,
    UnknownLdistTypeError,
    _age_fdist,  # noqa
    _ldist_attribute_table,  # noqa
    _ldist_codes_to_fdist_codes,  # noqa
    _ldist_row_to_fdist,  # noqa
    _this_years_fdist,  # noqa
    ldist_raster_to_fdist_raster,
    build_fdist_raster,
)

WILDFIRE_MODERATE = 2882
DEVELOPMENT = 2801
HERBICIDE = 2971
NOT_IN_TABLE = 1


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
        assert _ldist_row_to_fdist(dist_type, severity) == FDistResult(expected)

    @pytest.mark.parametrize("severity", ["Unburned/Low", "Increased Green"])
    def test_fire_only_severities_map_to_low(self, severity):
        assert _ldist_row_to_fdist("Fire", severity).fdist_code == 111

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
        assert _ldist_row_to_fdist(dist_type, severity) == FDistResult(0)

    @pytest.mark.parametrize("dist_type", sorted(LDIST_TYPE_UNREPRESENTABLE))
    def test_unrepresentable_is_zero_with_warning(self, dist_type):
        result = _ldist_row_to_fdist(dist_type, "Low")
        assert result.fdist_code == 0
        assert dist_type in result.warning

    @pytest.mark.parametrize("severity", [None, "Fill-NoData"])
    def test_missing_severity_is_zero_without_warning(self, severity):
        # Deliberate: no LDist row has this, so it isn't flagged.
        assert _ldist_row_to_fdist("Fire", severity) == FDistResult(0)

    def test_unknown_type_raises(self):
        with pytest.raises(UnknownLdistTypeError, match="Volcano"):
            _ldist_row_to_fdist("Volcano", "Low")

    def test_unknown_severity_raises(self):
        with pytest.raises(UnknownLdistSeverityError, match="Extreme"):
            _ldist_row_to_fdist("Fire", "Extreme")

    def test_unknown_errors_are_value_errors(self):
        assert issubclass(UnknownLdistTypeError, ValueError)
        assert issubclass(UnknownLdistSeverityError, ValueError)


class TestLdistCodesToFdistCodes:
    def test_every_row_of_the_packaged_table_converts(self):
        # Raises if any DIST_TYPE or SEVERITY isn't catalogued.
        codes, fdist, warnings = _ldist_codes_to_fdist_codes()
        assert len(codes) == len(fdist) == 133
        assert np.all(np.diff(codes) > 0)
        assert len(warnings) == 8

    def test_rows_with_missing_labels_are_no_disturbance(self):
        # Code 0 (background) has no DIST_TYPE or SEVERITY in the table.
        codes, fdist, _ = _ldist_codes_to_fdist_codes()
        assert fdist[np.flatnonzero(codes == 0)[0]] == 0

    def test_warnings_keyed_by_python_int_code(self):
        _, _, warnings = _ldist_codes_to_fdist_codes()
        assert HERBICIDE in warnings
        assert all(type(code) is int for code in warnings)

    def test_results_are_cached(self):
        assert _ldist_attribute_table() is _ldist_attribute_table()
        assert _ldist_codes_to_fdist_codes() is _ldist_codes_to_fdist_codes()


class TestLdistRasterToFdistRaster:
    def test_converts_known_codes_and_keeps_shape(self):
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

    @pytest.mark.parametrize("missing", [-10000, NOT_IN_TABLE, 99999])
    def test_code_missing_from_table_is_nodata_and_counted(self, missing):
        # Below the smallest code, between codes, and above the largest.
        result = ldist_raster_to_fdist_raster(
            np.array([WILDFIRE_MODERATE, missing, missing])
        )
        np.testing.assert_array_equal(result.fdist, [121, -9999, -9999])
        assert result.n_unmapped == 2

    def test_warns_only_for_codes_present(self):
        raster = np.array([[HERBICIDE, WILDFIRE_MODERATE], [0, HERBICIDE]])
        result = ldist_raster_to_fdist_raster(raster)
        assert list(result.warnings) == [HERBICIDE]
        assert "Herbicide" in result.warnings[HERBICIDE]
        np.testing.assert_array_equal(result.fdist, [[0, 121], [0, 0]])


class TestAgeFdist:
    @pytest.mark.parametrize(
        "code, expected",
        [
            (121, 122),  # one year old -> two
            (331, 332),
            (122, 0),  # already older: dropped
            (132, 0),
            (0, 0),
            (-9999, 0),  # nodata, even though -9999 % 10 == 1
        ],
    )
    def test_ages_only_one_year_old_codes(self, code, expected):
        assert _age_fdist(np.array([code])).tolist() == [expected]

    def test_keeps_shape(self):
        aged = _age_fdist(np.array([[121, 0], [-9999, 122]]))
        np.testing.assert_array_equal(aged, [[122, 0], [0, 0]])


class TestThisYearsFdist:
    @pytest.mark.parametrize(
        "code, expected",
        [(121, 121), (331, 331), (122, 0), (0, 0), (-9999, 0)],
    )
    def test_keeps_only_one_year_old_codes(self, code, expected):
        assert _this_years_fdist(np.array([code])).tolist() == [expected]


class TestBuildFdistRaster:
    def test_ldist_mode_is_the_converted_ldist(self):
        ldist = np.array([[WILDFIRE_MODERATE, HERBICIDE], [0, NOT_IN_TABLE]])
        result = build_fdist_raster("ldist", ldist=ldist)
        expected = ldist_raster_to_fdist_raster(ldist)
        np.testing.assert_array_equal(result.fdist, expected.fdist)
        assert result.n_unmapped == expected.n_unmapped == 1
        assert result.warnings == expected.warnings

    def test_ldist_and_last_year_fdist_mode(self):
        last_year = np.array([121, 131, 132, -9999, 0, 111, 0])
        ldist = np.array(
            [0, WILDFIRE_MODERATE, 0, WILDFIRE_MODERATE, 0, NOT_IN_TABLE, HERBICIDE]
        )
        result = build_fdist_raster(
            "ldist_and_last_year_fdist", ldist=ldist, fdist=last_year
        )
        # 121 aged; this year wins over 131; older 132 dropped; this year
        # over nodata; nothing; unknown LDist falls back to aged 111;
        # herbicide is no disturbance.
        np.testing.assert_array_equal(result.fdist, [122, 121, 0, 121, 0, 112, 0])
        assert result.n_unmapped == 1
        assert list(result.warnings) == [HERBICIDE]

    def test_fdist_mode_keeps_this_years_disturbances(self):
        this_year = np.array([121, 122, 0, -9999, 331])
        result = build_fdist_raster("fdist", fdist=this_year)
        np.testing.assert_array_equal(result.fdist, [121, 0, 0, 0, 331])
        assert result.n_unmapped == 0
        assert result.warnings == {}

    def test_modes_are_the_three_documented(self):
        assert DISTURBANCE_MODES == ("ldist", "ldist_and_last_year_fdist", "fdist")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="'fdsit'"):
            build_fdist_raster("fdsit", fdist=np.array([0]))

    @pytest.mark.parametrize(
        "mode, given, missing",
        [
            ("ldist", {}, "ldist"),
            ("ldist_and_last_year_fdist", {"fdist": np.array([0])}, "ldist"),
            ("ldist_and_last_year_fdist", {"ldist": np.array([0])}, "fdist"),
            ("fdist", {}, "fdist"),
        ],
    )
    def test_missing_input_raises(self, mode, given, missing):
        with pytest.raises(ValueError, match=f"needs {missing}"):
            build_fdist_raster(mode, **given)
