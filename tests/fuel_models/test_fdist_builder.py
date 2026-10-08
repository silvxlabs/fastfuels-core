"""Tests for :mod:`fastfuels_core.fuel_models.fdist_builder`.

FDist codes are ``100*D_TYPE + 10*D_SEVERITY + D_TIME``, and FDist nodata
is -9999.

Converting LDist: ``LDIST_TO_FDIST`` covers every known DIST_TYPE with every
known SEVERITY, including combinations no LANDFIRE table has had yet. Types
that aren't a disturbance become 0 without a warning. Anything else without
an FDist code -- Herbicide and the other unrepresentable types, an unknown
type, a missing severity, or a code not in the attribute table -- becomes 0
and is reported in the warnings.

Building the FDist the rules use: in ``"ldist_and_last_year_fdist"`` mode,
last year's one-year-old codes (ending in 1) are aged to two years, its
other disturbances are kept, and this year's converted LDist wins wherever
it has a disturbance. In ``"fdist"`` mode every disturbance is kept. In both,
0 and -9999 nodata become 0.

The attribute table is built by hand. Codes from the LANDFIRE 2025 table
keep their real values (2882 is Wildfire/Moderate, 2801 is Development,
2971 is Herbicide); 9001-9003 are made up for cases it doesn't have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.fuel_models.fdist_builder import (
    DISTURBANCE_MODES,
    LDIST_TO_FDIST,
    _age_fdist,  # noqa
    _ldist_raster_to_fdist_raster,  # noqa
    build_fdist_raster,
)

FILL_NODATA = -9999
BACKGROUND = 0
WATER = 2001
DEVELOPMENT = 2801
FIRE_UNBURNED_LOW = 2011  # FDist 111
WILDFIRE_MODERATE = 2882  # FDist 121
HERBICIDE = 2971
WEATHER_MODERATE = 9001  # not in the 2025 table; FDist 421
FIRE_NO_SEVERITY = 9002
VOLCANO = 9003  # an unknown DIST_TYPE
NOT_IN_TABLE = 1

TABLE = pd.DataFrame(
    [
        (FILL_NODATA, "Fill-NoData", "Fill-NoData"),
        (BACKGROUND, None, None),
        (WATER, "Water", None),
        (DEVELOPMENT, "Development", "Low"),
        (FIRE_UNBURNED_LOW, "Fire", "Unburned/Low"),
        (WILDFIRE_MODERATE, "Wildfire", "Moderate"),
        (HERBICIDE, "Herbicide", "Low"),
        (WEATHER_MODERATE, "Weather", "Moderate"),
        (FIRE_NO_SEVERITY, "Fire", None),
        (VOLCANO, "Volcano", "Low"),
    ],
    columns=["VALUE", "DIST_TYPE", "SEVERITY"],
)


class TestLdistToFdist:
    @pytest.mark.parametrize(
        "pair, expected",
        [
            (("Wildfire", "Moderate"), 121),
            (("Prescribed Fire", "Low"), 111),
            (("Fire", "Unburned/Low"), 111),
            (("Fire", "Increased Green"), 111),
            (("Fire", "High"), 131),
            (("Mechanical Add", "Moderate"), 221),
            (("Clearcut", "Moderate"), 321),  # not in the 2025 table
            (("Weather", "Moderate"), 421),  # not in the 2025 table
            (("Insects/Disease", "High"), 531),
            (("Mechanical Unknown", "Low"), 611),
            (("Mastication", "High"), 731),
        ],
    )
    def test_codes(self, pair, expected):
        assert LDIST_TO_FDIST[pair] == expected

    @pytest.mark.parametrize(
        "dist_type", ["Herbicide", "Insecticide", "Chemical", "Biological"]
    )
    def test_unrepresentable_types_are_absent(self, dist_type):
        assert not any(t == dist_type for t, _ in LDIST_TO_FDIST)

    def test_every_code_is_one_year_old(self):
        assert all(code % 10 == 1 for code in LDIST_TO_FDIST.values())


class TestLdistRasterToFdistRaster:
    def test_converts_and_keeps_shape(self):
        ldist = np.array([[WILDFIRE_MODERATE, DEVELOPMENT], [0, FIRE_UNBURNED_LOW]])
        fdist, warnings = _ldist_raster_to_fdist_raster(ldist, TABLE)
        np.testing.assert_array_equal(fdist, [[121, 0], [0, 111]])
        assert fdist.dtype == np.int32
        assert warnings == []

    def test_combination_new_to_landfire_converts(self):
        fdist, warnings = _ldist_raster_to_fdist_raster(
            np.array([WEATHER_MODERATE]), TABLE
        )
        assert fdist.tolist() == [421]
        assert warnings == []

    @pytest.mark.parametrize("code", [FILL_NODATA, BACKGROUND, WATER, DEVELOPMENT])
    def test_no_disturbance_is_zero_without_warning(self, code):
        fdist, warnings = _ldist_raster_to_fdist_raster(np.array([code]), TABLE)
        assert fdist.tolist() == [0]
        assert warnings == []

    @pytest.mark.parametrize(
        "code", [HERBICIDE, FIRE_NO_SEVERITY, VOLCANO, NOT_IN_TABLE]
    )
    def test_no_fdist_code_is_zero_with_warning(self, code):
        fdist, warnings = _ldist_raster_to_fdist_raster(
            np.array([WILDFIRE_MODERATE, code, code]), TABLE
        )
        assert fdist.tolist() == [121, 0, 0]
        assert warnings == [code]

    def test_warnings_are_sorted_codes_present_in_the_raster(self):
        ldist = np.array([[VOLCANO, HERBICIDE], [HERBICIDE, 0]])
        _, warnings = _ldist_raster_to_fdist_raster(ldist, TABLE)
        assert warnings == sorted([HERBICIDE, VOLCANO])
        assert all(type(code) is int for code in warnings)

    def test_blank_cells_read_from_a_file_count_as_missing(self):
        # A table read from CSV has NaN, not None, in blank cells.
        table = TABLE.astype({"DIST_TYPE": object, "SEVERITY": object}).fillna(np.nan)
        ldist = np.array([BACKGROUND, WATER, FIRE_NO_SEVERITY])
        fdist, warnings = _ldist_raster_to_fdist_raster(ldist, table)
        assert fdist.tolist() == [0, 0, 0]
        assert warnings == [FIRE_NO_SEVERITY]


class TestAgeFdist:
    @pytest.mark.parametrize(
        "code, expected",
        [
            (121, 122),  # one year old -> two
            (331, 332),
            (122, 122),  # already older: kept
            (132, 132),
            (0, 0),
            (-9999, 0),  # nodata, even though -9999 % 10 == 1
        ],
    )
    def test_ages_one_year_old_codes_and_keeps_the_rest(self, code, expected):
        assert _age_fdist(np.array([code])).tolist() == [expected]

    def test_keeps_shape(self):
        aged = _age_fdist(np.array([[121, 0], [-9999, 122]]))
        np.testing.assert_array_equal(aged, [[122, 0], [0, 122]])


class TestBuildFdistRaster:
    def test_ldist_and_last_year_fdist_mode(self):
        last_year = np.array([121, 131, 132, -9999, 0, 111, 0])
        ldist = np.array(
            [0, WILDFIRE_MODERATE, 0, WILDFIRE_MODERATE, 0, NOT_IN_TABLE, HERBICIDE]
        )
        fdist, warnings = build_fdist_raster(
            "ldist_and_last_year_fdist",
            fdist=last_year,
            ldist=ldist,
            ldist_attribute_table=TABLE,
        )
        # 121 aged; this year wins over 131; older 132 kept; this year over
        # nodata; nothing; a code not in the table falls back to last year's
        # aged 111; herbicide is no disturbance.
        np.testing.assert_array_equal(fdist, [122, 121, 132, 121, 0, 112, 0])
        assert warnings == [NOT_IN_TABLE, HERBICIDE]

    def test_fdist_mode_keeps_every_disturbance(self):
        fdist, warnings = build_fdist_raster(
            "fdist", fdist=np.array([121, 122, 0, -9999, 331])
        )
        np.testing.assert_array_equal(fdist, [121, 122, 0, 0, 331])
        assert warnings == []

    def test_modes(self):
        assert DISTURBANCE_MODES == ("ldist_and_last_year_fdist", "fdist")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="'fdsit'"):
            build_fdist_raster("fdsit", fdist=np.array([0]))

    @pytest.mark.parametrize(
        "given", [{}, {"ldist": np.array([0])}, {"ldist_attribute_table": TABLE}]
    )
    def test_ldist_mode_without_its_ldist_inputs_raises(self, given):
        with pytest.raises(ValueError, match="needs ldist and ldist_attribute_table"):
            build_fdist_raster(
                "ldist_and_last_year_fdist", fdist=np.array([0]), **given
            )

    def test_fdist_is_required(self):
        with pytest.raises(TypeError, match="fdist"):
            build_fdist_raster("fdist")
