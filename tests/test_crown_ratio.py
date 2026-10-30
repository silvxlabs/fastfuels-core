"""Tests for ``allometry.crown_ratio``, the uncompacted crown ratio
equations and the crown base height built on them.

The coefficient rows used as oracles are transcribed here: the refit's from
the 2026-10 national refit, the published equations' from the papers' printed
tables (not from ``REF_CROWN_RATIO.csv``), so a later edit to the file shows
up as a failing test rather than silently changing every inventory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry.crown_ratio import (
    EQUATIONS,
    FASTFUELS_2026,
    estimate_crown_base_height,
    estimate_uncompacted_crown_ratio,
)
from fastfuels_core.ref_data import DATA_PATH, REF_SPECIES
from fastfuels_core.units import CM_TO_IN, M_TO_FT

# (B0, B_CCR, B_LN_DIA, B_LN_HT); B_HD is zero in every refit row.
DOUGLAS_FIR = (0.06860, 4.57562, 0.29439, -0.57148)  # national SPCD 202
DOUGLAS_FIR_PNW = (0.27806, 4.59467, 0.40508, -0.64419)  # pacific_northwest 202
AILANTHUS = (-0.22833, 4.54171, 0.23954, -0.50671)  # national GENUS
SOFTWOOD = (-0.11907, 4.85791, 0.33718, -0.59226)  # national POOLED

AILANTHUS_VARIETY = 6051  # no species row, so it takes the genus row
ARAUCARIA = 6156  # softwood with no species or genus row: the pool


def logistic(coefs, ccr, dbh_cm, height_m):
    b0, b_ccr, b_ln_dia, b_ln_ht = coefs
    z = (
        b0
        + b_ccr * ccr
        + b_ln_dia * np.log(dbh_cm * CM_TO_IN)
        + b_ln_ht * np.log(height_m * M_TO_FT)
    )
    return 1.0 / (1.0 + np.exp(-z))


class TestLookup:
    @pytest.mark.parametrize(
        "spcd, pnw, coefs",
        [
            (202, False, DOUGLAS_FIR),
            (202, True, DOUGLAS_FIR_PNW),
            (AILANTHUS_VARIETY, False, AILANTHUS),
            (ARAUCARIA, False, SOFTWOOD),
            # No Pacific Northwest row: the national cascade applies
            (ARAUCARIA, True, SOFTWOOD),
        ],
    )
    def test_most_specific_row(self, spcd, pnw, coefs):
        got = estimate_uncompacted_crown_ratio(
            spcd, 0.4, 25.4, 20.0, pacific_northwest=pnw
        )
        assert got[0] == pytest.approx(logistic(coefs, 0.4, 25.4, 20.0), rel=1e-12)

    def test_hand_computed_value(self):
        # 0.0686 + 4.57562*0.4 + 0.29439*ln(10 in) - 0.57148*ln(65.617 ft)
        assert estimate_uncompacted_crown_ratio(202, 0.4, 25.4, 20.0)[
            0
        ] == pytest.approx(0.54630, abs=1e-5)

    def test_pacific_northwest_per_tree(self):
        got = estimate_uncompacted_crown_ratio(
            [202, 202], [0.4, 0.4], [25.4, 25.4], [20.0, 20.0], [False, True]
        )
        assert got[0] == pytest.approx(logistic(DOUGLAS_FIR, 0.4, 25.4, 20.0))
        assert got[1] == pytest.approx(logistic(DOUGLAS_FIR_PNW, 0.4, 25.4, 20.0))

    def test_every_reference_species_resolves(self):
        spcd = REF_SPECIES.index.to_numpy()
        n = len(spcd)
        for pnw in (False, True):
            got = estimate_uncompacted_crown_ratio(
                spcd, np.full(n, 0.4), np.full(n, 25.4), np.full(n, 20.0), pnw
            )
            assert np.isfinite(got).all()

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="not in the FIA reference"):
            estimate_uncompacted_crown_ratio(999999, 0.4, 25.4, 20.0)

    @pytest.mark.parametrize("spcd", [np.array(["202"]), 202.0, pd.Series([202])])
    def test_integral_species_codes_match_int(self, spcd):
        # A string code once missed the lookup and returned the compacted ratio
        expected = estimate_uncompacted_crown_ratio(202, 0.4, 30.0, 20.0)
        got = estimate_uncompacted_crown_ratio(spcd, 0.4, 30.0, 20.0)
        np.testing.assert_array_equal(got, expected)
        assert got[0] > 0.4

    @pytest.mark.parametrize("spcd", ["abc", 202.5, np.nan, None])
    def test_non_integral_species_raises(self, spcd):
        with pytest.raises(ValueError, match="integer FIA codes"):
            estimate_uncompacted_crown_ratio(spcd, 0.4, 30.0, 20.0)


class TestRules:
    def test_never_below_compacted(self):
        ccr = np.linspace(0.01, 1.0, 100)
        n = len(ccr)
        got = estimate_uncompacted_crown_ratio(
            np.full(n, 202), ccr, np.full(n, 80.0), np.full(n, 60.0)
        )
        assert (got >= ccr).all()
        assert (got <= 1.0).all()

    def test_full_crown_stays_full(self):
        assert estimate_uncompacted_crown_ratio(202, 1.0, 25.4, 20.0)[0] == 1.0

    def test_zero_crown_stays_zero(self):
        # A dead tree carries CR 0; the logistic alone would give it a crown
        assert estimate_uncompacted_crown_ratio(202, 0.0, 25.4, 20.0)[0] == 0.0

    @pytest.mark.parametrize(
        "dbh_cm, height_m", [(np.nan, 20.0), (0.0, 20.0), (25.4, np.nan), (25.4, 0.0)]
    )
    def test_zero_crown_is_zero_without_dbh_or_height(self, dbh_cm, height_m):
        assert estimate_uncompacted_crown_ratio(122, 0.0, dbh_cm, height_m)[0] == 0.0

    @pytest.mark.parametrize(
        "ccr, dbh_cm, height_m",
        [
            (np.nan, 25.4, 20.0),
            (0.4, np.nan, 20.0),
            (0.4, 0.0, 20.0),
            (0.4, 25.4, -1.0),
        ],
    )
    def test_missing_inputs_give_nan(self, ccr, dbh_cm, height_m):
        assert np.isnan(estimate_uncompacted_crown_ratio(202, ccr, dbh_cm, height_m)[0])

    def test_percent_raises(self):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            estimate_uncompacted_crown_ratio(202, 40.0, 25.4, 20.0)


class TestCrownBaseHeight:
    def test_matches_uncompacted_ratio(self):
        args = (
            [202, 202, AILANTHUS_VARIETY, ARAUCARIA],
            [0.4, 0.4, 0.7, 0.2],
            [25.4, 25.4, 12.0, 60.0],
            [20.0, 20.0, 9.5, 35.0],
            [False, True, False, True],
        )
        expected = np.asarray(args[3]) * (1.0 - estimate_uncompacted_crown_ratio(*args))
        np.testing.assert_allclose(
            estimate_crown_base_height(*args), expected, rtol=1e-12
        )

    def test_hand_computed_value(self):
        # Douglas-fir UCR 0.54630 at CCR 0.4, 25.4 cm, 20 m
        assert estimate_crown_base_height(202, 0.4, 25.4, 20.0)[0] == pytest.approx(
            20.0 * (1.0 - 0.54630), abs=1e-4
        )

    def test_zero_crown_is_full_height(self):
        assert estimate_crown_base_height(202, 0.0, 25.4, 20.0)[0] == 20.0

    @pytest.mark.parametrize("dbh_cm", [np.nan, 0.0])
    def test_zero_crown_without_dbh_is_full_height(self, dbh_cm):
        assert estimate_crown_base_height(122, 0.0, dbh_cm, 20.0)[0] == 20.0

    @pytest.mark.parametrize(
        "ccr, dbh_cm, height_m",
        [
            (np.nan, 25.4, 20.0),
            (0.4, 25.4, np.nan),
            (0.4, 25.4, 0.0),
            (0.4, 25.4, -1.0),
            (0.4, 0.0, 20.0),
            (0.0, 25.4, 0.0),
            (0.0, np.nan, np.nan),
            (0.0, np.nan, -1.0),
        ],
    )
    def test_missing_inputs_give_nan(self, ccr, dbh_cm, height_m):
        assert np.isnan(estimate_crown_base_height(202, ccr, dbh_cm, height_m)[0])

    def test_within_tree_height(self):
        ccr = np.linspace(0.0, 1.0, 101)
        n = len(ccr)
        height = np.linspace(1.0, 60.0, n)
        got = estimate_crown_base_height(np.full(n, 202), ccr, np.full(n, 40.0), height)
        assert (got >= 0.0).all()
        assert (got <= height).all()

    def test_scalar_returns_length_one_array(self):
        got = estimate_crown_base_height(202, 0.4, 25.4, 20.0)
        assert isinstance(got, np.ndarray)
        assert got.shape == (1,)

    def test_percent_raises(self):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            estimate_crown_base_height(202, 40.0, 25.4, 20.0)


PUBLISHED = [e for e in EQUATIONS if e != FASTFUELS_2026]


def published(b, ccr, dbh_cm, height_m, sign=-1):
    """A published equation evaluated in its own units (in, ft) from printed
    coefficients ``b = (b0, b_ccr, b_ln_dia, b_ln_ht, b_hd)``."""
    b0, b_ccr, b_ln_dia, b_ln_ht, b_hd = b
    dia_in = dbh_cm / 2.54
    ht_ft = height_m / 0.3048
    xb = (
        b0
        + b_ccr * ccr
        + b_ln_dia * np.log(dia_in)
        + b_ln_ht * np.log(ht_ft)
        + b_hd * ht_ft / dia_in
    )
    return 1.0 / (1.0 + np.exp(sign * xb))


def ucr(spcd, ccr, dbh_cm, height_m, equation):
    return estimate_uncompacted_crown_ratio(
        spcd, ccr, dbh_cm, height_m, equation=equation
    )[0]


IN = 2.54  # cm per inch
FT = 0.3048  # m per foot


class TestMonleon2004Eq1:
    # Table 2, p. 264: (intercept, CCR, ln dbh, ln height, H/D)
    DOUGLAS_FIR = (-0.4880, 4.9065, 0.3090, -0.4776, 0.0)
    WHITE_FIR = (2.3632, 4.6270, 2.5099, -2.9255, 0.3706)
    PINUS_SPP = (0.2071, 4.1248, 0.0, -0.2914, -0.0767)
    QUERCUS_SPP = (-0.1455, 5.2591, 0.0, -0.3223, -0.0949)
    JEFFREY_PINE = (-0.4129, 2.6760, 0.0, 0.0, 0.0)

    def test_worked_example(self):
        # p. 263: a 20 in, 120 ft Douglas-fir at CCR 0.60 has xb = 1.0951,
        # UCR = 0.75. The example prints the intercept as -0.4480; Table 2's
        # -0.4880 is the one that reproduces xb.
        got = ucr(202, 0.60, 20 * IN, 120 * FT, "monleon_2004_eq1")
        assert np.log(got / (1 - got)) == pytest.approx(1.0951, abs=5e-5)
        assert got == pytest.approx(0.7493, abs=5e-5)

    @pytest.mark.parametrize(
        "spcd, coefs",
        [
            (202, DOUGLAS_FIR),
            (15, WHITE_FIR),  # every term
            (101, PINUS_SPP),  # whitebark pine: genus model
            (801, QUERCUS_SPP),  # coast live oak: genus model
        ],
    )
    def test_hand_computed(self, spcd, coefs):
        got = ucr(spcd, 0.45, 30.0, 18.0, "monleon_2004_eq1")
        assert got == pytest.approx(published(coefs, 0.45, 30.0, 18.0), rel=1e-12)

    def test_ccr_above_090_is_ccr(self):
        assert ucr(202, 0.95, 30.0, 18.0, "monleon_2004_eq1") == 0.95
        assert ucr(202, 0.90, 30.0, 18.0, "monleon_2004_eq1") == pytest.approx(
            published(self.DOUGLAS_FIR, 0.90, 30.0, 18.0)
        )

    def test_prediction_below_ccr_is_kept(self):
        # Jeffrey pine at CCR 0.90: 1/(1+exp(-(-0.4129+2.676*0.9))) = 0.880
        got = ucr(116, 0.90, 30.0, 18.0, "monleon_2004_eq1")
        assert got == pytest.approx(published(self.JEFFREY_PINE, 0.9, 30, 18))
        assert got < 0.90

    def test_uncovered_species_is_nan(self):
        assert np.isnan(ucr(316, 0.45, 30.0, 18.0, "monleon_2004_eq1"))


class TestMonleon2004Eq2:
    # Table 4, p. 265: (intercept, CCR, ln dbh)
    DOUGLAS_FIR = (-1.8901, 5.1949, 0.0, 0.0, 0.0)
    WESTERN_HEMLOCK = (-1.7810, 5.8122, -0.1732, 0.0, 0.0)
    RED_FIR = (-0.2956, 3.1074, 0.0, 0.0, 0.0)
    PINUS_SPP = (-1.4072, 4.7545, -0.0911, 0.0, 0.0)
    # Table 2, p. 264: no height term, so not repeated in Table 4
    GRAND_FIR = (-1.7252, 5.3703, 0.0, 0.0, 0.0)
    SUBALPINE_FIR = (-2.1221, 4.6659, 0.4243, 0.0, 0.0)

    @pytest.mark.parametrize(
        "spcd, coefs",
        [
            (202, DOUGLAS_FIR),
            (263, WESTERN_HEMLOCK),
            (21, RED_FIR),  # Shasta red fir is in the red fir model
            (131, PINUS_SPP),  # loblolly pine: genus model
            (17, GRAND_FIR),
            (19, SUBALPINE_FIR),
        ],
    )
    def test_hand_computed(self, spcd, coefs):
        got = ucr(spcd, 0.45, 30.0, 18.0, "monleon_2004_eq2")
        assert got == pytest.approx(published(coefs, 0.45, 30.0, 18.0), rel=1e-12)

    def test_height_does_not_matter(self):
        a = ucr(202, 0.45, 30.0, 10.0, "monleon_2004_eq2")
        assert a == ucr(202, 0.45, 30.0, 40.0, "monleon_2004_eq2")

    def test_ccr_above_090_is_ccr(self):
        assert ucr(263, 0.91, 30.0, 18.0, "monleon_2004_eq2") == 0.91


class TestToneyReeves2009:
    # Table 3, p. 80: (a, b, c)
    PONDEROSA = (-1.6296, 4.5654, 0.0, 0.0, 0.0)
    DOUGLAS_FIR = (-1.4162, 5.3576, -0.1156, 0.0, 0.0)
    UTAH_JUNIPER = (-0.1831, 2.2050, 0.0, 0.0, 0.0)

    def test_ponderosa_hand_value(self):
        # -1.6296 + 4.5654 * 0.40 = 0.19656 -> 0.54898
        assert ucr(122, 0.40, 25.4, 15.0, "toney_reeves_2009") == pytest.approx(
            0.54898, abs=5e-6
        )

    @pytest.mark.parametrize(
        "spcd, coefs",
        [
            (122, PONDEROSA),
            (202, DOUGLAS_FIR),
            # Woodland, but no diameter term, so DRC is not needed
            (65, UTAH_JUNIPER),
        ],
    )
    def test_hand_computed(self, spcd, coefs):
        got = ucr(spcd, 0.45, 30.0, 18.0, "toney_reeves_2009")
        assert got == pytest.approx(published(coefs, 0.45, 30.0, 18.0), rel=1e-12)

    @pytest.mark.parametrize("spcd", [475, 66, 133, 814])
    def test_woodland_with_diameter_term_is_nan(self, spcd):
        # Fitted on diameter at the root collar; DBH is not substituted
        assert np.isnan(ucr(spcd, 0.45, 30.0, 18.0, "toney_reeves_2009"))

    @pytest.mark.parametrize("spcd", [131, 143])  # no genus or group models
    def test_uncovered_species_is_nan(self, spcd):
        assert np.isnan(ucr(spcd, 0.45, 30.0, 18.0, "toney_reeves_2009"))

    def test_ccr_above_090_is_ccr(self):
        assert ucr(122, 0.95, 30.0, 18.0, "toney_reeves_2009") == 0.95


class TestRandolph2010:
    # Table 3, p. 121: (a, b, c)
    YELLOW_POPLAR = (-1.2938, 3.9626, -0.0786, 0.0, 0.0)
    PISP = (-1.7131, 4.4212, -0.0290, 0.0, 0.0)
    QUSP = (-1.0633, 4.1355, -0.1464, 0.0, 0.0)
    JUSP_F = (-1.9268, 4.7840, 0.1571, 0.0, 0.0)
    HONEY_MESQUITE = (-0.0882, 3.5076, 0.0, 0.0, 0.0)
    CABBAGE_PALMETTO = (-1.5312, 7.4019, -0.4886, 0.0, 0.0)
    # Text, p. 121: ln(length) in ft, length being the tree height
    ASHE_JUNIPER = (5.0068, 3.42, 0.0, -1.7961, 0.0)
    JUSP_W = (5.4445, 3.5372, 0.0, -1.9477, 0.0)

    @pytest.mark.parametrize(
        "spcd, coefs",
        [
            (621, YELLOW_POPLAR),
            (131, PISP),  # loblolly pine: genus model
            (802, QUSP),  # white oak: genus model
            (68, JUSP_F),
            (67, JUSP_F),
            (756, HONEY_MESQUITE),  # woodland, but no diameter term
            (61, ASHE_JUNIPER),
            (58, JUSP_W),
            (65, JUSP_W),  # any woodland juniper but Ashe
        ],
    )
    def test_hand_computed(self, spcd, coefs):
        got = ucr(spcd, 0.45, 30.0, 8.0, "randolph_2010")
        assert got == pytest.approx(published(coefs, 0.45, 30.0, 8.0), rel=1e-12)

    def test_ashe_juniper_hand_value(self):
        # 5.0068 + 3.42 * 0.45 - 1.7961 * ln(20 ft) = 1.16517 -> 0.76227
        got = ucr(61, 0.45, 30.0, 20 * FT, "randolph_2010")
        assert got == pytest.approx(0.76227, abs=5e-6)

    def test_non_woodland_juniper_without_model_is_nan(self):
        assert np.isnan(ucr(64, 0.45, 30.0, 8.0, "randolph_2010"))

    def test_prediction_below_ccr_is_kept(self):
        # p. 122: cabbage palmetto at CCR 0.10 is underpredicted
        got = ucr(912, 0.10, 23 * IN, 8.0, "randolph_2010")
        assert got == pytest.approx(published(self.CABBAGE_PALMETTO, 0.1, 23 * IN, 8))
        assert got < 0.10

    def test_ccr_above_090_is_ccr(self):
        assert ucr(621, 0.92, 30.0, 18.0, "randolph_2010") == 0.92


class TestWestfall2019:
    # Table 2, p. 597: (b0, b1, b2) in 1 / (1 + exp(+(b0 + b1 CCR + b2 ln dbh)))
    SPRUCE_FIR = (1.3848, -4.9635, 0.1195, 0.0, 0.0)  # group 5
    RED_PINE = (1.4917, -5.2445, 0.1846, 0.0, 0.0)  # group 8
    SOFT_MAPLE = (1.2410, -4.4017, 0.1460, 0.0, 0.0)  # group 16
    OTHER_SOFTWOODS = (0.9622, -4.3375, 0.1404, 0.0, 0.0)  # group 7

    def test_sign_convention_hand_value(self):
        # 1 / (1 + exp(1.3848 - 4.9635 * 0.40 + 0.1195 * ln(10))) = 0.58065
        assert ucr(12, 0.40, 25.4, 15.0, "westfall_2019") == pytest.approx(
            0.58065, abs=5e-6
        )

    @pytest.mark.parametrize(
        "spcd, coefs",
        [
            (12, SPRUCE_FIR),
            (125, RED_PINE),  # FIA group 4 is split: red pine
            (316, SOFT_MAPLE),  # red maple, by FIA eastern species group
            (202, OTHER_SOFTWOODS),  # Douglas-fir is FIA group 9
        ],
    )
    def test_hand_computed(self, spcd, coefs):
        got = ucr(spcd, 0.45, 30.0, 18.0, "westfall_2019")
        want = published(coefs, 0.45, 30.0, 18.0, sign=1)
        assert got == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("spcd", [121, 221])  # longleaf pine, baldcypress
    def test_groups_the_paper_lacks_are_nan(self, spcd):
        assert np.isnan(ucr(spcd, 0.45, 30.0, 18.0, "westfall_2019"))

    def test_ccr_above_090_is_predicted(self):
        # No identity cutoff: at 10 in the prediction 0.955 exceeds CCR 0.95
        got = ucr(12, 0.95, 10 * IN, 15.0, "westfall_2019")
        assert got == pytest.approx(published(self.SPRUCE_FIR, 0.95, 10 * IN, 15, 1))
        assert got > 0.95

    def test_prediction_below_ccr_is_ccr(self):
        # At 40 in the prediction 0.947 is below CCR 0.95
        assert published(self.SPRUCE_FIR, 0.95, 40 * IN, 15, 1) < 0.95
        assert ucr(12, 0.95, 40 * IN, 15.0, "westfall_2019") == 0.95


class TestEveryEquation:
    @pytest.mark.parametrize("equation", EQUATIONS)
    def test_zero_crown_stays_zero(self, equation):
        # Also for species the equation does not cover
        got = estimate_uncompacted_crown_ratio(
            [202, 131, 316, 6156], 0.0, 30.0, 18.0, equation=equation
        )
        np.testing.assert_array_equal(got, 0.0)

    @pytest.mark.parametrize("equation", PUBLISHED)
    @pytest.mark.parametrize("ccr", [0.5, 0.95, 1.0])
    def test_uncovered_species_is_nan_at_every_ccr(self, equation, ccr):
        # Norfolk Island pine is in no paper, genus or group
        assert np.isnan(ucr(6156, ccr, 30.0, 18.0, equation))

    @pytest.mark.parametrize("equation", EQUATIONS)
    def test_percent_raises(self, equation):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            estimate_uncompacted_crown_ratio(202, 45.0, 30.0, 18.0, equation=equation)

    @pytest.mark.parametrize("equation", EQUATIONS)
    def test_unknown_species_raises(self, equation):
        with pytest.raises(ValueError, match="not in the FIA reference"):
            estimate_uncompacted_crown_ratio(999999, 0.4, 30.0, 18.0, equation=equation)

    @pytest.mark.parametrize("equation", EQUATIONS)
    @pytest.mark.parametrize(
        "ccr, dbh_cm, height_m",
        [(np.nan, 30.0, 18.0), (0.4, np.nan, 18.0), (0.4, 0.0, 18.0), (0.4, 30, -1)],
    )
    def test_missing_inputs_give_nan(self, equation, ccr, dbh_cm, height_m):
        # 202 is covered by every equation
        assert np.isnan(ucr(202, ccr, dbh_cm, height_m, equation))

    @pytest.mark.parametrize("equation", EQUATIONS)
    def test_result_in_unit_interval(self, equation):
        spcd = REF_SPECIES.index.to_numpy()
        n = len(spcd)
        for ccr in (0.01, 0.5, 0.9, 1.0):
            for dbh, height in ((2.6, 1.5), (30.0, 18.0), (250.0, 90.0)):
                got = estimate_uncompacted_crown_ratio(
                    spcd,
                    np.full(n, ccr),
                    np.full(n, dbh),
                    np.full(n, height),
                    equation=equation,
                )
                got = got[~np.isnan(got)]
                assert got.size
                assert ((got >= 0) & (got <= 1)).all()

    @pytest.mark.parametrize("equation", PUBLISHED)
    def test_pacific_northwest_with_published_raises(self, equation):
        with pytest.raises(ValueError, match="pacific_northwest applies only"):
            estimate_uncompacted_crown_ratio(
                202, 0.4, 30.0, 18.0, pacific_northwest=True, equation=equation
            )
        with pytest.raises(ValueError, match="pacific_northwest applies only"):
            estimate_crown_base_height(
                [202, 202], 0.4, 30.0, 18.0, [False, True], equation=equation
            )

    def test_unknown_equation_raises(self):
        with pytest.raises(ValueError, match="Unknown crown ratio equation"):
            estimate_uncompacted_crown_ratio(202, 0.4, 30.0, 18.0, equation="nope")

    def test_default_is_the_refit(self):
        args = ([202, 316, 6156], 0.4, 30.0, 18.0)
        np.testing.assert_array_equal(
            estimate_uncompacted_crown_ratio(*args),
            estimate_uncompacted_crown_ratio(*args, equation=FASTFUELS_2026),
        )

    @pytest.mark.parametrize("equation", PUBLISHED)
    def test_crown_base_height_matches_ratio(self, equation):
        spcd = [202, 122, 263, 131, 316, 12]
        cbh = estimate_crown_base_height(spcd, 0.45, 30.0, 18.0, equation=equation)
        u = estimate_uncompacted_crown_ratio(spcd, 0.45, 30.0, 18.0, equation=equation)
        np.testing.assert_array_equal(cbh, 18.0 * (1.0 - u))


def test_coefficient_file_integrity():
    rows = pd.read_csv(DATA_PATH / "REF_CROWN_RATIO.csv", dtype={"KEY": str})
    assert set(rows.SOURCE) == set(EQUATIONS)
    assert not rows.duplicated(["SOURCE", "REGION", "KEY_TYPE", "KEY"]).any()
    coefficients = ["B0", "B_CCR", "B_LN_DIA", "B_LN_HT", "B_HD"]
    assert rows[coefficients].notna().all().all()
    assert set(rows.RECOMMENDED) == {0, 1}

    refit = rows[rows.SOURCE == FASTFUELS_2026]
    assert len(refit) == 374
    assert set(refit.REGION) == {"national", "pacific_northwest"}
    assert set(refit[refit.REGION == "pacific_northwest"].KEY_TYPE) == {"SPCD"}
    assert set(refit[refit.KEY_TYPE == "POOLED"].KEY) == {"softwood", "hardwood"}
    assert (refit.EXP_SIGN == -1).all()

    papers = rows[rows.SOURCE != FASTFUELS_2026]
    assert papers.groupby("SOURCE").size().to_dict() == {
        "monleon_2004_eq1": 31,  # 30 models, red fir on two codes
        "monleon_2004_eq2": 31,  # 21 + 9 from Table 2, red fir on two codes
        "randolph_2010": 37,  # 36 models, forest junipers on two codes
        "toney_reeves_2009": 35,
        "westfall_2019": 27,
    }
    assert papers.TABLE.notna().all()
    assert set(papers.DIAMETER) == {"DBH", "DRC"}
    assert set(papers.KEY_TYPE) <= {"SPCD", "GENUS", "WOODLAND_GENUS", "SPGRP"}
    not_recommended = papers[papers.RECOMMENDED == 0].groupby("SOURCE").LABEL
    assert {k: set(v) for k, v in not_recommended} == {
        "monleon_2004_eq1": {
            "Abies lasiocarpa",
            "Abies magnifica",
            "Juniperus occidentalis",
            "Larix occidentalis",
            "Acer macrophyllum",
            "Quercus douglasii",
        },
        "monleon_2004_eq2": {
            "Abies lasiocarpa",
            "Abies magnifica",
            "Abies procera",
            "Juniperus occidentalis",
            "Acer macrophyllum",
            "Quercus douglasii",
        },
        "toney_reeves_2009": {"Pinus monophylla"},
        "randolph_2010": {
            "JUAS Juniperus ashei",
            "PRGL Prosopis glandulosa",
            "ROPS Robinia pseudoacacia",
        },
    }
    # Westfall's sign convention: more compacted crown, more uncompacted crown
    westfall = papers.SOURCE == "westfall_2019"
    assert (papers[westfall].EXP_SIGN == 1).all()
    assert (papers[westfall].B_CCR < 0).all()
    assert (papers[~westfall].EXP_SIGN == -1).all()
    assert (rows[rows.EXP_SIGN == -1].B_CCR > 0).all()

    spcd = rows[rows.KEY_TYPE == "SPCD"].KEY.astype(int)
    assert spcd.isin(REF_SPECIES.index).all()
    genus = rows[rows.KEY_TYPE.isin(["GENUS", "WOODLAND_GENUS"])].KEY
    assert genus.isin(REF_SPECIES.GENUS).all()
    spgrp = rows[rows.KEY_TYPE == "SPGRP"].KEY.astype(int)
    assert spgrp.isin(REF_SPECIES.E_SPGRPCD).all()
