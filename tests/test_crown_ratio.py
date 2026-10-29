"""Tests for ``allometry.crown_ratio``, the uncompacted crown ratio model
and the crown base height built on it.

The coefficient rows used as oracles are transcribed here from the 2026-10
national refit, so a later edit to ``REF_CROWN_RATIO.csv`` shows up as a
failing test rather than silently changing every inventory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry.crown_ratio import (
    crown_base_height,
    uncompacted_crown_ratio,
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
        got = uncompacted_crown_ratio(spcd, 0.4, 25.4, 20.0, pacific_northwest=pnw)
        assert got[0] == pytest.approx(logistic(coefs, 0.4, 25.4, 20.0), rel=1e-12)

    def test_hand_computed_value(self):
        # 0.0686 + 4.57562*0.4 + 0.29439*ln(10 in) - 0.57148*ln(65.617 ft)
        assert uncompacted_crown_ratio(202, 0.4, 25.4, 20.0)[0] == pytest.approx(
            0.54630, abs=1e-5
        )

    def test_pacific_northwest_per_tree(self):
        got = uncompacted_crown_ratio(
            [202, 202], [0.4, 0.4], [25.4, 25.4], [20.0, 20.0], [False, True]
        )
        assert got[0] == pytest.approx(logistic(DOUGLAS_FIR, 0.4, 25.4, 20.0))
        assert got[1] == pytest.approx(logistic(DOUGLAS_FIR_PNW, 0.4, 25.4, 20.0))

    def test_every_reference_species_resolves(self):
        spcd = REF_SPECIES.index.to_numpy()
        n = len(spcd)
        for pnw in (False, True):
            got = uncompacted_crown_ratio(
                spcd, np.full(n, 0.4), np.full(n, 25.4), np.full(n, 20.0), pnw
            )
            assert np.isfinite(got).all()

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="not in the FIA reference"):
            uncompacted_crown_ratio(999999, 0.4, 25.4, 20.0)


class TestRules:
    def test_never_below_compacted(self):
        ccr = np.linspace(0.01, 1.0, 100)
        n = len(ccr)
        got = uncompacted_crown_ratio(
            np.full(n, 202), ccr, np.full(n, 80.0), np.full(n, 60.0)
        )
        assert (got >= ccr).all()
        assert (got <= 1.0).all()

    def test_full_crown_stays_full(self):
        assert uncompacted_crown_ratio(202, 1.0, 25.4, 20.0)[0] == 1.0

    def test_zero_crown_stays_zero(self):
        # A dead tree carries CR 0; the logistic alone would give it a crown
        assert uncompacted_crown_ratio(202, 0.0, 25.4, 20.0)[0] == 0.0

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
        assert np.isnan(uncompacted_crown_ratio(202, ccr, dbh_cm, height_m)[0])

    def test_percent_raises(self):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            uncompacted_crown_ratio(202, 40.0, 25.4, 20.0)


class TestCrownBaseHeight:
    def test_matches_uncompacted_ratio(self):
        args = (
            [202, 202, AILANTHUS_VARIETY, ARAUCARIA],
            [0.4, 0.4, 0.7, 0.2],
            [25.4, 25.4, 12.0, 60.0],
            [20.0, 20.0, 9.5, 35.0],
            [False, True, False, True],
        )
        expected = np.asarray(args[3]) * (1.0 - uncompacted_crown_ratio(*args))
        np.testing.assert_allclose(crown_base_height(*args), expected, rtol=1e-12)

    def test_hand_computed_value(self):
        # Douglas-fir UCR 0.54630 at CCR 0.4, 25.4 cm, 20 m
        assert crown_base_height(202, 0.4, 25.4, 20.0)[0] == pytest.approx(
            20.0 * (1.0 - 0.54630), abs=1e-4
        )

    def test_zero_crown_is_full_height(self):
        assert crown_base_height(202, 0.0, 25.4, 20.0)[0] == 20.0

    @pytest.mark.parametrize(
        "ccr, dbh_cm, height_m",
        [
            (np.nan, 25.4, 20.0),
            (0.4, 25.4, np.nan),
            (0.4, 25.4, 0.0),
            (0.4, 25.4, -1.0),
            (0.4, 0.0, 20.0),
            (0.0, 25.4, 0.0),
        ],
    )
    def test_missing_inputs_give_nan(self, ccr, dbh_cm, height_m):
        assert np.isnan(crown_base_height(202, ccr, dbh_cm, height_m)[0])

    def test_within_tree_height(self):
        ccr = np.linspace(0.0, 1.0, 101)
        n = len(ccr)
        height = np.linspace(1.0, 60.0, n)
        got = crown_base_height(np.full(n, 202), ccr, np.full(n, 40.0), height)
        assert (got >= 0.0).all()
        assert (got <= height).all()

    def test_scalar_returns_length_one_array(self):
        got = crown_base_height(202, 0.4, 25.4, 20.0)
        assert isinstance(got, np.ndarray)
        assert got.shape == (1,)

    def test_percent_raises(self):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            crown_base_height(202, 40.0, 25.4, 20.0)


def test_coefficient_file_integrity():
    rows = pd.read_csv(DATA_PATH / "REF_CROWN_RATIO.csv", dtype={"KEY": str})
    assert not rows.duplicated(["REGION", "KEY_TYPE", "KEY"]).any()
    assert set(rows.REGION) == {"national", "pacific_northwest"}
    assert set(rows[rows.REGION == "pacific_northwest"].KEY_TYPE) == {"SPCD"}
    assert set(rows[rows.KEY_TYPE == "POOLED"].KEY) == {"softwood", "hardwood"}
    spcd = rows[rows.KEY_TYPE == "SPCD"].KEY.astype(int)
    assert spcd.isin(REF_SPECIES.index).all()
    assert rows[rows.KEY_TYPE == "GENUS"].KEY.isin(REF_SPECIES.GENUS).all()
    # More compacted crown always means more uncompacted crown
    assert (rows.B_CCR > 0).all()
