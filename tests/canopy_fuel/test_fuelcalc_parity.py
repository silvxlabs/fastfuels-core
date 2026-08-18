"""Parity of the canopy_fuel module against the FuelCalc 1.8 C source.

Every assertion here compares our code to
:mod:`tests.canopy_fuel.fuelcalc_reference`, a line-for-line
transcription of FuelCalc's own arithmetic. The point is not that we
must agree everywhere — we deliberately do not — but that every place we
disagree is *recorded*, so a future edit cannot silently move us toward
or away from the binary.

Two registries carry that record. :data:`EXPECTED_DIVERGENCES` lists the
crown-proportion Ids where we knowingly follow the primary sources
instead of the shipped table, and :data:`EXPECTED_TABLE_DIVERGENCES`
lists the species-table rows where we currently follow the User Guide
instead of the source. Both are asserted in both directions: everything
outside a registry must match exactly, and everything inside it must
still differ. Resolving a divergence therefore means deleting its entry,
not editing a tolerance.

The species-table check runs against ``fuelcalc_sr_esd.csv``, a frozen
parse of the live ``sr_ESD[]`` table in ``FC_DLL/NC_ESD.C`` (the rows
above ``#ifdef PRE_UN_CHANGE``, which archives a superseded copy). It is
committed so the suite is hermetic; regenerate it from a FuelCalc source
checkout if the upstream table changes.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry import brown
from fastfuels_core.canopy_fuel.metrics import (
    FT_TO_M,
    canopy_fuel_load,
    cbd_running_mean,
    cumulative_fuel_fraction,
    profile_threshold_heights,
    vertical_profile,
)
from fastfuels_core.canopy_fuel.ref_data import (
    fuelcalc_crown_class_factors,
    fuelcalc_species,
    fuelcalc_vdist,
)
from tests.canopy_fuel import fuelcalc_reference as fc

LAYER_FT_M = 0.3048

# Diameters spanning the fitted range, with every documented break and
# curve crossing hit exactly and from both sides.
DIA_IN = np.unique(
    np.round(
        np.concatenate(
            [
                np.arange(1.1, 60.0, 0.1),
                [
                    1.0001,
                    19.99,
                    20.0,
                    20.01,
                    31.0,
                    31.5,
                    32.0,
                    35.99,
                    36.0,
                    36.01,
                    39.99,
                    40.0,
                    40.01,
                ],
            ]
        ),
        4,
    )
)

SHARED_IDS = sorted(set(fc.ALL_IDS) & set(brown.P1_EQUATIONS) & set(brown.P2_EQUATIONS))

# (equation Id, quantity) -> why we differ from the shipped C table.
EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {
    ("WL", "p2"): (
        "NC_BM.C:69 uses 0.745*exp(-0.0632*d); the User Guide and our "
        "reading of Brown Table 16 give -0.0362. The shipped coefficient "
        "drives P2 below P1 at 38.6 in, and Brown flags every other "
        "crossing with an explicit Conditions entry but none for larch, "
        "so we treat -0.0632 as a transposition. Propagates to AL and QA."
    ),
    ("GF", "p2"): (
        "NC_BM.C:44 sets the Twg high-value to 0.286, identical to the "
        "Fol high-value, zeroing fine branchwood above 36 in. Brown "
        "Table 16 gives 0.378. FuelCalc bug."
    ),
    ("PP", "p2"): (
        "We hold the fine fraction at 0.01*CW past the 31.5 in curve "
        "crossing, per Brown's printed condition. FuelCalc has no "
        "override and lets BT_GetWC clamp the negative difference to 0."
    ),
}
EXPECTED_DIVERGENCES.update(
    {
        (eq_id, "fine"): reason
        for (eq_id, q), reason in list(EXPECTED_DIVERGENCES.items())
        if q == "p2"
    }
)

# SPCD -> (column, ours, theirs, why). Rows where our shipped species
# table follows the User Guide and the C source disagrees.
EXPECTED_TABLE_DIVERGENCES: dict[int, str] = {
    113: (
        "Limber Pine: FuelCalc ticket 176 (2018-10-25) moved every "
        "column to WB/LP/LP; the User Guide still prints the pre-2018 "
        "LP/PP/PP row that we transcribed."
    ),
}

# SPCDs present in the source table and absent from ours.
EXPECTED_TABLE_OMISSIONS: dict[int, str] = {
    312: (
        "Bigleaf Maple (ACMA3). The guide's Default Equation Table skips "
        "this row; the source has it."
    ),
}

# Two NRCS symbols in sr_ESD are not valid FIA symbols. The intended
# species is unambiguous from the common name and FOFEM mortality code.
SYMBOL_FIXUPS = {"SEQUO": 211, "QUMU": 431}


@pytest.fixture(scope="module")
def source_species_table() -> pd.DataFrame:
    """The frozen parse of sr_ESD[], keyed by FIA species code."""
    table = pd.read_csv(files("tests.canopy_fuel").joinpath("fuelcalc_sr_esd.csv"))
    ref = pd.read_csv(
        files("fastfuels_core.data").joinpath("REF_SPECIES.csv"),
        low_memory=False,
    )
    symbol_to_spcd = dict(zip(ref["SPECIES_SYMBOL"], ref["SPCD"]))
    symbol_to_spcd.update(SYMBOL_FIXUPS)
    table["SPCD"] = table["SYMBOL"].map(symbol_to_spcd)
    assert table["SPCD"].notna().all(), "unmapped NRCS symbol in sr_ESD"
    return table.astype({"SPCD": int}).set_index("SPCD")


def _ours(quantity: str, equation_id: str, dia: np.ndarray) -> np.ndarray:
    ids = np.full(dia.shape, equation_id, dtype=object)
    if quantity == "p1":
        return brown.foliage_fraction(ids, dia)
    if quantity == "p2":
        return brown.foliage_plus_fine_fraction(ids, dia)
    # Our fine share is a fraction of branchwood; FuelCalc's is a
    # fraction of total crown weight. (1 - P1) converts between them.
    return brown.fine_branchwood_share(ids, ids, dia) * (
        1.0 - brown.foliage_fraction(ids, dia)
    )


def _theirs(quantity: str, equation_id: str, dia: np.ndarray) -> np.ndarray:
    fn = {"p1": fc.p1, "p2": fc.p2, "fine": fc.fine_fraction_of_crown}[quantity]
    return np.array([fn(equation_id, float(d)) for d in dia])


class TestCrownProportionParity:
    """brown.py against NC_BM.C sr_BT[] and NC_BMSL.C sr_SL[]."""

    @pytest.mark.parametrize("quantity", ["p1", "p2", "fine"])
    @pytest.mark.parametrize("equation_id", SHARED_IDS)
    def test_matches_source(self, equation_id, quantity):
        if (equation_id, quantity) in EXPECTED_DIVERGENCES:
            pytest.skip(EXPECTED_DIVERGENCES[(equation_id, quantity)])
        np.testing.assert_allclose(
            _ours(quantity, equation_id, DIA_IN),
            _theirs(quantity, equation_id, DIA_IN),
            atol=1e-12,
            err_msg=f"{equation_id} {quantity} drifted from the C source",
        )

    @pytest.mark.parametrize("equation_id,quantity", sorted(EXPECTED_DIVERGENCES))
    def test_documented_divergence_still_diverges(self, equation_id, quantity):
        """Deleting a divergence requires deleting its registry entry."""
        ours = _ours(quantity, equation_id, DIA_IN)
        theirs = _theirs(quantity, equation_id, DIA_IN)
        assert np.abs(ours - theirs).max() > 1e-9, (
            f"{equation_id} {quantity} now matches FuelCalc. If that is "
            f"intended, drop its EXPECTED_DIVERGENCES entry."
        )

    def test_available_fuel_is_foliage_plus_half_fine(self):
        """_AvailFuel is Fol + 0.5*Twg with Twg the *differenced* weight.

        This is the arithmetic the FuelCalc guide misprints as
        ``pfol*CW + 0.5*pb*CW``. Pin it so the oracle cannot drift.
        """
        for eq_id in fc.BROWN_IDS:
            for d in (5.0, 12.0, 25.0):
                expected = fc.bt_get_wc(eq_id, "Fol", d) + 0.5 * fc.bt_get_wc(
                    eq_id, "Twg", d
                )
                assert fc.available_canopy_fuel_lb(eq_id, d) == expected
                crown = fc.bt_eq(eq_id, "Tot", d)
                naive = crown * (fc.p1(eq_id, d) + 0.5 * fc.p2(eq_id, d))
                assert naive > expected, "the misprinted formula is larger"

    def test_no_unaccounted_equation_ids(self):
        """Every Id we define is either in the source or explained here."""
        ours = set(brown.P1_EQUATIONS) & set(brown.P2_EQUATIONS)
        vestigial = {
            # sr_EFD has no AL entry at all; Subalpine Larch (LALY) uses
            # the WL Ids for all six components.
            "AL",
            # sr_EFD maps QA to QuakingAspen_MN (Loomis & Roussopoulos
            # 1978, NC_BM3.C), not to the whitebark/larch borrow the
            # guide's Appendix D prints. Ours is reachable: SPCD 746
            # resolves FOL and TWIG to QA.
            "QA",
        }
        assert ours - set(fc.ALL_IDS) == vestigial
        assert set(fc.ALL_IDS) - ours == set(), "source Id we do not define"

    @pytest.mark.parametrize("equation_id", SHARED_IDS)
    def test_fine_share_conversion_is_consistent(self, equation_id):
        """Our branchwood-relative share and FuelCalc's crown-relative
        one differ only by the (1 - P1) factor, by construction."""
        ids = np.full(DIA_IN.shape, equation_id, dtype=object)
        p1 = brown.foliage_fraction(ids, DIA_IN)
        share = brown.fine_branchwood_share(ids, ids, DIA_IN)
        assert ((share >= 0.0) & (share <= 1.0)).all()
        np.testing.assert_allclose(
            share * (1.0 - p1), _ours("fine", equation_id, DIA_IN), atol=1e-15
        )


class TestVerticalDistributionParity:
    """metrics.cumulative_fuel_fraction against NC_VD.C."""

    @pytest.mark.parametrize("vdist_code", sorted(fc.VDIST_CUBICS))
    def test_cubic_coefficients_match_source(self, vdist_code):
        row = fuelcalc_vdist().loc[vdist_code]
        assert (row.B1, row.B2, row.B3) == fc.VDIST_CUBICS[vdist_code]

    def test_ps_cubic_does_not_close(self):
        """PS is the one cubic whose coefficients do not sum to 1.

        Reinhardt et al. (2006) Table 4 rounds it to 1.0001. The table
        is a transcription, so it keeps the published value; the
        closure happens where FuelCalc does it, in the layer weights.
        """
        assert sum(fc.VDIST_CUBICS["PS"]) == pytest.approx(1.0001, abs=1e-9)
        others = [c for c in fc.VDIST_CUBICS if c != "PS"]
        for code in others:
            assert sum(fc.VDIST_CUBICS[code]) == pytest.approx(1.0, abs=1e-12)
        # SPCD 135 (Arizona pine) is the only species that reaches it.
        reaches_ps = (
            fuelcalc_species().index[fuelcalc_species()["VDIST_CODE"] == "PS"].tolist()
        )
        assert reaches_ps == [135]

    def test_ps_layer_weights_conserve_mass(self):
        """VD_Calc closes the crown at the top; so must we.

        FuelCalc's top-of-crown branch is ``pcWT = 1 - pw(layer
        bottom)`` (``_Top`` in ``FC_DLL/NC_VD.C``), which absorbs the
        1.0001. A difference of raw cumulatives would hand every
        Arizona pine 100.01% of its fuel.
        """
        trees = pd.DataFrame(
            {
                "x": [5.0, 15.0, 25.0],
                "y": [-5.0, -5.0, -15.0],
                "height": [12.0, 20.0, 7.3],
                "crown_ratio": [0.4, 0.65, 0.9],
                "fia_species_code": [135, 135, 135],
                "dbh": [25.0, 40.0, 12.0],
            }
        )
        fuel = np.array([10.0, 25.0, 4.0])
        transform = (10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        profile = vertical_profile(
            trees,
            fuel,
            transform,
            (3, 3),
            horizontal_distribution="stem",
        )
        cell_volume = 10.0 * 10.0 * FT_TO_M
        assert profile.sum() * cell_volume == pytest.approx(fuel.sum(), rel=1e-12)

    def test_cumulative_fraction_matches_crown_fraction(self):
        ph = np.linspace(0.0, 1.0, 51)
        for spcd, code in (
            (122, "PP"),
            (202, "DF"),
            (108, "LP"),
            (242, "IC"),
            (746, "UN"),
        ):
            np.testing.assert_allclose(
                cumulative_fuel_fraction(np.full(ph.shape, spcd), ph),
                [fc.crown_fraction(code, float(p)) for p in ph],
                atol=1e-12,
            )

    @pytest.mark.parametrize("vdist_code", sorted(set(fc.VDIST_CUBICS) - {"PS"}))
    def test_layer_weights_match_vd_calc(self, vdist_code):
        """Our difference-of-clipped-cumulatives reproduces VD_Calc's
        three-case loop over random crown geometries."""
        rng = np.random.default_rng(20260817)
        b1, b2, b3 = fc.VDIST_CUBICS[vdist_code]
        for _ in range(120):
            top = float(rng.uniform(1.0, 40.0))
            base = float(rng.uniform(0.0, top))
            n = int(np.ceil(top / LAYER_FT_M)) + 3
            theirs = np.array(fc.vd_calc(vdist_code, base, top, LAYER_FT_M, 1.0, n))
            bounds = np.arange(n + 1) * LAYER_FT_M
            ph = np.clip((bounds - base) / max(top - base, 1e-9), 0.0, 1.0)
            ours = np.diff(b1 * ph + b2 * ph**2 + b3 * ph**3)
            np.testing.assert_allclose(
                ours,
                theirs,
                atol=1e-12,
                err_msg=f"{vdist_code} crown [{base}, {top}]",
            )
            assert ours.sum() == pytest.approx(1.0, abs=1e-12)

    def test_whole_crown_inside_one_layer_is_a_point_mass(self):
        """VD_Calc's _Bot branch clamps its ratio to 1.0, putting the
        entire crown in the layer holding the crown base — which is what
        our zero-length-crown handling does."""
        weights = fc.vd_calc("PP", 1.28, 1.40, LAYER_FT_M, 1.0, 8)
        assert weights[4] == pytest.approx(1.0)
        assert sum(weights) == pytest.approx(1.0)


class TestProfileReductionParity:
    """metrics reductions against NC_PTL.C _BulkDensity().

    Run with the window matched to FuelCalc's five 1-ft layers, which is
    the configuration the `fuelcalc_compat` API example targets. Our
    shipped defaults (3.0 m window, no CBH/CH smoothing) are a different
    and deliberate choice; see fastfuels-core#95.
    """

    WINDOW = 5 * LAYER_FT_M

    @staticmethod
    def _profiles(n=200, seed=20260817):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            n_layers = int(rng.integers(12, 90))
            profile = np.zeros(n_layers)
            base = int(rng.integers(0, max(1, n_layers - 6)))
            top = int(rng.integers(base + 2, n_layers))
            profile[base:top] = rng.uniform(0.001, 0.35, top - base)
            yield profile, base

    def test_cbd_matches_when_canopy_is_off_the_ground(self):
        """With the crown base clear of the ground, the maximum running
        mean agrees exactly: every window FuelCalc evaluates that we do
        not is either zero-padded above the canopy or below the crown."""
        for profile, base in self._profiles():
            if base < 3:
                continue
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            ours = cbd_running_mean(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                window=self.WINDOW,
            )[0, 0]
            assert ours == pytest.approx(reference.cbd, abs=1e-12)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FuelCalc truncates the running mean at the ground — the "
            "window at layer 0 averages 3 layers, at layer 1 averages 4 "
            "— so a crown reaching the ground can produce a maximum we "
            "never evaluate. Ours only takes full-width interior "
            "windows. Affects cells whose crown base is within 2 ft of "
            "the ground. fastfuels-core#95 D10."
        ),
    )
    def test_cbd_matches_when_canopy_reaches_the_ground(self):
        for profile, base in self._profiles():
            if base >= 3:
                continue
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            ours = cbd_running_mean(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                window=self.WINDOW,
            )[0, 0]
            assert ours == pytest.approx(reference.cbd, abs=1e-12)

    def test_cbh_matches_up_to_the_deliberate_anchor_offset(self):
        """With the bounds in place, the only CBH difference left is the
        layer-anchor convention: ours is the bottom of the layer,
        FuelCalc's the top, so ours sits exactly one layer lower. Before
        the bounds landed this gap was three layers."""
        for profile, _ in self._profiles(n=60):
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            cbh, _ = profile_threshold_heights(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                threshold=0.012,
                relative_fraction=0.1,
                smoothing_window=self.WINDOW,
            )
            assert cbh[0, 0] + LAYER_FT_M == pytest.approx(reference.cbh, abs=1e-12)

    def test_chm_matches(self):
        """Canopy height already shares FuelCalc's top-of-layer anchor,
        so with the bounds in place it agrees exactly."""
        for profile, _ in self._profiles(n=60):
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            _, chm = profile_threshold_heights(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                threshold=0.012,
                relative_fraction=0.1,
                smoothing_window=self.WINDOW,
            )
            assert chm[0, 0] == pytest.approx(reference.sh, abs=1e-12)

    def test_layer_anchor_convention_is_deliberate(self):
        """CBH anchors to the bottom of its layer, CH to the top.

        FuelCalc labels every layer by its top for both heights
        (NC_PTL.C:663-667), so its CBH sits one layer above ours. Ours
        spans the qualifying layers instead, and that is the property
        the rest of the module depends on: canopy depth is the true
        depth of qualifying canopy, load over depth recovers the mean
        bulk density, and a single qualifying layer still has a
        positive depth. A top or midpoint anchor on both ends breaks
        all three, so this is not a knob.
        """
        layer = LAYER_FT_M
        profile = np.zeros((10, 1, 1))
        profile[4:7, 0, 0] = 0.05
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=layer, relative_fraction=None
        )
        assert cbh[0, 0] == pytest.approx(4 * layer)  # bottom of layer 4
        assert chm[0, 0] == pytest.approx(7 * layer)  # top of layer 6
        depth = chm[0, 0] - cbh[0, 0]
        assert depth == pytest.approx(3 * layer), "one layer per qualifier"

        load = canopy_fuel_load(profile, layer_depth=layer)[0, 0]
        assert load / depth == pytest.approx(0.05), "mean density recovered"

        single = np.zeros((10, 1, 1))
        single[4, 0, 0] = 0.05
        lo, hi = profile_threshold_heights(
            single, layer_depth=layer, relative_fraction=None
        )
        assert hi[0, 0] - lo[0, 0] == pytest.approx(layer), (
            "a single qualifying layer must have positive depth; a top "
            "or midpoint anchor would make this zero and divide by zero "
            "in a load_over_depth CBD"
        )

        reference = fc.bulk_density(list(profile[:, 0, 0]), layer, spread=1)
        assert reference.cbh == pytest.approx(
            cbh[0, 0] + layer
        ), "FuelCalc's CBH is exactly one layer above ours"
        assert reference.sh == pytest.approx(
            chm[0, 0]
        ), "canopy height already agrees; only the base anchor differs"

    def test_threshold_rule_matches(self):
        """min(max_smoothed/10, 0.012) — the one part of the height rule
        we already reproduce exactly, given the same smoothed profile."""
        for profile, _ in self._profiles(n=60):
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            smoothed_max = cbd_running_mean(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                window=self.WINDOW,
            )[0, 0]
            if smoothed_max != pytest.approx(reference.cbd, abs=1e-12):
                continue  # covered by the ground-window xfail above
            assert min(smoothed_max / 10.0, 0.012) == pytest.approx(
                reference.threshold, abs=1e-15
            )

    def test_spread_and_threshold_constants(self):
        assert fc.RUNNING_MEAN_SPREAD == 5  # FC_DLL.h:634
        assert fc.CBD_CRITICAL_THRESHOLD == 0.012  # FC_DLL.h:639
        assert self.WINDOW == pytest.approx(1.524, abs=1e-9)


class TestSpeciesTableParity:
    """FUELCALC_SPECIES_TABLE.csv against the frozen sr_ESD[] parse."""

    COLUMNS = [
        "INCL_CBD",
        "TOTAL",
        "FOL",
        "TWIG",
        "IN1",
        "IN3",
        "IN3PLUS",
        "VDIST_CODE",
        "CROWN_REDUC_CODE",
    ]

    def test_rows_match_source(self, source_species_table):
        ours = fuelcalc_species()
        shared = sorted(set(ours.index) & set(source_species_table.index))
        mismatched = {
            spcd: [
                f"{c}: ours {ours.loc[spcd, c]!r} != source "
                f"{source_species_table.loc[spcd, c]!r}"
                for c in self.COLUMNS
                if ours.loc[spcd, c] != source_species_table.loc[spcd, c]
            ]
            for spcd in shared
        }
        mismatched = {k: v for k, v in mismatched.items() if v}
        unexpected = set(mismatched) - set(EXPECTED_TABLE_DIVERGENCES)
        assert (
            not unexpected
        ), "species table rows drifted from the C source: " + "; ".join(
            f"SPCD {s}: {mismatched[s]}" for s in sorted(unexpected)
        )

    @pytest.mark.parametrize("spcd", sorted(EXPECTED_TABLE_DIVERGENCES))
    def test_documented_row_still_differs(self, spcd, source_species_table):
        ours = fuelcalc_species()
        assert any(
            ours.loc[spcd, c] != source_species_table.loc[spcd, c] for c in self.COLUMNS
        ), (
            f"SPCD {spcd} now matches the source. If that is intended, "
            f"drop its EXPECTED_TABLE_DIVERGENCES entry."
        )

    def test_omissions_are_documented(self, source_species_table):
        missing = set(source_species_table.index) - set(fuelcalc_species().index)
        assert missing == set(EXPECTED_TABLE_OMISSIONS), (
            f"species present in the source and absent from ours: "
            f"{sorted(missing - set(EXPECTED_TABLE_OMISSIONS))}; "
            f"documented but no longer missing: "
            f"{sorted(set(EXPECTED_TABLE_OMISSIONS) - missing)}"
        )

    def test_no_rows_we_invented(self, source_species_table):
        extra = set(fuelcalc_species().index) - set(source_species_table.index)
        assert not extra, f"SPCDs not in the FuelCalc source: {sorted(extra)}"

    def test_crown_class_factors_match_source(self):
        ours = fuelcalc_crown_class_factors()
        source = {
            "WF": (0.85, 0.85, 0.35, 0.3, 0.5),
            "PP": (0.55, 0.55, 0.3, 0.15, 0.5),
            "PS": (0.3, 0.3, 0.15, 0.1, 0.5),
            "IC": (1.1, 1.1, 0.75, 0.4, 0.5),
            "DF": (1.15, 1.15, 1.15, 0.75, 0.5),
            "LP": (0.6, 0.6, 0.6, 0.3, 0.5),
            "WL": (1.0, 0.45, 0.30, 0.20, 0.5),
            "WP": (0.80, 0.90, 0.60, 0.35, 0.7),
            "WC": (1.0, 1.0, 1.0, 0.60, 0.75),
            "PJ": (1.0, 1.0, 1.0, 1.0, 1.0),
        }
        columns = ["DOMINANT", "CODOMINANT", "INTERMEDIATE", "SUPPRESSED", "OTHER_NONE"]
        for code, expected in source.items():
            assert tuple(ours.loc[code, columns]) == expected

    def test_grand_fir_crown_class_row_is_unreachable(self):
        """sr_CCT's GF row is commented out in NC_CC.C ("doesn't seem to
        get used"), correctly: every GF-biomass species routes to WF for
        crown reduction. We carry the row; nothing may reference it."""
        assert "GF" in fuelcalc_crown_class_factors().index
        assert "GF" not in set(fuelcalc_species()["CROWN_REDUC_CODE"])
