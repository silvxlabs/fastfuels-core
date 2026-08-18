"""Parity of the canopy_fuel module against the FuelCalc 1.8 C source.

Every assertion here compares our code to
:mod:`tests.canopy_fuel.fuelcalc_reference`, which implements the same
equations with FuelCalc's algorithmic structure. What is being tested is
the *algorithm* — accumulative differencing, available fuel as foliage
plus half the fine branchwood, the vertical-distribution layer weights,
the running-mean bulk density and its threshold heights — against the
canonical implementation of it. The equations themselves are pinned to
their primary sources in :mod:`tests.canopy_fuel.test_brown_table16`.

:data:`EXPECTED_TABLE_DIVERGENCES` lists the species-table rows where we
currently follow the User Guide instead of the C source, and
:data:`EXPECTED_TABLE_OMISSIONS` the rows we do not carry. Both are
asserted in both directions: everything outside a registry must match
exactly, and everything inside it must still differ. Resolving one
therefore means deleting its entry, not editing a tolerance.

The species-table check runs against ``fuelcalc_sr_esd.csv``, a frozen
parse of the live ``sr_ESD[]`` table in ``FC_DLL/NC_ESD.C`` (the rows
above ``#ifdef PRE_UN_CHANGE``, which archives a superseded copy). It is
committed so the suite is hermetic; regenerate it from a FuelCalc source
checkout if the upstream table changes.
"""

from __future__ import annotations

from importlib.resources import files

import math

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry import brown, fvs
from fastfuels_core.canopy_fuel.metrics import (
    FT_TO_M,
    available_canopy_fuel,
    canopy_cover,
    canopy_fuel_load,
    crown_class_factor,
    cbd_running_mean,
    cumulative_fuel_fraction,
    profile_threshold_heights,
    vertical_profile,
)
from fastfuels_core.units import conversion_factor
from fastfuels_core.canopy_fuel.ref_data import (
    fuelcalc_crown_class_factors,
    fuelcalc_crown_width,
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
        np.testing.assert_allclose(
            _ours(quantity, equation_id, DIA_IN),
            _theirs(quantity, equation_id, DIA_IN),
            atol=1e-12,
            err_msg=f"{equation_id} {quantity} drifted from the C source",
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

    def test_brown_1978_arm_reproduces_available_fuel_exactly(self):
        """The whole per-tree chain, ours against the reference.

        Under ``equations="brown_1978"`` both sides take crown weight
        from Brown Table 1 and split it with Brown Table 16, so this is
        an absolute comparison of kilograms per tree rather than of a
        proportion -- the thing that could not be checked while the only
        biomass arm was NSVB. Height is passed but unread: Brown's
        diameter-only forms do not use it.
        """
        species = fuelcalc_species()
        for eq_id in sorted(brown.CROWN_WEIGHT_EQUATIONS):
            spcd = int(species.index[species["TOTAL"] == eq_id][0])
            dia_in = np.round(np.arange(1.05, 45.0, 0.05), 4)
            trees = pd.DataFrame(
                {
                    "fia_species_code": spcd,
                    "dbh": dia_in * conversion_factor("inch", "cm"),
                    "height": 20.0,
                    "crown_ratio": 0.5,
                }
            )
            ours = available_canopy_fuel(trees, equations="brown_1978")
            theirs = np.array(
                [
                    fc.available_canopy_fuel_lb(eq_id, float(d))
                    * conversion_factor("lb", "kg")
                    for d in dia_in
                ]
            )
            np.testing.assert_allclose(
                ours,
                theirs,
                rtol=1e-11,
                atol=1e-11,
                err_msg=f"{eq_id} available fuel drifted from the reference",
            )

    def test_brown_1978_and_nsvb_arms_disagree(self):
        """They are different biomass models, not two names for one.

        A regression that quietly routed brown_1978 back to NSVB would
        pass every other test in this class.
        """
        trees = pd.DataFrame(
            {
                "fia_species_code": [122, 202, 108],
                "dbh": [40.0, 40.0, 40.0],
                "height": [18.0, 18.0, 18.0],
                "crown_ratio": [0.5, 0.5, 0.5],
            }
        )
        assert not np.allclose(
            available_canopy_fuel(trees, equations="brown_1978"),
            available_canopy_fuel(trees, equations="nsvb"),
            rtol=0.01,
        )

    @pytest.mark.parametrize(
        "crown_class", ["D", "C", "I", "S", "O", "E", "SC", "N", "", "?"]
    )
    def test_crown_class_dispatch_matches_cc_adj(self, crown_class):
        """Every code, including the three CC_Adj folds and the fallthrough."""
        species = fuelcalc_species()
        spcd = species.index.to_numpy()
        ours = crown_class_factor(spcd, np.full(spcd.shape, crown_class))
        theirs = np.array(
            [
                fc.crown_class_factor(
                    species.loc[code, "CROWN_REDUC_CODE"], crown_class
                )
                for code in spcd
            ]
        )
        np.testing.assert_allclose(ours, theirs, atol=1e-12)

    def test_crown_class_folds_are_not_identity(self):
        """O, E and SC must land on C, D and I -- not on Other/none.

        Every crown-class row a species actually resolves to has
        Dominant equal to Codominant, so O and E cannot be told apart
        through a real species; what is observable is that all three
        folds differ from the Other/none column they would take if the
        remap were skipped.
        """
        species = fuelcalc_species()
        spcd = int(species.index[species["CROWN_REDUC_CODE"] == "WF"][0])
        other = crown_class_factor(np.array([spcd]), np.array(["N"]))[0]
        for alias, target in {"O": "C", "E": "D", "SC": "I"}.items():
            folded = crown_class_factor(np.array([spcd]), np.array([alias]))[0]
            direct = crown_class_factor(np.array([spcd]), np.array([target]))[0]
            assert folded == direct, alias
            assert folded != other, alias

    def test_omitting_crown_class_takes_the_other_column(self):
        species = fuelcalc_species()
        spcd = species.index.to_numpy()
        np.testing.assert_allclose(
            crown_class_factor(spcd),
            crown_class_factor(spcd, np.full(spcd.shape, "N")),
            atol=1e-12,
        )

    def test_adjusted_available_fuel_matches_the_reference(self):
        """The factor scales available fuel, as PTL_SetBioMass does.

        FuelCalc multiplies each crown component by the factor before
        summing; available fuel is linear in the components, so scaling
        the sum is the same arithmetic.
        """
        species = fuelcalc_species()
        for eq_id in sorted(brown.CROWN_WEIGHT_EQUATIONS):
            spcd = int(species.index[species["TOTAL"] == eq_id][0])
            reduc = species.loc[spcd, "CROWN_REDUC_CODE"]
            for crown_class in ("D", "C", "I", "S", "N"):
                dia_in = np.array([3.0, 11.0, 24.0, 38.0])
                trees = pd.DataFrame(
                    {
                        "fia_species_code": spcd,
                        "dbh": dia_in * conversion_factor("inch", "cm"),
                        "height": 20.0,
                        "crown_ratio": 0.5,
                        "cc": crown_class,
                    }
                )
                ours = available_canopy_fuel(
                    trees,
                    equations="brown_1978",
                    crown_class_adjustment="fuelcalc_table",
                    crown_class_column="cc",
                )
                theirs = np.array(
                    [
                        fc.available_canopy_fuel_lb(eq_id, float(d))
                        * fc.crown_class_factor(reduc, crown_class)
                        * conversion_factor("lb", "kg")
                        for d in dia_in
                    ]
                )
                np.testing.assert_allclose(
                    ours,
                    theirs,
                    rtol=1e-11,
                    atol=1e-11,
                    err_msg=f"{eq_id}/{crown_class}",
                )

    def test_adjustment_defaults_to_inert(self):
        trees = pd.DataFrame(
            {
                "fia_species_code": [122, 202],
                "dbh": [30.0, 30.0],
                "height": [18.0, 18.0],
                "crown_ratio": [0.5, 0.5],
            }
        )
        np.testing.assert_array_equal(
            available_canopy_fuel(trees),
            available_canopy_fuel(trees, crown_class_adjustment="none"),
        )
        assert not np.allclose(
            available_canopy_fuel(trees),
            available_canopy_fuel(
                trees.assign(cc="D"),
                crown_class_adjustment="fuelcalc_table",
                crown_class_column="cc",
            ),
        )

    @staticmethod
    def _two_trees():
        return pd.DataFrame(
            {
                "fia_species_code": [122, 202],
                "dbh": [30.0, 30.0],
                "height": [18.0, 18.0],
                "crown_ratio": [0.5, 0.5],
                "cc": ["D", "S"],
            }
        )

    def test_unknown_adjustment_raises(self):
        with pytest.raises(ValueError, match="crown_class_adjustment"):
            available_canopy_fuel(self._two_trees(), crown_class_adjustment="fuelcalc")

    def test_none_means_no_adjustment(self):
        """``None`` is the natural Python spelling and must not raise.

        ``crown_class_column`` beside it takes a real ``None``, so
        accepting only the string here is a trap.
        """
        trees = self._two_trees()
        np.testing.assert_array_equal(
            available_canopy_fuel(trees, crown_class_adjustment=None),
            available_canopy_fuel(trees, crown_class_adjustment="none"),
        )

    def test_a_column_that_would_be_ignored_raises(self):
        """Naming the column says the inventory has crown position.

        Applying no adjustment to it would throw away the only input
        that makes the adjustment more than a constant.
        """
        with pytest.raises(ValueError, match="would be ignored"):
            available_canopy_fuel(self._two_trees(), crown_class_column="cc")

    def test_the_adjustment_requires_a_column(self):
        """The two arguments are one decision.

        Allowing the adjustment without the column would apply the
        Other/none factor to everything, which halves 50 of the 54
        species — a silent blanket scaling wearing the name of a
        crown-class adjustment.
        """
        with pytest.raises(ValueError, match="needs crown_class_column"):
            available_canopy_fuel(
                self._two_trees(), crown_class_adjustment="fuelcalc_table"
            )

    def test_uniform_other_none_is_reachable_deliberately(self):
        """A column of "N" is FuelCalc's blank crown class field.

        The behaviour the bare flag used to give is still available; it
        just has to be asked for where a reader can see it.
        """
        trees = self._two_trees()
        got = available_canopy_fuel(
            trees.assign(cc="N"),
            crown_class_adjustment="fuelcalc_table",
            crown_class_column="cc",
        )
        expected = available_canopy_fuel(trees) * crown_class_factor(
            trees["fia_species_code"].to_numpy()
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_a_missing_column_names_the_parameter(self):
        with pytest.raises(ValueError, match="crown_class_column"):
            available_canopy_fuel(
                self._two_trees(),
                crown_class_adjustment="fuelcalc_table",
                crown_class_column="not_a_column",
            )

    def test_the_fallback_is_nearly_a_constant(self):
        """Without crown position the adjustment loses its content.

        50 of the 54 species take 0.5, so turning the table on with no
        column is close to halving every tree. Pinned because it is the
        difference between a species-and-position adjustment and a
        blanket scale factor, and it is invisible from the call site.
        """
        spcd = fuelcalc_species().index.to_numpy()
        factors = crown_class_factor(spcd)
        values, counts = np.unique(factors, return_counts=True)
        assert dict(zip(np.round(values, 2), counts)) == {0.5: 50, 0.75: 1, 1.0: 3}

    def test_no_unaccounted_equation_ids(self):
        """Every Id we define is either in the source or explained here."""
        ours = set(brown.P1_EQUATIONS) & set(brown.P2_EQUATIONS)
        vestigial = {
            # sr_EFD has no AL entry at all; Subalpine Larch (LALY) uses
            # the WL Ids for all six components. Unreachable, so it is
            # harmless -- unlike QA, which resolved SPCD 746 to the same
            # unsourced borrow and has been dropped.
            "AL",
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

    The running mean is over a slab of fixed depth, so its denominator
    is that depth wherever the slab sits. See the note in
    ``fuelcalc_reference.bulk_density`` for why that is the published
    quantity and what the C does instead.
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

    def test_cbd_matches(self):
        """The maximum running mean agrees exactly, at any crown base.

        Every window the reference evaluates that we do not is a partial
        one at an end of the profile, whose sum is contained in some
        full window at the same denominator, so it can never carry the
        maximum.
        """
        for profile, _ in self._profiles():
            reference = fc.bulk_density(list(profile), LAYER_FT_M)
            ours = cbd_running_mean(
                profile[:, None, None],
                layer_depth=LAYER_FT_M,
                window=self.WINDOW,
            )[0, 0]
            assert ours == pytest.approx(reference.cbd, abs=1e-12)

    def test_cbd_is_invariant_to_how_high_the_canopy_sits(self):
        """A slab of fuel has one bulk density wherever it sits.

        The running mean is over a fixed depth, so translating a canopy
        vertically cannot change its CBD. FuelCalc's C shrinks the
        denominator at the ground and reports 1.0 for the profile below
        resting on layer 0 against 0.6 higher up; the fixed-depth mean
        that Reinhardt et al. (2006) define gives 0.6 for both.
        """
        slab = np.array([1.0, 1.0, 1.0])
        densities = set()
        for offset in range(0, 6):
            profile = np.zeros(12)
            profile[offset : offset + len(slab)] = slab
            densities.add(
                round(
                    float(
                        cbd_running_mean(
                            profile[:, None, None],
                            layer_depth=LAYER_FT_M,
                            window=self.WINDOW,
                        )[0, 0]
                    ),
                    12,
                )
            )
        assert densities == {0.6}

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
        columns = ["DOMINANT", "CODOMINANT", "INTERMEDIATE", "SUPPRESSED", "OTHER_NONE"]
        for code, expected in fc.CROWN_CLASS_FACTORS.items():
            assert tuple(ours.loc[code, columns]) == expected

    def test_grand_fir_crown_class_row_is_unreachable(self):
        """sr_CCT's GF row is commented out in NC_CC.C ("doesn't seem to
        get used"), correctly: every GF-biomass species routes to WF for
        crown reduction. We carry the row; nothing may reference it."""
        assert "GF" in fuelcalc_crown_class_factors().index
        assert "GF" not in set(fuelcalc_species()["CROWN_REDUC_CODE"])


class TestCanopyCoverParity:
    """Crown width and the overlap correction against NC_CA.C.

    Crookston & Stage (1999), RMRS-GTR-24, reached through FFE-FVS.
    ``crown_overlap`` is the estimator FuelCalc uses; it is the one to
    select for reproducing FuelCalc or LANDFIRE, and the weaker of the
    two cover methods for inventories that carry stem positions.
    """

    TRANSFORM = (30.0, 0.0, 0.0, 0.0, -30.0, 0.0)
    CELL_SQ_M = 30.0 * 30.0
    FT2_PER_M2 = 1.0 / (0.3048**2)

    def test_coefficient_table_matches_source(self):
        ours = fuelcalc_crown_width()
        assert set(ours.index) == set(fc.CROWN_WIDTH_COEFFICIENTS)
        for eq, (a, b, ratio) in fc.CROWN_WIDTH_COEFFICIENTS.items():
            row = ours.loc[eq]
            assert (row.A, row.B, row.RATIO) == (a, b, ratio), eq

    def test_every_species_resolves_to_a_row(self):
        species = fuelcalc_species()
        assert set(species["COVER_EQ"]) <= set(fuelcalc_crown_width().index)

    @pytest.mark.parametrize("cover_eq", sorted(fc.CROWN_WIDTH_COEFFICIENTS))
    def test_crown_width_matches_ca_crnarea(self, cover_eq):
        """Both branches, either side of the 4.5 ft split."""
        dia = np.round(np.arange(0.2, 40.0, 0.2), 4)
        for height in (2.0, 4.5, 4.6, 60.0):
            ours = fvs.crown_width(
                np.full(dia.shape, cover_eq), dia, np.full(dia.shape, height)
            )
            theirs = np.array(
                [
                    2.0
                    * math.sqrt(fc.ca_crown_area(cover_eq, float(d), height) / math.pi)
                    for d in dia
                ]
            )
            np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)

    def test_cover_matches_ca_overlap(self):
        """One cell, crowns wholly inside it, against the C estimator."""
        species = fuelcalc_species()
        rng = np.random.default_rng(20260818)
        for _ in range(25):
            n = int(rng.integers(1, 30))
            spcd = rng.choice(species.index.to_numpy(), n)
            trees = pd.DataFrame(
                {
                    "x": rng.uniform(12.0, 18.0, n),
                    "y": -rng.uniform(12.0, 18.0, n),
                    "fia_species_code": spcd,
                    "dbh": rng.uniform(2.0, 25.0, n),
                    "height": rng.uniform(6.0, 30.0, n),
                    "crown_ratio": rng.uniform(0.3, 0.9, n),
                }
            )
            ours = canopy_cover(
                trees,
                self.TRANSFORM,
                (1, 1),
                crown_radius_equations="fuelcalc",
                method="crown_overlap",
            )[0, 0]
            total_sqft = sum(
                fc.ca_crown_area(
                    int(species.loc[int(c), "COVER_EQ"]),
                    float(d) * conversion_factor("cm", "inch"),
                    float(h) * conversion_factor("m", "foot"),
                )
                for c, d, h in zip(spcd, trees["dbh"], trees["height"])
            )
            theirs = fc.ca_overlap(total_sqft, self.CELL_SQ_M * self.FT2_PER_M2)
            assert ours == pytest.approx(theirs, rel=1e-9)

    def test_crown_overlap_cannot_see_arrangement(self):
        """Its defining property, and its limitation.

        The estimator reads only total crown area, so translating stems
        within a cell cannot move it. The union sees the difference,
        which is the whole reason both methods exist.
        """
        rng = np.random.default_rng(11)
        n = 25
        layouts = {
            "random": (rng.uniform(4, 26, n), -rng.uniform(4, 26, n)),
            "grid": (
                np.tile(np.linspace(4, 26, 5), 5),
                -np.repeat(np.linspace(4, 26, 5), 5),
            ),
            "clumped": (rng.normal(15, 1.5, n), -rng.normal(15, 1.5, n)),
        }
        overlap, union = [], []
        for x, y in layouts.values():
            trees = pd.DataFrame(
                {
                    "x": np.clip(x, 3, 27),
                    "y": -np.clip(-y, 3, 27),
                    "fia_species_code": 122,
                    "dbh": np.full(n, 25.0),
                    "height": np.full(n, 15.0),
                    "crown_ratio": np.full(n, 0.5),
                }
            )
            overlap.append(
                canopy_cover(trees, self.TRANSFORM, (1, 1), method="crown_overlap")[
                    0, 0
                ]
            )
            union.append(canopy_cover(trees, self.TRANSFORM, (1, 1))[0, 0])
        assert max(overlap) - min(overlap) < 1e-9
        assert max(union) - min(union) > 20.0

    def test_methods_agree_when_nothing_can_overlap(self):
        """One crown in a cell: no overlap to resolve, so both are exact.

        1 - exp(-p) != p, so they agree only in the limit; a crown
        covering under 2% of the cell puts the two within a tenth of a
        point.
        """
        trees = pd.DataFrame(
            {
                "x": [15.0],
                "y": [-15.0],
                "fia_species_code": [122],
                "dbh": [8.0],
                "height": [7.0],
                "crown_ratio": [0.5],
            }
        )
        union = canopy_cover(trees, self.TRANSFORM, (1, 1))[0, 0]
        overlap = canopy_cover(trees, self.TRANSFORM, (1, 1), method="crown_overlap")[
            0, 0
        ]
        assert union < 2.0
        assert overlap == pytest.approx(union, abs=0.1)

    def test_unknown_method_raises(self):
        trees = pd.DataFrame(
            {
                "x": [15.0],
                "y": [-15.0],
                "fia_species_code": [122],
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.5],
            }
        )
        with pytest.raises(ValueError, match="canopy cover method"):
            canopy_cover(trees, self.TRANSFORM, (1, 1), method="crookston")

    def test_cover_counts_species_excluded_from_bulk_density(self):
        """Cover is not gated by the species inclusion flag.

        ``PTL_CanCov`` (``NC_PTL2.C:44``) loops over every live record,
        skipping only the dead. The inclusion flag is read in exactly
        one place in the whole source, ``NC_PTL.C:731``, inside the loop
        that builds the bulk-density profile. So a hardwood contributes
        to cover while contributing nothing to CBD.

        Pinned ahead of the ``fuelcalc_default`` species exclusion
        slice, which must gate cbd/cbh/chm/cfl and leave cc alone.
        """
        excluded = fuelcalc_species()
        excluded = excluded[excluded["INCL_CBD"] == "No"]
        assert not excluded.empty
        spcd = int(excluded.index[0])

        trees = pd.DataFrame(
            {
                "x": [15.0],
                "y": [-15.0],
                "fia_species_code": [spcd],
                "dbh": [30.0],
                "height": [18.0],
                "crown_ratio": [0.6],
            }
        )
        for method in ("crown_union", "crown_overlap"):
            assert (
                canopy_cover(trees, self.TRANSFORM, (1, 1), method=method)[0, 0] > 0.0
            ), method
