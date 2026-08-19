"""Tests for :mod:`fastfuels_core.canopy_fuel.available_fuel`.

Available canopy fuel is ``foliage_fraction * foliage +
branchwood_fraction * fine branchwood``. These tests pin the
composition, the fractions that weight it, and which species the
allometry covers. The equations behind the two weights are pinned
against their published sources in :mod:`tests.canopy_fuel.
test_brown_table1`, :mod:`~tests.canopy_fuel.test_brown_table16` and
:mod:`~tests.canopy_fuel.test_fuelcalc_parity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry import brown, nsvb
from fastfuels_core.canopy_fuel.available_fuel import (
    available_canopy_fuel,
    crown_class_factor,
    small_tree_crown_components,
)
from fastfuels_core.canopy_fuel.ref_data import (
    fuelcalc_small_tree_biomass,
    fuelcalc_species,
)
from tests.canopy_fuel.builders import random_stand


def two_trees(**extra):
    """One ponderosa and one Douglas-fir, both 30 cm at 18 m."""
    return pd.DataFrame(
        {
            "fia_species_code": [122, 202],
            "dbh": [30.0, 30.0],
            "height": [18.0, 18.0],
            "crown_ratio": [0.5, 0.5],
            **{k: [v, v] for k, v in extra.items()},
        }
    )


def one_tree(species_code, dbh_cm, height_m, **extra):
    """A single tree with the columns the allometry path reads."""
    return pd.DataFrame(
        {
            "fia_species_code": [species_code],
            "dbh": [dbh_cm],
            "height": [height_m],
            "crown_ratio": [0.5],
            **{k: [v] for k, v in extra.items()},
        }
    )


class TestComposition:
    def test_equals_foliage_plus_half_the_fine_branchwood(self):
        """Hand-composed from the NSVB weights and Brown's fine share.

        202 Douglas-fir at dbh 25.4 cm (10 in), height 15.24 m. Brown
        Table 16 gives DF's P1 and P2 at 10 in, and the fine share of
        branchwood is ``(P2 - P1) / (1 - P1)``.
        """
        trees = one_tree(202, 25.4, 15.24)
        p1 = 0.484 * np.exp(-0.0210 * 10.0)
        p2 = 0.729 * np.exp(-0.0233 * 10.0)
        fine_share = (p2 - p1) / (1.0 - p1)
        expected = nsvb.foliage_biomass(
            [202], [25.4], [15.24]
        ) + 0.5 * fine_share * nsvb.branch_biomass([202], [25.4], [15.24])
        np.testing.assert_allclose(available_canopy_fuel(trees), expected, rtol=1e-10)

    def test_the_two_fractions_decompose_the_default(self):
        """Default fuel is the foliage term plus half the branchwood term."""
        trees = random_stand(20)
        foliage_only = available_canopy_fuel(
            trees, foliage_fraction=1.0, branchwood_fraction=0.0
        )
        fine_only = available_canopy_fuel(
            trees, foliage_fraction=0.0, branchwood_fraction=1.0
        )
        np.testing.assert_allclose(
            available_canopy_fuel(trees),
            foliage_only + 0.5 * fine_only,
            rtol=1e-12,
        )

    def test_every_tree_carries_foliage(self):
        trees = random_stand(20)
        assert (
            available_canopy_fuel(trees, foliage_fraction=1.0, branchwood_fraction=0.0)
            > 0
        ).all()

    def test_the_fine_branchwood_term_is_never_negative(self):
        trees = random_stand(20)
        assert (
            available_canopy_fuel(trees, foliage_fraction=0.0, branchwood_fraction=1.0)
            >= 0
        ).all()

    def test_one_fuel_value_per_tree(self):
        assert available_canopy_fuel(random_stand(7)).shape == (7,)

    def test_an_empty_stand_gives_an_empty_result(self):
        """Reachable through exclude_hardwoods on an all-hardwood stand."""
        fuel = available_canopy_fuel(random_stand(0))
        assert fuel.shape == (0,)


class TestFuelColumn:
    def test_a_named_column_is_returned_verbatim(self):
        trees = random_stand(5)
        trees["acf_kg"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_equal(
            available_canopy_fuel(trees, fuel_column="acf_kg"),
            trees["acf_kg"].to_numpy(),
        )

    def test_it_bypasses_allometry_entirely(self):
        """Species the equations do not cover still pass through."""
        trees = one_tree(999, 20.0, 15.0, acf_kg=7.0)
        np.testing.assert_array_equal(
            available_canopy_fuel(trees, fuel_column="acf_kg"), [7.0]
        )


class TestSpeciesCoverage:
    """Species the equations do not reach must say so, not guess."""

    def test_a_species_outside_the_table_raises(self):
        trees = random_stand(3)
        trees.loc[1, "fia_species_code"] = 999
        with pytest.raises(ValueError, match="999"):
            available_canopy_fuel(trees)

    def test_pinyon_juniper_raises(self):
        """106 two-needle pinyon resolves to PY, which has no equations."""
        with pytest.raises(ValueError, match="PY"):
            available_canopy_fuel(one_tree(106, 20.0, 8.0))

    def test_an_eastern_newer_species_raises(self):
        """833 northern red oak is in the table; its Ids have no equations."""
        with pytest.raises(ValueError, match="RO"):
            available_canopy_fuel(one_tree(833, 30.0, 20.0))

    def test_quaking_aspen_raises(self):
        """746 used to borrow whitebark pine P1 and western larch P2.

        That pairing is sanctioned by neither Brown nor Snell & Little,
        so the Id was dropped; aspen's real source is Loomis &
        Roussopoulos 1978 (NC-156). Raising beats an unsourced number.
        """
        with pytest.raises(ValueError, match="QA"):
            available_canopy_fuel(one_tree(746, 20.0, 15.0))


def test_unknown_equations_raises():
    with pytest.raises(ValueError, match="bogus"):
        available_canopy_fuel(random_stand(3), equations="bogus")


class TestEquationsArm:
    def test_brown_1978_and_nsvb_do_not_agree(self):
        """They are different biomass models, not two names for one.

        A regression that quietly routed brown_1978 back to NSVB would
        pass every parity test: both arms would match the same oracle
        on the proportions and differ only in the weight they scale.
        """
        trees = pd.concat(
            [two_trees(), one_tree(108, 40.0, 18.0)], ignore_index=True
        ).assign(dbh=40.0)
        assert not np.allclose(
            available_canopy_fuel(trees, equations="brown_1978"),
            available_canopy_fuel(trees, equations="nsvb"),
            rtol=0.01,
        )


class TestCrownClassFactor:
    """The multiplier itself. Its values are pinned against FuelCalc in
    :mod:`tests.canopy_fuel.test_fuelcalc_parity`; what is here is how
    codes map onto columns."""

    def test_the_three_aliases_fold_onto_real_classes(self):
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

    def test_omitting_the_class_takes_the_other_none_column(self):
        spcd = fuelcalc_species().index.to_numpy()
        np.testing.assert_allclose(
            crown_class_factor(spcd),
            crown_class_factor(spcd, np.full(spcd.shape, "N")),
            atol=1e-12,
        )

    def test_the_fallback_is_nearly_a_constant(self):
        """Without crown position the adjustment loses its content.

        50 of the 54 species take 0.5, so turning the table on with no
        column is close to halving every tree. Pinned because it is the
        difference between a species-and-position adjustment and a
        blanket scale factor, and it is invisible from the call site.
        """
        values, counts = np.unique(
            crown_class_factor(fuelcalc_species().index.to_numpy()),
            return_counts=True,
        )
        assert dict(zip(np.round(values, 2), counts)) == {0.5: 50, 0.75: 1, 1.0: 3}


class TestCrownClassArguments:
    """``crown_class_adjustment`` and ``crown_class_column`` are one
    decision: an adjustment needs the data it adjusts by, and the data
    is pointless without the adjustment. Both half-specified forms fail
    silently if allowed through, so both raise."""

    def test_no_adjustment_is_the_default(self):
        trees = two_trees()
        np.testing.assert_array_equal(
            available_canopy_fuel(trees),
            available_canopy_fuel(trees, crown_class_adjustment="none"),
        )

    def test_the_adjustment_changes_the_answer(self):
        """Otherwise the tests above would pass on a no-op."""
        trees = two_trees(cc="D")
        assert not np.allclose(
            available_canopy_fuel(trees),
            available_canopy_fuel(
                trees,
                crown_class_adjustment="reinhardt_2006",
                crown_class_column="cc",
            ),
        )

    def test_none_means_no_adjustment(self):
        """``None`` is the natural Python spelling and must not raise.

        ``crown_class_column`` beside it takes a real ``None``, so
        accepting only the string here is a trap.
        """
        trees = two_trees(cc="D")
        np.testing.assert_array_equal(
            available_canopy_fuel(trees, crown_class_adjustment=None),
            available_canopy_fuel(trees, crown_class_adjustment="none"),
        )

    def test_an_unknown_adjustment_raises(self):
        """The arm is named for the paper, not for FuelCalc.

        The multipliers are Reinhardt, Scott, Gray & Keane (2006), the
        same paper the vertical distribution cubics come from, so the
        value matches ``vertical_distribution="reinhardt_2006"``.
        FuelCalc is one program that applies them.
        """
        with pytest.raises(ValueError, match="crown_class_adjustment"):
            available_canopy_fuel(
                two_trees(cc="D"), crown_class_adjustment="fuelcalc_table"
            )

    def test_a_column_that_would_be_ignored_raises(self):
        """Naming the column says the inventory has crown position.

        Applying no adjustment to it would throw away the only input
        that makes the adjustment more than a constant.
        """
        with pytest.raises(ValueError, match="would be ignored"):
            available_canopy_fuel(two_trees(cc="D"), crown_class_column="cc")

    def test_the_adjustment_without_a_column_raises(self):
        """Allowing it would apply the Other/none factor to everything,
        which halves 50 of the 54 species — a silent blanket scaling
        wearing the name of a crown-class adjustment."""
        with pytest.raises(ValueError, match="needs crown_class_column"):
            available_canopy_fuel(
                two_trees(cc="D"), crown_class_adjustment="reinhardt_2006"
            )

    def test_a_column_that_is_not_in_the_frame_raises(self):
        with pytest.raises(ValueError, match="crown_class_column"):
            available_canopy_fuel(
                two_trees(cc="D"),
                crown_class_adjustment="reinhardt_2006",
                crown_class_column="not_a_column",
            )

    def test_the_uniform_fallback_is_still_reachable_deliberately(self):
        """A column of "N" is FuelCalc's blank crown class field.

        The behaviour the bare flag used to give is still available; it
        just has to be asked for where a reader can see it.
        """
        trees = two_trees(cc="N")
        got = available_canopy_fuel(
            trees,
            crown_class_adjustment="reinhardt_2006",
            crown_class_column="cc",
        )
        expected = available_canopy_fuel(trees) * crown_class_factor(
            trees["fia_species_code"].to_numpy()
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12)


IN_TO_CM = 2.54
FT_TO_M = 0.3048
LB_TO_KG = 0.45359237


def sapling(dia_in, height_ft, spcd=202):
    """One tree given the way small trees are measured: inches and feet."""
    return pd.DataFrame(
        {
            "fia_species_code": [spcd],
            "dbh": [dia_in * IN_TO_CM],
            "height": [height_ft * FT_TO_M],
            "crown_ratio": [0.9],
        }
    )


class TestSmallTreeComponents:
    """Under an inch the crown equations give way to a lookup.

    Brown fitted Table 1 to trees over one inch, and several of its
    forms are additive in diameter squared, so below that the intercept
    dominates and crown weight barely varies with diameter. The table
    is read by equation code and by height in one-foot classes.
    """

    def test_a_sapling_takes_the_table_not_the_equations(self):
        """Douglas-fir at 0.5 in and 6 ft: the DF row, class 6.

        The table gives 1.032 lb of foliage and 0.466 lb of 1-hour
        branchwood there, so available fuel is 1.032 + 0.466 / 2.
        """
        got = available_canopy_fuel(sapling(0.5, 6.0), equations="brown_1978")
        expected = (1.032 + 0.466 / 2) * LB_TO_KG
        np.testing.assert_allclose(got, [expected], rtol=1e-9)

    def test_it_does_not_apply_to_the_nsvb_arm(self):
        """NSVB is fitted nationally and is not out of range here."""
        trees = sapling(0.5, 6.0)
        table = available_canopy_fuel(trees, equations="brown_1978")
        assert available_canopy_fuel(trees, equations="nsvb") != pytest.approx(table)

    @pytest.mark.parametrize("dia_in", [0.1, 0.5, 1.0])
    def test_it_covers_diameters_up_to_and_including_an_inch(self, dia_in):
        """The same height gives the same weight at any sapling diameter.

        The table is keyed by height alone, so a tree's diameter selects
        which side of the cutoff it falls on and nothing more.
        """
        got = available_canopy_fuel(sapling(dia_in, 6.0), equations="brown_1978")
        one_inch = available_canopy_fuel(sapling(1.0, 6.0), equations="brown_1978")
        np.testing.assert_allclose(got, one_inch, rtol=1e-12)

    def test_just_over_an_inch_goes_back_to_the_equations(self):
        table = available_canopy_fuel(sapling(1.0, 6.0), equations="brown_1978")
        equations = available_canopy_fuel(sapling(1.01, 6.0), equations="brown_1978")
        assert equations != pytest.approx(table)

    def test_a_stand_can_straddle_the_cutoff(self):
        """Both arms in one call, each tree taking its own."""
        trees = pd.concat([sapling(0.5, 6.0), sapling(4.0, 20.0)], ignore_index=True)
        got = available_canopy_fuel(trees, equations="brown_1978")
        np.testing.assert_allclose(
            got[0],
            available_canopy_fuel(sapling(0.5, 6.0), equations="brown_1978")[0],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            got[1],
            available_canopy_fuel(sapling(4.0, 20.0), equations="brown_1978")[0],
            rtol=1e-12,
        )


class TestSmallTreeHeightClasses:
    """Height picks the row: ``h <= 1`` is class 1, over 9 ft is class 10."""

    def sapling_fuel(self, height_ft):
        return float(
            available_canopy_fuel(sapling(0.5, height_ft), equations="brown_1978")[0]
        )

    @pytest.mark.parametrize("height_ft", [1.0, 2.0, 3.0, 6.0, 9.0])
    def test_a_whole_foot_height_stays_in_its_own_class(self, height_ft):
        """The class boundary is inclusive, and must survive the units.

        Heights arrive in metres and are converted back to feet, so a
        tree measured at a whole number of feet lands a few ulp either
        side of it. Without rounding, half of them take the class above
        -- and saplings are measured to the foot.
        """
        assert self.sapling_fuel(height_ft) == pytest.approx(
            self.sapling_fuel(height_ft - 0.01)
        )
        assert self.sapling_fuel(height_ft) != pytest.approx(
            self.sapling_fuel(height_ft + 0.01)
        )

    def test_it_rises_with_height(self):
        heights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0]
        fuel = [self.sapling_fuel(h) for h in heights]
        assert fuel == sorted(fuel)
        assert len(set(fuel)) == len(fuel)

    def test_everything_over_nine_feet_shares_the_top_class(self):
        assert self.sapling_fuel(9.5) == pytest.approx(self.sapling_fuel(40.0))

    def test_a_seedling_under_a_foot_takes_the_bottom_class(self):
        assert self.sapling_fuel(0.5) == pytest.approx(self.sapling_fuel(1.0))


class TestSmallTreeSpeciesCoverage:
    """The table carries fewer codes than the equations do."""

    def test_a_species_the_table_covers_uses_its_own_row(self):
        """Subalpine fir differs from Douglas-fir at the same size."""
        subalpine_fir, douglas_fir = 19, 202
        assert available_canopy_fuel(
            sapling(0.5, 3.0, spcd=subalpine_fir), equations="brown_1978"
        ) != pytest.approx(
            available_canopy_fuel(
                sapling(0.5, 3.0, spcd=douglas_fir), equations="brown_1978"
            )
        )

    def test_the_table_covers_every_species_the_equations_do(self):
        """So the arm never has to fall back for a species it accepts.

        ``brown_1978`` raises for anything outside Brown's eleven
        conifers, and the small-tree table carries exactly those eleven.
        A new crown-weight equation without a matching table row would
        silently start returning Douglas-fir weights; this catches it.
        """
        assert set(brown.CROWN_WEIGHT_EQUATIONS) == set(
            fuelcalc_small_tree_biomass().index.get_level_values("CODE")
        )

    def test_an_uncovered_code_falls_back_to_douglas_fir(self):
        """Reachable through the stage function, which takes any code."""
        uncovered = small_tree_crown_components(
            np.array(["ZZ"]), np.array([0.5]), np.array([6.0])
        )
        douglas_fir = small_tree_crown_components(
            np.array(["DF"]), np.array([0.5]), np.array([6.0])
        )
        np.testing.assert_allclose(uncovered, douglas_fir, rtol=1e-12)

    def test_a_tree_over_the_cutoff_takes_nothing_from_the_table(self):
        """The stage returns zeros there; the caller chooses which trees."""
        foliage, twig = small_tree_crown_components(
            np.array(["DF"]), np.array([1.01]), np.array([6.0])
        )
        assert foliage == 0.0 and twig == 0.0

    def test_the_crown_class_factor_still_scales_a_sapling(self):
        trees = sapling(0.5, 6.0).assign(cc="I")
        plain = available_canopy_fuel(trees, equations="brown_1978")
        adjusted = available_canopy_fuel(
            trees,
            equations="brown_1978",
            crown_class_adjustment="reinhardt_2006",
            crown_class_column="cc",
        )
        expected = plain * crown_class_factor(
            trees["fia_species_code"].to_numpy(), trees["cc"].to_numpy()
        )
        np.testing.assert_allclose(adjusted, expected, rtol=1e-12)
