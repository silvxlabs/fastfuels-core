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

from fastfuels_core.allometry import nsvb
from fastfuels_core.canopy_fuel.available_fuel import available_canopy_fuel
from tests.canopy_fuel.builders import random_stand


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
