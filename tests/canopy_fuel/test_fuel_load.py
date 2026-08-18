"""Tests for :mod:`fastfuels_core.canopy_fuel.fuel_load`."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.fuel_load import canopy_fuel_load
from fastfuels_core.canopy_fuel.profile import vertical_profile
from tests.canopy_fuel.builders import (
    CELL_AREA,
    SHAPE,
    TRANSFORM,
    column_profile,
    stand_on_lattice,
)


def test_integrates_the_profile_over_depth():
    """[0, 2, 4] at 0.5 m layers is 3 kg/m2."""
    np.testing.assert_allclose(
        canopy_fuel_load(column_profile([0, 2, 4]), layer_depth=0.5), [[3.0]]
    )


def test_recovers_the_mass_that_went_into_the_profile():
    trees = stand_on_lattice(50, seed=11)
    fuel = np.full(len(trees), 3.0)
    profile = vertical_profile(
        trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="stem"
    )
    cfl = canopy_fuel_load(profile)
    np.testing.assert_allclose((cfl * CELL_AREA).sum(), fuel.sum(), rtol=1e-3)


def test_an_empty_profile_carries_no_load():
    assert canopy_fuel_load(np.zeros((5, 4, 5))).sum() == 0.0
