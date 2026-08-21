"""NSVB -> Jenkins fallback for the woodland species group.

NSVB has no crown model for the woodland group (Jenkins group 10:
junipers, pinyon, oak, mesquite). ``available_canopy_fuel`` with the
``nsvb`` equations prices those trees with Jenkins instead of raising, and
prices every other tree with NSVB — in one vectorized call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.allometry import jenkins, nsvb
from fastfuels_core.canopy_fuel.available_fuel import available_canopy_fuel

# Utah juniper / singleleaf pinyon are Jenkins group 10; ponderosa and
# Douglas-fir are priced directly by NSVB.
WOODLAND = [65, 106]
NSVB_DIRECT = [122, 202]


def _fuel(trees, **kwargs):
    return available_canopy_fuel(
        trees,
        equations="nsvb",
        branchwood_size_partition="none",
        crown_class_adjustment="none",
        foliage_fraction=1.0,
        branchwood_fraction=0.5,
        **kwargs,
    )


def test_woodland_species_priced_without_error():
    trees = pd.DataFrame(
        {
            "dbh": [20.0, 15.0],
            "height": [6.0, 5.0],
            "crown_ratio": [0.6, 0.5],
            "fia_species_code": WOODLAND,
        }
    )
    fuel = _fuel(trees)
    assert np.all(np.isfinite(fuel))
    assert np.all(fuel > 0)


def test_woodland_fuel_matches_jenkins():
    trees = pd.DataFrame(
        {
            "dbh": [22.0],
            "height": [7.0],
            "crown_ratio": [0.55],
            "fia_species_code": [65],
        }
    )
    dbh = np.array([22.0])
    expected = (
        jenkins.foliage_biomass(np.array([65]), dbh)[0]
        + 0.5 * jenkins.branch_biomass(np.array([65]), dbh)[0]
    )
    assert _fuel(trees)[0] == expected


def test_mixed_stand_routes_each_species_correctly():
    # A conifer and a woodland tree in one call: the conifer must match
    # NSVB and the woodland tree must match Jenkins.
    trees = pd.DataFrame(
        {
            "dbh": [30.0, 20.0],
            "height": [18.0, 6.0],
            "crown_ratio": [0.5, 0.6],
            "fia_species_code": [122, 65],
        }
    )
    fuel = _fuel(trees)

    pp = np.array([122])
    conifer = (
        nsvb.foliage_biomass(pp, np.array([30.0]), np.array([18.0]))[0]
        + 0.5 * nsvb.branch_biomass(pp, np.array([30.0]), np.array([18.0]))[0]
    )
    ju = np.array([65])
    woodland = (
        jenkins.foliage_biomass(ju, np.array([20.0]))[0]
        + 0.5 * jenkins.branch_biomass(ju, np.array([20.0]))[0]
    )
    np.testing.assert_allclose(fuel, [conifer, woodland])


def test_all_conifer_stand_untouched_by_fallback():
    # With no woodland species, every tree is priced by NSVB directly.
    trees = pd.DataFrame(
        {
            "dbh": [30.0, 25.0],
            "height": [18.0, 15.0],
            "crown_ratio": [0.5, 0.5],
            "fia_species_code": NSVB_DIRECT,
        }
    )
    fuel = _fuel(trees)
    spcd = np.array(NSVB_DIRECT)
    dbh = trees["dbh"].to_numpy()
    ht = trees["height"].to_numpy()
    expected = nsvb.foliage_biomass(spcd, dbh, ht) + 0.5 * nsvb.branch_biomass(
        spcd, dbh, ht
    )
    np.testing.assert_allclose(fuel, expected)
