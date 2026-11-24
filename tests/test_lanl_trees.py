"""Reproduce LANL Trees crown bulk density against its own output.

The fixture (tests/data/lanl_trees/) holds grids written by LANL Trees. Core
rebuilds each one with subgrid voxelization: a "paraboloid" crown,
LinearHeightQuadraticRadialDensity integrated over 10x10x10 subcells, the stem
offset within its cell, and z cells aligned to the ground.
"""

# Core imports
import math
from pathlib import Path

# Internal imports
from fastfuels_core.trees import Tree
from fastfuels_core.voxelization import (
    LinearHeightQuadraticRadialDensity,
    VoxelizedTree,
)
from fastfuels_core.voxelization import subgrid

# External imports
import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "data" / "lanl_trees" / "lanl_trees.npz"
DATA = np.load(FIXTURE)
CASES = [str(case) for case in DATA["case"]]
GRADIENT = LinearHeightQuadraticRadialDensity()

# LANL computes in float32 and core in float64. Densities agree to float32
# rounding, except that a subcell lying on the crown surface can fall inside
# under one and outside under the other, moving its voxel by one subcell's
# share: at most 1.5 * bulk density / 1000, since the gradient is at most 1.
FLOAT32_TOLERANCE = 1e-5


def _lanl_tree(height, crown_base_height, crown_diameter, max_diameter_height):
    return Tree(
        species_code=122,
        status_code=1,
        diameter=30.0,
        height=height,
        crown_base_height=crown_base_height,
        crown_profile_model_type="paraboloid",
        max_crown_radius=crown_diameter / 2.0,
        max_crown_diameter_height=max_diameter_height,
        crown_fuel_load=1.0,
    )


def _crown_grid(tree, x, y, hr, vr):
    """Weighted grid of one crown and the (k, j, i) index of its first voxel."""
    i, j = math.floor(x / hr), math.floor(y / hr)
    grid = subgrid.discretize_crown_profile(
        tree,
        hr,
        vr,
        n_subgrid=10,
        z_origin=0.0,
        stem_offset=(x - (i + 0.5) * hr, y - (j + 0.5) * hr),
        density_field=GRADIENT,
    )
    half = grid.shape[1] // 2
    return grid, (math.floor(tree.crown_base_height / vr), j - half, i - half)


def _rebuild(case):
    trees = DATA[f"{case}__trees"]
    hr, vr = DATA[f"{case}__resolution"]
    lanl = DATA[f"{case}__bulk_density"].astype(float)
    core = np.zeros_like(lanl)
    for x, y, height, cbh, crown_diameter, hd, bulk_density in trees:
        tree = _lanl_tree(height, cbh, crown_diameter, hd)
        grid, (k, j, i) = _crown_grid(tree, x, y, hr, vr)
        nz, n = grid.shape[0], grid.shape[1]
        core[k : k + nz, j : j + n, i : i + n] += bulk_density * grid
    return core, lanl, trees


@pytest.mark.parametrize("case", CASES)
def test_matches_lanl_trees(case):
    core, lanl, trees = _rebuild(case)
    diff = np.abs(core - lanl)
    one_subcell = 1.5 * trees[:, 6].max() / 1000.0

    np.testing.assert_array_equal(core > 0, lanl > 0)
    assert diff.sum() / lanl.sum() < FLOAT32_TOLERANCE
    assert diff.max() <= one_subcell + FLOAT32_TOLERANCE * lanl.max()


def test_bulk_density_mass_distributes_to_lanl_grid():
    # A crown bulk density gives the tree's mass; distributing that mass over
    # the weighted grid recovers bulk density times the grid.
    case = next(c for c in CASES if c.startswith("mid_peak"))
    (x, y, height, cbh, crown_diameter, hd, bulk_density) = DATA[f"{case}__trees"][0]
    hr, vr = DATA[f"{case}__resolution"]
    grid, _ = _crown_grid(_lanl_tree(height, cbh, crown_diameter, hd), x, y, hr, vr)
    mass = bulk_density * grid.sum() * hr * hr * vr
    tree = Tree(
        species_code=122,
        status_code=1,
        diameter=30.0,
        height=height,
        crown_base_height=cbh,
        crown_profile_model_type="paraboloid",
        max_crown_radius=crown_diameter / 2.0,
        max_crown_diameter_height=hd,
        crown_fuel_load=mass,
    )
    distributed = VoxelizedTree(tree, grid, hr, vr, z_origin=0.0).distribute_biomass()
    np.testing.assert_allclose(distributed, bulk_density * grid, rtol=1e-12)
