# Internal imports
from fastfuels_core.trees import Tree
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.voxelization import (
    voxelize_tree,
    VoxelizedTree,
    LinearHeightQuadraticRadialDensity,
)
from fastfuels_core.voxelization import subgrid
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile
from fastfuels_core.voxelization._coords import _get_vertical_tree_coords

# External imports
import numpy as np
import pytest


def _paraboloid_tree(height=31.5, cbh=3.12, crown_dia=9.34, mass=375.08):
    hd = float(
        BetaCrownProfile(
            species_code=122, crown_base_height=cbh, crown_length=height - cbh
        ).get_max_radius_height()
    )
    return Tree(
        species_code=122,
        status_code=1,
        diameter=72.6,
        height=height,
        crown_ratio=(height - cbh) / height,
        crown_profile_model_type="paraboloid",
        max_crown_radius=crown_dia / 2.0,
        max_crown_diameter_height=hd,
        crown_fuel_load=mass,
    )


class TestVerticalCoords:
    def test_none_is_crown_base_centered(self):
        # default: crown base at the center of the first cell
        z = _get_vertical_tree_coords(1.0, 10.0, 4.0, z_origin=None)
        assert z[0] == pytest.approx(4.0)  # first cell centered on the crown base
        assert np.allclose(np.diff(z), 1.0)

    def test_float_aligns_boundaries_to_reference(self):
        # z_origin=0: cell boundaries on integers, centers on half-integers
        z = _get_vertical_tree_coords(1.0, 10.4, 4.2, z_origin=0.0)
        assert np.allclose(z, [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

    def test_crown_base_falls_inside_first_cell_not_centered(self):
        z = _get_vertical_tree_coords(1.0, 10.4, 4.2, z_origin=0.0)
        cbh = 4.2
        assert z[0] - 0.5 <= cbh <= z[0] + 0.5  # inside first cell
        assert z[0] != pytest.approx(cbh)  # but not at its center

    def test_phase_is_shared_across_trees(self):
        # any z_origin=0 grid has centers at half-integers, regardless of cbh
        for cbh in (3.1, 4.0, 5.73, 8.19):
            z = _get_vertical_tree_coords(1.0, cbh + 5.0, cbh, z_origin=0.0)
            assert np.allclose(z % 1.0, 0.5)


class TestBackendsAcceptZOrigin:
    def test_marching_squares_grid_matches_coords(self):
        tree = _paraboloid_tree()
        grid = discretize_crown_profile(tree, 2.0, 1.0, z_origin=0.0)
        z = _get_vertical_tree_coords(1.0, tree.height, tree.crown_base_height, 0.0)
        assert grid.shape[0] == len(z)

    def test_subgrid_grid_matches_coords(self):
        tree = _paraboloid_tree()
        grid = subgrid.discretize_crown_profile(tree, 2.0, 1.0, z_origin=0.0)
        z = _get_vertical_tree_coords(1.0, tree.height, tree.crown_base_height, 0.0)
        assert grid.shape[0] == len(z)

    def test_default_none_unchanged(self):
        tree = _paraboloid_tree()
        assert np.array_equal(
            discretize_crown_profile(tree, 2.0, 1.0),
            discretize_crown_profile(tree, 2.0, 1.0, z_origin=None),
        )


class TestVoxelizeTreeThreadsZOrigin:
    def test_voxelize_tree_stores_and_uses_z_origin(self):
        tree = _paraboloid_tree()
        vt = voxelize_tree(tree, 2.0, 1.0, alpha=0.0, beta=0.0, rho=1.0, z_origin=0.0)
        assert vt.z_origin == 0.0
        z, _ = vt._voxel_coordinates()
        # density-field heights align to the same half-integer grid
        assert np.allclose(z.ravel() % 1.0, 0.5)

    def test_mass_conserved_with_z_origin(self):
        tree = _paraboloid_tree()
        vt = voxelize_tree(tree, 2.0, 1.0, alpha=0.0, beta=0.0, rho=1.0, z_origin=0.0)
        bulk = vt.distribute_biomass(LinearHeightQuadraticRadialDensity())
        assert bulk.sum() * 2.0 * 2.0 * 1.0 == pytest.approx(tree.foliage_biomass)

    def test_voxelized_tree_default_z_origin_is_none(self):
        tree = _paraboloid_tree()
        vt = VoxelizedTree(tree, np.zeros((3, 3, 3)), 2.0, 1.0)
        assert vt.z_origin is None
