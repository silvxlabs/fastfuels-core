# Core imports
import math

# Internal imports
from tests.utils import make_random_tree
from fastfuels_core.trees import Tree
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.voxelization import (
    VoxelizedTree,
    LinearHeightQuadraticRadialDensity,
)
from fastfuels_core.voxelization import subgrid
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile

# External imports
import pytest
import numpy as np


HR, VR = 2.0, 1.0


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


class TestSubgridDiscretizeCrownProfile:
    def test_shape_matches_marching_squares(self):
        tree = _paraboloid_tree()
        sampled = subgrid.discretize_crown_profile(tree, HR, VR)
        marching = discretize_crown_profile(tree, HR, VR)
        assert sampled.shape == marching.shape

    def test_volume_fraction_in_unit_range(self):
        grid = subgrid.discretize_crown_profile(_paraboloid_tree(), HR, VR)
        assert np.all(grid >= 0.0) and np.all(grid <= 1.0)

    def test_symmetric_about_stem(self):
        grid = subgrid.discretize_crown_profile(_paraboloid_tree(), HR, VR)
        assert np.allclose(grid, np.flip(grid, axis=1))
        assert np.allclose(grid, np.flip(grid, axis=2))

    def test_n_subgrid_one_samples_cell_centers(self):
        # n_subgrid=1 -> occupancy is 0/1 per voxel (single center sample)
        grid = subgrid.discretize_crown_profile(_paraboloid_tree(), HR, VR, n_subgrid=1)
        assert set(np.unique(grid)).issubset({0.0, 1.0})
        assert grid.sum() > 0

    def test_invalid_n_subgrid_raises(self):
        with pytest.raises(ValueError, match="n_subgrid"):
            subgrid.discretize_crown_profile(_paraboloid_tree(), HR, VR, n_subgrid=0)

    def test_approximates_paraboloid_crown_volume(self):
        # Analytic paraboloid crown volume = pi * D^2 * L / 8.
        tree = _paraboloid_tree()
        d = 2 * tree.max_crown_radius
        length = tree.height - tree.crown_base_height
        analytic = math.pi * d**2 * length / 8.0
        volume = (
            subgrid.discretize_crown_profile(tree, HR, VR, n_subgrid=10).sum()
            * HR
            * HR
            * VR
        )
        assert volume == pytest.approx(analytic, rel=0.05)

    def test_converges_to_marching_squares(self):
        # Sampled occupancy should approach the exact marching-squares volume
        # fraction as the subgrid is refined.
        tree = _paraboloid_tree()
        exact = discretize_crown_profile(tree, HR, VR).sum()
        errors = [
            abs(
                subgrid.discretize_crown_profile(tree, HR, VR, n_subgrid=n).sum()
                - exact
            )
            for n in (2, 6, 16)
        ]
        assert errors[0] > errors[1] > errors[2]

    def test_composes_with_voxelized_tree_and_conserves_mass(self):
        tree = _paraboloid_tree()
        grid = subgrid.discretize_crown_profile(tree, HR, VR)
        vt = VoxelizedTree(tree, grid, HR, VR)
        for field in (None, LinearHeightQuadraticRadialDensity()):
            bd = vt.distribute_biomass(field)
            assert bd.sum() * HR * HR * VR == pytest.approx(tree.foliage_biomass)

    def test_works_for_allometric_crowns(self):
        # Not just geometric profiles: the r <= R(z) test works for any crown.
        tree = make_random_tree(
            species_code=122, height=20.0, crown_ratio=0.6, crown_profile_model="beta"
        )
        grid = subgrid.discretize_crown_profile(tree, HR, VR)
        assert grid.shape == discretize_crown_profile(tree, HR, VR).shape
        assert grid.sum() > 0
