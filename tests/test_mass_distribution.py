# Internal imports
from fastfuels_core.trees import Tree
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.voxelization import (
    voxelize_tree,
    DensityField,
    UniformDensity,
    GradientDensity,
    LinearHeightQuadraticRadialDensity,
)
from fastfuels_core.voxelization.mass_distribution import (
    _linear_height_quadratic_radial,
)

# External imports
import pytest
import numpy as np


HR, VR = 2.0, 1.0


def _paraboloid_tree(height=31.5, cbh=3.12, crown_dia=9.34, mass=375.08):
    """A deterministic paraboloid tree (Hd at the beta mode), like tree_43050."""
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
        crown_base_height=cbh,
        crown_profile_model_type="paraboloid",
        max_crown_radius=crown_dia / 2.0,
        max_crown_diameter_height=hd,
        crown_fuel_load=mass,
    )


def _full_occupancy(tree):
    """Deterministic, fully-occupied crown (no stochastic thinning)."""
    return voxelize_tree(tree, HR, VR, alpha=0.0, beta=0.0, rho=1.0, seed=1)


def _mass(grid):
    return float(grid.sum()) * HR * HR * VR


# --------------------------------------------------------------------------- #
# UniformDensity
# --------------------------------------------------------------------------- #


class TestUniformDensity:
    def test_matches_legacy_formula_exactly(self):
        tree = _paraboloid_tree()
        vt = _full_occupancy(tree)
        legacy = vt.grid * (tree.foliage_biomass / (vt.grid.sum() * HR * HR * VR))
        assert np.array_equal(vt.distribute_biomass(UniformDensity()), legacy)

    def test_is_the_default(self):
        vt = _full_occupancy(_paraboloid_tree())
        assert np.array_equal(
            vt.distribute_biomass(), vt.distribute_biomass(UniformDensity())
        )

    def test_conserves_mass(self):
        tree = _paraboloid_tree()
        vt = _full_occupancy(tree)
        assert _mass(vt.distribute_biomass(UniformDensity())) == pytest.approx(
            tree.foliage_biomass
        )

    def test_bulk_density_proportional_to_occupancy(self):
        # Defining property of uniform density: bulk density is one constant
        # times the volume fraction, so bd / occupancy is the same everywhere.
        vt = _full_occupancy(_paraboloid_tree())
        bd = vt.distribute_biomass(UniformDensity())
        occ = vt.grid
        ratio = bd[occ > 0] / occ[occ > 0]
        assert np.allclose(ratio, ratio[0])

    def test_empty_grid_returns_zeros_not_nan(self):
        tree = _paraboloid_tree()
        vt = _full_occupancy(tree)
        vt.grid = np.zeros_like(vt.grid)
        out = vt.distribute_biomass(UniformDensity())
        assert np.all(out == 0.0) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# GradientDensity
# --------------------------------------------------------------------------- #


class TestGradientDensity:
    def test_constant_weight_equals_uniform(self):
        vt = _full_occupancy(_paraboloid_tree())
        const = GradientDensity(lambda r, z, tree: np.ones_like(r * z))
        assert np.allclose(
            vt.distribute_biomass(const), vt.distribute_biomass(UniformDensity())
        )

    def test_conserves_mass_for_arbitrary_weight(self):
        tree = _paraboloid_tree()
        vt = _full_occupancy(tree)
        # a lopsided weight that still must integrate to the crown mass
        field = GradientDensity(lambda r, z, tree: z**2 + r + 1.0)
        assert _mass(vt.distribute_biomass(field)) == pytest.approx(
            tree.foliage_biomass
        )

    def test_empty_grid_returns_zeros_not_nan(self):
        vt = _full_occupancy(_paraboloid_tree())
        vt.grid = np.zeros_like(vt.grid)
        out = vt.distribute_biomass(GradientDensity(lambda r, z, tree: r + z + 1))
        assert np.all(out == 0.0) and np.all(np.isfinite(out))

    def test_is_a_density_field(self):
        assert isinstance(GradientDensity(lambda r, z, t: r), DensityField)


# --------------------------------------------------------------------------- #
# LinearHeightQuadraticRadialDensity (the LANL gradient)
# --------------------------------------------------------------------------- #


class TestLinearHeightQuadraticRadialDensity:
    def test_conserves_mass(self):
        tree = _paraboloid_tree()
        vt = _full_occupancy(tree)
        out = vt.distribute_biomass(LinearHeightQuadraticRadialDensity())
        assert _mass(out) == pytest.approx(tree.foliage_biomass)

    def test_is_top_weighted_relative_to_uniform(self):
        vt = _full_occupancy(_paraboloid_tree())
        z = vt.voxel_height.ravel()
        uniform = vt.distribute_biomass(UniformDensity())
        lanl = vt.distribute_biomass(LinearHeightQuadraticRadialDensity())

        def com(g):
            return (g.sum(axis=(1, 2)) * z).sum() / g.sum()

        assert com(lanl) > com(uniform)

    def test_is_outer_weighted_relative_to_uniform(self):
        vt = _full_occupancy(_paraboloid_tree())
        r = vt.radial_distance
        r = np.broadcast_to(r, vt.grid.shape)
        uniform = vt.distribute_biomass(UniformDensity())
        lanl = vt.distribute_biomass(LinearHeightQuadraticRadialDensity())
        # mass-weighted mean radius is larger for the quadratic-radial gradient
        assert (lanl * r).sum() / lanl.sum() > (uniform * r).sum() / uniform.sum()

    def test_weight_matches_analytic_formula(self):
        tree = _paraboloid_tree()
        hb, ht = tree.crown_base_height, tree.height
        hd = tree.crown_profile_model.get_max_radius_height()
        d = 2.0 * tree.max_crown_radius
        r = np.array([0.0, 1.5, 3.0])
        z = np.array([hb, (hb + ht) / 2, ht])
        expected = ((z - hb) + 4.0 * (ht - hd) * r**2 / d**2) / (ht - hb)
        assert np.allclose(_linear_height_quadratic_radial(r, z, tree), expected)

    def test_weight_zero_at_crown_base_center(self):
        tree = _paraboloid_tree()
        w = _linear_height_quadratic_radial(
            np.array(0.0), np.array(tree.crown_base_height), tree
        )
        assert w == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Point weight and crown mean
# --------------------------------------------------------------------------- #


def _paraboloid_with_peak(fraction):
    """A paraboloid crown whose widest height is ``fraction`` up the crown."""
    hb, ht = 4.0, 20.0
    return Tree(
        species_code=122,
        status_code=1,
        diameter=40.0,
        height=ht,
        crown_base_height=hb,
        crown_profile_model_type="paraboloid",
        max_crown_radius=3.0,
        max_crown_diameter_height=hb + fraction * (ht - hb),
        crown_fuel_load=100.0,
    )


class TestPointWeightAndCrownMean:
    def test_uniform_weight_and_crown_mean_are_one(self):
        tree = _paraboloid_tree()
        assert UniformDensity().weight(np.array(1.0), np.array(10.0), tree) == 1.0
        assert UniformDensity().crown_mean(tree) == 1.0

    def test_gradient_weight_is_the_weight_function(self):
        tree = _paraboloid_tree()
        r = np.array([0.0, 1.5, 3.0])
        z = np.array([5.0, 15.0, 25.0])
        np.testing.assert_array_equal(
            LinearHeightQuadraticRadialDensity().weight(r, z, tree),
            _linear_height_quadratic_radial(r, z, tree),
        )

    def test_crown_weight_reads_the_point_weight_at_voxel_centers(self):
        vt = _full_occupancy(_paraboloid_tree())
        field = LinearHeightQuadraticRadialDensity()
        np.testing.assert_array_equal(
            field.crown_weight(vt),
            field.weight(vt.radial_distance, vt.voxel_height, vt.tree),
        )

    @pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_lanl_gradient_averages_two_thirds_over_a_paraboloid(self, fraction):
        # Why LANL Trees scales its gradient by 3/2.
        tree = _paraboloid_with_peak(fraction)
        mean = LinearHeightQuadraticRadialDensity().crown_mean(tree)
        assert mean == pytest.approx(2.0 / 3.0, rel=1e-12)

    def test_constant_weight_has_that_crown_mean(self):
        field = GradientDensity(
            lambda r, z, tree: np.full(np.broadcast(r, z).shape, 2.5)
        )
        assert field.crown_mean(_paraboloid_tree()) == pytest.approx(2.5, rel=1e-12)

    @pytest.mark.parametrize("profile", ["beta", "purves", "ellipsoid", "cone"])
    def test_crown_mean_matches_a_fine_voxelization(self, profile):
        from fastfuels_core.voxelization import subgrid

        tree = Tree(
            species_code=122,
            status_code=1,
            diameter=40.0,
            height=20.0,
            crown_base_height=6.0,
            crown_profile_model_type=profile,
            max_crown_radius=3.0,
            crown_fuel_load=100.0,
        )
        field = LinearHeightQuadraticRadialDensity()
        hr = vr = 0.25
        volume = subgrid.discretize_crown_profile(tree, hr, vr, n_subgrid=4)
        weighted = subgrid.discretize_crown_profile(
            tree, hr, vr, n_subgrid=4, density_field=field
        )
        # The weighted grid is scaled by the crown mean, so it averages 1.
        assert weighted.sum() / volume.sum() == pytest.approx(1.0, rel=5e-3)

    def test_field_without_a_point_weight_says_so(self):
        class CenterOnly(DensityField):
            def crown_weight(self, vt):
                return 1.0

        with pytest.raises(NotImplementedError, match="point weight"):
            CenterOnly().weight(np.array(0.0), np.array(0.0), _paraboloid_tree())
        with pytest.raises(NotImplementedError, match="crown mean"):
            CenterOnly().crown_mean(_paraboloid_tree())
