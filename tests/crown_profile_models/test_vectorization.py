"""
tests/crown_profile_models/test_vectorization.py

A profile built for many trees at once agrees, tree by tree, with profiles
built for each tree alone.
"""

# Internal imports
from fastfuels_core.crown_profile_models.abc import CrownProfileModel
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.crown_profile_models.cone import ConeCrownProfile
from fastfuels_core.crown_profile_models.cylinder import CylinderCrownProfile
from fastfuels_core.crown_profile_models.ellipsoid import EllipsoidCrownProfile
from fastfuels_core.crown_profile_models.paraboloid import ParaboloidCrownProfile
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile

# External imports
import numpy as np
import pytest

SPCD = np.array([122, 202, 747, 131, 316])
DBH = np.array([25.0, 40.0, 12.0, 33.0, 18.0])
HEIGHT = np.array([15.0, 30.0, 9.0, 24.0, 12.0])
CROWN_RATIO = np.array([0.5, 0.3, 0.8, 0.45, 0.6])
CBH = HEIGHT * (1.0 - CROWN_RATIO)
RADIUS = np.array([2.0, 3.5, 1.2, 2.8, 1.6])
# Includes both single-lobe limits (Hd == Hb, Hd == Ht).
HD = np.array([CBH[0], HEIGHT[1], (CBH[2] + HEIGHT[2]) / 2, CBH[3] + 2.0, CBH[4]])

PROFILES = {
    "purves": lambda i: PurvesCrownProfile(SPCD[i], DBH[i], HEIGHT[i], CROWN_RATIO[i]),
    "beta": lambda i: BetaCrownProfile(SPCD[i], CBH[i], HEIGHT[i] - CBH[i]),
    "cone": lambda i: ConeCrownProfile(CBH[i], HEIGHT[i], RADIUS[i]),
    "cylinder": lambda i: CylinderCrownProfile(CBH[i], HEIGHT[i], RADIUS[i]),
    "ellipsoid": lambda i: EllipsoidCrownProfile(CBH[i], HEIGHT[i], RADIUS[i]),
    "ellipsoid_hd": lambda i: EllipsoidCrownProfile(
        CBH[i], HEIGHT[i], RADIUS[i], HD[i]
    ),
    "paraboloid": lambda i: ParaboloidCrownProfile(CBH[i], HEIGHT[i], RADIUS[i]),
    "paraboloid_hd": lambda i: ParaboloidCrownProfile(
        CBH[i], HEIGHT[i], RADIUS[i], HD[i]
    ),
}

N_TREES = SPCD.size
ALL = slice(None)


@pytest.fixture(params=sorted(PROFILES))
def factory(request):
    return PROFILES[request.param]


def test_every_profile_subclass_is_covered():
    covered = {type(PROFILES[name](0)) for name in PROFILES}
    assert covered == set(CrownProfileModel.__subclasses__())


def test_get_max_radius_height_is_abstract():
    assert "get_max_radius_height" in CrownProfileModel.__abstractmethods__


@pytest.mark.parametrize("method", ["get_max_radius", "get_max_radius_height"])
class TestPerTreeMethods:
    def test_single_tree_returns_float(self, factory, method):
        for i in range(N_TREES):
            assert isinstance(getattr(factory(i), method)(), float)

    def test_vectorized_returns_1d(self, factory, method):
        result = getattr(factory(ALL), method)()
        assert isinstance(result, np.ndarray)
        assert result.shape == (N_TREES,)

    def test_vectorized_matches_per_tree(self, factory, method):
        vectorized = getattr(factory(ALL), method)()
        per_tree = [getattr(factory(i), method)() for i in range(N_TREES)]
        np.testing.assert_allclose(vectorized, per_tree, rtol=1e-12)


def test_max_radius_height_attains_max_radius(factory):
    profile = factory(ALL)
    hd = profile.get_max_radius_height()
    r_max = profile.get_max_radius()
    # Per-tree heights through the [n_trees, 1] broadcast: the diagonal holds
    # each tree's radius at its own max-radius height.
    radius = np.diag(profile.get_radius_at_height(hd))
    np.testing.assert_allclose(radius, r_max, rtol=1e-6)


def test_radius_at_height_matches_per_tree(factory):
    z = np.linspace(0.0, HEIGHT.max() + 1.0, 97)
    vectorized = factory(ALL).get_radius_at_height(z)
    assert vectorized.shape == (N_TREES, z.size)
    for i in range(N_TREES):
        np.testing.assert_allclose(
            vectorized[i], factory(i).get_radius_at_height(z), rtol=1e-12
        )
