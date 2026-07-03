"""
tests/crown_profile_models/test_cylinder.py
"""

# Internal imports
from fastfuels_core.crown_profile_models.cylinder import CylinderCrownProfile

# External imports
import numpy as np
import pytest

HB, HT, R = 10.0, 20.0, 3.0


@pytest.fixture
def model():
    return CylinderCrownProfile(crown_base_height=HB, height=HT, max_crown_radius=R)


class TestGetMaxRadius:
    def test_returns_r(self, model):
        assert model.get_max_radius() == pytest.approx(R)

    def test_max_radius_height_is_midpoint(self, model):
        # Radius is uniform, so the representative height is the crown midpoint.
        assert model.get_max_radius_height() == pytest.approx((HB + HT) / 2)


class TestGetRadiusAtHeight:
    def test_uniform_within_crown(self, model):
        z = np.linspace(HB, HT, 50)
        assert np.allclose(model.get_radius_at_height(z), R)

    def test_at_base_and_top(self, model):
        assert model.get_radius_at_height(HB) == pytest.approx(R)
        assert model.get_radius_at_height(HT) == pytest.approx(R)

    def test_outside_crown_is_zero(self, model):
        assert model.get_radius_at_height(HB - 1.0) == 0.0
        assert model.get_radius_at_height(HT + 1.0) == 0.0
        assert model.get_radius_at_height(-5.0) == 0.0
        assert model.get_radius_at_height(1e6) == 0.0

    def test_scalar_returns_float(self, model):
        assert isinstance(model.get_radius_at_height(15.0), float)

    def test_1d_returns_1d(self, model):
        z = np.linspace(HB, HT, 20)
        r = model.get_radius_at_height(z)
        assert isinstance(r, np.ndarray)
        assert r.shape == z.shape

    def test_multiple_trees_returns_2d(self):
        model = CylinderCrownProfile(
            crown_base_height=np.array([10.0, 5.0]),
            height=np.array([20.0, 15.0]),
            max_crown_radius=np.array([3.0, 2.0]),
        )
        z = np.linspace(0, 25, 30)
        r = model.get_radius_at_height(z)
        assert r.shape == (2, len(z))
        # Within each crown the radius equals that tree's R.
        assert r[0].max() == pytest.approx(3.0)
        assert r[1].max() == pytest.approx(2.0)
