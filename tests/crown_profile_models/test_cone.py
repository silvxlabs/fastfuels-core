"""
tests/crown_profile_models/test_cone.py
"""

# Internal imports
from fastfuels_core.crown_profile_models.cone import ConeCrownProfile

# External imports
import numpy as np
import pytest

HB, HT, R = 10.0, 20.0, 3.0


@pytest.fixture
def model():
    return ConeCrownProfile(crown_base_height=HB, height=HT, max_crown_radius=R)


class TestGetMaxRadius:
    def test_returns_r(self, model):
        assert model.get_max_radius() == pytest.approx(R)

    def test_max_radius_height_is_base(self, model):
        assert model.get_max_radius_height() == pytest.approx(HB)


class TestGetRadiusAtHeight:
    def test_widest_at_base(self, model):
        assert model.get_radius_at_height(HB) == pytest.approx(R)

    def test_zero_at_top(self, model):
        assert model.get_radius_at_height(HT) == pytest.approx(0.0)

    def test_half_at_midpoint(self, model):
        assert model.get_radius_at_height((HB + HT) / 2) == pytest.approx(R / 2)

    def test_linear_profile(self, model):
        z = np.linspace(HB, HT, 50)
        expected = R * (HT - z) / (HT - HB)
        assert np.allclose(model.get_radius_at_height(z), expected)

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
        assert np.all(r >= 0.0)
        assert np.all(r <= R + 1e-9)

    def test_multiple_trees_returns_2d(self):
        model = ConeCrownProfile(
            crown_base_height=np.array([10.0, 5.0]),
            height=np.array([20.0, 15.0]),
            max_crown_radius=np.array([3.0, 2.0]),
        )
        z = np.linspace(0, 25, 30)
        r = model.get_radius_at_height(z)
        assert r.shape == (2, len(z))
        assert np.all(r <= np.array([[3.0], [2.0]]) + 1e-9)

    def test_zero_length_crown_is_safe(self):
        model = ConeCrownProfile(
            crown_base_height=10.0, height=10.0, max_crown_radius=3.0
        )
        # No division-by-zero blow-up; a degenerate crown has no interior.
        r = model.get_radius_at_height(np.linspace(0, 20, 10))
        assert np.all(np.isfinite(r))
