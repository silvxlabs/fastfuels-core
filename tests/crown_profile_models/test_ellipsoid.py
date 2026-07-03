"""
tests/crown_profile_models/test_ellipsoid.py
"""

# Internal imports
from fastfuels_core.crown_profile_models.ellipsoid import EllipsoidCrownProfile

# External imports
import numpy as np
import pytest

HB, HT, R = 10.0, 20.0, 3.0
MID = (HB + HT) / 2.0


@pytest.fixture
def model():
    # Hd omitted -> defaults to the crown midpoint (symmetric spheroid).
    return EllipsoidCrownProfile(crown_base_height=HB, height=HT, max_crown_radius=R)


class TestConstruction:
    def test_default_hd_is_midpoint(self, model):
        assert model.max_crown_diameter_height.item() == pytest.approx(MID)

    def test_out_of_range_hd_raises(self):
        with pytest.raises(ValueError):
            EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=HB - 1.0)
        with pytest.raises(ValueError):
            EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=HT + 1.0)


class TestGetMaxRadius:
    def test_returns_r(self, model):
        assert model.get_max_radius() == pytest.approx(R)

    def test_max_radius_height_is_hd(self):
        m = EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=13.0)
        assert m.get_max_radius_height() == pytest.approx(13.0)


class TestGetRadiusAtHeight:
    def test_peak_at_hd(self, model):
        assert model.get_radius_at_height(MID) == pytest.approx(R)

    def test_zero_at_ends(self, model):
        assert model.get_radius_at_height(HB) == pytest.approx(0.0)
        assert model.get_radius_at_height(HT) == pytest.approx(0.0)

    def test_matches_ellipse_formula(self, model):
        z = np.linspace(HB, HT, 101)
        hd = MID
        lower = R * np.sqrt(np.clip(1 - ((hd - z) / (hd - HB)) ** 2, 0, 1))
        upper = R * np.sqrt(np.clip(1 - ((z - hd) / (HT - hd)) ** 2, 0, 1))
        expected = np.where(z <= hd, lower, upper)
        assert np.allclose(model.get_radius_at_height(z), expected)

    def test_spheroid_symmetry_when_hd_is_midpoint(self, model):
        for s in [1.0, 2.0, 3.0, 4.9]:
            assert model.get_radius_at_height(MID - s) == pytest.approx(
                model.get_radius_at_height(MID + s)
            )

    def test_argmax_coincides_with_hd(self):
        m = EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=13.0)
        z = np.linspace(HB, HT, 1001)
        r = m.get_radius_at_height(z)
        assert z[np.argmax(r)] == pytest.approx(13.0, abs=0.05)

    def test_outside_crown_is_zero(self, model):
        assert model.get_radius_at_height(HB - 1.0) == 0.0
        assert model.get_radius_at_height(HT + 1.0) == 0.0
        assert model.get_radius_at_height(-5.0) == 0.0
        assert model.get_radius_at_height(1e6) == 0.0

    # ── Single-lobe limiting cases ────────────────────────────────────────
    def test_single_lobe_widest_at_base(self):
        m = EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=HB)
        assert m.get_radius_at_height(HB) == pytest.approx(R)
        assert m.get_radius_at_height(HT) == pytest.approx(0.0)
        z = np.linspace(HB, HT, 50)
        r = m.get_radius_at_height(z)
        assert np.all(np.diff(r) <= 1e-9)
        assert np.all(np.isfinite(r))

    def test_single_lobe_widest_at_top(self):
        m = EllipsoidCrownProfile(HB, HT, R, max_crown_diameter_height=HT)
        assert m.get_radius_at_height(HT) == pytest.approx(R)
        assert m.get_radius_at_height(HB) == pytest.approx(0.0)
        z = np.linspace(HB, HT, 50)
        r = m.get_radius_at_height(z)
        assert np.all(np.diff(r) >= -1e-9)
        assert np.all(np.isfinite(r))

    # ── Return shapes ─────────────────────────────────────────────────────
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
        model = EllipsoidCrownProfile(
            crown_base_height=np.array([10.0, 5.0]),
            height=np.array([20.0, 15.0]),
            max_crown_radius=np.array([3.0, 2.0]),
            max_crown_diameter_height=np.array([15.0, 10.0]),
        )
        z = np.linspace(0, 25, 30)
        r = model.get_radius_at_height(z)
        assert r.shape == (2, len(z))
        assert np.all(r <= np.array([[3.0], [2.0]]) + 1e-9)
