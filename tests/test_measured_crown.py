# Internal imports
from fastfuels_core.trees import Tree
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.crown_profile_models.cone import ConeCrownProfile
from fastfuels_core.crown_profile_models.cylinder import CylinderCrownProfile
from fastfuels_core.crown_profile_models.ellipsoid import EllipsoidCrownProfile
from fastfuels_core.crown_profile_models.paraboloid import ParaboloidCrownProfile
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.voxelization import (
    MeasuredCrown,
    VoxelizedTree,
    sample_occupied_cells,
    voxelize_measured_crown,
)
from fastfuels_core.voxelization.measured_crown import _N_PROFILE_SAMPLES

# External imports
import math
import pytest
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq

CBH = 4.0
HEIGHT = 20.0
HD = 11.0
RADIUS = 5.0


def _flat_profiles():
    return {
        "purves": PurvesCrownProfile(122, 40.0, HEIGHT, (HEIGHT - CBH) / HEIGHT),
        "cone": ConeCrownProfile(CBH, HEIGHT, RADIUS),
        "cylinder": CylinderCrownProfile(CBH, HEIGHT, RADIUS),
        "single_ellipsoid": EllipsoidCrownProfile(CBH, HEIGHT, RADIUS, CBH),
        "single_paraboloid": ParaboloidCrownProfile(CBH, HEIGHT, RADIUS, CBH),
    }


def _disc(n=21, res=1.0):
    """A disc footprint of ``n x n`` cells centred on the stem at (0, 0)."""
    origin = (-n * res / 2, n * res / 2)
    rows, cols = np.indices((n, n))
    x = origin[0] + (cols + 0.5) * res
    y = origin[1] - (rows + 0.5) * res
    footprint = np.hypot(x, y) <= (n // 2) * res
    return footprint, origin, res


def _rho(footprint, origin, res, stem):
    """Independent ρ = d / (d + e) for each cell."""
    rows, cols = np.indices(footprint.shape)
    x = origin[0] + (cols + 0.5) * res
    y = origin[1] - (rows + 0.5) * res
    d = np.hypot(x - stem[0], y - stem[1])
    padded = np.zeros((footprint.shape[0] + 2, footprint.shape[1] + 2), bool)
    padded[1:-1, 1:-1] = footprint
    e = (distance_transform_edt(padded)[1:-1, 1:-1] - 0.5) * res
    return d / (d + e)


def _column_volume(result, footprint, top, res):
    active = footprint & (top > CBH)
    return float(np.sum((top - result.bottom)[active]) * res * res)


def _voxelize(footprint, top, origin, res, profile, **kwargs):
    kwargs.setdefault("stem_xy", (0.0, 0.0))
    kwargs.setdefault("crown_base_height", CBH)
    kwargs.setdefault("output_origin", (0.0, 0.0))
    kwargs.setdefault("output_resolution", (1.0, 1.0))
    return voxelize_measured_crown(
        footprint=footprint,
        top=top,
        source_origin=origin,
        source_resolution=res,
        profile=profile,
        **kwargs,
    )


class TestProfileBottom:
    @pytest.mark.parametrize("name", list(_flat_profiles()))
    def test_widest_at_base_gives_flat_bottom(self, name):
        profile = _flat_profiles()[name]
        footprint, origin, res = _disc()
        rng = np.random.default_rng(0)
        # Mix of tall tops and tops that make the thickness rule bind.
        top = np.where(rng.random(footprint.shape) < 0.7, 18.0, CBH + 0.4)
        result = _voxelize(footprint, top, origin, res, profile)
        unconstrained = footprint & (top - 1.0 >= CBH)
        assert unconstrained.sum() > 100
        assert np.all(result.bottom[unconstrained] == CBH)

    @pytest.mark.parametrize(
        "profile, inverse",
        [
            (
                EllipsoidCrownProfile(CBH, HEIGHT, RADIUS, HD),
                lambda rho: HD - (HD - CBH) * np.sqrt(1 - rho**2),
            ),
            (
                ParaboloidCrownProfile(CBH, HEIGHT, RADIUS, HD),
                lambda rho: CBH + (HD - CBH) * rho**2,
            ),
        ],
        ids=["dual_ellipsoid", "dual_paraboloid"],
    )
    def test_dual_profile_bottom_inverts_lower_branch(self, profile, inverse):
        footprint, origin, res = _disc()
        top = np.full(footprint.shape, 40.0)
        result = _voxelize(footprint, top, origin, res, profile)
        rho = _rho(footprint, origin, res, (0.0, 0.0))[footprint]
        expected = inverse(rho)
        step = (HD - CBH) / (_N_PROFILE_SAMPLES - 1)
        np.testing.assert_allclose(result.bottom[footprint], expected, atol=step)
        # The bottom rises from the crown base at the stem toward the edge.
        centre = footprint.shape[0] // 2
        assert result.bottom[centre, centre] == CBH
        assert result.bottom[footprint].max() > CBH + 0.5 * (HD - CBH)

    def test_beta_bottom_inverts_lower_branch(self):
        profile = BetaCrownProfile(122, CBH, HEIGHT - CBH)
        hd = profile.get_max_radius_height()
        r_max = profile.get_max_radius()
        footprint, origin, res = _disc()
        top = np.full(footprint.shape, 40.0)
        result = _voxelize(footprint, top, origin, res, profile)
        rho = _rho(footprint, origin, res, (0.0, 0.0))[footprint]

        def invert(r):
            if r == 0:
                return CBH
            return brentq(
                lambda z: profile.get_radius_at_height(z) / r_max - r, CBH, hd
            )

        expected = np.array([invert(r) for r in rho])
        step = (hd - CBH) / (_N_PROFILE_SAMPLES - 1)
        np.testing.assert_allclose(result.bottom[footprint], expected, atol=step)
        assert result.bottom[footprint].max() > CBH + 1.0


class TestThickness:
    @pytest.mark.parametrize("min_thickness", [None, 0.3, 2.5])
    def test_min_thickness_rule_in_every_cell(self, min_thickness):
        profile = EllipsoidCrownProfile(CBH, HEIGHT, RADIUS, HD)
        footprint, origin, res = _disc()
        rng = np.random.default_rng(1)
        top = rng.uniform(CBH - 2.0, HEIGHT, footprint.shape)
        result = _voxelize(
            footprint, top, origin, res, profile, min_thickness=min_thickness
        )
        mt = 1.0 if min_thickness is None else min_thickness
        # Same cells, no thickness rule: the profile bottom alone.
        tall = _voxelize(footprint, np.full(footprint.shape, 1e3), origin, res, profile)
        active = footprint & (top > CBH)
        expected = np.maximum(np.minimum(tall.bottom[active], top[active] - mt), CBH)
        np.testing.assert_array_equal(result.bottom[active], expected)
        thickness = top[active] - result.bottom[active]
        assert np.all(thickness >= np.minimum(mt, top[active] - CBH) - 1e-12)
        assert np.all(np.isnan(result.bottom[~active]))

    def test_tops_at_or_below_base_contribute_nothing(self):
        profile = CylinderCrownProfile(CBH, HEIGHT, RADIUS)
        footprint = np.ones((1, 3), bool)
        top = np.array([[CBH - 1.0, CBH, CBH + 2.0]])
        result = _voxelize(
            footprint, top, (-1.5, 0.5), 1.0, profile, output_origin=(-1.5, 0.5)
        )
        assert np.isnan(result.bottom[0, :2]).all()
        assert result.volume_fraction.shape == (2, 1, 1)
        assert result.offset == (4, 0, 2)
        np.testing.assert_allclose(result.volume_fraction.sum(), 2.0)

    def test_all_empty_returns_none(self):
        profile = CylinderCrownProfile(CBH, HEIGHT, RADIUS)
        footprint, origin, res = _disc()
        top = np.full(footprint.shape, CBH)
        assert _voxelize(footprint, top, origin, res, profile) is None
        assert (
            _voxelize(np.zeros((3, 3), bool), np.zeros((3, 3)), origin, res, profile)
            is None
        )


def _stepped_top(shape, rng):
    return np.round(rng.uniform(CBH + 0.2, HEIGHT, shape) * 2) / 2


def _sloped_top(shape, rng):
    rows, cols = np.indices(shape)
    return 6.0 + 0.37 * rows + 0.23 * cols + rng.uniform(0, 0.1)


class TestVolumeExactness:
    @pytest.mark.parametrize("chm_res", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("voxel_res", [1.0, 2.0])
    @pytest.mark.parametrize(
        "shift", [(0.0, 0.0, 1.0), (0.3, -0.7, 0.7), (1.13, 0.41, 0.35)]
    )
    @pytest.mark.parametrize("top_kind", ["stepped", "sloped"])
    def test_volume_matches_columns(self, chm_res, voxel_res, shift, top_kind):
        rng = np.random.default_rng(7)
        n = int(16 / chm_res)
        footprint, origin, res = _disc(n, chm_res)
        make_top = _stepped_top if top_kind == "stepped" else _sloped_top
        top = make_top(footprint.shape, rng)
        dx, dy, vr = shift
        result = _voxelize(
            footprint,
            top,
            origin,
            res,
            ParaboloidCrownProfile(CBH, HEIGHT, RADIUS, HD),
            output_origin=(-20.0 + dx, 20.0 + dy),
            output_resolution=(voxel_res, vr),
        )
        voxel_volume = voxel_res * voxel_res * vr
        grid_volume = result.volume_fraction.sum() * voxel_volume
        expected = _column_volume(result, footprint, top, res)
        assert math.isclose(grid_volume, expected, rel_tol=1e-9)
        assert result.volume_fraction.min() >= 0.0
        assert result.volume_fraction.max() <= 1.0
        # Grid is trimmed to the crown: no empty boundary slabs.
        vf = result.volume_fraction
        assert vf[0].any() and vf[-1].any()
        assert vf[:, 0].any() and vf[:, -1].any()
        assert vf[:, :, 0].any() and vf[:, :, -1].any()


def _overlap(left, right, other_left, other_right):
    return max(0.0, min(right, other_right) - max(left, other_left))


def _covering_indices(lower, upper, origin, step):
    return (
        math.floor((lower - origin) / step),
        math.ceil((upper - origin) / step),
    )


def _c1_construct_envelope(
    roof, bottom, source_resolution, source_origin, output_resolution, output_origin
):
    """Test-only reference copy of the integration rule in C1's
    ``_construct_envelope`` (crown-geometry research, ``prepared_crown.py``).

    Kept as C1 wrote it -- a loop per source cell and output voxel over a
    southwest origin with rows increasing northward -- with two changes: the
    spatially constant ``base_hag_m`` becomes each cell's ``bottom``, and the
    envelope accumulates into a dict keyed by global ``(z, y, x)`` voxel index
    instead of a pre-bounded array. Support states, bounds, limits and
    accounting are not copied.
    """
    envelope = {}
    source_dx = source_dy = source_resolution
    source_x, source_y = source_origin
    output_dx, output_dy, output_dz = output_resolution
    output_x, output_y, output_z = output_origin
    for row, col in np.ndindex(roof.shape):
        base = bottom[row, col]
        if np.isnan(base):
            continue
        source_x0 = source_x + col * source_dx
        source_x1 = source_x0 + source_dx
        source_y0 = source_y + row * source_dy
        source_y1 = source_y0 + source_dy
        ix0, ix1 = _covering_indices(source_x0, source_x1, output_x, output_dx)
        iy0, iy1 = _covering_indices(source_y0, source_y1, output_y, output_dy)
        for iy in range(iy0, iy1):
            target_y0 = output_y + iy * output_dy
            y_overlap = _overlap(source_y0, source_y1, target_y0, target_y0 + output_dy)
            for ix in range(ix0, ix1):
                target_x0 = output_x + ix * output_dx
                x_overlap = _overlap(
                    source_x0, source_x1, target_x0, target_x0 + output_dx
                )
                area_fraction = (x_overlap / output_dx) * (y_overlap / output_dy)
                roof_value = float(roof[row, col])
                if roof_value <= base:
                    continue
                iz0, iz1 = _covering_indices(base, roof_value, output_z, output_dz)
                for iz in range(iz0, iz1):
                    target_z0 = output_z + iz * output_dz
                    z_overlap = _overlap(
                        base, roof_value, target_z0, target_z0 + output_dz
                    )
                    key = (iz, iy, ix)
                    envelope[key] = envelope.get(key, 0.0) + area_fraction * (
                        z_overlap / output_dz
                    )
    return {key: min(max(value, 0.0), 1.0) for key, value in envelope.items()}


class TestMatchesC1Reference:
    @pytest.mark.parametrize("seed", range(8))
    def test_random_inputs(self, seed):
        rng = np.random.default_rng(seed)
        rows, cols = rng.integers(3, 12, size=2)
        res = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        hr = float(rng.choice([0.5, 1.0, 2.0, 3.0]))
        vr = float(rng.choice([0.4, 1.0, 1.7]))
        footprint = rng.random((rows, cols)) < 0.7
        top = rng.uniform(CBH - 1.0, HEIGHT, (rows, cols))
        source_origin = tuple(rng.uniform(-10, 10, 2))
        output_origin = tuple(rng.uniform(-10, 10, 2))
        stem = (
            source_origin[0] + rng.uniform(0, cols * res),
            source_origin[1] - rng.uniform(0, rows * res),
        )
        result = _voxelize(
            footprint,
            top,
            source_origin,
            res,
            BetaCrownProfile(122, CBH, HEIGHT - CBH),
            stem_xy=stem,
            output_origin=output_origin,
            output_resolution=(hr, vr),
        )
        if result is None:
            assert not (footprint & (top > CBH)).any()
            return

        # C1 is southwest-origin with rows increasing north: flip the source
        # rows, and put its output origin n_rows below ours so C1 row iy is our
        # row n_rows - 1 - iy.
        n_rows = 1000
        reference = _c1_construct_envelope(
            np.flipud(top),
            np.flipud(result.bottom),
            res,
            (source_origin[0], source_origin[1] - rows * res),
            (hr, hr, vr),
            (output_origin[0], output_origin[1] - n_rows * hr, 0.0),
        )
        expected = {
            (iz, n_rows - 1 - iy, ix): v for (iz, iy, ix), v in reference.items()
        }
        z0, r0, c0 = result.offset
        actual = {
            (z0 + k, r0 + i, c0 + j): v
            for (k, i, j), v in np.ndenumerate(result.volume_fraction)
        }
        for key in expected.keys() | actual.keys():
            assert math.isclose(
                actual.get(key, 0.0), expected.get(key, 0.0), abs_tol=1e-9
            ), key


class TestOffset:
    @pytest.mark.parametrize(
        "cell_x, cell_y, expected_rc",
        [
            (3.25, -5.75, (5, 3)),  # inside voxel (row 5, col 3)
            (-2.75, 1.25, (-2, -3)),  # north-west of the output origin
            (-0.75, -0.25, (0, -1)),  # west of the origin only
        ],
    )
    def test_one_cell_crown_lands_in_its_voxel(self, cell_x, cell_y, expected_rc):
        res = 0.5
        origin = (cell_x - res / 2, cell_y + res / 2)
        result = _voxelize(
            np.ones((1, 1), bool),
            np.array([[7.5]]),
            origin,
            res,
            CylinderCrownProfile(CBH, HEIGHT, RADIUS),
            stem_xy=(cell_x, cell_y),
            output_origin=(0.0, 0.0),
            output_resolution=(1.0, 1.0),
        )
        assert isinstance(result, MeasuredCrown)
        assert result.offset == (4, *expected_rc)
        assert result.volume_fraction.shape == (4, 1, 1)
        np.testing.assert_allclose(
            result.volume_fraction[:, 0, 0], [0.25, 0.25, 0.25, 0.125]
        )

    def test_z_offset_counts_voxels_from_ground(self):
        result = _voxelize(
            np.ones((1, 1), bool),
            np.array([[9.0]]),
            (0.0, 0.0),
            1.0,
            CylinderCrownProfile(CBH, HEIGHT, RADIUS),
            crown_base_height=5.2,
            output_resolution=(1.0, 2.0),
        )
        assert result.offset == (2, 0, 0)
        np.testing.assert_allclose(result.volume_fraction[:, 0, 0], [0.4, 1.0, 0.5])


class TestMassConservation:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_distribute_biomass_conserves_foliage_mass(self, seed):
        tree = Tree(
            species_code=122,
            status_code=1,
            diameter=40.0,
            height=HEIGHT,
            crown_ratio=(HEIGHT - CBH) / HEIGHT,
            crown_profile_model_type="ellipsoid",
            max_crown_radius=RADIUS,
            max_crown_diameter_height=HD,
            crown_fuel_load=123.4,
        )
        footprint, origin, res = _disc(21, 0.5)
        rng = np.random.default_rng(seed)
        top = rng.uniform(CBH + 1.0, HEIGHT, footprint.shape)
        hr, vr = 1.0, 1.0
        result = _voxelize(
            footprint,
            top,
            origin,
            res,
            tree.crown_profile_model,
            output_origin=(-10.3, 10.6),
            output_resolution=(hr, vr),
        )
        sampled = sample_occupied_cells(result.volume_fraction, 0.5, 0.5, seed=seed)
        density = VoxelizedTree(tree, sampled, hr, vr).distribute_biomass()
        assert density.shape == result.volume_fraction.shape
        assert math.isclose(
            density.sum() * hr * hr * vr, tree.foliage_biomass, rel_tol=1e-9
        )


class TestValidation:
    def _call(self, **overrides):
        kwargs = dict(
            footprint=np.ones((2, 2), bool),
            top=np.full((2, 2), 10.0),
            source_origin=(0.0, 0.0),
            source_resolution=1.0,
            stem_xy=(1.0, -1.0),
            crown_base_height=CBH,
            profile=CylinderCrownProfile(CBH, HEIGHT, RADIUS),
            output_origin=(0.0, 0.0),
            output_resolution=(1.0, 1.0),
        )
        kwargs.update(overrides)
        return voxelize_measured_crown(**kwargs)

    def test_valid_call(self):
        assert self._call() is not None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"top": np.full((2, 3), 10.0)},
            {"footprint": np.ones(4, bool), "top": np.full(4, 10.0)},
            {"source_resolution": 0.0},
            {"source_resolution": -1.0},
            {"source_resolution": (1.0, 2.0)},
            {"output_resolution": (0.0, 1.0)},
            {"output_resolution": (1.0, -1.0)},
            {"min_thickness": 0.0},
            {"crown_base_height": -0.1},
            {"top": np.array([[10.0, np.nan], [10.0, 10.0]])},
            {"top": np.array([[10.0, np.inf], [10.0, 10.0]])},
        ],
    )
    def test_rejects(self, overrides):
        with pytest.raises(ValueError):
            self._call(**overrides)

    def test_non_finite_top_outside_footprint_is_allowed(self):
        footprint = np.array([[True, False], [True, True]])
        top = np.array([[10.0, np.nan], [10.0, 10.0]])
        assert self._call(footprint=footprint, top=top) is not None

    def test_square_resolution_pair_is_accepted(self):
        assert self._call(source_resolution=(1.0, 1.0)) is not None

    def test_stem_outside_footprint(self):
        result = self._call(stem_xy=(50.0, 50.0))
        assert result is not None
