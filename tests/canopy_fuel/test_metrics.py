"""Tests for fastfuels_core.canopy_fuel.metrics."""

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.allometry import nsvb
from fastfuels_core.allometry.brown import (
    fine_branchwood_share,
    foliage_fraction,
    foliage_plus_fine_fraction,
)
from fastfuels_core.canopy_fuel.metrics import (
    FT_TO_M,
    available_canopy_fuel,
    canopy_cover,
    canopy_fuel_load,
    compute_canopy_metrics,
    cbd_running_mean,
    cumulative_fuel_fraction,
    disk_rect_overlap_area,
    max_crown_radius,
    profile_threshold_heights,
    vertical_profile,
)
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species
from fastfuels_core.trees import Tree


def random_stand(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "fia_species_code": rng.choice([202, 122, 73, 17, 351, 15, 108], n),
            "dbh": rng.uniform(2.5, 90, n),
            "height": rng.uniform(2, 40, n),
            "crown_ratio": rng.uniform(0.1, 0.9, n),
        }
    )


class TestMaxCrownRadius:
    def test_matches_per_tree_reference(self):
        """Vectorized radii equal the per-tree Tree.max_crown_radius path."""
        trees = random_stand(500)
        vectorized = max_crown_radius(trees)
        reference = np.array(
            [
                Tree(
                    species_code=row.fia_species_code,
                    status_code=1,
                    diameter=row.dbh,
                    height=row.height,
                    crown_ratio=row.crown_ratio,
                ).max_crown_radius
                for row in trees.itertuples()
            ]
        )
        np.testing.assert_allclose(vectorized, reference, rtol=1e-12)

    def test_shapes_and_positivity(self):
        trees = random_stand(50)
        r = max_crown_radius(trees)
        assert r.shape == (50,)
        assert (r > 0).all()

    def test_single_tree_returns_1d(self):
        r = max_crown_radius(random_stand(1))
        assert r.shape == (1,)

    def test_column_override(self):
        trees = random_stand(10)
        trees["crad"] = np.arange(10, dtype=float) + 1.0
        r = max_crown_radius(trees, crown_radius_column="crad")
        np.testing.assert_array_equal(r, trees["crad"].to_numpy())

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            max_crown_radius(random_stand(5), crown_radius_column="nope")


class TestBrownProportions:
    """Hand-computed anchors from Brown 1978 Table 16 (p. 53) and
    Snell & Little 1983 Table 3 (p. 6). P1 is the foliage fraction of
    total crown weight; P2 is ACCUMULATIVE foliage + 0-1/4 in
    branchwood, so the fine fraction is P2 - P1."""

    def test_ponderosa_p1_p2_10in(self):
        p1 = foliage_fraction(np.array(["PP"]), np.array([10.0]))
        p2 = foliage_plus_fine_fraction(np.array(["PP"]), np.array([10.0]))
        np.testing.assert_allclose(p1, 0.558 * np.exp(-0.475), rtol=1e-12)
        np.testing.assert_allclose(p2, 0.625 * np.exp(-0.511), rtol=1e-12)
        # Accumulative: P2 > P1 below the crossing; fine is their difference.
        assert p2[0] > p1[0]

    def test_ponderosa_fine_share_is_small(self):
        # The fine share of branchwood for PP@10in is (P2-P1)/(1-P1)
        # ~= 0.0428 — NOT P2/(1-P1) ~= 0.57. Guards the accumulative
        # semantics this module exists to get right.
        share = fine_branchwood_share(
            np.array(["PP"]), np.array(["PP"]), np.array([10.0])
        )
        p1 = 0.558 * np.exp(-0.475)
        p2 = 0.625 * np.exp(-0.511)
        np.testing.assert_allclose(share, (p2 - p1) / (1 - p1), rtol=1e-12)
        assert share[0] < 0.1

    def test_ponderosa_crossover_guard(self):
        # Past d ~31.5 in the fitted curves cross; Brown holds the fine
        # fraction at 0.01.
        p1 = foliage_fraction(np.array(["PP"]), np.array([35.0]))
        p2 = foliage_plus_fine_fraction(np.array(["PP"]), np.array([35.0]))
        np.testing.assert_allclose(p2 - p1, 0.01, atol=1e-12)

    def test_grand_fir_reciprocal_and_break(self):
        ids = np.array(["GF", "GF"])
        dia = np.array([20.0, 40.0])
        p1 = foliage_fraction(ids, dia)
        p2 = foliage_plus_fine_fraction(ids, dia)
        np.testing.assert_allclose(p1[0], 1.0 / (1.592 + 0.0529 * 20.0))
        np.testing.assert_allclose(p2[0], 1.0 / (1.15 + 0.0416 * 20.0))
        # dia > 36 in -> Table 16 constants
        assert p1[1] == 0.286 and p2[1] == 0.378

    def test_douglas_fir_break_from_primary_source(self):
        # DF > 36 in: P1 = 0.227, P2 = 0.315 — in Brown Table 16 but
        # omitted from the FuelCalc guide.
        p1 = foliage_fraction(np.array(["DF"]), np.array([40.0]))
        p2 = foliage_plus_fine_fraction(np.array(["DF"]), np.array([40.0]))
        assert p1[0] == 0.227 and p2[0] == 0.315

    def test_lodgepole_linear_clamps_to_zero(self):
        share = fine_branchwood_share(
            np.array(["LP"]), np.array(["LP"]), np.array([60.0])
        )
        assert share[0] == 0.0

    def test_quaking_aspen_has_no_equations(self):
        # SPCD 746 used to resolve to the QA Id, which borrowed
        # whitebark pine's P1 and western larch's P2 -- a pairing
        # neither Brown nor SL-83 sanctions. Dropped, so aspen now
        # raises rather than returning an unsourced number. Its real
        # source is Loomis & Roussopoulos 1978 (NC-156).
        with pytest.raises(ValueError, match="QA"):
            fine_branchwood_share(np.array(["QA"]), np.array(["QA"]), np.array([8.0]))
        aspen = pd.DataFrame(
            {
                "fia_species_code": [746],
                "dbh": [20.0],
                "height": [15.0],
                "crown_ratio": [0.5],
            }
        )
        with pytest.raises(ValueError, match="QA"):
            available_canopy_fuel(aspen)

    def test_giant_chinkapin_from_sl83(self):
        # GC is absent from the FuelCalc guide but present in SL-83
        # Table 3: f(1) = 1/(1.6048 + 0.5630 d^0.6828).
        p1 = foliage_fraction(np.array(["GC"]), np.array([10.0]))
        np.testing.assert_allclose(
            p1, 1.0 / (1.6048 + 0.5630 * 10.0**0.6828), rtol=1e-12
        )

    def test_fine_never_negative(self):
        # Cross-Id subtraction and curve crossings must never yield a
        # negative fine share, over a broad diameter sweep.
        dia = np.linspace(1.0, 80.0, 200)
        for eq_id in ["PP", "GF", "DF", "LP", "WP", "WB", "ES", "WH", "AL", "RA"]:
            ids = np.full(dia.shape, eq_id)
            share = fine_branchwood_share(ids, ids, dia)
            assert (share >= 0.0).all() and (share <= 1.0).all(), eq_id

    def test_unsupported_id_raises(self):
        with pytest.raises(ValueError, match="PY"):
            foliage_fraction(np.array(["PY"]), np.array([8.0]))


class TestAvailableCanopyFuel:
    def test_composition_and_units(self):
        """ACF equals hand-composed NSVB + Brown terms."""
        # 202 Douglas-fir, dbh 25.4 cm = 10 in, height 15.24 m = 50 ft
        trees = pd.DataFrame(
            {
                "fia_species_code": [202],
                "dbh": [25.4],
                "height": [15.24],
                "crown_ratio": [0.5],
            }
        )
        acf = available_canopy_fuel(trees)
        # Brown Table 16 DF at 10 in: fine share = (P2 - P1)/(1 - P1).
        p1 = 0.484 * np.exp(-0.0210 * 10.0)
        p2 = 0.729 * np.exp(-0.0233 * 10.0)
        fine_share = (p2 - p1) / (1.0 - p1)
        expected = nsvb.foliage_biomass(
            [202], [25.4], [15.24]
        ) + 0.5 * fine_share * nsvb.branch_biomass([202], [25.4], [15.24])
        np.testing.assert_allclose(acf, expected, rtol=1e-10)

    def test_fractions(self):
        trees = random_stand(20)
        foliage_only = available_canopy_fuel(
            trees, foliage_fraction=1.0, branchwood_fraction=0.0
        )
        fine_only = available_canopy_fuel(
            trees, foliage_fraction=0.0, branchwood_fraction=1.0
        )
        default = available_canopy_fuel(trees)
        np.testing.assert_allclose(default, foliage_only + 0.5 * fine_only, rtol=1e-12)
        assert (foliage_only > 0).all() and (fine_only >= 0).all()

    def test_fuel_column_passthrough(self):
        trees = random_stand(5)
        trees["acf_kg"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_equal(
            available_canopy_fuel(trees, fuel_column="acf_kg"),
            trees["acf_kg"].to_numpy(),
        )

    def test_unknown_species_raises(self):
        trees = random_stand(3)
        trees.loc[1, "fia_species_code"] = 999
        with pytest.raises(ValueError, match="999"):
            available_canopy_fuel(trees)

    def test_pinyon_juniper_raises(self):
        trees = random_stand(2)
        trees["fia_species_code"] = [106, 202]
        with pytest.raises(ValueError, match="PY"):
            available_canopy_fuel(trees)

    def test_eastern_newer_species_raises(self):
        # 833 northern red oak is in the species table but its equation
        # Ids (RO/GC) have no printed equations in the guide.
        trees = random_stand(1)
        trees["fia_species_code"] = [833]
        with pytest.raises(ValueError, match="RO"):
            available_canopy_fuel(trees)


class TestCumulativeFuelFraction:
    """Anchors from the FuelCalc 1.7 guide p. 72 cubics and the worked
    example on pp. 80-81."""

    def test_guide_worked_example(self):
        # PP at ph = 0.25: pw = 2.3637*0.25**2 - 1.3637*0.25**3, which
        # the guide's worked example rounds to 0.13.
        pw = cumulative_fuel_fraction(np.array([122]), np.array([0.25]))
        expected = 2.3637 * 0.25**2 - 1.3637 * 0.25**3
        np.testing.assert_allclose(pw, expected, rtol=1e-12)
        np.testing.assert_allclose(pw, 0.13, atol=0.005)

    def test_douglas_fir_cubic(self):
        pw = cumulative_fuel_fraction(np.array([202]), np.array([0.5]))
        np.testing.assert_allclose(pw, 2.3284 * 0.25 - 1.3284 * 0.125, rtol=1e-12)

    def test_endpoints_all_table_species(self):
        # pw(0) = 0 and pw(1) = 1 for every species in the table (PS
        # sums to 1.0001 in the published coefficients).
        spcd = fuelcalc_species().index.to_numpy()
        zeros = cumulative_fuel_fraction(spcd, np.zeros(len(spcd)))
        ones = cumulative_fuel_fraction(spcd, np.ones(len(spcd)))
        np.testing.assert_array_equal(zeros, 0.0)
        np.testing.assert_allclose(ones, 1.0, atol=2e-4)

    def test_hardwood_and_unknown_are_uniform(self):
        # 746 quaking aspen maps to UN; 999 is absent from the table and
        # falls back to uniform.
        ph = np.array([0.3, 0.3])
        pw = cumulative_fuel_fraction(np.array([746, 999]), ph)
        np.testing.assert_array_equal(pw, ph)

    def test_nondecreasing_within_crown(self):
        # Cumulative fractions never decrease going up the crown.
        # Tolerance covers PS, whose published fit dips ~1e-4 near the
        # crown top (coefficients sum to 1.0001).
        ph_grid = np.linspace(0.0, 1.0, 101)
        for spcd in [122, 202, 108, 81, 15, 746]:
            pw = cumulative_fuel_fraction(np.full(ph_grid.shape, spcd), ph_grid)
            assert (np.diff(pw) >= -1e-3).all(), spcd

    def test_2d_broadcast_and_clipping(self):
        # (n_trees, n_levels) evaluation, with out-of-range ph clipped.
        spcd = np.array([122, 202])
        ph = np.array([[-0.5, 0.25, 1.5], [0.0, 0.5, 1.0]])
        pw = cumulative_fuel_fraction(spcd, ph)
        assert pw.shape == (2, 3)
        assert pw[0, 0] == 0.0  # clipped to ph=0
        np.testing.assert_allclose(pw[0, 2], 1.0, atol=1e-4)  # clipped to 1
        np.testing.assert_allclose(pw[1, 1], 2.3284 * 0.25 - 1.3284 * 0.125, rtol=1e-12)


# 30 m north-up lattice: 4x5 cells anchored at (1000, 5000).
TRANSFORM = (30.0, 0.0, 1000.0, 0.0, -30.0, 5000.0)
SHAPE = (4, 5)
CELL_AREA = 900.0


def stand_on_lattice(n, seed=0):
    """Random stand with stems strictly inside the 4x5 test lattice."""
    trees = random_stand(n, seed)
    rng = np.random.default_rng(seed + 1)
    trees["x"] = rng.uniform(1000.1, 1149.9, n)
    trees["y"] = rng.uniform(4880.1, 4999.9, n)
    return trees


def naive_profile(trees, fuel, n_layers, layer_depth, vertical_distribution):
    """Per-tree, per-layer Python-loop reference for the stem path."""
    profile = np.zeros((n_layers, *SHAPE))
    for i, tree in enumerate(trees.itertuples()):
        col = int(np.floor((tree.x - 1000.0) / 30.0))
        row = int(np.floor((5000.0 - tree.y) / 30.0))
        length = tree.height * tree.crown_ratio
        base = tree.height - length
        for k in range(n_layers):
            z_lo, z_hi = k * layer_depth, (k + 1) * layer_depth
            if length > 0:
                ph_lo = np.clip((z_lo - base) / length, 0.0, 1.0)
                ph_hi = np.clip((z_hi - base) / length, 0.0, 1.0)
            else:
                ph_lo, ph_hi = float(z_lo > base), float(z_hi > base)
            if vertical_distribution == "reinhardt_2006":
                spcd = np.array([tree.fia_species_code, tree.fia_species_code])
                pw = cumulative_fuel_fraction(spcd, np.array([ph_lo, ph_hi]))
                weight = pw[1] - pw[0]
            else:
                weight = ph_hi - ph_lo
            profile[k, row, col] += fuel[i] * weight
    return profile / (CELL_AREA * layer_depth)


class TestVerticalProfile:
    def test_mass_conservation(self):
        trees = stand_on_lattice(300)
        fuel = np.abs(np.random.default_rng(2).normal(10.0, 3.0, len(trees)))
        for vdist in ("reinhardt_2006", "uniform"):
            profile = vertical_profile(
                trees,
                fuel,
                TRANSFORM,
                SHAPE,
                horizontal_distribution="stem",
                vertical_distribution=vdist,
            )
            total = profile.sum() * CELL_AREA * FT_TO_M
            np.testing.assert_allclose(total, fuel.sum(), rtol=1e-3)

    def test_matches_naive_reference(self):
        trees = stand_on_lattice(80, seed=5)
        fuel = np.full(len(trees), 4.0)
        n_layers = int(np.ceil(trees["height"].max() / 0.3048))
        for vdist in ("reinhardt_2006", "uniform"):
            fast = vertical_profile(
                trees,
                fuel,
                TRANSFORM,
                SHAPE,
                n_layers=n_layers,
                horizontal_distribution="stem",
                vertical_distribution=vdist,
            )
            slow = naive_profile(trees, fuel, n_layers, 0.3048, vdist)
            np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-12)

    def test_single_tree_hand_fixture(self):
        # One uniform-crown tree: base 6 m, top 12 m, fuel 9 kg, 3 m
        # layers -> layers 2 and 3 each get half the fuel.
        trees = pd.DataFrame(
            {
                "x": [1045.0],
                "y": [4915.0],
                "height": [12.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
            }
        )
        profile = vertical_profile(
            trees,
            np.array([9.0]),
            TRANSFORM,
            SHAPE,
            n_layers=4,
            layer_depth=3.0,
            vertical_distribution="uniform",
            horizontal_distribution="stem",
        )
        col, row = 1, 2  # x=1045 -> col 1; y=4915 -> row 2
        expected_density = 4.5 / (CELL_AREA * 3.0)
        np.testing.assert_allclose(profile[2, row, col], expected_density)
        np.testing.assert_allclose(profile[3, row, col], expected_density)
        assert profile[0].sum() == 0.0 and profile[1].sum() == 0.0

    def test_zero_crown_length_is_point_mass(self):
        trees = pd.DataFrame(
            {
                "x": [1010.0],
                "y": [4990.0],
                "height": [10.0],
                "crown_ratio": [0.0],
                "fia_species_code": [122],
            }
        )
        profile = vertical_profile(
            trees,
            np.array([5.0]),
            TRANSFORM,
            SHAPE,
            n_layers=4,
            layer_depth=3.0,
            vertical_distribution="uniform",
            horizontal_distribution="stem",
        )
        # Crown base = 10 m -> layer 3 (9-12 m) holds everything.
        np.testing.assert_allclose(profile[3, 0, 0], 5.0 / (CELL_AREA * 3.0))
        np.testing.assert_allclose(profile.sum(), 5.0 / (CELL_AREA * 3.0))

    def test_stem_on_cell_boundary(self):
        # A stem exactly on x=1030 belongs to the east cell (col 1) by
        # the half-open convention.
        trees = pd.DataFrame(
            {
                "x": [1030.0],
                "y": [4970.0],
                "height": [9.0],
                "crown_ratio": [1.0],
                "fia_species_code": [122],
            }
        )
        profile = vertical_profile(
            trees,
            np.array([1.0]),
            TRANSFORM,
            SHAPE,
            n_layers=3,
            layer_depth=3.0,
            vertical_distribution="uniform",
            horizontal_distribution="stem",
        )
        assert profile[:, 1, 1].sum() > 0
        assert profile[:, :, 0].sum() == 0.0

    def test_out_of_bounds_stem_raises(self):
        trees = stand_on_lattice(3)
        trees.loc[1, "x"] = 999.0  # west of the lattice
        with pytest.raises(ValueError, match="outside the lattice"):
            vertical_profile(
                trees,
                np.ones(3),
                TRANSFORM,
                SHAPE,
                horizontal_distribution="stem",
            )

    def test_batching_matches_single_pass(self, monkeypatch):
        import fastfuels_core.canopy_fuel.metrics as m

        trees = stand_on_lattice(150, seed=9)
        fuel = np.full(len(trees), 2.0)
        one_pass = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="stem"
        )
        monkeypatch.setattr(m, "_PROFILE_BATCH_BYTES", 10_000)  # tiny batches
        batched = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="stem"
        )
        np.testing.assert_allclose(batched, one_pass, rtol=1e-12)

    def test_empty_stand(self):
        profile = vertical_profile(
            random_stand(0),
            np.zeros(0),
            TRANSFORM,
            SHAPE,
            n_layers=2,
            horizontal_distribution="stem",
        )
        assert profile.shape == (2, *SHAPE) and profile.sum() == 0.0


def column_profile(values):
    """A (n_layers, 1, 1) profile from a list of layer densities."""
    return np.asarray(values, dtype=float).reshape(-1, 1, 1)


class TestCbdRunningMean:
    def test_hand_fixture(self):
        # Window means over [0,0,2,4,6,0] at 1 m layers, 3 m window:
        # 2/3, 2, 4, 10/3 -> max 4.
        profile = column_profile([0, 0, 2, 4, 6, 0])
        cbd = cbd_running_mean(profile, layer_depth=1.0, window=3.0)
        np.testing.assert_allclose(cbd, [[4.0]])

    def test_window_none_is_max_layer(self):
        profile = column_profile([0, 0, 2, 4, 6, 0])
        cbd = cbd_running_mean(profile, layer_depth=1.0, window=None)
        np.testing.assert_allclose(cbd, [[6.0]])

    def test_shallow_profile_zero_padded(self):
        # Two 3 kg/m3 layers under a 3-layer window dilute to 2.
        profile = column_profile([3.0, 3.0])
        cbd = cbd_running_mean(profile, layer_depth=1.0, window=3.0)
        np.testing.assert_allclose(cbd, [[2.0]])

    def test_smoothed_never_exceeds_max_layer(self):
        rng = np.random.default_rng(3)
        profile = rng.uniform(0, 1, (30, 4, 5))
        smoothed = cbd_running_mean(profile, layer_depth=0.3048, window=3.0)
        unsmoothed = cbd_running_mean(profile, layer_depth=0.3048, window=None)
        assert (smoothed <= unsmoothed + 1e-12).all()


class TestProfileThresholdHeights:
    def test_flat_threshold(self):
        profile = column_profile([0, 0.005, 0.02, 0.05, 0.02, 0.005])
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=None
        )
        np.testing.assert_allclose(cbh, [[2.0]])
        np.testing.assert_allclose(chm, [[5.0]])

    def test_relative_rule_engages(self):
        # max = 0.05 -> effective threshold min(0.005, 0.012) = 0.005,
        # pulling CBH down and CH up relative to the flat rule. Edge
        # layers sit at 0.006, clear of float noise in 0.1 * 0.05.
        profile = column_profile([0, 0.006, 0.02, 0.05, 0.02, 0.006])
        cbh, chm = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=0.1
        )
        np.testing.assert_allclose(cbh, [[1.0]])
        np.testing.assert_allclose(chm, [[6.0]])

    def test_empty_cell_is_nan(self):
        profile = np.zeros((4, 2, 2))
        profile[1, 0, 0] = 0.5
        cbh, chm = profile_threshold_heights(profile, layer_depth=1.0)
        assert np.isnan(cbh[1, 1]) and np.isnan(chm[1, 1])
        np.testing.assert_allclose(cbh[0, 0], 1.0)
        np.testing.assert_allclose(chm[0, 0], 2.0)

    def test_smoothing_window_is_bounded_by_the_fuel(self):
        # Smoothing spreads density past the ends of the canopy, so the
        # scan alone would put canopy in layers 1 and 5, which hold none.
        # The bounds pull both ends back to the layers that do.
        profile = column_profile([0, 0, 0.03, 0.03, 0, 0, 0])
        cbh, chm = profile_threshold_heights(
            profile,
            layer_depth=1.0,
            threshold=0.012,
            relative_fraction=None,
            smoothing_window=3.0,
        )
        np.testing.assert_allclose(cbh, [[2.0]])
        np.testing.assert_allclose(chm, [[4.0]])

    def test_smoothing_can_dilute_a_thin_canopy_to_nothing(self):
        # [0, 0.03, 0, 0] smooths to [0.015, 0.01, 0.01, 0]: truncating
        # the window at the profile floor averages layer 0 over two
        # layers instead of three, so the one layer clearing 0.012 holds
        # no fuel while the one holding fuel falls short. The qualifying
        # span and the fuel span are disjoint, which is no canopy.
        profile = column_profile([0, 0.03, 0, 0])
        cbh_raw, chm_raw = profile_threshold_heights(
            profile, layer_depth=1.0, threshold=0.012, relative_fraction=None
        )
        np.testing.assert_allclose(cbh_raw, [[1.0]])
        np.testing.assert_allclose(chm_raw, [[2.0]])
        cbh_s, chm_s = profile_threshold_heights(
            profile,
            layer_depth=1.0,
            threshold=0.012,
            relative_fraction=None,
            smoothing_window=3.0,
        )
        assert np.isnan(cbh_s).all() and np.isnan(chm_s).all()

    def test_bounds_are_inert_without_smoothing(self):
        # Every qualifying layer holds fuel by construction, so the
        # bounds cannot move an unsmoothed answer.
        rng = np.random.default_rng(11)
        profile = rng.uniform(0, 0.05, (40, 6, 7)) * (
            rng.uniform(0, 1, (40, 6, 7)) > 0.6
        )
        cbh, chm = profile_threshold_heights(profile, layer_depth=0.3048)
        qualifies = profile > 0
        lowest = qualifies.argmax(axis=0) * 0.3048
        highest = (40 - 1 - qualifies[::-1].argmax(axis=0) + 1) * 0.3048
        defined = ~np.isnan(cbh)
        assert (cbh[defined] >= lowest[defined] - 1e-12).all()
        assert (chm[defined] <= highest[defined] + 1e-12).all()

    def test_cbh_never_exceeds_chm(self):
        rng = np.random.default_rng(4)
        profile = rng.uniform(0, 0.05, (40, 6, 7)) * (
            rng.uniform(0, 1, (40, 6, 7)) > 0.7
        )
        cbh, chm = profile_threshold_heights(profile, layer_depth=0.3048)
        defined = ~np.isnan(cbh)
        assert (cbh[defined] < chm[defined]).all()


class TestCanopyFuelLoad:
    def test_hand_fixture(self):
        profile = column_profile([0, 2, 4])
        np.testing.assert_allclose(canopy_fuel_load(profile, layer_depth=0.5), [[3.0]])

    def test_consistent_with_profile_mass(self):
        trees = stand_on_lattice(50, seed=11)
        fuel = np.full(len(trees), 3.0)
        profile = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="stem"
        )
        cfl = canopy_fuel_load(profile)
        np.testing.assert_allclose((cfl * CELL_AREA).sum(), fuel.sum(), rtol=1e-3)


class TestDiskRectOverlap:
    """Layer-4 geometry cross-validation for the analytic circle-rectangle
    intersection against brute-force supersampled integration."""

    @staticmethod
    def brute_force(cx, cy, r, x0, x1, y0, y1, n=2000):
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        gx, gy = np.meshgrid(xs, ys)
        inside = (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
        return inside.mean() * (x1 - x0) * (y1 - y0)

    def test_random_configurations(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            cx, cy = rng.uniform(-5, 5, 2)
            r = rng.uniform(0.3, 6.0)
            x0, y0 = rng.uniform(-6, 4, 2)
            x1, y1 = x0 + rng.uniform(0.5, 6.0), y0 + rng.uniform(0.5, 6.0)
            analytic = disk_rect_overlap_area(
                np.array([cx]),
                np.array([cy]),
                np.array([r]),
                np.array([x0]),
                np.array([x1]),
                np.array([y0]),
                np.array([y1]),
            )[0]
            brute = self.brute_force(cx, cy, r, x0, x1, y0, y1)
            assert abs(analytic - brute) < max(0.01 * np.pi * r * r, 1e-3)

    def test_disk_fully_inside_rect(self):
        area = disk_rect_overlap_area(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([-5.0]),
            np.array([5.0]),
            np.array([-5.0]),
            np.array([5.0]),
        )
        np.testing.assert_allclose(area, np.pi, rtol=1e-12)

    def test_rect_fully_inside_disk(self):
        area = disk_rect_overlap_area(
            np.array([0.0]),
            np.array([0.0]),
            np.array([10.0]),
            np.array([-1.0]),
            np.array([1.0]),
            np.array([-1.0]),
            np.array([1.0]),
        )
        np.testing.assert_allclose(area, 4.0, rtol=1e-12)

    def test_disjoint(self):
        area = disk_rect_overlap_area(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([2.0]),
            np.array([3.0]),
        )
        np.testing.assert_allclose(area, 0.0, atol=1e-12)

    def test_quarter_disk(self):
        # Rectangle covering exactly the first quadrant.
        area = disk_rect_overlap_area(
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.0]),
            np.array([0.0]),
            np.array([5.0]),
            np.array([0.0]),
            np.array([5.0]),
        )
        np.testing.assert_allclose(area, np.pi, rtol=1e-12)

    def test_quadrant_partition_sums_to_disk(self):
        # The four quadrant rectangles partition the disk exactly.
        r = 3.0
        quads = [(-9, 0, -9, 0), (0, 9, -9, 0), (-9, 0, 0, 9), (0, 9, 0, 9)]
        total = sum(
            disk_rect_overlap_area(
                np.array([0.5]),
                np.array([-0.25]),
                np.array([r]),
                np.array([float(x0)]),
                np.array([float(x1)]),
                np.array([float(y0)]),
                np.array([float(y1)]),
            )[0]
            for x0, x1, y0, y1 in quads
        )
        np.testing.assert_allclose(total, np.pi * r * r, rtol=1e-10)


class TestCrownProjected:
    def test_mass_conservation_interior_trees(self):
        # Crowns well inside the lattice: every kg lands somewhere.
        trees = stand_on_lattice(200, seed=13)
        trees["x"] = np.clip(trees["x"], 1010, 1140)
        trees["y"] = np.clip(trees["y"], 4890, 4990)
        fuel = np.abs(np.random.default_rng(14).normal(8.0, 2.0, len(trees)))
        profile = vertical_profile(trees, fuel, TRANSFORM, SHAPE)
        total = profile.sum() * CELL_AREA * FT_TO_M
        np.testing.assert_allclose(total, fuel.sum(), rtol=1e-3)

    def test_boundary_crown_loses_overhang(self):
        # A crown centered on the west boundary loses ~half its fuel.
        trees = pd.DataFrame(
            {
                "x": [1000.5],
                "y": [4940.0],
                "height": [10.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
                "crad": [3.0],
            }
        )
        profile = vertical_profile(
            trees,
            np.array([10.0]),
            TRANSFORM,
            SHAPE,
            vertical_distribution="uniform",
            crown_radius_column="crad",
        )
        total = profile.sum() * CELL_AREA * FT_TO_M
        # Independent segment formula: the lost slice is the circular
        # segment west of the chord 0.5 m from center (r = 3).
        d, r = 0.5, 3.0
        lost = r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)
        expected = 10.0 * (1.0 - lost / (np.pi * r * r))
        np.testing.assert_allclose(total, expected, rtol=1e-3)

    def test_crown_splits_across_cells(self):
        # Crown of radius 3 m centered 1 m east of the x=1030 cell edge
        # puts most fuel in the east cell, the rest in the west cell.
        trees = pd.DataFrame(
            {
                "x": [1031.0],
                "y": [4915.0],
                "height": [9.0],
                "crown_ratio": [1.0],
                "fia_species_code": [122],
                "crad": [3.0],
            }
        )
        profile = vertical_profile(
            trees,
            np.array([6.0]),
            TRANSFORM,
            SHAPE,
            vertical_distribution="uniform",
            crown_radius_column="crad",
        )
        column_totals = profile.sum(axis=0) * CELL_AREA * FT_TO_M
        west, east = column_totals[2, 0], column_totals[2, 1]
        np.testing.assert_allclose(west + east, 6.0, rtol=1e-6)
        assert east > west > 0
        # The west share equals the analytic overlap fraction.
        area_west = disk_rect_overlap_area(
            np.array([1031.0]),
            np.array([4915.0]),
            np.array([3.0]),
            np.array([1000.0]),
            np.array([1030.0]),
            np.array([4910.0]),
            np.array([4940.0]),
        )[0]
        np.testing.assert_allclose(west, 6.0 * area_west / (np.pi * 9.0), rtol=1e-6)

    def test_small_crown_matches_stem(self):
        # A crown fully inside one cell distributes identically to stem.
        trees = stand_on_lattice(40, seed=17)
        trees["x"] = 1000.0 + (np.floor((trees["x"] - 1000.0) / 30.0) * 30.0) + 15.0
        trees["y"] = 5000.0 - (np.floor((5000.0 - trees["y"]) / 30.0) * 30.0) - 15.0
        trees["crad"] = 2.0  # well inside the 30 m cell around its center
        fuel = np.full(len(trees), 5.0)
        projected = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, crown_radius_column="crad"
        )
        stem = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="stem"
        )
        np.testing.assert_allclose(projected, stem, rtol=1e-9, atol=1e-12)


class TestCanopyCover:
    def test_single_crown_matches_analytic_area(self):
        # One crown fully inside one cell: cover = pi r^2 / cell area.
        trees = pd.DataFrame(
            {
                "x": [1045.0],
                "y": [4915.0],
                "height": [10.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
                "crad": [4.0],
            }
        )
        # supersample=200 (0.15 m pixels) so discretization noise sits
        # well under the tolerance and the analytic area is the anchor.
        cover = canopy_cover(
            trees, TRANSFORM, SHAPE, crown_radius_column="crad", supersample=200
        )
        expected = 100.0 * np.pi * 16.0 / CELL_AREA
        np.testing.assert_allclose(cover[2, 1], expected, rtol=0.01)
        assert cover.sum() == cover[2, 1]  # nothing anywhere else

    def test_union_not_sum(self):
        # Two identical crowns cover exactly what one covers.
        one = pd.DataFrame(
            {
                "x": [1045.0],
                "y": [4915.0],
                "height": [10.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
                "crad": [5.0],
            }
        )
        two = pd.concat([one, one], ignore_index=True)
        cover_one = canopy_cover(one, TRANSFORM, SHAPE, crown_radius_column="crad")
        cover_two = canopy_cover(two, TRANSFORM, SHAPE, crown_radius_column="crad")
        np.testing.assert_array_equal(cover_one, cover_two)

    def test_nested_crown_adds_nothing(self):
        base = {
            "x": [1045.0],
            "y": [4915.0],
            "height": [10.0],
            "crown_ratio": [0.5],
            "fia_species_code": [122],
        }
        big = pd.DataFrame({**base, "crad": [6.0]})
        nested = pd.concat(
            [big, pd.DataFrame({**base, "crad": [2.0]})], ignore_index=True
        )
        np.testing.assert_array_equal(
            canopy_cover(big, TRANSFORM, SHAPE, crown_radius_column="crad"),
            canopy_cover(nested, TRANSFORM, SHAPE, crown_radius_column="crad"),
        )

    def test_disjoint_crowns_sum(self):
        trees = pd.DataFrame(
            {
                "x": [1040.0, 1052.0],
                "y": [4915.0, 4915.0],
                "height": [10.0, 10.0],
                "crown_ratio": [0.5, 0.5],
                "fia_species_code": [122, 122],
                "crad": [3.0, 3.0],
            }
        )
        cover = canopy_cover(
            trees, TRANSFORM, SHAPE, crown_radius_column="crad", supersample=200
        )
        expected = 100.0 * 2 * np.pi * 9.0 / CELL_AREA
        np.testing.assert_allclose(cover[2, 1], expected, rtol=0.01)

    def test_cross_validates_against_disk_rect_overlap(self):
        # A crown straddling cells: per-cell rasterized cover matches the
        # analytic overlap areas within supersampling tolerance.
        trees = pd.DataFrame(
            {
                "x": [1031.0],
                "y": [4941.0],
                "height": [10.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
                "crad": [5.0],
            }
        )
        cover = canopy_cover(trees, TRANSFORM, SHAPE, crown_radius_column="crad")
        for row, col in [(1, 0), (1, 1), (2, 0), (2, 1)]:
            x_lo = 1000.0 + col * 30.0
            y_hi = 5000.0 - row * 30.0
            analytic = disk_rect_overlap_area(
                np.array([1031.0]),
                np.array([4941.0]),
                np.array([5.0]),
                np.array([x_lo]),
                np.array([x_lo + 30.0]),
                np.array([y_hi - 30.0]),
                np.array([y_hi]),
            )[0]
            np.testing.assert_allclose(
                cover[row, col], 100.0 * analytic / CELL_AREA, atol=0.35
            )

    def test_strip_chunking_is_invisible(self, monkeypatch):
        import fastfuels_core.canopy_fuel.metrics as m

        trees = stand_on_lattice(60, seed=21)
        whole = canopy_cover(trees, TRANSFORM, SHAPE)
        monkeypatch.setattr(m, "_COVER_STRIP_BYTES", SHAPE[1] * 40**2)  # 1-row strips
        striped = canopy_cover(trees, TRANSFORM, SHAPE)
        np.testing.assert_array_equal(whole, striped)

    def test_bounds_and_empty(self):
        cover = canopy_cover(stand_on_lattice(100, seed=22), TRANSFORM, SHAPE)
        assert (cover >= 0.0).all() and (cover <= 100.0).all()
        assert canopy_cover(random_stand(0), TRANSFORM, SHAPE).sum() == 0.0


def band_template(bands):
    """A griddle-style georeferenced template on the 4x5 test lattice."""
    import rioxarray  # noqa: F401
    from affine import Affine
    import xarray as xr

    ds = xr.Dataset(
        {b: (("y", "x"), np.full(SHAPE, np.nan, dtype=np.float32)) for b in bands},
        coords={
            "y": 5000.0 - 30.0 * (np.arange(SHAPE[0]) + 0.5),
            "x": 1000.0 + 30.0 * (np.arange(SHAPE[1]) + 0.5),
        },
    )
    ds.rio.write_crs("EPSG:32611", inplace=True)
    ds.rio.write_transform(Affine(30.0, 0.0, 1000.0, 0.0, -30.0, 5000.0), inplace=True)
    return ds


class TestComputeCanopyMetrics:
    def hand_stand(self):
        # One tree, cell (row 2, col 1): crown 6-12 m, 9 kg via
        # fuel_column, crown radius 4 m.
        return pd.DataFrame(
            {
                "x": [1045.0],
                "y": [4915.0],
                "height": [12.0],
                "crown_ratio": [0.5],
                "fia_species_code": [122],
                "acf": [9.0],
                "crad": [4.0],
            }
        )

    def test_hand_computed_end_to_end(self):
        ds = compute_canopy_metrics(
            self.hand_stand(),
            band_template(["cbd", "cbh", "chm", "cc", "cfl"]),
            fuel_column="acf",
            crown_radius_column="crad",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            layer_depth=3.0,
            cbd_window=None,
        )
        # 4.5 kg in each of layers 2-3 -> density 4.5/(900*3).
        density = 4.5 / (CELL_AREA * 3.0)
        np.testing.assert_allclose(ds.cbd.values[2, 1], density, rtol=1e-6)
        np.testing.assert_allclose(ds.cbh.values[2, 1], 6.0)
        np.testing.assert_allclose(ds.chm.values[2, 1], 12.0)
        np.testing.assert_allclose(ds.cfl.values[2, 1], 9.0 / CELL_AREA, rtol=1e-6)
        np.testing.assert_allclose(
            ds.cc.values[2, 1], 100.0 * np.pi * 16.0 / CELL_AREA, rtol=0.05
        )
        # Empty cells: 0 for densities and cover, NaN for heights.
        assert ds.cbd.values[0, 0] == 0.0 and ds.cc.values[0, 0] == 0.0
        assert np.isnan(ds.cbh.values[0, 0]) and np.isnan(ds.chm.values[0, 0])

    def test_cc_only_needs_no_biomass_columns(self):
        # Canopy cover alone must not touch the allometry path: no dbh,
        # no fuel column.
        trees = self.hand_stand().drop(columns=["acf"])
        ds = compute_canopy_metrics(
            trees, band_template(["cc"]), crown_radius_column="crad"
        )
        assert float(ds.cc.values[2, 1]) > 0.0

    def test_min_tree_height_filters(self):
        trees = pd.concat([self.hand_stand(), self.hand_stand()], ignore_index=True)
        trees.loc[1, "height"] = 1.0  # below the cutoff
        ds = compute_canopy_metrics(
            trees,
            band_template(["cfl"]),
            fuel_column="acf",
            min_tree_height=2.0,
            horizontal_distribution="stem",
            vertical_distribution="uniform",
        )
        np.testing.assert_allclose(ds.cfl.values[2, 1], 9.0 / CELL_AREA, rtol=1e-6)

    def test_empty_inventory(self):
        ds = compute_canopy_metrics(
            random_stand(0).assign(acf=[], crad=[]),
            band_template(["cbd", "cbh", "cc"]),
            fuel_column="acf",
            crown_radius_column="crad",
        )
        assert (ds.cbd.values == 0).all() and (ds.cc.values == 0).all()
        assert np.isnan(ds.cbh.values).all()

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            compute_canopy_metrics(self.hand_stand(), band_template(["cbd", "bogus"]))

    def test_default_pipeline_mass_consistency(self):
        # Full default path (NSVB + reinhardt + crown_projected) on a
        # real stand kept clear of the boundary: total CFL mass equals
        # total available fuel.
        trees = stand_on_lattice(120, seed=23)
        trees["x"] = np.clip(trees["x"], 1010, 1140)
        trees["y"] = np.clip(trees["y"], 4890, 4990)
        ds = compute_canopy_metrics(trees, band_template(["cfl", "cbd"]))
        fuel = available_canopy_fuel(trees)
        np.testing.assert_allclose(
            (ds.cfl.values * CELL_AREA).sum(), fuel.sum(), rtol=1e-3
        )
        assert (ds.cbd.values >= 0).all()


class TestExcludeHardwoods:
    """Hardwoods leave the bulk-density profile but stay in cover."""

    @staticmethod
    def _dataset():
        import xarray as xr
        from affine import Affine

        d = xr.Dataset(
            {
                k: (("y", "x"), np.full((1, 1), np.nan, np.float32))
                for k in ("cbd", "cbh", "cc")
            },
            coords={"y": [-15.0], "x": [15.0]},
        )
        return d.rio.write_crs("EPSG:5070").rio.write_transform(
            Affine(30, 0, 0, 0, -30, 0)
        )

    @staticmethod
    def _mixed_stand():
        # 202 Douglas-fir (conifer), 351 red alder (hardwood).
        return pd.DataFrame(
            {
                "x": [10.0, 20.0],
                "y": [-10.0, -20.0],
                "fia_species_code": [202, 351],
                "dbh": [35.0, 30.0],
                "height": [20.0, 14.0],
                "crown_ratio": [0.6, 0.7],
            }
        )

    def test_cbd_drops_and_cover_does_not(self):
        trees = self._mixed_stand()
        both = compute_canopy_metrics(trees, self._dataset())
        conifer_only = compute_canopy_metrics(
            trees, self._dataset(), exclude_hardwoods=True
        )
        assert conifer_only["cbd"].values[0, 0] < both["cbd"].values[0, 0]
        assert conifer_only["cc"].values[0, 0] == pytest.approx(both["cc"].values[0, 0])

    def test_inert_on_an_all_conifer_stand(self):
        trees = self._mixed_stand()
        trees = trees[trees["fia_species_code"] == 202]
        a = compute_canopy_metrics(trees, self._dataset())
        b = compute_canopy_metrics(trees, self._dataset(), exclude_hardwoods=True)
        for band in ("cbd", "cbh", "cc"):
            np.testing.assert_array_equal(a[band].values, b[band].values)

    def test_all_hardwood_stand_has_cover_but_no_cbd(self):
        trees = self._mixed_stand()
        trees = trees[trees["fia_species_code"] == 351]
        out = compute_canopy_metrics(trees, self._dataset(), exclude_hardwoods=True)
        assert out["cc"].values[0, 0] > 0.0
        assert out["cbd"].values[0, 0] == 0.0
        assert np.isnan(out["cbh"].values[0, 0])
