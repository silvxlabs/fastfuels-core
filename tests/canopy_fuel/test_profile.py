"""Tests for :mod:`fastfuels_core.canopy_fuel.profile`.

Two quantities: :func:`cumulative_fuel_fraction`, the vertical shape of
one crown, and :func:`vertical_profile`, the per-cell accumulation of
every crown. The accumulation is checked three ways -- against a
per-tree Python reference, against hand-computed single-tree fixtures,
and by conservation of mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area
from fastfuels_core.units import FT_TO_M
from fastfuels_core.canopy_fuel.profile import (
    cumulative_fuel_fraction,
    vertical_profile,
)
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species
from tests.canopy_fuel.builders import (
    CELL_AREA,
    SHAPE,
    TRANSFORM,
    interior_stand,
    random_stand,
    single_tree,
    stand_on_lattice,
)

VERTICAL_DISTRIBUTIONS = ("reinhardt_2006", "uniform")


def stem_profile(trees, fuel, **kwargs):
    """The profile with horizontal attribution switched off."""
    kwargs.setdefault("horizontal_distribution", "stem")
    return vertical_profile(trees, fuel, TRANSFORM, SHAPE, **kwargs)


def total_mass(profile, layer_depth=FT_TO_M):
    """Recover kilograms from a bulk-density profile."""
    return profile.sum() * CELL_AREA * layer_depth


class TestCumulativeFuelFraction:
    """The Reinhardt et al. (2006) cubics, anchored on the FuelCalc guide."""

    def test_matches_the_guide_worked_example(self):
        """PP at ph = 0.25; the guide (pp. 80-81) rounds pw to 0.13."""
        pw = cumulative_fuel_fraction(np.array([122]), np.array([0.25]))
        np.testing.assert_allclose(pw, 2.3637 * 0.25**2 - 1.3637 * 0.25**3, rtol=1e-12)
        np.testing.assert_allclose(pw, 0.13, atol=0.005)

    def test_evaluates_the_douglas_fir_cubic(self):
        pw = cumulative_fuel_fraction(np.array([202]), np.array([0.5]))
        np.testing.assert_allclose(pw, 2.3284 * 0.25 - 1.3284 * 0.125, rtol=1e-12)

    def test_the_crown_base_holds_no_fuel(self):
        spcd = fuelcalc_species().index.to_numpy()
        got = cumulative_fuel_fraction(spcd, np.zeros(len(spcd)))
        np.testing.assert_array_equal(got, 0.0)

    def test_the_crown_top_holds_all_of_it(self):
        """Tolerance covers PS, whose published coefficients sum to 1.0001."""
        spcd = fuelcalc_species().index.to_numpy()
        got = cumulative_fuel_fraction(spcd, np.ones(len(spcd)))
        np.testing.assert_allclose(got, 1.0, atol=2e-4)

    @pytest.mark.parametrize("spcd", [746, 999], ids=["hardwood", "absent"])
    def test_species_without_a_cubic_are_uniform(self, spcd):
        """746 quaking aspen maps to UN; 999 is not in the table at all."""
        ph = np.array([0.3])
        np.testing.assert_array_equal(
            cumulative_fuel_fraction(np.array([spcd]), ph), ph
        )

    @pytest.mark.parametrize("spcd", [122, 202, 108, 81, 15, 746])
    def test_the_fraction_never_decreases_up_the_crown(self, spcd):
        """Tolerance covers PS, whose fit dips ~1e-4 near the crown top."""
        ph = np.linspace(0.0, 1.0, 101)
        pw = cumulative_fuel_fraction(np.full(ph.shape, spcd), ph)
        assert (np.diff(pw) >= -1e-3).all()

    def test_it_broadcasts_over_a_grid_of_heights(self):
        pw = cumulative_fuel_fraction(
            np.array([122, 202]), np.array([[0.0, 0.25, 1.0], [0.0, 0.5, 1.0]])
        )
        assert pw.shape == (2, 3)
        np.testing.assert_allclose(pw[1, 1], 2.3284 * 0.25 - 1.3284 * 0.125, rtol=1e-12)

    def test_heights_outside_the_crown_are_clipped(self):
        pw = cumulative_fuel_fraction(np.array([122, 122]), np.array([-0.5, 1.5]))
        assert pw[0] == 0.0
        np.testing.assert_allclose(pw[1], 1.0, atol=1e-4)


def naive_profile(trees, fuel, n_layers, layer_depth, vertical_distribution):
    """Per-tree, per-layer Python-loop reference for the stem path.

    Deliberately written as nested loops over trees and layers, the way
    the formula reads, so it shares no code with the vectorized
    implementation it checks.
    """
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


class TestVerticalProfileAgainstReferences:
    @pytest.mark.parametrize("vdist", VERTICAL_DISTRIBUTIONS)
    def test_conserves_mass(self, vdist):
        trees = stand_on_lattice(300)
        fuel = np.abs(np.random.default_rng(2).normal(10.0, 3.0, len(trees)))
        profile = stem_profile(trees, fuel, vertical_distribution=vdist)
        np.testing.assert_allclose(total_mass(profile), fuel.sum(), rtol=1e-3)

    @pytest.mark.parametrize("vdist", VERTICAL_DISTRIBUTIONS)
    def test_matches_the_per_tree_python_reference(self, vdist):
        trees = stand_on_lattice(80, seed=5)
        fuel = np.full(len(trees), 4.0)
        n_layers = int(np.ceil(trees["height"].max() / FT_TO_M))
        fast = stem_profile(trees, fuel, n_layers=n_layers, vertical_distribution=vdist)
        slow = naive_profile(trees, fuel, n_layers, FT_TO_M, vdist)
        # The profile is stored float32, so the tolerance is float32-level.
        np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-8)


class TestVerticalPlacement:
    """Single-tree fixtures whose layers can be worked out by hand."""

    def test_a_uniform_crown_splits_evenly_over_the_layers_it_spans(self):
        """Crown 6-12 m, 9 kg, 3 m layers: layers 2 and 3 take half each."""
        profile = stem_profile(
            single_tree(),
            np.array([9.0]),
            n_layers=4,
            layer_depth=3.0,
            vertical_distribution="uniform",
        )
        density = 4.5 / (CELL_AREA * 3.0)
        np.testing.assert_allclose(profile[2, 2, 1], density)
        np.testing.assert_allclose(profile[3, 2, 1], density)

    def test_nothing_lands_below_the_crown_base(self):
        profile = stem_profile(
            single_tree(),
            np.array([9.0]),
            n_layers=4,
            layer_depth=3.0,
            vertical_distribution="uniform",
        )
        assert profile[0].sum() == 0.0 and profile[1].sum() == 0.0

    def test_a_zero_length_crown_is_a_point_mass_at_its_base(self):
        """Crown base = tree top = 10 m, so layer 3 (9-12 m) takes it all."""
        profile = stem_profile(
            single_tree(x=1010.0, y=4990.0, height=10.0, crown_ratio=0.0),
            np.array([5.0]),
            n_layers=4,
            layer_depth=3.0,
            vertical_distribution="uniform",
        )
        expected = 5.0 / (CELL_AREA * 3.0)
        np.testing.assert_allclose(profile[3, 0, 0], expected)
        np.testing.assert_allclose(profile.sum(), expected)


class TestStemAttribution:
    def test_a_stem_on_a_cell_edge_belongs_to_the_east_cell(self):
        """Cell bounds are half-open, so x = 1030 is col 1, not col 0."""
        profile = stem_profile(
            single_tree(x=1030.0, y=4970.0, height=9.0, crown_ratio=1.0),
            np.array([1.0]),
            n_layers=3,
            layer_depth=3.0,
            vertical_distribution="uniform",
        )
        assert profile[:, 1, 1].sum() > 0
        assert profile[:, :, 0].sum() == 0.0

    def test_a_stem_outside_the_lattice_raises(self):
        """A domain-bounded inventory cannot have one; the lattice is wrong."""
        trees = stand_on_lattice(3)
        trees.loc[1, "x"] = 999.0
        with pytest.raises(ValueError, match="outside the lattice"):
            stem_profile(trees, np.ones(3))


class TestCrownProjectedAttribution:
    def test_conserves_mass_for_crowns_inside_the_lattice(self):
        trees = interior_stand(200, seed=13)
        fuel = np.abs(np.random.default_rng(14).normal(8.0, 2.0, len(trees)))
        profile = vertical_profile(
            trees, fuel, TRANSFORM, SHAPE, horizontal_distribution="crown_projected"
        )
        np.testing.assert_allclose(total_mass(profile), fuel.sum(), rtol=1e-3)

    def test_a_crown_fully_inside_one_cell_matches_stem_attribution(self):
        trees = stand_on_lattice(40, seed=17)
        trees["x"] = 1000.0 + np.floor((trees["x"] - 1000.0) / 30.0) * 30.0 + 15.0
        trees["y"] = 5000.0 - np.floor((5000.0 - trees["y"]) / 30.0) * 30.0 - 15.0
        trees["crad"] = 2.0  # well inside the 30 m cell around its center
        fuel = np.full(len(trees), 5.0)
        np.testing.assert_allclose(
            vertical_profile(
                trees,
                fuel,
                TRANSFORM,
                SHAPE,
                crown_radius_column="crad",
                horizontal_distribution="crown_projected",
            ),
            stem_profile(trees, fuel),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_a_crown_overhanging_the_boundary_loses_its_overhang(self):
        """Checked against the circular-segment formula, not our own geometry."""
        trees = single_tree(x=1000.5, y=4940.0, height=10.0, crad=3.0)
        profile = vertical_profile(
            trees,
            np.array([10.0]),
            TRANSFORM,
            SHAPE,
            vertical_distribution="uniform",
            crown_radius_column="crad",
            horizontal_distribution="crown_projected",
        )
        d, r = 0.5, 3.0
        lost = r * r * np.arccos(d / r) - d * np.sqrt(r * r - d * d)
        expected = 10.0 * (1.0 - lost / (np.pi * r * r))
        np.testing.assert_allclose(total_mass(profile), expected, rtol=1e-3)


class TestCrownStraddlingACellEdge:
    """One 3 m crown centered 1 m east of the x = 1030 edge, 6 kg."""

    @staticmethod
    def column_totals():
        trees = single_tree(x=1031.0, y=4915.0, height=9.0, crown_ratio=1.0, crad=3.0)
        profile = vertical_profile(
            trees,
            np.array([6.0]),
            TRANSFORM,
            SHAPE,
            vertical_distribution="uniform",
            crown_radius_column="crad",
            horizontal_distribution="crown_projected",
        )
        return profile.sum(axis=0) * CELL_AREA * FT_TO_M

    def test_the_two_cells_together_hold_the_whole_tree(self):
        totals = self.column_totals()
        np.testing.assert_allclose(totals[2, 0] + totals[2, 1], 6.0, rtol=1e-6)

    def test_the_cell_holding_the_stem_takes_the_larger_share(self):
        totals = self.column_totals()
        assert totals[2, 1] > totals[2, 0] > 0

    def test_the_split_is_the_analytic_overlap_fraction(self):
        west = self.column_totals()[2, 0]
        area_west = disk_rect_overlap_area(
            *(
                np.array([v])
                for v in (1031.0, 4915.0, 3.0, 1000.0, 1030.0, 4910.0, 4940.0)
            )
        )[0]
        np.testing.assert_allclose(west, 6.0 * area_west / (np.pi * 9.0), rtol=1e-6)


class TestProfileEdgeCases:
    def test_an_empty_stand_gives_an_empty_profile(self):
        profile = stem_profile(random_stand(0), np.zeros(0), n_layers=2)
        assert profile.shape == (2, *SHAPE)
        assert profile.sum() == 0.0

    def test_batching_does_not_change_the_result(self, monkeypatch):
        """Scatter-adds commute, so the batch size is invisible to float32."""
        import fastfuels_core.canopy_fuel.profile as m

        trees = stand_on_lattice(150, seed=9)
        fuel = np.full(len(trees), 2.0)
        one_pass = stem_profile(trees, fuel)
        monkeypatch.setattr(m, "_PROFILE_BATCH_BYTES", 10_000)
        np.testing.assert_allclose(stem_profile(trees, fuel), one_pass, rtol=1e-6)

    def test_an_unknown_vertical_distribution_raises(self):
        with pytest.raises(ValueError, match="vertical_distribution"):
            stem_profile(random_stand(0), np.zeros(0), vertical_distribution="bogus")

    def test_an_unknown_horizontal_distribution_raises(self):
        with pytest.raises(ValueError, match="horizontal_distribution"):
            vertical_profile(
                random_stand(0),
                np.zeros(0),
                TRANSFORM,
                SHAPE,
                horizontal_distribution="bogus",
            )

    def test_a_rotated_transform_raises(self):
        with pytest.raises(ValueError, match="Rotated"):
            vertical_profile(
                random_stand(0),
                np.zeros(0),
                (30.0, 1.0, 1000.0, 0.0, -30.0, 5000.0),
                SHAPE,
            )
