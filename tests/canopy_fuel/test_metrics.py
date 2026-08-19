"""Tests for :mod:`fastfuels_core.canopy_fuel.metrics`, the stage chain.

Each stage is tested against its own contract in its own module; what is
left for the orchestrator is that it wires them together correctly,
honours the band selection, and applies the two stand-level filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.canopy_fuel.available_fuel import available_canopy_fuel
from fastfuels_core.canopy_fuel.metrics import compute_canopy_metrics
from tests.canopy_fuel.builders import (
    CELL_AREA,
    band_template,
    interior_stand,
    random_stand,
    single_tree,
)

# Hand fixture: one tree in cell (2, 1) with crown 6-12 m, 9 kg of fuel
# supplied directly, and a 4 m crown radius. Three metre layers put half
# the fuel in layer 2 and half in layer 3.
HAND_LAYER_DEPTH = 3.0
HAND_FUEL_KG = 9.0
HAND_CROWN_RADIUS_M = 4.0


@pytest.fixture
def hand_stand():
    return single_tree(acf=HAND_FUEL_KG, crad=HAND_CROWN_RADIUS_M)


@pytest.fixture
def hand_metrics(hand_stand):
    """The hand fixture run through every band, with allometry bypassed."""
    return compute_canopy_metrics(
        hand_stand,
        band_template(["cbd", "cbh", "chm", "cc", "cfl"]),
        fuel_column="acf",
        crown_radius_column="crad",
        horizontal_distribution="stem",
        vertical_distribution="uniform",
        layer_depth=HAND_LAYER_DEPTH,
        cbd_window=None,
    )


class TestHandComputedBands:
    def test_cbd_is_the_density_of_the_layers_the_crown_spans(self, hand_metrics):
        density = (HAND_FUEL_KG / 2) / (CELL_AREA * HAND_LAYER_DEPTH)
        np.testing.assert_allclose(hand_metrics.cbd.values[2, 1], density, rtol=1e-6)

    def test_cbh_is_the_crown_base(self, hand_metrics):
        np.testing.assert_allclose(hand_metrics.cbh.values[2, 1], 6.0)

    def test_chm_is_the_tree_top(self, hand_metrics):
        np.testing.assert_allclose(hand_metrics.chm.values[2, 1], 12.0)

    def test_cfl_is_the_fuel_spread_over_the_cell(self, hand_metrics):
        np.testing.assert_allclose(
            hand_metrics.cfl.values[2, 1], HAND_FUEL_KG / CELL_AREA, rtol=1e-6
        )

    def test_cc_is_the_crown_disk_over_the_cell(self, hand_metrics):
        np.testing.assert_allclose(
            hand_metrics.cc.values[2, 1],
            100.0 * np.pi * HAND_CROWN_RADIUS_M**2 / CELL_AREA,
            rtol=0.05,
        )


class TestEmptyCellConventions:
    """Zero density is physical; no canopy has no base or top."""

    @pytest.mark.parametrize("band", ["cbd", "cfl", "cc"])
    def test_densities_and_cover_are_zero(self, hand_metrics, band):
        assert hand_metrics[band].values[0, 0] == 0.0

    @pytest.mark.parametrize("band", ["cbh", "chm"])
    def test_heights_are_nan(self, hand_metrics, band):
        assert np.isnan(hand_metrics[band].values[0, 0])


class TestBandSelection:
    def test_only_the_requested_bands_are_computed(self, hand_stand):
        """Cover alone must not enter the allometry path.

        The hand stand carries no ``dbh``, so a run that reached the
        biomass equations would raise rather than return.
        """
        ds = compute_canopy_metrics(
            hand_stand.drop(columns=["acf"]),
            band_template(["cc"]),
            crown_radius_column="crad",
        )
        assert float(ds.cc.values[2, 1]) > 0.0

    def test_an_unknown_band_raises(self, hand_stand):
        with pytest.raises(ValueError, match="bogus"):
            compute_canopy_metrics(hand_stand, band_template(["cbd", "bogus"]))


class TestStandFilters:
    def test_trees_below_min_tree_height_are_dropped(self, hand_stand):
        trees = pd.concat([hand_stand, hand_stand], ignore_index=True)
        trees.loc[1, "height"] = 1.0
        ds = compute_canopy_metrics(
            trees,
            band_template(["cfl"]),
            fuel_column="acf",
            min_tree_height=2.0,
            horizontal_distribution="stem",
            vertical_distribution="uniform",
        )
        np.testing.assert_allclose(
            ds.cfl.values[2, 1], HAND_FUEL_KG / CELL_AREA, rtol=1e-6
        )

    def test_an_empty_inventory_leaves_zero_densities(self):
        ds = compute_canopy_metrics(
            random_stand(0).assign(acf=[], crad=[]),
            band_template(["cbd", "cbh", "cc"]),
            fuel_column="acf",
            crown_radius_column="crad",
        )
        assert (ds.cbd.values == 0).all() and (ds.cc.values == 0).all()

    def test_an_empty_inventory_leaves_undefined_heights(self):
        ds = compute_canopy_metrics(
            random_stand(0).assign(acf=[], crad=[]),
            band_template(["cbh"]),
            fuel_column="acf",
            crown_radius_column="crad",
        )
        assert np.isnan(ds.cbh.values).all()


def test_the_default_pipeline_conserves_fuel_mass():
    """NSVB + Reinhardt cubics + crown_projected, end to end.

    Every stage conserves mass on its own; this checks the chain does
    too, on a stand kept clear of the lattice boundary.
    """
    trees = interior_stand(120, seed=23)
    ds = compute_canopy_metrics(trees, band_template(["cfl", "cbd"]))
    np.testing.assert_allclose(
        (ds.cfl.values * CELL_AREA).sum(),
        available_canopy_fuel(trees).sum(),
        rtol=1e-3,
    )


def test_the_default_pipeline_gives_non_negative_bulk_density():
    ds = compute_canopy_metrics(interior_stand(120, seed=23), band_template(["cbd"]))
    assert (ds.cbd.values >= 0).all()


class TestExcludeHardwoods:
    """Hardwoods leave the bulk-density profile but stay in cover.

    The crown fire models CBD feeds are built for conifer crowns, so a
    hardwood understorey would otherwise raise CBD and lower CBH. Its
    canopy still occupies ground, so cover counts it either way.
    """

    BANDS = ["cbd", "cbh", "cc"]

    @staticmethod
    def mixed_stand():
        """202 Douglas-fir (conifer) and 351 red alder (hardwood), one cell."""
        return pd.concat(
            [
                single_tree(
                    x=1040.0,
                    y=4915.0,
                    fia_species_code=202,
                    dbh=35.0,
                    height=20.0,
                    crown_ratio=0.6,
                ),
                single_tree(
                    x=1050.0,
                    y=4925.0,
                    fia_species_code=351,
                    dbh=30.0,
                    height=14.0,
                    crown_ratio=0.7,
                ),
            ],
            ignore_index=True,
        )

    def metrics(self, trees, **kwargs):
        return compute_canopy_metrics(trees, band_template(self.BANDS), **kwargs)

    def test_dropping_the_hardwood_lowers_cbd(self):
        trees = self.mixed_stand()
        both = self.metrics(trees)
        conifer_only = self.metrics(trees, exclude_hardwoods=True)
        assert conifer_only.cbd.values[2, 1] < both.cbd.values[2, 1]

    def test_dropping_the_hardwood_leaves_cover_alone(self):
        trees = self.mixed_stand()
        both = self.metrics(trees)
        conifer_only = self.metrics(trees, exclude_hardwoods=True)
        assert conifer_only.cc.values[2, 1] == pytest.approx(both.cc.values[2, 1])

    @pytest.mark.parametrize("band", BANDS)
    def test_it_is_inert_on_an_all_conifer_stand(self, band):
        trees = self.mixed_stand().query("fia_species_code == 202")
        np.testing.assert_array_equal(
            self.metrics(trees)[band].values,
            self.metrics(trees, exclude_hardwoods=True)[band].values,
        )

    def test_an_all_hardwood_stand_still_has_cover(self):
        trees = self.mixed_stand().query("fia_species_code == 351")
        out = self.metrics(trees, exclude_hardwoods=True)
        assert out.cc.values[2, 1] > 0.0

    def test_an_all_hardwood_stand_has_no_bulk_density(self):
        trees = self.mixed_stand().query("fia_species_code == 351")
        out = self.metrics(trees, exclude_hardwoods=True)
        assert out.cbd.values[2, 1] == 0.0
        assert np.isnan(out.cbh.values[2, 1])


class TestCrownRadiusReachesEveryStage:
    """The radius allometry drives attribution as well as cover.

    ``crown_projected`` spreads a tree's fuel over its crown footprint,
    so the allometry that sizes that footprint changes ``cbd``,
    ``cbh``, ``chm`` and ``cfl`` as surely as it changes ``cc``. The
    tree here sits 2 m inside a cell corner so its disk straddles four
    cells and the split is radius-dependent; a tree well inside one cell
    would put all its weight there under any radius and hide the wiring.
    """

    BANDS = ["cbd", "cfl", "cc"]

    @staticmethod
    def straddling_tree():
        """One ponderosa near the corner of cell (2, 1), with a dbh."""
        return single_tree(x=1032.0, y=4912.0, dbh=30.0, height=20.0)

    def metrics(self, equations):
        return compute_canopy_metrics(
            self.straddling_tree(),
            band_template(self.BANDS),
            crown_radius_equations=equations,
        )

    @pytest.mark.parametrize("band", BANDS)
    def test_the_allometry_changes_every_band_it_feeds(self, band):
        purves = self.metrics("purves")[band].values
        crookston = self.metrics("crookston_stage")[band].values
        assert not np.allclose(purves, crookston, equal_nan=True), (
            f"{band} is unchanged by crown_radius_equations; the "
            f"allometry is not reaching the stage that produces it."
        )

    def test_a_radius_column_still_overrides_the_allometry(self):
        trees = self.straddling_tree().assign(crad=3.0)
        by_column = {
            equations: compute_canopy_metrics(
                trees,
                band_template(self.BANDS),
                crown_radius_column="crad",
                crown_radius_equations=equations,
            ).cbd.values
            for equations in ("purves", "crookston_stage")
        }
        np.testing.assert_array_equal(by_column["purves"], by_column["crookston_stage"])

    def test_the_stem_arm_does_not_read_the_radius(self):
        """It puts the whole tree in one cell, so nothing to size."""
        stem = {
            equations: compute_canopy_metrics(
                self.straddling_tree(),
                band_template(["cbd"]),
                horizontal_distribution="stem",
                crown_radius_equations=equations,
            ).cbd.values
            for equations in ("purves", "crookston_stage")
        }
        np.testing.assert_array_equal(stem["purves"], stem["crookston_stage"])
