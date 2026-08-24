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
        crown_class_adjustment="none",
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


class TestPerBandThresholds:
    """``cbh`` and ``chm`` each read their own scan when asked to."""

    def run(self, hand_stand, **thresholds):
        return compute_canopy_metrics(
            hand_stand,
            band_template(["cbh", "chm"]),
            fuel_column="acf",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            layer_depth=HAND_LAYER_DEPTH,
            crown_class_adjustment="none",
            cbh_relative_fraction=None,
            chm_relative_fraction=None,
            **thresholds,
        )

    def test_a_threshold_only_chm_fails_leaves_cbh_alone(self, hand_stand):
        ds = self.run(hand_stand, cbh_threshold=0.0001, chm_threshold=1.0)
        np.testing.assert_allclose(ds.cbh.values[2, 1], 6.0)
        assert np.isnan(ds.chm.values[2, 1])

    def test_a_threshold_only_cbh_fails_leaves_chm_alone(self, hand_stand):
        ds = self.run(hand_stand, cbh_threshold=1.0, chm_threshold=0.0001)
        assert np.isnan(ds.cbh.values[2, 1])
        np.testing.assert_allclose(ds.chm.values[2, 1], 12.0)


class TestCrownBaseStatisticCbh:
    """The ``"mean"``, ``"percentile"``, and ``"minimum"`` cbh methods swap
    the threshold scan for a plain statistic of the per-tree crown bases,
    reading none of the ``cbh_`` scan settings.
    """

    @staticmethod
    def two_crowns():
        # Both stems in cell (2, 1): crown bases 4 m (8 m tree) and 10 m
        # (20 m tree), with 1 and 3 kg of fuel supplied directly.
        return pd.concat(
            [
                single_tree(x=1045.0, y=4915.0, height=8.0, crown_ratio=0.5, acf=1.0),
                single_tree(x=1045.0, y=4915.0, height=20.0, crown_ratio=0.5, acf=3.0),
            ],
            ignore_index=True,
        )

    def cbh(self, **kwargs):
        ds = compute_canopy_metrics(
            self.two_crowns(),
            band_template(["cbh"]),
            fuel_column="acf",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            crown_class_adjustment="none",
            **kwargs,
        )
        return ds.cbh.values[2, 1]

    def test_mean_is_one_tree_one_vote(self):
        # (4 + 10) / 2 = 7.0, a value no threshold scan of this profile can
        # produce, so it also proves the dispatch switched.
        assert self.cbh(cbh_method="mean") == pytest.approx(7.0)

    def test_weighted_mean_uses_available_fuel(self):
        # (1*4 + 3*10) / (1 + 3) = 8.5 m.
        assert self.cbh(cbh_method="mean", cbh_weight_by_fuel=True) == pytest.approx(
            8.5
        )

    def test_minimum_is_the_lowest_crown_base(self):
        assert self.cbh(cbh_method="minimum") == pytest.approx(4.0)

    def test_percentile_reads_the_percentile(self):
        assert self.cbh(cbh_method="percentile", cbh_percentile=50) == pytest.approx(
            7.0
        )

    def test_percentile_without_a_value_raises(self):
        with pytest.raises(ValueError, match="requires a percentile"):
            self.cbh(cbh_method="percentile")

    def test_an_unknown_cbh_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            self.cbh(cbh_method="bogus")


class TestHeightPercentileChm:
    """``chm_method="height_percentile"`` swaps the threshold scan for a
    per-cell percentile of tree heights, reading none of the ``chm_``
    scan settings.
    """

    @staticmethod
    def two_heights():
        # 10 m and 30 m trees in cell (2, 1).
        return pd.concat(
            [
                single_tree(x=1045.0, y=4915.0, height=10.0, acf=1.0),
                single_tree(x=1045.0, y=4915.0, height=30.0, acf=1.0),
            ],
            ignore_index=True,
        )

    def chm(self, **kwargs):
        ds = compute_canopy_metrics(
            self.two_heights(),
            band_template(["chm"]),
            fuel_column="acf",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            crown_class_adjustment="none",
            chm_method="height_percentile",
            **kwargs,
        )
        return ds.chm.values[2, 1]

    def test_the_hundredth_percentile_is_the_tallest_tree(self):
        assert self.chm(chm_percentile=100.0) == pytest.approx(30.0)

    def test_the_default_percentile_is_the_ninety_ninth(self):
        # 99th of [10, 30] linearly interpolates to 29.8 m.
        assert self.chm() == pytest.approx(29.8)

    def test_an_unknown_chm_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            compute_canopy_metrics(
                self.two_heights(),
                band_template(["chm"]),
                fuel_column="acf",
                crown_class_adjustment="none",
                chm_method="bogus",
            )


class TestLoadOverDepthCbd:
    """``cbd_method="load_over_depth"`` divides canopy fuel load by one of
    the four canopy depths ``cbd_depth`` selects.
    """

    DEPTHS = [
        "canopy_depth",
        "mean_crown_length",
        "biomass_percentile",
        "height_percentile",
    ]

    @staticmethod
    def two_trees():
        # 10 m and 30 m ponderosa in cell (2, 1), 2 kg of fuel each.
        return pd.concat(
            [
                single_tree(x=1045.0, y=4915.0, height=10.0, crown_ratio=0.5, acf=2.0),
                single_tree(x=1045.0, y=4915.0, height=30.0, crown_ratio=0.5, acf=2.0),
            ],
            ignore_index=True,
        )

    def cbd(self, **kwargs):
        # Only cbd is requested, so canopy_depth also proves it computes
        # its own chm - cbh scan rather than reading the other bands.
        ds = compute_canopy_metrics(
            self.two_trees(),
            band_template(["cbd"]),
            fuel_column="acf",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            crown_class_adjustment="none",
            cbd_method="load_over_depth",
            **kwargs,
        )
        return ds.cbd.values

    def test_mean_crown_length_divides_the_load_by_the_mean_crown(self):
        # CFL = 4 kg / 900 m2; mean crown length = mean(5, 15) = 10 m.
        cbd = self.cbd(cbd_depth="mean_crown_length")
        assert cbd[2, 1] == pytest.approx((4.0 / CELL_AREA) / 10.0)

    @pytest.mark.parametrize("depth", DEPTHS)
    def test_every_depth_gives_a_positive_density(self, depth):
        assert self.cbd(cbd_depth=depth)[2, 1] > 0.0

    @pytest.mark.parametrize("depth", DEPTHS)
    def test_an_empty_cell_is_zero_density(self, depth):
        assert self.cbd(cbd_depth=depth)[0, 0] == 0.0

    def test_load_over_depth_is_lower_than_the_running_mean_maximum(self):
        # The average-density convention sits below the running-mean peak.
        load_over = self.cbd(cbd_depth="canopy_depth")[2, 1]
        running_mean = compute_canopy_metrics(
            self.two_trees(),
            band_template(["cbd"]),
            fuel_column="acf",
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            crown_class_adjustment="none",
        ).cbd.values[2, 1]
        assert 0.0 < load_over < running_mean

    def test_an_unknown_cbd_method_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            compute_canopy_metrics(
                self.two_trees(),
                band_template(["cbd"]),
                fuel_column="acf",
                crown_class_adjustment="none",
                cbd_method="bogus",
            )

    def test_an_unknown_cbd_depth_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            self.cbd(cbd_depth="bogus")


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
            crown_class_adjustment="none",
        )
        assert float(ds.cc.values[2, 1]) > 0.0

    def test_an_unknown_band_raises(self, hand_stand):
        with pytest.raises(ValueError, match="bogus"):
            compute_canopy_metrics(
                hand_stand,
                band_template(["cbd", "bogus"]),
                crown_class_adjustment="none",
            )


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
            crown_class_adjustment="none",
        )
        np.testing.assert_allclose(
            ds.cfl.values[2, 1], HAND_FUEL_KG / CELL_AREA, rtol=1e-6
        )

    def test_a_tree_exactly_at_min_tree_height_is_kept(self, hand_stand):
        trees = hand_stand.copy()
        trees["height"] = 2.0
        ds = compute_canopy_metrics(
            trees,
            band_template(["cfl"]),
            fuel_column="acf",
            min_tree_height=2.0,
            horizontal_distribution="stem",
            vertical_distribution="uniform",
            crown_class_adjustment="none",
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
            crown_class_adjustment="none",
        )
        assert (ds.cbd.values == 0).all() and (ds.cc.values == 0).all()

    def test_an_empty_inventory_leaves_undefined_heights(self):
        ds = compute_canopy_metrics(
            random_stand(0).assign(acf=[], crad=[]),
            band_template(["cbh"]),
            fuel_column="acf",
            crown_radius_column="crad",
            crown_class_adjustment="none",
        )
        assert np.isnan(ds.cbh.values).all()


# Brown's equations cover eleven Rocky Mountain conifers; the random
# stand draws from a wider list, so mass-conservation tests of the
# default pipeline take the subset it can price.
BROWN_SPECIES = [202, 122, 73, 17, 108]


def brown_stand(n, seed=0):
    """An interior stand Brown's arm covers, with crown positions."""
    trees = interior_stand(n, seed=seed)
    trees = trees[trees["fia_species_code"].isin(BROWN_SPECIES)]
    return trees.assign(crown_class="C")


def test_the_default_pipeline_conserves_fuel_mass():
    """Brown + Reinhardt cubics + stem attribution, end to end.

    Every stage conserves mass on its own; this checks the chain does
    too, on a stand kept clear of the lattice boundary.
    """
    trees = brown_stand(120, seed=23)
    ds = compute_canopy_metrics(
        trees, band_template(["cfl", "cbd"]), crown_class_column="crown_class"
    )
    np.testing.assert_allclose(
        (ds.cfl.values * CELL_AREA).sum(),
        available_canopy_fuel(trees, crown_class_column="crown_class").sum(),
        rtol=1e-3,
    )


def test_the_default_pipeline_gives_non_negative_bulk_density():
    ds = compute_canopy_metrics(
        interior_stand(120, seed=23),
        band_template(["cbd"]),
        crown_class_adjustment="none",
    )
    assert (ds.cbd.values >= 0).all()


class TestExcludeHardwoods:
    """Hardwoods leave the bulk-density profile but stay in cover.

    Whether excluding one lowers CBD depends on where its crown sits:
    CBD is a maximum over the profile, so a hardwood below the densest
    layer does not set it and dropping it changes nothing. The stand
    here puts the two crowns in the same layers, which is the case the
    flag exists for. Cover counts the hardwood either way, since its
    canopy occupies ground whatever the crown fire model does with it.
    """

    BANDS = ["cbd", "cbh", "cc"]

    @staticmethod
    def mixed_stand():
        """202 Douglas-fir and 351 red alder, crowns overlapping, one cell."""
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
                    height=19.0,
                    crown_ratio=0.7,
                ),
            ],
            ignore_index=True,
        )

    def metrics(self, trees, **kwargs):
        # NSVB, because Brown's arm has no equations for red alder, and
        # inclusion by default, so each test asks for the exclusion it
        # is about rather than inheriting it.
        kwargs.setdefault("exclude_hardwoods", False)
        return compute_canopy_metrics(
            trees,
            band_template(self.BANDS),
            equations="nsvb",
            crown_class_adjustment="none",
            **kwargs,
        )

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
            crown_class_adjustment="none",
            horizontal_distribution="crown_projected",
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
                crown_class_adjustment="none",
                horizontal_distribution="crown_projected",
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
                crown_class_adjustment="none",
            ).cbd.values
            for equations in ("purves", "crookston_stage")
        }
        np.testing.assert_array_equal(stem["purves"], stem["crookston_stage"])


class TestUnknownSpeciesAreNotSilentlyDropped:
    """Excluding hardwoods must not also swallow unpriced species.

    The exclusion reads a flag from the FuelCalc species table, and a
    code the table does not carry has no flag to read. Dropping those
    rows would lose their fuel with no signal, and it is the default
    path, so it would be the common case rather than an opt-in trap.
    """

    @staticmethod
    def stand():
        """One ponderosa and one code no species table carries."""
        return pd.concat(
            [
                single_tree(dbh=30.0, fia_species_code=122, crown_class="C"),
                single_tree(
                    x=1050.0,
                    y=4925.0,
                    dbh=30.0,
                    fia_species_code=9999,
                    crown_class="C",
                ),
            ],
            ignore_index=True,
        )

    def test_it_raises_under_the_exclusion(self):
        with pytest.raises(ValueError, match="9999"):
            compute_canopy_metrics(
                self.stand(),
                band_template(["cfl"]),
                crown_class_column="crown_class",
            )

    def test_it_raises_without_the_exclusion_too(self):
        """The allometry rejects the same code, so neither path is quiet."""
        with pytest.raises(ValueError, match="9999"):
            compute_canopy_metrics(
                self.stand(),
                band_template(["cfl"]),
                crown_class_column="crown_class",
                exclude_hardwoods=False,
            )

    def test_a_known_hardwood_is_still_dropped_quietly(self):
        """Exclusion is for species the table covers and flags as out."""
        trees = pd.concat(
            [
                single_tree(dbh=30.0, fia_species_code=122, crown_class="C"),
                single_tree(
                    x=1050.0,
                    y=4925.0,
                    dbh=30.0,
                    fia_species_code=351,
                    crown_class="C",
                ),
            ],
            ignore_index=True,
        )
        ds = compute_canopy_metrics(
            trees, band_template(["cfl"]), crown_class_column="crown_class"
        )
        assert float(ds.cfl.values[2, 1]) > 0.0
