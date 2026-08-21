"""Chain the canopy fuel stages into a georeferenced metric Dataset.

Each stage lives in its own module, named for the quantity it produces:
:mod:`crown_radius`, :mod:`available_fuel`, :mod:`profile`,
:mod:`bulk_density`, :mod:`canopy_height`, :mod:`fuel_load` and
:mod:`cover`. Every stage is public and usable on its own; this
module only wires them together and writes the result into a
caller-provided lattice.

The caller owns the output lattice — nothing here creates grids or
chooses resolutions. All per-tree math is vectorized; trees are never
iterated in Python.

Expected tree columns follow the v2 FastFuels inventory convention:
``x``, ``y``, ``height``, ``crown_ratio``, and, depending on options,
``dbh``, ``fia_species_code``, and caller-named biomass / crown radius
columns. Units: meters, centimeters (dbh), kilograms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr

from fastfuels_core.canopy_fuel.available_fuel import (
    FUELCALC_CROWN_CLASS_ADJUSTMENT,
    available_canopy_fuel,
)
from fastfuels_core.canopy_fuel.bulk_density import FUELCALC_EDGE, cbd_running_mean
from fastfuels_core.canopy_fuel.cover import canopy_cover
from fastfuels_core.canopy_fuel.canopy_height import (
    MEAN_CROWN_BASE_METHOD,
    mean_crown_base_height,
    profile_threshold_heights,
    validate_cbh_method,
)
from fastfuels_core.canopy_fuel.fuel_load import canopy_fuel_load
from fastfuels_core.canopy_fuel.profile import FUELCALC_LAYER_DEPTH, vertical_profile
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species

# Five one-foot layers, the running-mean depth FuelCalc smooths over.
FUELCALC_WINDOW = 5 * FUELCALC_LAYER_DEPTH

KNOWN_BANDS = ("cbd", "cbh", "chm", "cc", "cfl")
PROFILE_BANDS = ("cbd", "cbh", "chm", "cfl")


def _conifers_only(trees: pd.DataFrame) -> pd.DataFrame:
    """Drop broadleaf species, by the species table's CBD inclusion flag.

    Species the table does not carry raise rather than dropping out.
    They have no inclusion flag to read, so silently excluding them
    would lose their fuel without saying so, and the caller would have
    no way to tell a hardwood the flag excluded from a species code
    this package cannot price at all.
    """
    spcd = trees["fia_species_code"].to_numpy()
    species = fuelcalc_species()
    unknown = np.setdiff1d(spcd, species.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FuelCalc "
            f"species table, so they have no canopy fuel inclusion flag. "
            f"Drop them before calling, or pass exclude_hardwoods=False "
            f"to keep every species in the profile."
        )
    conifer = species["INCL_CBD"].reindex(spcd)
    return trees[(conifer == "Yes").to_numpy()]


def compute_canopy_metrics(
    trees: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    fuel_column: str | None = None,
    crown_radius_column: str | None = None,
    crown_radius_equations: str = "crookston_stage",
    cover_method: str = "crown_overlap",
    cover_height_threshold: float = 2.0,
    equations: str = "brown_1978",
    crown_class_adjustment: str = FUELCALC_CROWN_CLASS_ADJUSTMENT,
    crown_class_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
    branchwood_size_partition: str = "brown_proportions",
    min_tree_height: float = 0.0,
    exclude_hardwoods: bool = True,
    layer_depth: float = FUELCALC_LAYER_DEPTH,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "stem",
    cbd_method: str = "maximum_running_mean",
    cbd_window: float | None = FUELCALC_WINDOW,
    cbd_window_edge: str = FUELCALC_EDGE,
    cbh_method: str = "bulk_density_threshold",
    cbh_threshold: float = 0.012,
    cbh_relative_fraction: float | None = 0.1,
    cbh_smoothing_window: float | None = FUELCALC_WINDOW,
    cbh_smoothing_edge: str = FUELCALC_EDGE,
    chm_method: str = "bulk_density_threshold",
    chm_threshold: float = 0.012,
    chm_relative_fraction: float | None = 0.1,
    chm_smoothing_window: float | None = FUELCALC_WINDOW,
    chm_smoothing_edge: str = FUELCALC_EDGE,
) -> xr.Dataset:
    """Fill a georeferenced Dataset with canopy fuel metrics.

    The caller provides ``dataset`` with one variable per requested band
    — any of ``cbd``, ``cbh``, ``chm``, ``cc``, ``cfl`` — on a north-up
    lattice with CRS and transform set via rioxarray. Only the variables
    present are computed. Tree coordinates must be in the dataset's CRS;
    only live trees should be passed. Trees shorter than
    ``min_tree_height`` (m) are excluded.

    ``exclude_hardwoods`` drops broadleaf species from the bulk-density
    profile — ``cbd``, ``cbh``, ``chm`` and ``cfl`` — while leaving
    ``cc`` computed over every tree, since broadleaf canopy still
    occupies ground even where it is not treated as crown-fire fuel.
    The crown fire models CBD feeds are built for conifer crowns, so a
    hardwood understorey would otherwise raise CBD and lower CBH.

    The keyword arguments are the stages' own, grouped by the stage
    that reads them; see each stage for semantics.

    :func:`~fastfuels_core.canopy_fuel.available_fuel.available_canopy_fuel`
        ``fuel_column``, ``equations``, ``crown_class_adjustment``,
        ``crown_class_column``, ``foliage_fraction``,
        ``branchwood_fraction``, ``branchwood_size_partition``.
    :func:`~fastfuels_core.canopy_fuel.profile.vertical_profile`
        ``layer_depth``, ``vertical_distribution``,
        ``horizontal_distribution``, and under ``crown_projected`` the
        crown radius pair below.
    :func:`~fastfuels_core.canopy_fuel.bulk_density.cbd_running_mean`
        ``cbd_window``, ``cbd_window_edge``.
    :func:`~fastfuels_core.canopy_fuel.canopy_height.profile_threshold_heights`
        ``cbh_threshold``, ``cbh_relative_fraction``,
        ``cbh_smoothing_window``, ``cbh_smoothing_edge``, and the same
        four under ``chm_``. One scan produces both heights when the
        two sets agree, which they do by default; give ``chm`` its own
        settings and it is read off a second scan. ``cbh_method``
        selects the base-height stage: ``"bulk_density_threshold"``
        (the default, the scan above) or ``"mean_crown_base"``, the
        fuel-weighted mean per-tree crown base
        (:func:`~fastfuels_core.canopy_fuel.canopy_height.mean_crown_base_height`),
        which ignores the ``cbh_`` scan settings.
    :func:`~fastfuels_core.canopy_fuel.cover.canopy_cover`
        ``cover_method``, ``cover_height_threshold``, and the crown
        radius pair.
    :func:`~fastfuels_core.canopy_fuel.crown_radius.max_crown_radius`
        ``crown_radius_column``, ``crown_radius_equations`` — read by
        cover and by the ``crown_projected`` profile.

    Each stage is an independent choice, so a run is a point in that
    space rather than one fixed method. Every default is FuelCalc 1.7's,
    which is the parameterization
    ``tests/canopy_fuel/test_fuelcalc_comparison.py`` verifies against
    the numbers FuelCalc itself reported. ``equations="nsvb"`` with
    ``branchwood_size_partition="none"`` reaches every species NSVB
    prices, free of the FuelCalc species table (with
    ``exclude_hardwoods=False``, which also reads that table); a caller
    with tree positions finer than a crown takes ``horizontal_distribution="crown_projected"`` and
    ``cover_method="crown_union"``, which use those positions where
    FuelCalc's stand-table methods cannot.

    ``crown_class_adjustment`` defaults to ``"fuelcalc_table"`` and so
    requires ``crown_class_column``; see :func:`available_canopy_fuel`
    for why it will not fall back silently.

    Cells with no canopy come back as 0 for ``cbd``, ``cfl``, and ``cc``
    (zero density is physical) and NaN for ``cbh`` and ``chm`` (no
    canopy has no base or top); masking is the caller's choice.

    Returns
    -------
    xarray.Dataset
        ``dataset`` with its band variables filled in place.
    """
    bands = set(dataset.data_vars)
    unknown = bands - set(KNOWN_BANDS)
    if unknown:
        raise ValueError(
            f"Unknown dataset variable(s) {sorted(unknown)}; expected a "
            f"subset of {sorted(KNOWN_BANDS)}."
        )
    validate_cbh_method(cbh_method)

    t = dataset.rio.transform()
    transform = (t.a, t.b, t.c, t.d, t.e, t.f)
    shape = (dataset.sizes["y"], dataset.sizes["x"])

    trees = trees[trees["height"].to_numpy() >= min_tree_height]

    # Hardwoods are dropped from the bulk-density profile only, never
    # from cover: broadleaf canopy still occupies ground.
    fuel_trees = _conifers_only(trees) if exclude_hardwoods else trees

    if bands & set(PROFILE_BANDS):
        fuel = available_canopy_fuel(
            fuel_trees,
            fuel_column=fuel_column,
            equations=equations,
            crown_class_adjustment=crown_class_adjustment,
            crown_class_column=crown_class_column,
            foliage_fraction=foliage_fraction,
            branchwood_fraction=branchwood_fraction,
            branchwood_size_partition=branchwood_size_partition,
        )
        profile = vertical_profile(
            fuel_trees,
            fuel,
            transform,
            shape,
            layer_depth=layer_depth,
            vertical_distribution=vertical_distribution,
            horizontal_distribution=horizontal_distribution,
            crown_radius_column=crown_radius_column,
            crown_radius_equations=crown_radius_equations,
        )
        if "cbd" in bands:
            dataset["cbd"].data[...] = cbd_running_mean(
                profile,
                layer_depth=layer_depth,
                window=cbd_window,
                edge=cbd_window_edge,
            )
        scans = {}

        def threshold_heights(threshold, relative_fraction, window, edge):
            key = (threshold, relative_fraction, window, edge)
            if key not in scans:
                scans[key] = profile_threshold_heights(
                    profile,
                    layer_depth=layer_depth,
                    threshold=threshold,
                    relative_fraction=relative_fraction,
                    smoothing_window=window,
                    smoothing_edge=edge,
                )
            return scans[key]

        if "cbh" in bands:
            if cbh_method == MEAN_CROWN_BASE_METHOD:
                dataset["cbh"].data[...] = mean_crown_base_height(
                    fuel_trees, fuel, transform, shape
                )
            else:
                dataset["cbh"].data[...] = threshold_heights(
                    cbh_threshold,
                    cbh_relative_fraction,
                    cbh_smoothing_window,
                    cbh_smoothing_edge,
                )[0]
        if "chm" in bands:
            dataset["chm"].data[...] = threshold_heights(
                chm_threshold,
                chm_relative_fraction,
                chm_smoothing_window,
                chm_smoothing_edge,
            )[1]
        if "cfl" in bands:
            dataset["cfl"].data[...] = canopy_fuel_load(
                profile, layer_depth=layer_depth
            )

    if "cc" in bands:
        dataset["cc"].data[...] = canopy_cover(
            trees,
            transform,
            shape,
            crown_radius_column=crown_radius_column,
            crown_radius_equations=crown_radius_equations,
            method=cover_method,
            height_threshold=cover_height_threshold,
        )

    return dataset
