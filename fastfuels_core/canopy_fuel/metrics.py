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

import pandas as pd
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr

from fastfuels_core.canopy_fuel.available_fuel import (
    NO_CROWN_CLASS_ADJUSTMENT,
    available_canopy_fuel,
)
from fastfuels_core.canopy_fuel.bulk_density import (
    FUELCALC_EDGE,
    SLAB_EDGE,
    cbd_running_mean,
)
from fastfuels_core.canopy_fuel.cover import canopy_cover
from fastfuels_core.canopy_fuel.canopy_height import profile_threshold_heights
from fastfuels_core.canopy_fuel.fuel_load import canopy_fuel_load
from fastfuels_core.canopy_fuel.profile import FT_TO_M, vertical_profile
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species

KNOWN_BANDS = ("cbd", "cbh", "chm", "cc", "cfl")
PROFILE_BANDS = ("cbd", "cbh", "chm", "cfl")


def _conifers_only(trees: pd.DataFrame) -> pd.DataFrame:
    """Drop broadleaf species, by the species table's CBD inclusion flag."""
    conifer = fuelcalc_species()["INCL_CBD"].reindex(
        trees["fia_species_code"].to_numpy()
    )
    return trees[(conifer == "Yes").to_numpy()]


def compute_canopy_metrics(
    trees: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    fuel_column: str | None = None,
    crown_radius_column: str | None = None,
    crown_radius_equations: str = "purves",
    cover_method: str = "crown_union",
    cover_height_threshold: float = 2.0,
    equations: str = "nsvb",
    crown_class_adjustment: str = NO_CROWN_CLASS_ADJUSTMENT,
    crown_class_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
    min_tree_height: float = 0.0,
    exclude_hardwoods: bool = False,
    layer_depth: float = FT_TO_M,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "crown_projected",
    cbd_window: float | None = 3.0,
    cbh_threshold: float = 0.012,
    cbh_relative_fraction: float | None = 0.1,
    threshold_smoothing_window: float | None = None,
    cbd_window_edge: str = SLAB_EDGE,
    threshold_smoothing_edge: str = FUELCALC_EDGE,
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

    Chains the public stages, one module each:
    :mod:`available_fuel` → :mod:`profile` → :mod:`bulk_density` /
    :mod:`canopy_height` / :mod:`fuel_load`, plus :mod:`cover`. See each
    stage for parameter semantics.

    Each stage is an independent choice, so a run is a point in that
    space rather than one fixed method. The defaults are
    FastFuels-native (NSVB biomass, a 3.0 m bulk-density window,
    unsmoothed heights); ``equations="brown_1978"`` with
    ``cbd_window=1.524`` and ``threshold_smoothing_window=1.524`` moves
    the biomass and reduction stages onto FuelCalc's.

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
        )
        if "cbd" in bands:
            dataset["cbd"].data[...] = cbd_running_mean(
                profile,
                layer_depth=layer_depth,
                window=cbd_window,
                edge=cbd_window_edge,
            )
        if bands & {"cbh", "chm"}:
            cbh, chm = profile_threshold_heights(
                profile,
                layer_depth=layer_depth,
                threshold=cbh_threshold,
                relative_fraction=cbh_relative_fraction,
                smoothing_window=threshold_smoothing_window,
                smoothing_edge=threshold_smoothing_edge,
            )
            if "cbh" in bands:
                dataset["cbh"].data[...] = cbh
            if "chm" in bands:
                dataset["chm"].data[...] = chm
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
