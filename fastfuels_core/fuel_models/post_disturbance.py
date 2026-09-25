"""Post-disturbance fuel models, updated the way LANDFIRE's LFTFC does.

Chains the other ``fuel_models`` stages into one call:

1. LDist codes to FDist codes (:mod:`disturbance_crosswalk`)
2. Pixels with an FDist code above 0 are the disturbed ones
3. Map zone for each cell (:mod:`lf_zone_lookup`)
4. Match each disturbed pixel to one Master_Rulesets row
   (:mod:`ruleset_lookup`)
5. Everywhere without a new value -- undisturbed pixels, and disturbed
   pixels with no matching rule -- keeps last year's fuel model.

Every stage is usable on its own; this module only orders them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from affine import Affine

from fastfuels_core.fuel_models.disturbance_crosswalk import (
    ldist_raster_to_fdist_raster,
)
from fastfuels_core.fuel_models.lf_zone_lookup import lookup_lf_zones
from fastfuels_core.fuel_models.ruleset_lookup import RulesetIndex, match_rulesets


@dataclass(frozen=True)
class FuelModelUpdate:
    """The outcome of :func:`update_fuel_models`.

    Attributes
    ----------
    output : numpy.ndarray
        The updated fuel model grid, same shape and dtype as last year's.
    n_disturbed : int
        Number of pixels with a disturbance FDist can encode, i.e. the
        pixels the rules were applied to.
    n_unmatched : int
        Disturbed pixels with no matching Master_Rulesets row. They kept
        last year's fuel model.
    n_unmapped : int
        Pixels whose LDist code isn't in the LDist attribute table. They
        are treated as undisturbed and kept last year's fuel model.
    warnings : dict of int to str
        LDist code -> reason, for each code present that is a real
        disturbance FDist can't encode (e.g. Herbicide). Those pixels are
        treated as undisturbed.
    """

    output: np.ndarray
    n_disturbed: int
    n_unmatched: int
    n_unmapped: int
    warnings: dict[int, str]


def update_fuel_models(
    previous: np.ndarray,
    *,
    fuel_model: str,
    ldist: np.ndarray,
    ldist_nodata: float | None = None,
    fvt: np.ndarray,
    fvc: np.ndarray,
    fvh: np.ndarray,
    bps: np.ndarray,
    transform: Affine,
    crs,
    index: RulesetIndex,
) -> FuelModelUpdate:
    """Update last year's fuel model grid for this year's disturbances.

    Parameters
    ----------
    previous : numpy.ndarray
        Last year's grid for the fuel model being updated.
    fuel_model : str
        Its Master_Rulesets column name, e.g. ``"FBFM40"``.
    ldist : numpy.ndarray
        LANDFIRE Limited Annual Disturbance codes.
    ldist_nodata : int or float, optional
        LDist's declared nodata value, e.g. ``ldist.rio.nodata``. Those
        pixels are treated as undisturbed rather than counted as unmapped.
    fvt, fvc, fvh, bps : numpy.ndarray
        LANDFIRE FVT, FVC, FVH and BPS raster values.
    transform : affine.Affine
        The grid's transform, e.g. ``ldist.rio.transform()``.
    crs
        The grid's CRS, e.g. ``ldist.rio.crs``.
    index : RulesetIndex
        From :func:`~fastfuels_core.fuel_models.ruleset_lookup.build_ruleset_index`.
        Build once, reuse across calls.

    Returns
    -------
    FuelModelUpdate
        The updated grid, plus counts of disturbed, unmatched and unmapped
        pixels and warnings for disturbances that couldn't be encoded.

    Raises
    ------
    ValueError
        A grid's shape differs from ``ldist``'s, or ``fuel_model`` isn't a
        Master_Rulesets column.

    Notes
    -----
    Every grid must be on the same grid as ``ldist``. A pixel keeps last
    year's value wherever there is no new one: it wasn't disturbed, its
    disturbance has no FDist encoding, it matched no rule, or its matched
    row has no value for ``fuel_model``.
    """
    if fuel_model not in index.table.columns:
        raise ValueError(f"Unknown fuel model column: {fuel_model!r}")

    shape = np.shape(ldist)
    grids = {"previous": previous, "fvt": fvt, "fvc": fvc, "fvh": fvh, "bps": bps}
    for name, grid in grids.items():
        if np.shape(grid) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(grid)}, expected {shape} to match ldist."
            )

    crosswalk = ldist_raster_to_fdist_raster(np.asarray(ldist), nodata=ldist_nodata)
    disturbed = crosswalk.fdist > 0
    n_disturbed = int(np.count_nonzero(disturbed))
    output = np.array(previous, copy=True)
    n_unmatched = 0

    # With nothing disturbed there is nothing to match, and the zone lookup
    # can be skipped entirely.
    if n_disturbed:
        zone = lookup_lf_zones(transform, shape, crs)
        result = match_rulesets(
            zone=zone[disturbed],
            evt=np.asarray(fvt)[disturbed],
            dist=crosswalk.fdist[disturbed],
            cover=np.asarray(fvc)[disturbed],
            height=np.asarray(fvh)[disturbed],
            bpsrf=np.asarray(bps)[disturbed],
            index=index,
            output_column=fuel_model,
        )
        n_unmatched = result.n_unmatched
        has_value = result.matched & ~pd.isna(result.output)
        output[disturbed] = np.where(has_value, result.output, output[disturbed])

    return FuelModelUpdate(
        output=output,
        n_disturbed=n_disturbed,
        n_unmatched=n_unmatched,
        n_unmapped=crosswalk.n_unmapped,
        warnings=crosswalk.warnings,
    )
