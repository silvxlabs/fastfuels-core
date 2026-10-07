"""Update a fuel model grid from LANDFIRE's Master_Rulesets, the way LFTFC does.

Every pixel is matched to one Master_Rulesets row (:mod:`ruleset_lookup`)
by its map zone, vegetation type, FDist code, cover, height and biophysical
setting, and takes that row's fuel model. A pixel with no matching rule
keeps last year's fuel model code.

The caller supplies the FDist raster (see :mod:`fdist_builder`) and the map
zone of each pixel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.fuel_models.ruleset_lookup import match_rulesets


def update_fuel_models(
    previous: np.ndarray,
    *,
    fuel_model: str,
    dist: np.ndarray,
    zone: np.ndarray,
    fvt: np.ndarray,
    fvc: np.ndarray,
    fvh: np.ndarray,
    bps: np.ndarray,
    rules: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Update last year's fuel model grid with Master_Rulesets.

    Parameters
    ----------
    previous : numpy.ndarray
        Last year's grid for the fuel model being updated.
    fuel_model : str
        Its Master_Rulesets column name, e.g. ``"FBFM40"``.
    dist : numpy.ndarray
        FDist code per pixel (0 for no disturbance), e.g. from
        :func:`~fastfuels_core.fuel_models.fdist_builder.build_fdist_raster`.
    zone : numpy.ndarray
        LANDFIRE map zone per pixel.
    fvt, fvc, fvh, bps : numpy.ndarray
        LANDFIRE FVT, FVC, FVH and BPS raster values.
    rules : pandas.DataFrame
        The Master_Rulesets table.

    Returns
    -------
    output : numpy.ndarray
        The updated grid, same shape and dtype as ``previous``.
    matched : numpy.ndarray
        Boolean raster: True where a pixel's value came from a rule, False
        where it kept last year's.

    Raises
    ------
    ValueError
        ``fuel_model`` isn't a Master_Rulesets column, or a grid's shape
        differs from ``previous``'s.

    Notes
    -----
    Every grid must be on the same grid as ``previous``. A pixel keeps last
    year's value only where it matched no rule.
    """
    if fuel_model not in rules.columns:
        raise ValueError(f"Unknown fuel model column: {fuel_model!r}")

    shape = np.shape(previous)
    grids = {"dist": dist, "zone": zone, "fvt": fvt, "fvc": fvc, "fvh": fvh, "bps": bps}
    for name, grid in grids.items():
        if np.shape(grid) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(grid)}, expected {shape} to match previous."
            )

    new_values, matched = match_rulesets(
        zone=zone,
        evt=fvt,
        dist=dist,
        cover=fvc,
        height=fvh,
        bpsrf=bps,
        rules=rules,
        output_column=fuel_model,
    )

    output = np.array(previous, copy=True)
    output[matched] = new_values[matched]
    return output, matched
