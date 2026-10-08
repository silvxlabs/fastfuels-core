"""Update a fuel model grid for disturbances, the way LANDFIRE's LFTFC does.

Pixels with a disturbance (an FDist code above 0) are matched to one
Master_Rulesets row each (:mod:`ruleset_lookup`), and take that row's fuel
model. Every other pixel -- undisturbed, or disturbed with no matching rule
-- keeps last year's.

The caller supplies the FDist raster (see :mod:`fdist_builder`) and the map
zone of each pixel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.fuel_models.ruleset_lookup import match_rulesets


# Master_Rulesets' value for "no fuel model code".
_NO_FUEL_MODEL = 9999


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
    """Update last year's fuel model grid for this year's disturbances.

    Parameters
    ----------
    previous : numpy.ndarray
        Last year's grid for the fuel model being updated.
    fuel_model : str
        Its Master_Rulesets column of integer fuel model codes, e.g.
        ``"FBFM40_code"``.
    dist : numpy.ndarray
        FDist code per pixel, e.g. from
        :func:`~fastfuels_core.fuel_models.fdist_builder.build_fdist_raster`.
        Pixels above 0 are the disturbed ones.
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
    updated : numpy.ndarray
        Boolean raster: True where a pixel took a new value from a rule.

    Raises
    ------
    ValueError
        ``fuel_model`` isn't a numeric Master_Rulesets column, or a grid's
        shape differs from ``previous``'s.

    Notes
    -----
    Every grid must be on the same grid as ``previous``. A pixel keeps last
    year's value wherever there is no new one: it wasn't disturbed, it
    matched no rule, or its rule's value is empty or 9999 (no fuel model
    code). The number of disturbed pixels that kept last year's value is
    ``((dist > 0) & ~updated).sum()``.
    """
    if fuel_model not in rules.columns:
        raise ValueError(f"Unknown fuel model column: {fuel_model!r}")

    if not pd.api.types.is_numeric_dtype(rules[fuel_model]):
        raise ValueError(
            f"Fuel model column {fuel_model!r} isn't numeric "
            f"(dtype {rules[fuel_model].dtype}); use a column of integer codes."
        )

    shape = np.shape(previous)
    grids = {"dist": dist, "zone": zone, "fvt": fvt, "fvc": fvc, "fvh": fvh, "bps": bps}
    for name, grid in grids.items():
        if np.shape(grid) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(grid)}, expected {shape} to match previous."
            )

    disturbed = np.asarray(dist) > 0
    new_values, matched = match_rulesets(
        zone=np.asarray(zone)[disturbed],
        evt=np.asarray(fvt)[disturbed],
        dist=np.asarray(dist)[disturbed],
        cover=np.asarray(fvc)[disturbed],
        height=np.asarray(fvh)[disturbed],
        bpsrf=np.asarray(bps)[disturbed],
        rules=rules,
        output_column=fuel_model,
    )
    # A matched rule may still have no value: an empty cell (<NA> when the
    # code column is built with .astype("Int64")) or 9999. Those pixels keep
    # last year's value. The checks go through pandas because <NA> != 9999
    # is <NA>, which NumPy can't use as a mask.
    values = pd.Series(new_values)
    has_value = matched & (values.notna() & (values != _NO_FUEL_MODEL)).to_numpy()

    output = np.array(previous, copy=True)
    output[disturbed] = np.where(has_value, new_values, output[disturbed])
    updated = np.zeros(shape, dtype=bool)
    updated[disturbed] = has_value
    return output, updated
