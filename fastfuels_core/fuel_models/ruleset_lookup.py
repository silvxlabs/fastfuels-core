"""
Master_Rulesets lookup
======================

Rule matching against LANDFIRE's Master_Rulesets table, the rules LFTFC
uses to assign fuel models after a disturbance. Given per-pixel Zone, EVT,
DIST, Cover, Height and BPS codes, finds each pixel's matching
Master_Rulesets row and returns the requested output column (e.g. FBFM13,
FBFM40, FCCS).

The table is not packaged (about 626K rows); the caller loads it and
passes it to :func:`match_rulesets`.

Notes
-----
(Zone, EVT, DIST) is an exact-match key that narrows each pixel to a
handful of candidate rows. A candidate qualifies when the pixel's Cover
and Height fall within its ``Cover_Low``/``Cover_High`` and
``Height_Low``/``Height_High`` ranges (inclusive), and the pixel's BPS
value equals its ``BPSRF``, or its ``BPSRF`` is "any". "any" accepts every
BPS value, including nodata.

When several candidates qualify, the winner is chosen in this order, most
preferred first:

1. BPSRF: an exact match over "any"
2. OnOff: On over Off
3. Wildcard: "any" over a specific value

Remaining ties go to the row earliest in the table. Wildcard and OnOff are
properties of a row, used only to rank candidates; pixels don't carry them.

A pixel whose (Zone, EVT, DIST) has no rows, or that no candidate
qualifies for, is a real gap in the ruleset. Its output is -9999.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BPSRF_ANY = "any"
_WILDCARD_ANY = "any"
_ONOFF_ON = "On"

_GROUP_KEY_COLUMNS = ["Zone", "EVT", "DIST"]
_PIXEL_COLUMNS = ["Zone", "EVT", "DIST", "cover", "height", "bps"]

# Output nodata sentinel -- is safely outside the valid range of every
# output column here: FBFM13/FBFM40/FCCS codes and the canopy columns
# are never negative.
_OUTPUT_NODATA = -9999


def _as_int64(name: str, values: np.ndarray) -> np.ndarray:
    """Flatten one input raster to int64, refusing values that aren't whole numbers."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64).ravel()
    if np.issubdtype(values.dtype, np.floating):
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN or infinite values; fill nodata with an "
                f"integer sentinel (e.g. -9999) before matching."
            )
        if not (values == np.trunc(values)).all():
            raise ValueError(f"{name} contains non-integer values.")
        return values.astype(np.int64).ravel()
    raise TypeError(f"{name} must be an integer or float array, got {values.dtype}.")


def match_rulesets(
    zone: np.ndarray,
    evt: np.ndarray,
    dist: np.ndarray,
    cover: np.ndarray,
    height: np.ndarray,
    bpsrf: np.ndarray,
    rules: pd.DataFrame,
    output_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each pixel's inputs to one Master_Rulesets row.

    The six input rasters must have the same shape and describe the same
    grid, so ``zone[row, col]``, ``evt[row, col]`` and so on are all the
    same pixel. Integer or whole-valued float rasters are accepted.

    Parameters
    ----------
    zone : numpy.ndarray
        LANDFIRE map zone per pixel.
    evt : numpy.ndarray
        FVT raster VALUE per pixel (Master_Rulesets' ``EVT`` column).
    dist : numpy.ndarray
        FDist code per pixel, e.g. ``build_fdist_raster(...)`` from
        :mod:`~fastfuels_core.fuel_models.fdist_builder`.
    cover : numpy.ndarray
        FVC raster VALUE per pixel (an integer cover class code, the same
        codes Master_Rulesets' ``Cover_Low``/``Cover_High`` use).
    height : numpy.ndarray
        FVH raster VALUE per pixel (an integer height class code, the same
        codes Master_Rulesets' ``Height_Low``/``Height_High`` use).
    bpsrf : numpy.ndarray
        BPS raster VALUE per pixel.
    rules : pandas.DataFrame
        The Master_Rulesets table, with ``Zone``, ``EVT``, ``DIST``,
        ``Cover_Low``, ``Cover_High``, ``Height_Low``, ``Height_High``,
        ``BPSRF``, ``OnOff`` and ``Wildcard`` columns, plus
        ``output_column``.
    output_column : str
        The Master_Rulesets column to return, e.g. ``"FBFM13"``.

    Returns
    -------
    output : numpy.ndarray
        ``output_column``'s value from each pixel's matched row, same shape
        as the inputs. -9999 where nothing matched (None in a text column).
    matched : numpy.ndarray
        Boolean raster, same shape as the inputs: True where a pixel matched
        a row.

    Raises
    ------
    ValueError
        The output column isn't in the table, the inputs differ in shape, or
        a float input has NaN, infinite or non-integer values.
    TypeError
        An input isn't an integer or float array.

    Notes
    -----
    A pixel with a nodata input needs no special handling. Its nodata code
    matches no rule, so it comes back unmatched like a real gap in the
    ruleset, with one exception: a nodata BPS value still matches rows
    whose ``BPSRF`` is "any".

    Matching runs once per distinct combination of the six input values,
    not once per pixel.

    Examples
    --------
    Two rules share a key; the one with an exact BPSRF wins where it
    applies, and a pixel in a zone with no rules is unmatched.

    >>> rules = pd.DataFrame({
    ...     "Zone": [1, 1], "EVT": [7000, 7000], "DIST": [0, 0],
    ...     "Cover_Low": [0, 0], "Cover_High": [999, 999],
    ...     "Height_Low": [0, 0], "Height_High": [999, 999],
    ...     "BPSRF": ["any", "11"], "OnOff": ["On", "On"],
    ...     "Wildcard": ["any", "any"], "FBFM13": [2, 8],
    ... })
    >>> output, matched = match_rulesets(
    ...     zone=np.array([1, 1, 2]),
    ...     evt=np.array([7000, 7000, 7000]),
    ...     dist=np.array([0, 0, 0]),
    ...     cover=np.array([150, 150, 150]),
    ...     height=np.array([110, 110, 110]),
    ...     bpsrf=np.array([11, 12, 11]),
    ...     rules=rules,
    ...     output_column="FBFM13",
    ... )
    >>> output.tolist()
    [8, 2, -9999]
    >>> matched.tolist()
    [True, True, False]
    """
    if output_column not in rules.columns:
        raise ValueError(f"Unknown output column: {output_column!r}")

    inputs = {
        "zone": zone,
        "evt": evt,
        "dist": dist,
        "cover": cover,
        "height": height,
        "bpsrf": bpsrf,
    }
    shape = np.shape(zone)
    for name, values in inputs.items():
        if np.shape(values) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(values)}, expected {shape} to match zone."
            )

    # One row per pixel, and one per distinct combination of inputs, since
    # identical pixels always match the same rule.
    pixels = pd.DataFrame(
        {
            column: _as_int64(name, values)
            for column, (name, values) in zip(_PIXEL_COLUMNS, inputs.items())
        }
    )
    combos = pixels.drop_duplicates().reset_index(drop=True)
    combos["combo"] = np.arange(len(combos))

    # 1. Every rule with the same Zone, EVT and DIST is a candidate.
    rules = rules.reset_index(drop=True).assign(rule_row=lambda df: np.arange(len(df)))
    candidates = combos.merge(rules, on=_GROUP_KEY_COLUMNS)

    # 2. Rank candidates: exact BPSRF beats "any", then On beats Off, then
    #    Wildcard "any" beats a specific value. Lowest rank wins.
    bpsrf_any = candidates["BPSRF"].astype(str) == _BPSRF_ANY
    candidates["rank"] = (
        bpsrf_any.astype(int) * 4
        + (candidates["OnOff"] != _ONOFF_ON).astype(int) * 2
        + (candidates["Wildcard"].astype(str) != _WILDCARD_ANY).astype(int)
    )

    # 3. Keep the candidates the combination qualifies for.
    bpsrf_value = pd.to_numeric(candidates["BPSRF"].where(~bpsrf_any), errors="coerce")
    qualifies = (
        (candidates["cover"] >= candidates["Cover_Low"])
        & (candidates["cover"] <= candidates["Cover_High"])
        & (candidates["height"] >= candidates["Height_Low"])
        & (candidates["height"] <= candidates["Height_High"])
        & (bpsrf_any | (bpsrf_value == candidates["bps"]))
    )

    # 4. Best qualifying rule per combination. Equal ranks go to the rule
    #    earliest in the table.
    best = (
        candidates[qualifies]
        .sort_values(["rank", "rule_row"])
        .drop_duplicates("combo")
        .set_index("combo")[output_column]
    )

    # 5. Give each pixel its combination's answer.
    pixel_combo = pixels.merge(combos, on=_PIXEL_COLUMNS, how="left")["combo"]
    matched = pixel_combo.isin(best.index).to_numpy()
    output = best.reindex(pixel_combo).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(rules[output_column]):
        output = output.where(matched, _OUTPUT_NODATA).astype(
            rules[output_column].dtype
        )
    else:
        output = output.astype(object).where(matched, None)

    return output.to_numpy().reshape(shape), matched.reshape(shape)
