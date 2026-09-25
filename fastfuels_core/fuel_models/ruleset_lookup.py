"""
Master_Rulesets lookup
======================

Vectorized rule matching against LANDFIRE's Master_Rulesets table, the
rules LFTFC uses to assign fuel models after a disturbance. Given
per-pixel Zone, EVT, DIST, Cover, Height and BPS codes, finds each pixel's
matching Master_Rulesets row and returns the requested output columns
(e.g. FBFM13, FBFM40, FCCS).

The table is not packaged (about 626K rows); the caller loads it and
builds a :class:`RulesetIndex` once with :func:`build_ruleset_index`, then
reuses it for every :func:`match_rulesets` call.

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

Wildcard and OnOff are properties of a row, used only to rank candidates;
pixels don't carry them.

A pixel whose (Zone, EVT, DIST) has no rows, or that no candidate
qualifies for, is a real gap in the ruleset. Its output is -9999.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_BPSRF_ANY = "any"
_WILDCARD_ANY = "any"
_ONOFF_ON = "On"

_GROUP_KEY_COLUMNS = ["Zone", "EVT", "DIST"]

# Output nodata sentinel -- is safely outside the valid range of every
# output column here: FBFM13/FBFM40/FCCS codes and the canopy columns
# are never negative.
_OUTPUT_NODATA = -9999

# Any candidate that fails the Cover/Height/BPSRF check gets this score --
# higher than the max real score (bpsrf_rank*4 + onoff_rank*2 + wildcard_rank,
# max 7), so it never wins an argmin.
_UNQUALIFIED_SCORE = 99


@dataclass(frozen=True)
class _RulesetGroup:
    """Candidate rows sharing one (Zone, EVT, DIST) key -- a handful of rows."""

    cover_low: np.ndarray
    cover_high: np.ndarray
    height_low: np.ndarray
    height_high: np.ndarray
    bpsrf: np.ndarray  # int64; meaningless where bpsrf_is_any is True
    bpsrf_is_any: np.ndarray
    is_on: np.ndarray
    wildcard_is_any: np.ndarray
    row_index: np.ndarray  # positional index into RulesetIndex.table


@dataclass(frozen=True)
class RulesetIndex:
    """The whole Master_Rulesets table, reorganized for fast lookup.

    Rows are grouped by their (Zone, EVT, DIST) values, so finding a
    pixel's candidate rows is a single dictionary lookup. Build once with
    :func:`build_ruleset_index` and reuse across :func:`match_rulesets`
    calls.

    Attributes
    ----------
    groups : dict of tuple to _RulesetGroup
        Candidate rows keyed by their exact (Zone, EVT, DIST) tuple.
    table : pandas.DataFrame
        The full table, kept so any output column can be gathered by name.
    """

    groups: dict[tuple[int, int, int], _RulesetGroup]
    table: pd.DataFrame


def build_ruleset_index(rulesets_df: pd.DataFrame) -> RulesetIndex:
    """Group Master_Rulesets by the exact (Zone, EVT, DIST) key.

    Parameters
    ----------
    rulesets_df : pandas.DataFrame
        The full Master_Rulesets table. Must have ``Zone``, ``EVT``,
        ``DIST``, ``Cover_Low``, ``Cover_High``, ``Height_Low``,
        ``Height_High``, ``BPSRF``, ``OnOff`` and ``Wildcard`` columns,
        plus whichever output columns will be requested.

    Returns
    -------
    RulesetIndex
        The index to pass to :func:`match_rulesets`.

    Notes
    -----
    Each group ends up with only a handful of candidate rows, differing by
    Cover, Height, BPSRF, OnOff and Wildcard. Building the index is a
    groupby over the whole table, so do it once per running service and
    reuse the result rather than rebuilding it per raster.
    """
    df = rulesets_df.reset_index(drop=True)

    bpsrf_is_any = df["BPSRF"].astype(str) == _BPSRF_ANY
    bpsrf_numeric = (
        pd.to_numeric(df["BPSRF"].where(~bpsrf_is_any), errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )
    is_on = (df["OnOff"] == _ONOFF_ON).to_numpy()
    wildcard_is_any = (df["Wildcard"].astype(str) == _WILDCARD_ANY).to_numpy()
    cover_low = df["Cover_Low"].to_numpy()
    cover_high = df["Cover_High"].to_numpy()
    height_low = df["Height_Low"].to_numpy()
    height_high = df["Height_High"].to_numpy()
    bpsrf_is_any_arr = bpsrf_is_any.to_numpy()

    groups: dict[tuple[int, int, int], _RulesetGroup] = {}
    for raw_key, positions in df.groupby(_GROUP_KEY_COLUMNS).indices.items():
        key = tuple(int(v) for v in raw_key)
        idx = np.asarray(positions, dtype=np.int64)
        groups[key] = _RulesetGroup(
            cover_low=cover_low[idx],
            cover_high=cover_high[idx],
            height_low=height_low[idx],
            height_high=height_high[idx],
            bpsrf=bpsrf_numeric[idx],
            bpsrf_is_any=bpsrf_is_any_arr[idx],
            is_on=is_on[idx],
            wildcard_is_any=wildcard_is_any[idx],
            row_index=idx,
        )

    return RulesetIndex(groups=groups, table=df)


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


@dataclass(frozen=True)
class MatchResult:
    """The rasters :func:`match_rulesets` computed for its output column.

    Attributes
    ----------
    output : numpy.ndarray
        The output column's value from each pixel's matched row, same shape
        as the inputs. Unmatched pixels are -9999 in a numeric column and
        None in a text column.
    matched : numpy.ndarray
        Boolean raster, same shape as the inputs: True where a pixel
        matched a Master_Rulesets row.
    n_unmatched : int
        Number of pixels with no matching Master_Rulesets row.
    """

    output: np.ndarray
    matched: np.ndarray
    n_unmatched: int


def match_rulesets(
    zone: np.ndarray,
    evt: np.ndarray,
    dist: np.ndarray,
    cover: np.ndarray,
    height: np.ndarray,
    bpsrf: np.ndarray,
    index: RulesetIndex,
    output_column: str,
) -> MatchResult:
    """Match each pixel's inputs to one Master_Rulesets row.

    The six input rasters must have the same shape and describe the same
    grid, so ``zone[row, col]``, ``evt[row, col]`` and so on are all the
    same pixel. Integer or whole-valued float rasters are accepted.

    Parameters
    ----------
    zone : numpy.ndarray
        LANDFIRE map zone per pixel, e.g. from
        :func:`~fastfuels_core.fuel_models.lf_zone_lookup.lookup_lf_zones`.
    evt : numpy.ndarray
        FVT raster VALUE per pixel (Master_Rulesets' ``EVT`` column).
    dist : numpy.ndarray
        FDist code per pixel, e.g. ``ldist_raster_to_fdist_raster(...).fdist``
        from :mod:`~fastfuels_core.fuel_models.disturbance_crosswalk`.
    cover : numpy.ndarray
        FVC raster VALUE per pixel (an integer cover class code, the same
        codes Master_Rulesets' ``Cover_Low``/``Cover_High`` use).
    height : numpy.ndarray
        FVH raster VALUE per pixel (an integer height class code, the same
        codes Master_Rulesets' ``Height_Low``/``Height_High`` use).
    bpsrf : numpy.ndarray
        BPS raster VALUE per pixel.
    index : RulesetIndex
        From :func:`build_ruleset_index`. Build once, reuse across calls.
    output_column : str
        The Master_Rulesets column to return, e.g. ``"FBFM40"``.

    Returns
    -------
    MatchResult
        The output column's raster, same shape as the inputs, plus which
        pixels matched a row and how many didn't.

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
    not once per pixel, so its cost depends on how varied the landscape
    is rather than on raster size.

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
    >>> result = match_rulesets(
    ...     zone=np.array([1, 1, 2]),
    ...     evt=np.array([7000, 7000, 7000]),
    ...     dist=np.array([0, 0, 0]),
    ...     cover=np.array([150, 150, 150]),
    ...     height=np.array([110, 110, 110]),
    ...     bpsrf=np.array([11, 12, 11]),
    ...     index=build_ruleset_index(rules),
    ...     output_column="FBFM13",
    ... )
    >>> result.output.tolist()
    [8, 2, -9999]
    >>> result.n_unmatched
    1
    """
    if output_column not in index.table.columns:
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
    stacked = np.stack(
        [_as_int64(name, values) for name, values in inputs.items()], axis=1
    )
    unique_rows, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_unique = unique_rows.shape[0]

    matched_row_index = np.full(n_unique, -1, dtype=np.int64)

    # unique_rows is lexicographically sorted with (Zone, EVT, DIST) as the
    # first 3 columns, so rows sharing a key are contiguous -- one diff finds
    # every run boundary instead of a second groupby.
    keys = unique_rows[:, :3]
    boundaries = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    run_starts = np.concatenate(([0], boundaries))
    run_ends = np.concatenate((boundaries, [n_unique]))

    for start, end in zip(run_starts, run_ends):
        key = (int(keys[start, 0]), int(keys[start, 1]), int(keys[start, 2]))
        group = index.groups.get(key)
        if group is None:
            continue  # stays unmatched (-1)

        run_cover = unique_rows[start:end, 3]
        run_height = unique_rows[start:end, 4]
        run_bpsrf = unique_rows[start:end, 5]

        cover_ok = (run_cover[:, None] >= group.cover_low[None, :]) & (
            run_cover[:, None] <= group.cover_high[None, :]
        )
        height_ok = (run_height[:, None] >= group.height_low[None, :]) & (
            run_height[:, None] <= group.height_high[None, :]
        )
        bpsrf_ok = group.bpsrf_is_any[None, :] | (
            run_bpsrf[:, None] == group.bpsrf[None, :]
        )
        qualifies = cover_ok & height_ok & bpsrf_ok

        # Candidate-row rank, most preferred = 0: BPSRF dominant (x4), then
        # OnOff (x2), then Wildcard (x1) -- reproduces the sequential
        # narrowing (BPSRF, then OnOff, then Wildcard) as a single
        # lexicographic score.
        bpsrf_rank = group.bpsrf_is_any.astype(np.int8)
        onoff_rank = (~group.is_on).astype(np.int8)
        wildcard_rank = (~group.wildcard_is_any).astype(np.int8)
        candidate_score = bpsrf_rank * 4 + onoff_rank * 2 + wildcard_rank

        score_matrix = np.where(qualifies, candidate_score[None, :], _UNQUALIFIED_SCORE)
        best_col = np.argmin(score_matrix, axis=1)
        best_score = score_matrix[np.arange(end - start), best_col]
        matched = best_score < _UNQUALIFIED_SCORE

        matched_row_index[start:end] = np.where(matched, group.row_index[best_col], -1)

    is_matched = matched_row_index >= 0
    safe_index = np.clip(matched_row_index, 0, len(index.table) - 1)

    col_values = index.table[output_column].to_numpy()
    if np.issubdtype(col_values.dtype, np.number):
        gathered = col_values[safe_index].copy()
        gathered[~is_matched] = _OUTPUT_NODATA
    else:
        gathered = col_values[safe_index].astype(object)
        gathered[~is_matched] = None

    matched = is_matched[inverse].reshape(shape)
    return MatchResult(
        output=gathered[inverse].reshape(shape),
        matched=matched,
        n_unmatched=int(np.count_nonzero(~matched)),
    )
