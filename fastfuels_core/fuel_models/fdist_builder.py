"""
FDist rasters for the Master_Rulesets rules
===========================================

Master_Rulesets' ``DIST`` column uses LANDFIRE's FDist codes:

    FDist code = 100 * D_TYPE + 10 * D_SEVERITY + D_TIME

where D_TIME is the time since disturbance. :func:`build_fdist_raster`
builds the FDist raster the rules are applied to, in one of three
``DISTURBANCE_MODES``:

- ``"ldist"``: this year's LANDFIRE Limited Annual Disturbance (LDist),
  converted to FDist with the packaged LANDFIRE LDist attribute table.
- ``"ldist_and_last_year_fdist"``: this year's converted LDist, with last
  year's one-year-old FDist disturbances aged to two years wherever this
  year has none. For products updated two years at a time (FCCS).
- ``"fdist"``: this year's disturbances from this year's LANDFIRE FDist.

Notes
-----
LANDFIRE does not publish the LDist to FDist conversion, so it is a
best-effort approximation built from name/definition matching and small
real-data samples, not an authoritative source:

- Most DIST_TYPE groupings (e.g. Clearcut/Harvest/Thinning -> Mechanical
  Remove) follow name/definition similarity to FDist's D_TYPE categories.
- Herbicide, Insecticide, Chemical and Biological are the exception.
  Pixels with these types were sampled and checked at the same locations
  in LANDFIRE's FDist output; none appeared there, showing either no
  disturbance or an unrelated older disturbance. That is not proof at
  scale, but it is enough to treat them as unrepresentable rather than
  force them into an ill-fitting D_TYPE.
- SEVERITY's "Unburned/Low" and "Increased Green" (fire only) are mapped
  to Low as an approximation.
- D_TIME is always 1 for converted LDist, i.e. a disturbance this year.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd


class UnknownLdistTypeError(ValueError):
    """A DIST_TYPE value not in any known group.

    Raised for a new LDist category that hasn't been catalogued, not for one
    of the deliberately excluded ones (see ``LDIST_TYPE_UNREPRESENTABLE``).
    """


class UnknownLdistSeverityError(ValueError):
    """A SEVERITY value not in any known group."""


DISTURBANCE_MODES = ("ldist", "ldist_and_last_year_fdist", "fdist")

LDIST_TYPE_TO_FDIST_TYPE: dict[str, int] = {
    "Fire": 1,
    "Wildfire": 1,
    "Wildland Fire Use": 1,
    "Prescribed Fire": 1,
    "Mechanical Add": 2,
    "Mechanical Remove": 3,
    "Clearcut": 3,
    "Harvest": 3,
    "Thinning": 3,
    "Weather": 4,
    "Insects": 5,
    "Disease": 5,
    "Insects/Disease": 5,
    "Mechanical Unknown": 6,
    "Mastication": 7,
}

# Not a vegetation disturbance -- FDist code 0, no warning.
LDIST_TYPE_NO_DISTURBANCE: frozenset[str] = frozenset(
    {
        "Water",
        "Development",
        "Fill-NoData",
    }
)

# A real disturbance LANDFIRE's own FDist doesn't encode into any D_TYPE
# either (confirmed against real FDist samples) -- FDist code 0, flagged so
# the caller can warn instead of silently dropping it.
LDIST_TYPE_UNREPRESENTABLE: frozenset[str] = frozenset(
    {
        "Herbicide",
        "Insecticide",
        "Chemical",
        "Biological",
    }
)

LDIST_SEVERITY_TO_FDIST_SEVERITY: dict[str, int] = {
    "Unburned/Low": 1,
    "Increased Green": 1,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
}


@lru_cache(maxsize=1)
def _ldist_attribute_table() -> pd.DataFrame:
    """LANDFIRE 2025 Limited Annual Disturbance (LDist) Attribute Data Table.

    Read from disk on first call and cached.

    Returns
    -------
    pandas.DataFrame
        One row per LDist code, with ``VALUE``, ``DIST_TYPE``, ``SEVERITY``
        and LANDFIRE's other attribute columns.

    Notes
    -----
    Original file: ``LF2025_LDist25_20260108.csv`` (published 2026-01-08).
    """
    return pd.read_csv(files("fastfuels_core.data") / "LF2025_LDIST_ATTRIBUTES.csv")


@dataclass(frozen=True)
class FDistResult:
    """The outcome of converting one LDist attribute-table row to an FDist code.

    Attributes
    ----------
    fdist_code : int
        The encoded FDist code, or 0 for no disturbance or an
        unrepresentable one.
    warning : str or None
        None when there is no disturbance, or the row's type doesn't apply
        to vegetation (e.g. Water, Development). An explanatory message
        when ``fdist_code`` is 0 even though the row is a real disturbance
        the encoding can't carry (e.g. Herbicide).
    """

    fdist_code: int
    warning: str | None = None


def _ldist_row_to_fdist(
    dist_type: str | None,
    severity: str | None,
) -> FDistResult:
    """Convert one LDist attribute-table row to an FDist code.

    Parameters
    ----------
    dist_type : str or None
        The row's DIST_TYPE label, or None if missing.
    severity : str or None
        The row's SEVERITY label, or None if missing.

    Returns
    -------
    FDistResult
        The FDist code, or 0 with ``warning`` explaining why when a real
        disturbance can't be encoded.

    Raises
    ------
    UnknownLdistTypeError
        ``dist_type`` isn't in any known group.
    UnknownLdistSeverityError
        ``severity`` isn't in any known group.

    Notes
    -----
    D_TIME (time since disturbance) is fixed at 1. That matches LANDFIRE's
    FDist output more closely than computing it from the LDist
    ``CALENDAR_YR``.

    A real disturbance with a missing severity (None or "Fill-NoData")
    returns 0 without a warning. No row in the _ldist_attribute_table has one.

    Examples
    --------
    >>> _ldist_row_to_fdist("Wildfire", "Moderate")
    FDistResult(fdist_code=121, warning=None)
    >>> _ldist_row_to_fdist("Herbicide", "Low")
    FDistResult(fdist_code=0, warning="'Herbicide' has no FDist D_TYPE equivalent")
    """
    if dist_type is None or dist_type in LDIST_TYPE_NO_DISTURBANCE:
        return FDistResult(0)

    if dist_type in LDIST_TYPE_UNREPRESENTABLE:
        return FDistResult(0, warning=f"{dist_type!r} has no FDist D_TYPE equivalent")

    d_type = LDIST_TYPE_TO_FDIST_TYPE.get(dist_type)
    if d_type is None:
        raise UnknownLdistTypeError(f"Unrecognized LDist DIST_TYPE: {dist_type!r}")

    if severity is None or severity == "Fill-NoData":
        return FDistResult(0)

    d_severity = LDIST_SEVERITY_TO_FDIST_SEVERITY.get(severity)
    if d_severity is None:
        raise UnknownLdistSeverityError(f"Unrecognized LDist SEVERITY: {severity!r}")

    # Using 1 (one year) -- more accurate against real FDist data than
    # computing time since disturbance from LDist CALENDAR_YR.
    tsd = 1

    return FDistResult(100 * d_type + 10 * d_severity + tsd)


@lru_cache(maxsize=1)
def _ldist_codes_to_fdist_codes() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """The FDist code for every LDist code in the LDist attribute table.

    Computed from :func:`_ldist_attribute_table` on first call and cached.

    Returns
    -------
    sorted_codes : numpy.ndarray
        Every LDist code in the table, ascending.
    fdist_codes : numpy.ndarray
        int32 FDist code for each entry in ``sorted_codes``.
    warnings : dict of int to str
        LDist code -> reason, for each code that is a real disturbance
        but was set to 0. Keyed by code so a caller can keep only the
        codes present in a given raster.

    Raises
    ------
    UnknownLdistTypeError
        A row's DIST_TYPE isn't in any known group.
    UnknownLdistSeverityError
        A row's SEVERITY isn't in any known group.
    """
    rows = _ldist_attribute_table().sort_values("VALUE")
    warnings: dict[int, str] = {}
    fdist_codes = np.empty(len(rows), dtype=np.int32)

    for i, row in enumerate(rows.itertuples()):
        dist_type = row.DIST_TYPE if pd.notna(row.DIST_TYPE) else None
        severity = row.SEVERITY if pd.notna(row.SEVERITY) else None

        result = _ldist_row_to_fdist(dist_type, severity)
        fdist_codes[i] = result.fdist_code
        if result.warning is not None:
            warnings[int(row.VALUE)] = result.warning

    return rows["VALUE"].to_numpy(), fdist_codes, warnings


@dataclass(frozen=True)
class FDistRaster:
    """An FDist raster, with diagnostics from converting LDist.

    Attributes
    ----------
    fdist : numpy.ndarray
        FDist code raster, same shape as the input. Where LDist was
        converted, a pixel whose LDist code isn't in the attribute table is
        -9999.
    n_unmapped : int
        Number of pixels whose LDist code isn't in the attribute table. 0
        when no LDist was converted.
    warnings : dict of int to str
        LDist code -> reason, for each code present in an LDist raster that was
        set to 0 even though it is a real disturbance (e.g. Herbicide).
        Empty when there are none.
    """

    fdist: np.ndarray
    n_unmapped: int
    warnings: dict[int, str]


def ldist_raster_to_fdist_raster(ldist_codes: np.ndarray) -> FDistRaster:
    """Convert an LDist code raster to an FDist code raster.

    Uses the packaged LANDFIRE LDist attribute table.

    Parameters
    ----------
    ldist_codes : numpy.ndarray
        The raw LDist raster, one LDist code per pixel.

    Returns
    -------
    FDistRaster
        The FDist code raster, the number of pixels whose code isn't in the
        attribute table, and a warning for each code present that is a
        real disturbance but was set to 0.

    Examples
    --------
    >>> result = ldist_raster_to_fdist_raster(np.array([[2882, 2801], [0, 2971]]))
    >>> result.fdist.tolist()
    [[121, 0], [0, 0]]
    >>> result.n_unmapped
    0
    >>> result.warnings
    {2971: "'Herbicide' has no FDist D_TYPE equivalent"}
    """
    ldist_codes = np.asarray(ldist_codes)
    sorted_codes, fdist_codes, all_warnings = _ldist_codes_to_fdist_codes()

    # Each pixel's position in sorted_codes. A code that isn't in the table
    # lands next to where it would be, so the equality check catches it.
    idx = np.clip(np.searchsorted(sorted_codes, ldist_codes), 0, len(sorted_codes) - 1)
    in_table = sorted_codes[idx] == ldist_codes
    fdist = np.where(in_table, fdist_codes[idx], -9999).astype(np.int32)

    warnings = {
        code: msg for code, msg in all_warnings.items() if np.any(ldist_codes == code)
    }
    return FDistRaster(
        fdist=fdist, n_unmapped=int(np.count_nonzero(~in_table)), warnings=warnings
    )


def _is_one_year_old(fdist: np.ndarray) -> np.ndarray:
    """True where an FDist code is a disturbance with time since disturbance 1.

    The ``> 0`` check matters: FDist nodata is -9999, and ``-9999 % 10 == 1``.
    """
    return (fdist > 0) & (fdist % 10 == 1)


def _age_fdist(fdist: np.ndarray) -> np.ndarray:
    """Last year's FDist a year later: ``...1`` codes become ``...2``, the rest 0.

    Only last year's one-year-old disturbances carry forward; older ones,
    undisturbed pixels and nodata become 0.
    """
    fdist = np.asarray(fdist)
    return np.where(_is_one_year_old(fdist), fdist + 1, 0)


def _this_years_fdist(fdist: np.ndarray) -> np.ndarray:
    """This year's disturbances from an FDist: ``...1`` codes kept, the rest 0."""
    fdist = np.asarray(fdist)
    return np.where(_is_one_year_old(fdist), fdist, 0)


def build_fdist_raster(
    disturbance: str,
    *,
    ldist: np.ndarray | None = None,
    fdist: np.ndarray | None = None,
) -> FDistRaster:
    """The FDist raster the Master_Rulesets rules are applied to.

    Parameters
    ----------
    disturbance : str
        One of ``DISTURBANCE_MODES``:

        ``"ldist"``
            This year's LDist, converted to FDist.
        ``"ldist_and_last_year_fdist"``
            This year's LDist, converted to FDist, wherever it's a
            disturbance; everywhere else, last year's one-year-old
            disturbances aged to two years. Used when a product (FCCS) is
            updated two years at a time.
        ``"fdist"``
            This year's disturbances (time since disturbance 1) from this
            year's LANDFIRE FDist.
    ldist : numpy.ndarray, optional
        This year's LDist. Needed for ``"ldist"`` and
        ``"ldist_and_last_year_fdist"``.
    fdist : numpy.ndarray, optional
        LANDFIRE FDist: last year's for ``"ldist_and_last_year_fdist"``,
        this year's for ``"fdist"``.

    Returns
    -------
    FDistRaster
        The FDist raster, plus ``n_unmapped`` and ``warnings`` from the LDist
        conversion (0 and empty in ``"fdist"`` mode).

    Raises
    ------
    ValueError
        ``disturbance`` isn't one of ``DISTURBANCE_MODES``, or the input it
        needs wasn't given.
    """
    if disturbance not in DISTURBANCE_MODES:
        raise ValueError(
            f"Unknown disturbance mode {disturbance!r}; "
            f"expected one of {DISTURBANCE_MODES}."
        )
    if disturbance != "fdist" and ldist is None:
        raise ValueError(f"disturbance={disturbance!r} needs ldist.")
    if disturbance != "ldist" and fdist is None:
        raise ValueError(f"disturbance={disturbance!r} needs fdist.")

    if disturbance == "fdist":
        return FDistRaster(fdist=_this_years_fdist(fdist), n_unmapped=0, warnings={})

    converted = ldist_raster_to_fdist_raster(ldist)
    if disturbance == "ldist":
        return converted

    # This year's disturbance wins; elsewhere, last year's carried forward.
    combined = np.where(converted.fdist > 0, converted.fdist, _age_fdist(fdist))
    return FDistRaster(
        fdist=combined, n_unmapped=converted.n_unmapped, warnings=converted.warnings
    )
