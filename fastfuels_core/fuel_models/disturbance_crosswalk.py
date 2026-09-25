"""
LDist to FDist crosswalk
========================

Converts LANDFIRE Limited Annual Disturbance (LDist) codes into the FDist
codes that Master_Rulesets' ``DIST`` column expects:

    FDist code = 100 * D_TYPE + 10 * D_SEVERITY + D_TIME

with D_TIME fixed at 1 (see :func:`ldist_row_to_fdist`).

Most callers need only :func:`ldist_raster_to_fdist_raster`, which uses the
packaged LF2025 attribute table.

Notes
-----
LANDFIRE does not publish this crosswalk directly, so this is a best-effort
approximation built from name/definition matching and small real-data
samples, not an authoritative source:

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
def _lf2025_ldist_attributes() -> pd.DataFrame:
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
    """The outcome of crosswalking one LDist attribute-table row to an FDist code.

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


def ldist_row_to_fdist(
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
    returns 0 without a warning. No row in the LF2025 table has one.

    Examples
    --------
    >>> ldist_row_to_fdist("Wildfire", "Moderate")
    FDistResult(fdist_code=121, warning=None)
    >>> ldist_row_to_fdist("Herbicide", "Low")
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


def _build_fdist_crosswalk(
    decode_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Build an LDist code -> FDist code crosswalk from an attribute table.

    Parameters
    ----------
    decode_table : pandas.DataFrame
        An LDist attribute table with ``VALUE``, ``DIST_TYPE`` and
        ``SEVERITY`` columns, e.g. :func:`_lf2025_ldist_attributes`.

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

    Notes
    -----
    The crosswalk's size depends on the number of codes in the table, not
    on raster size, so build it once and reuse it.
    """
    rows = decode_table.sort_values("VALUE")
    warnings: dict[int, str] = {}
    fdist_codes = np.empty(len(rows), dtype=np.int32)

    for i, row in enumerate(rows.itertuples()):
        dist_type = row.DIST_TYPE if pd.notna(row.DIST_TYPE) else None
        severity = row.SEVERITY if pd.notna(row.SEVERITY) else None

        result = ldist_row_to_fdist(dist_type, severity)
        fdist_codes[i] = result.fdist_code
        if result.warning is not None:
            warnings[int(row.VALUE)] = result.warning

    return rows["VALUE"].to_numpy(), fdist_codes, warnings


@lru_cache(maxsize=1)
def _fdist_crosswalk() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """The LF2025 LDist -> FDist crosswalk, built on first call and cached.

    Returns
    -------
    tuple
        ``(sorted_codes, fdist_codes, warnings)``, as returned by
        :func:`_build_fdist_crosswalk` for :func:`_lf2025_ldist_attributes`.
    """
    return _build_fdist_crosswalk(_lf2025_ldist_attributes())


def _apply_fdist_crosswalk(
    ldist_codes: np.ndarray,
    sorted_codes: np.ndarray,
    fdist_codes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Convert an LDist code raster to FDist codes with a prebuilt crosswalk.

    Parameters
    ----------
    ldist_codes : numpy.ndarray
        The raw LDist raster, one LDist code per pixel.
    sorted_codes : numpy.ndarray
        Every LDist code from :func:`_build_fdist_crosswalk`, ascending.
    fdist_codes : numpy.ndarray
        The FDist code for each entry in ``sorted_codes``.

    Returns
    -------
    fdist : numpy.ndarray
        int32 FDist code raster, same shape as ``ldist_codes``. A pixel
        whose LDist code isn't in ``sorted_codes`` is -9999.
    n_unmapped : int
        Number of pixels whose LDist code isn't in ``sorted_codes``, so the
        caller can report one total instead of one warning per pixel.
    """
    idx = np.clip(np.searchsorted(sorted_codes, ldist_codes), 0, len(sorted_codes) - 1)
    in_table = sorted_codes[idx] == ldist_codes

    out = np.where(in_table, fdist_codes[idx], -9999).astype(np.int32)
    return out, int((~in_table).sum())


@dataclass(frozen=True)
class FDistRaster:
    """The outcome of crosswalking an entire LDist raster to FDist codes.

    Attributes
    ----------
    fdist : numpy.ndarray
        int32 FDist code raster, same shape as the input. A pixel whose
        LDist code isn't in the attribute table is -9999.
    n_unmapped : int
        Number of pixels whose LDist code isn't in the attribute table.
    warnings : dict of int to str
        LDist code -> reason, for each code present in the raster that was
        set to 0 even though it is a real disturbance (e.g. Herbicide).
        Empty when there are none.
    """

    fdist: np.ndarray
    n_unmapped: int
    warnings: dict[int, str]


def ldist_raster_to_fdist_raster(
    ldist_codes: np.ndarray,
    nodata: float | None = None,
) -> FDistRaster:
    """Convert an LDist code raster to an FDist code raster.

    Uses the packaged LANDFIRE 2025 LDist attribute table
    (``LF2025_LDIST_ATTRIBUTES.csv``).

    Parameters
    ----------
    ldist_codes : numpy.ndarray
        The raw LDist raster, one LDist code per pixel.
    nodata : int or float, optional
        The raster's declared nodata value, e.g. ``ldist.rio.nodata``. Its
        pixels are treated as no disturbance (FDist 0), like LANDFIRE's own
        Fill-NoData code (-9999), rather than counted as unmapped.

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
    if nodata is not None:
        ldist_codes = np.where(ldist_codes == nodata, -9999, ldist_codes)

    sorted_codes, fdist_codes, build_warnings = _fdist_crosswalk()
    out, n_unmapped = _apply_fdist_crosswalk(ldist_codes, sorted_codes, fdist_codes)

    warnings = {
        code: msg for code, msg in build_warnings.items() if np.any(ldist_codes == code)
    }
    return FDistRaster(fdist=out, n_unmapped=n_unmapped, warnings=warnings)
