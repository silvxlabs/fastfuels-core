"""
FDist rasters for the Master_Rulesets rules
===========================================

Master_Rulesets' ``DIST`` column uses LANDFIRE's FDist codes:

    FDist code = 100 * D_TYPE + 10 * D_SEVERITY + D_TIME

where D_TIME is the time since disturbance. :func:`build_fdist_raster`
builds the FDist raster the rules are applied to, in one of two
``DISTURBANCE_MODES``:

- ``"ldist_and_last_year_fdist"``: Last year's FDist with its one-year-old
    disturbances aged to two years and with this year's LDist, converted to
    FDist, replacing it wherever this year has a disturbance.
- ``"fdist"``: this year's LANDFIRE FDist.

Notes
-----
LANDFIRE does not publish the LDist to FDist conversion, so
``LDIST_TO_FDIST`` is a best-effort approximation built from
name/definition matching and small real-data samples, not an authoritative
source. It is built from every combination of a DIST_TYPE and a SEVERITY
below, so a combination new to a later LANDFIRE table is still covered:

- Most DIST_TYPE groupings (e.g. Clearcut/Harvest/Thinning -> Mechanical
  Remove) follow name/definition similarity to FDist's D_TYPE categories.
- Herbicide, Insecticide, Chemical and Biological are left out. Pixels with
  these types were sampled and checked at the same locations in LANDFIRE's
  FDist output; none appeared there, showing either no disturbance or an
  unrelated older disturbance. That is not proof at scale, but it is enough
  to treat them as unrepresentable rather than force them into an
  ill-fitting D_TYPE.
- SEVERITY's "Unburned/Low" and "Increased Green" (fire only) are mapped
  to Low as an approximation.
- D_TIME is always 1 for converted LDist, i.e. a disturbance this year.

Aging last year's FDist has two known limits:

- LANDFIRE may revise a past disturbance's severity between releases
  (e.g. 121 one year, 132 the next). Aging last year's code can't
  anticipate that.
- Only one-year-old codes are aged (D_TIME 1 becomes 2). D_TIME 2 covers
  2-5 years and 3 covers 6-10 years, so moving from 2 to 3 needs the year
  of the disturbance, which the class alone doesn't give. Codes ending in
  2 stay 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DISTURBANCE_MODES = ("ldist_and_last_year_fdist", "fdist")

# FDist D_TYPE for each LDist DIST_TYPE. Herbicide, Insecticide, Chemical and
# Biological are deliberately absent (see the module notes).
_FDIST_TYPE = {
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

# FDist D_SEVERITY for each LDist SEVERITY.
_FDIST_SEVERITY = {
    "Unburned/Low": 1,
    "Increased Green": 1,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
}

# LDist DIST_TYPEs that aren't a vegetation disturbance: FDist 0, no warning.
# None is a blank DIST_TYPE (background).
_NO_DISTURBANCE = {None, "Water", "Development", "Fill-NoData"}

# Every (DIST_TYPE, SEVERITY) pair -> FDist code, with D_TIME 1.
LDIST_TO_FDIST: dict[tuple[str, str], int] = {
    (dist_type, severity): 100 * d_type + 10 * d_severity + 1
    for dist_type, d_type in _FDIST_TYPE.items()
    for severity, d_severity in _FDIST_SEVERITY.items()
}


def _label(value):
    """An attribute-table cell, with blanks as None."""
    return None if pd.isna(value) else value


def _ldist_raster_to_fdist_raster(
    ldist: np.ndarray, attribute_table: pd.DataFrame
) -> tuple[np.ndarray, list[int]]:
    """Convert an LDist code raster to FDist codes.

    Each pixel's LDist code is looked up in ``attribute_table`` (its
    ``VALUE`` column) to get the code's DIST_TYPE and SEVERITY. A type that
    isn't a disturbance becomes 0; otherwise the pair is looked up in
    ``LDIST_TO_FDIST``. A code that isn't in the table, or whose pair isn't
    in ``LDIST_TO_FDIST``, becomes 0 and is listed in the warnings.

    Parameters
    ----------
    ldist : numpy.ndarray
        The LDist raster, one LDist code per pixel.
    attribute_table : pandas.DataFrame
        The LANDFIRE LDist attribute table, with ``VALUE``, ``DIST_TYPE``
        and ``SEVERITY`` columns.

    Returns
    -------
    fdist : numpy.ndarray
        int32 FDist raster, same shape as ``ldist``. 0 where there is no
        disturbance or no FDist code.
    warnings : list of int
        LDist codes present in ``ldist`` that got no FDist code, ascending.

    Examples
    --------
    >>> table = pd.DataFrame({
    ...     "VALUE": [0, 2801, 2882, 2971],
    ...     "DIST_TYPE": [None, "Development", "Wildfire", "Herbicide"],
    ...     "SEVERITY": [None, "Low", "Moderate", "Low"],
    ... })
    >>> fdist, warnings = _ldist_raster_to_fdist_raster(
    ...     np.array([[2882, 2801], [0, 2971]]), table
    ... )
    >>> fdist.tolist()
    [[121, 0], [0, 0]]
    >>> warnings
    [2971]
    """
    fdist_by_code = pd.Series(
        [
            (
                0
                if dist_type in _NO_DISTURBANCE
                else LDIST_TO_FDIST.get((dist_type, severity))
            )
            for dist_type, severity in zip(
                map(_label, attribute_table["DIST_TYPE"]),
                map(_label, attribute_table["SEVERITY"]),
            )
        ],
        index=attribute_table["VALUE"].to_numpy(),
    )

    codes = pd.Series(np.ravel(ldist))
    fdist = codes.map(fdist_by_code)
    warnings = sorted(codes[fdist.isna()].unique().tolist())
    return (
        fdist.fillna(0).astype(np.int32).to_numpy().reshape(np.shape(ldist)),
        warnings,
    )


def _age_fdist(fdist: np.ndarray) -> np.ndarray:
    """Last year's FDist a year later: ``...1`` codes become ``...2``.

    Other disturbance codes are kept as they are. Undisturbed pixels (0) and
    nodata (-9999) become 0.
    """
    fdist = np.asarray(fdist)
    fdist = np.where(fdist > 0, fdist, 0)
    one_year_old = fdist % 10 == 1
    return np.where(one_year_old, fdist + 1, fdist)


def build_fdist_raster(
    disturbance: str,
    *,
    fdist: np.ndarray,
    ldist: np.ndarray | None = None,
    ldist_attribute_table: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[int]]:
    """The FDist raster the Master_Rulesets rules are applied to.

    Parameters
    ----------
    disturbance : str
        One of ``DISTURBANCE_MODES``:

        ``"ldist_and_last_year_fdist"``
            This year's LDist, converted to FDist, wherever it has a
            disturbance; everywhere else, last year's FDist, with its
            one-year-old disturbances aged to two years.
        ``"fdist"``
            This year's LANDFIRE FDist.
    fdist : numpy.ndarray
        LANDFIRE FDist: last year's for ``"ldist_and_last_year_fdist"``,
        this year's for ``"fdist"``.
    ldist : numpy.ndarray, optional
        This year's LDist. Needed for ``"ldist_and_last_year_fdist"``.
    ldist_attribute_table : pandas.DataFrame, optional
        The LANDFIRE LDist attribute table, with ``VALUE``, ``DIST_TYPE`` and
        ``SEVERITY`` columns. Needed whenever ``ldist`` is.

    Returns
    -------
    fdist : numpy.ndarray
        The FDist raster. 0 where there is no disturbance.
    warnings : list of int
        LDist codes present in ``ldist`` that got no FDist code: not in the
        attribute table, or a disturbance type and severity with no FDist
        equivalent (e.g. Herbicide). Their pixels are 0. Empty in
        ``"fdist"`` mode.

    Raises
    ------
    ValueError
        ``disturbance`` isn't one of ``DISTURBANCE_MODES``, or an input it
        needs wasn't given.
    """
    if disturbance not in DISTURBANCE_MODES:
        raise ValueError(
            f"Unknown disturbance mode {disturbance!r}; "
            f"expected one of {DISTURBANCE_MODES}."
        )
    if disturbance != "fdist" and (ldist is None or ldist_attribute_table is None):
        raise ValueError(
            f"disturbance={disturbance!r} needs ldist and ldist_attribute_table."
        )

    if disturbance == "fdist":
        return np.where(fdist > 0, fdist, 0), []

    converted, warnings = _ldist_raster_to_fdist_raster(ldist, ldist_attribute_table)

    # This year's disturbance wins; elsewhere, last year's carried forward.
    return np.where(converted > 0, converted, _age_fdist(fdist)), warnings
