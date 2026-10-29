"""Uncompacted crown ratio from FIA compacted crown ratio.

FIA's ``CR`` is the compacted crown ratio: lower live branches are moved up
to fill gaps in the crown before the crew estimates the ratio. The
uncompacted crown ratio (FIA ``UNCRCD``) is measured from the lowest live
foliage instead, so it is the ratio that places the crown base. The two
differ by about 13 percentage points on a typical tree.

One logistic model predicts the uncompacted ratio::

    ucr = 1 / (1 + exp(-(b0 + b_ccr * ccr + b_ln_dia * ln(dia_in)
                         + b_ln_ht * ln(ht_ft) + b_hd * ht_ft / dia_in)))

the form of the published regional equations (Monleon et al. 2004, Toney
& Reeves 2009, Randolph 2010, Westfall et al. 2019). The coefficients in
``REF_CROWN_RATIO`` are a national refit of that form to FIA trees with
both ratios recorded: a fractional logit per species, each shrunk toward
its genus and each genus toward a softwood or hardwood pool, with species
overrides for the Pacific Northwest (CA, OR, WA) where the national rows
are biased. On held-out FIA plots it is about 10.6 percentage points RMSE
against 11.7 for the published regional equations.

A species takes the most specific row available: its Pacific Northwest
row (when asked for), its national row, its genus row, then the pool for
its softwood/hardwood class. The result is never below the compacted
ratio, and a compacted ratio of zero stays zero (no crown, not a compact
one). Diameters are in centimeters and heights in meters, the FastFuels
convention; the model's own inches and feet are converted here.

``crown_base_height`` turns the uncompacted ratio into the height of the
crown base in meters.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fastfuels_core.ref_data import DATA_PATH, REF_SPECIES
from fastfuels_core.units import CM_TO_IN, M_TO_FT

PACIFIC_NORTHWEST_STATES = frozenset({"CA", "OR", "WA"})

_COEFFICIENTS = ["B0", "B_CCR", "B_LN_DIA", "B_LN_HT", "B_HD"]


@lru_cache(maxsize=1)
def _coefficient_rows() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "REF_CROWN_RATIO.csv", dtype={"KEY": str})


@lru_cache(maxsize=2)
def _species_coefficients(pacific_northwest: bool) -> pd.DataFrame:
    """Coefficients per FIA species code, the lookup cascade already applied."""
    rows = _coefficient_rows()
    national = rows[rows.REGION == "national"]

    def keyed(key_type: str, region_rows: pd.DataFrame = national) -> pd.DataFrame:
        found = region_rows[region_rows.KEY_TYPE == key_type]
        return found.set_index("KEY")[_COEFFICIENTS]

    # Four reference species have no softwood/hardwood class; the fit pooled
    # them with the hardwoods.
    pool = (
        REF_SPECIES["SFTWD_HRDWD"].fillna("H").map({"S": "softwood", "H": "hardwood"})
    )
    table = keyed("POOLED").reindex(pool).set_axis(REF_SPECIES.index)
    overrides = [
        keyed("GENUS").reindex(REF_SPECIES["GENUS"]).set_axis(REF_SPECIES.index),
        keyed("SPCD").set_axis(keyed("SPCD").index.astype(int)),
    ]
    if pacific_northwest:
        regional = keyed("SPCD", rows[rows.REGION == "pacific_northwest"])
        overrides.append(regional.set_axis(regional.index.astype(int)))
    for override in overrides:
        table.update(override)
    return table


def uncompacted_crown_ratio(
    species_code: np.ndarray,
    compacted_crown_ratio: np.ndarray,
    dbh_cm: np.ndarray,
    height_m: np.ndarray,
    pacific_northwest: bool | np.ndarray = False,
) -> np.ndarray:
    """Per-tree uncompacted crown ratio (fraction) from the compacted ratio.

    ``pacific_northwest`` (scalar or per tree) selects the CA/OR/WA species
    rows; see ``PACIFIC_NORTHWEST_STATES``. A missing compacted ratio, or a
    missing or non-positive diameter or height, gives NaN. Raises ValueError
    for a compacted ratio outside [0, 1] (a percentage passed by mistake) and
    for species codes outside the FIA reference table.
    """
    spcd = np.atleast_1d(np.asarray(species_code))
    ccr = np.atleast_1d(np.asarray(compacted_crown_ratio, dtype=np.float64))
    dia_in = np.atleast_1d(np.asarray(dbh_cm, dtype=np.float64)) * CM_TO_IN
    ht_ft = np.atleast_1d(np.asarray(height_m, dtype=np.float64)) * M_TO_FT
    pnw = np.broadcast_to(np.asarray(pacific_northwest, dtype=bool), spcd.shape)

    out_of_range = (ccr < 0) | (ccr > 1)
    if out_of_range.any():
        raise ValueError(
            f"Compacted crown ratio must be a fraction in [0, 1]; got "
            f"{np.unique(ccr[out_of_range])[:5].tolist()}."
        )
    unknown = np.setdiff1d(spcd, REF_SPECIES.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FIA reference "
            f"table, so they have no crown ratio coefficients."
        )

    national = _species_coefficients(False).reindex(spcd).to_numpy()
    regional = _species_coefficients(True).reindex(spcd).to_numpy()
    b0, b_ccr, b_ln_dia, b_ln_ht, b_hd = np.where(pnw[:, None], regional, national).T

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (dia_in > 0) & (ht_ft > 0)
        dia_in = np.where(valid, dia_in, np.nan)
        ht_ft = np.where(valid, ht_ft, np.nan)
        z = b0 + b_ccr * ccr + b_ln_dia * np.log(dia_in) + b_ln_ht * np.log(ht_ft)
        z = z + b_hd * ht_ft / dia_in
        ucr = 1.0 / (1.0 + np.exp(-z))
    ucr = np.fmax(ucr, ccr)
    ucr = np.where(ccr == 0.0, 0.0, ucr)
    return np.where(valid, np.clip(ucr, 0.0, 1.0), np.nan)


def crown_base_height(
    species_code: np.ndarray,
    compacted_crown_ratio: np.ndarray,
    dbh_cm: np.ndarray,
    height_m: np.ndarray,
    pacific_northwest: bool | np.ndarray = False,
) -> np.ndarray:
    """Per-tree crown base height (m) from the compacted crown ratio.

    The height of the lowest live foliage, the base of the uncompacted crown
    from ``uncompacted_crown_ratio``. A compacted ratio of zero gives the
    full height; unknown inputs give NaN, and the same ValueErrors apply.
    """
    ht_m = np.atleast_1d(np.asarray(height_m, dtype=np.float64))
    ucr = uncompacted_crown_ratio(
        species_code, compacted_crown_ratio, dbh_cm, height_m, pacific_northwest
    )
    return ht_m * (1.0 - ucr)
