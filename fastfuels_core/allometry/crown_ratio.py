"""Uncompacted crown ratio from FIA compacted crown ratio.

FIA's ``CR`` is the compacted crown ratio: lower live branches are moved up
to fill gaps in the crown before the crew estimates the ratio. The
uncompacted crown ratio (FIA ``UNCRCD``) is measured from the lowest live
foliage instead, so it is the ratio that places the crown base. The two
differ by about 13 percentage points on a typical tree.

Every equation here is a logistic model of the uncompacted ratio::

    xb  = b0 + b_ccr * ccr + b_ln_dia * ln(dia_in) + b_ln_ht * ln(ht_ft)
          + b_hd * ht_ft / dia_in
    ucr = 1 / (1 + exp(exp_sign * xb))

with ``exp_sign`` -1 for every equation but Westfall et al. (2019), whose
paper writes the opposite sign. The coefficients are rows of
``REF_CROWN_RATIO``, one ``SOURCE`` per equation, stored as printed in each
paper and with the table and page they came from in ``TABLE``.
Diameters are in centimeters and heights in meters, the FastFuels
convention; the equations' own inches and feet are converted here.

``equation`` selects the coefficients:

``"fastfuels_2026"`` (default)
    A national refit of the logistic form to FIA trees with both ratios
    recorded: a fractional logit per species, each shrunk toward its genus
    and each genus toward a softwood or hardwood pool, with species
    overrides for the Pacific Northwest (CA, OR, WA) where the national rows
    are biased (``pacific_northwest``). On held-out FIA plots it is about
    10.6 percentage points RMSE against 11.7 for the published regional
    equations. A species takes its Pacific Northwest row (when asked for),
    its national row, its genus row, then its softwood/hardwood pool, so
    every FIA species has a value. The result is never below the compacted
    ratio.

``"monleon_2004_eq1"``, ``"monleon_2004_eq2"``
    Monleon, V.J., D. Azuma, and D. Gedney. 2004. Equations for predicting
    uncompacted crown ratio based on compacted crown ratio and tree
    attributes. Western Journal of Applied Forestry 19(4):260-267.
    https://doi.org/10.1093/wjaf/19.4.260. CA, OR, WA: 7,578 live trees
    with dbh >= 5 in on 341 Forest Health Monitoring plots, 2000-2001,
    fitted to trees with CCR <= 0.90. Eq. 1 takes CCR, ln(dbh), ln(height)
    and height/dbh (ft/in), each species keeping the terms its stepwise
    selection retained (Table 2, p. 264: 28 species, Pinus spp. and Quercus
    spp.). Eq. 2 takes CCR and ln(dbh) (Table 4, p. 265: 19 species and the
    two genera). Table 4 lists only the species whose Table 2 model has a
    height term; the other nine (grand and subalpine fir, incense-cedar,
    Engelmann spruce, Jeffrey and sugar pine, red alder, tanoak, interior
    live oak) already have an Eq. 2-form model in Table 2, which Eq. 2 uses.
    Other pines and oaks take the Pinus spp. or Quercus spp. model; other
    species give NaN. The authors do not recommend Eq. 1 for subalpine fir,
    red fir, western juniper, western larch, bigleaf maple and blue oak, or
    Eq. 2 for red fir, noble fir, western juniper, bigleaf maple and blue
    oak; Eq. 2's subalpine fir model, taken from Table 2, keeps its flag
    (``RECOMMENDED`` 0). Post-rule: CCR > 0.90 gives UCR = CCR. The
    model can predict UCR < CCR (the paper reports it for bigleaf maple and
    Jeffrey pine); that is kept.

``"toney_reeves_2009"``
    Toney, C., and M.C. Reeves. 2009. Equations to convert compacted crown
    ratio to uncompacted crown ratio for trees in the Interior West.
    Western Journal of Applied Forestry 24(2):76-82.
    https://doi.org/10.1093/wjaf/24.2.76. ID, MT, WY, NV, UT, CO, AZ, NM:
    201,855 trees with diameter >= 5 in on 11,766 Interior West FIA plots,
    1978-2005, fitted to CCR <= 0.90. Takes CCR and ln(diameter); 35
    species, no genus or group models (Table 3, p. 80), so other species
    give NaN. Diameter is at the root collar (DRC) for the nine woodland
    species; the four whose equation has a diameter term (curlleaf
    mountain-mahogany, Rocky Mountain juniper, singleleaf pinyon, Gambel
    oak) give NaN, since FastFuels carries DBH only. Singleleaf pinyon is
    not recommended (``RECOMMENDED`` 0); the authors also advise caution
    below an efficiency of about 0.49, which covers the pinyons and
    junipers (0.18-0.42) and several others. Post-rule: CCR > 0.90 gives
    UCR = CCR; the paper's alternative, CCR + 0.02, is not implemented.
    UCR < CCR is kept.

``"randolph_2010"``
    Randolph, K.C. 2010. Equations relating compacted and uncompacted live
    crown ratio for common tree species in the South. Southern Journal of
    Applied Forestry 34(3):118-123. https://doi.org/10.1093/sjaf/34.3.118.
    AL, AR, FL, GA, KY, LA, NC, SC, TN, TX, VA: live trees >= 5 in on FIA
    phase 3 plots, 2003-2008, fitted to CCR <= 0.90. Takes CCR and
    ln(diameter) for 24 species and 10 genera (Table 3, p. 121), and CCR
    and ln(tree length, ft), which is the tree height, for Ashe juniper and
    the other woodland junipers (text, p. 121). Eastern and southern
    redcedar take the forest juniper model; other junipers give NaN;
    species of the ten modeled genera without their own row take the genus
    model; other species give NaN. Ashe juniper, honey mesquite and black
    locust are not recommended (``RECOMMENDED`` 0); 14 of the 36 models
    have an efficiency below 0.49. Post-rule: CCR > 0.90 gives UCR = CCR.
    UCR < CCR is kept.

``"westfall_2019"``
    Westfall, J.A., M.B.E. Westfall, and K.C. Randolph. 2019. Modeling
    relations between compacted and uncompacted crown ratio for the
    northern United States. Forest Science 65(5):593-601.
    https://doi.org/10.1093/forsci/fxz029; erratum Forest Science 67(2):241,
    https://doi.org/10.1093/forsci/fxz067, corrects an author name only.
    The 24 Northern Research Station states (CT, DE, IA, IL, IN, KS, MA,
    MD, ME, MI, MN, MO, ND, NE, NH, NJ, NY, OH, PA, RI, SD, VT, WI, WV):
    234,854 trees with dbh >= 1 in, all CCR. Takes CCR and ln(dbh) in
    ``1 / (1 + exp(+xb))`` (Table 2, p. 597), with the condition random
    effects at zero, as for prediction without them (Table 3). 27 species
    groups, which follow FIA's eastern species groups (``E_SPGRPCD``),
    eastern white and red pine split into their own groups (the paper lists
    group names, not species). Western species FIA places in those groups
    take them, e.g. Douglas-fir in other eastern softwoods; species in FIA
    groups the paper lacks (longleaf/slash pine, cypress, woodland and
    tropical groups) give NaN. Post-rule: UCR = CCR only when the
    prediction is below CCR, at every CCR.

For every equation a compacted ratio of zero gives zero (no crown, not a
compact one), whatever the post-rule or species coverage would give. A
compacted ratio outside [0, 1] raises, as does a species code outside the
FIA reference table; a missing compacted ratio, a missing or non-positive
diameter or height, or a species or input the equation does not cover gives
NaN. Each published equation was fitted in its own region; outside it,
and for the genus and group fallbacks, it is an extrapolation.

``estimate_crown_base_height`` turns the uncompacted ratio into the height
of the crown base in meters.
"""

from __future__ import annotations

from functools import cache, lru_cache

import numpy as np
import pandas as pd

from fastfuels_core.ref_data import DATA_PATH, REF_SPECIES
from fastfuels_core.units import CM_TO_IN, M_TO_FT

PACIFIC_NORTHWEST_STATES = frozenset({"CA", "OR", "WA"})

FASTFUELS_2026 = "fastfuels_2026"
EQUATIONS = (
    FASTFUELS_2026,
    "monleon_2004_eq1",
    "monleon_2004_eq2",
    "toney_reeves_2009",
    "randolph_2010",
    "westfall_2019",
)

# Equations whose paper sets UCR = CCR above this compacted ratio
_IDENTITY_ABOVE_090 = frozenset(
    {"monleon_2004_eq1", "monleon_2004_eq2", "toney_reeves_2009", "randolph_2010"}
)
# Equations that never go below the compacted ratio
_NOT_BELOW_CCR = frozenset({FASTFUELS_2026, "westfall_2019"})

_COEFFICIENTS = ["B0", "B_CCR", "B_LN_DIA", "B_LN_HT", "B_HD", "EXP_SIGN"]


@lru_cache(maxsize=1)
def _coefficient_rows() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "REF_CROWN_RATIO.csv", dtype={"KEY": str})


def _keyed(rows: pd.DataFrame, key_type: str) -> pd.DataFrame:
    found = rows[rows.KEY_TYPE == key_type]
    columns = _COEFFICIENTS + ["NEEDS_DRC"] * ("NEEDS_DRC" in rows)
    table = found.set_index("KEY")[columns]
    if key_type in ("SPCD", "SPGRP"):
        table.index = table.index.astype(int)
    return table


def _by_species(table: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Rows of ``table`` looked up by a per-species key, indexed by SPCD."""
    return table.reindex(key).set_axis(REF_SPECIES.index)


@cache
def _species_coefficients(equation: str, pacific_northwest: bool) -> pd.DataFrame:
    """Coefficients per FIA species code, the lookup cascade already applied.

    Species the equation does not cover, or whose row needs a diameter at
    the root collar, have NaN coefficients.
    """
    rows = _coefficient_rows()
    rows = rows[rows.SOURCE == equation]
    if equation == FASTFUELS_2026:
        national = rows[rows.REGION == "national"]
        # Four reference species have no softwood/hardwood class; the fit
        # pooled them with the hardwoods.
        pool = REF_SPECIES["SFTWD_HRDWD"].fillna("H")
        pool = pool.map({"S": "softwood", "H": "hardwood"})
        table = _by_species(_keyed(national, "POOLED"), pool)
        overrides = [
            _by_species(_keyed(national, "GENUS"), REF_SPECIES["GENUS"]),
            _keyed(national, "SPCD"),
        ]
        if pacific_northwest:
            overrides.append(_keyed(rows[rows.REGION == "pacific_northwest"], "SPCD"))
    else:
        # DBH cannot stand in for a diameter at the root collar
        needs_drc = (rows.DIAMETER == "DRC") & ((rows.B_LN_DIA != 0) | (rows.B_HD != 0))
        rows = rows.assign(NEEDS_DRC=needs_drc.astype(float))
        columns = _COEFFICIENTS + ["NEEDS_DRC"]
        table = pd.DataFrame(np.nan, index=REF_SPECIES.index, columns=columns)
        woodland = REF_SPECIES["GENUS"].where(REF_SPECIES["WOODLAND"] == "Y")
        overrides = [
            _by_species(_keyed(rows, "SPGRP"), REF_SPECIES["E_SPGRPCD"]),
            _by_species(_keyed(rows, "GENUS"), REF_SPECIES["GENUS"]),
            _by_species(_keyed(rows, "WOODLAND_GENUS"), woodland),
            _keyed(rows, "SPCD"),
        ]
    for override in overrides:
        table.update(override)
    if "NEEDS_DRC" in table:
        table.loc[table.pop("NEEDS_DRC") == 1] = np.nan
    return table


def _species_codes(species_code) -> np.ndarray:
    """Species codes as int64; raises ValueError for non-integral codes."""
    spcd = np.atleast_1d(np.asarray(species_code))
    if np.issubdtype(spcd.dtype, np.integer):
        return spcd.astype(np.int64)
    try:
        as_float = spcd.astype(np.float64)
    except (TypeError, ValueError):
        as_float = np.array([np.nan])
    if not (np.isfinite(as_float).all() and (as_float % 1 == 0).all()):
        raise ValueError(
            f"Species codes must be integer FIA codes; got "
            f"{np.unique(spcd.astype(str))[:5].tolist()}."
        )
    return as_float.astype(np.int64)


def _check_equation(equation: str, pacific_northwest) -> None:
    if equation not in EQUATIONS:
        raise ValueError(
            f"Unknown crown ratio equation {equation!r}; expected one of "
            f"{list(EQUATIONS)}."
        )
    if equation != FASTFUELS_2026 and np.any(pacific_northwest):
        raise ValueError(
            f"pacific_northwest applies only to {FASTFUELS_2026!r}, not {equation!r}."
        )


def estimate_uncompacted_crown_ratio(
    species_code: np.ndarray,
    compacted_crown_ratio: np.ndarray,
    dbh_cm: np.ndarray,
    height_m: np.ndarray,
    pacific_northwest: bool | np.ndarray = False,
    equation: str = FASTFUELS_2026,
) -> np.ndarray:
    """Per-tree uncompacted crown ratio (fraction) from the compacted ratio.

    Parameters
    ----------
    species_code : array_like
        FIA species codes (SPCD).
    compacted_crown_ratio : array_like
        FIA compacted crown ratio ``CR`` as a fraction in [0, 1].
    dbh_cm : array_like
        Diameter at breast height (cm).
    height_m : array_like
        Total tree height (m).
    pacific_northwest : bool or array_like, optional
        Scalar or per tree; selects the CA/OR/WA species rows of
        ``"fastfuels_2026"`` (see ``PACIFIC_NORTHWEST_STATES``). Must be
        False for the published equations.
    equation : str, optional
        One of ``EQUATIONS``: ``"fastfuels_2026"`` (default),
        ``"monleon_2004_eq1"`` and ``"monleon_2004_eq2"`` (CA, OR, WA),
        ``"toney_reeves_2009"`` (Interior West), ``"randolph_2010"`` (South)
        or ``"westfall_2019"`` (North). See the module docstring for each
        equation's reference, coverage, post-rule and caveats.

    Returns
    -------
    numpy.ndarray
        Uncompacted crown ratio in [0, 1]. Zero where the compacted ratio is
        zero. NaN for a missing compacted ratio, a missing or non-positive
        diameter or height, a species the equation does not cover, or a
        species whose equation needs a diameter at the root collar.

    Raises
    ------
    ValueError
        For an unknown ``equation``, ``pacific_northwest`` with a published
        equation, a compacted ratio outside [0, 1] (a percentage passed by
        mistake), species codes that are not integers, and species codes
        outside the FIA reference table.
    """
    _check_equation(equation, pacific_northwest)
    spcd = _species_codes(species_code)
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

    coefficients = _species_coefficients(equation, False).reindex(spcd).to_numpy()
    if pnw.any():
        regional = _species_coefficients(equation, True).reindex(spcd).to_numpy()
        coefficients = np.where(pnw[:, None], regional, coefficients)
    b0, b_ccr, b_ln_dia, b_ln_ht, b_hd, exp_sign = coefficients.T

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        valid = (dia_in > 0) & (ht_ft > 0)
        dia_in = np.where(valid, dia_in, np.nan)
        ht_ft = np.where(valid, ht_ft, np.nan)
        xb = b0 + b_ccr * ccr + b_ln_dia * np.log(dia_in) + b_ln_ht * np.log(ht_ft)
        xb = xb + b_hd * ht_ft / dia_in
        ucr = 1.0 / (1.0 + np.exp(exp_sign * xb))
    if equation in _IDENTITY_ABOVE_090:
        # Only for species the equation covers
        ucr = np.where((ccr > 0.90) & np.isfinite(b0), ccr, ucr)
    if equation in _NOT_BELOW_CCR:
        ucr = np.maximum(ucr, ccr)
    ucr = np.where(valid, np.clip(ucr, 0.0, 1.0), np.nan)
    return np.where(ccr == 0.0, 0.0, ucr)


def estimate_crown_base_height(
    species_code: np.ndarray,
    compacted_crown_ratio: np.ndarray,
    dbh_cm: np.ndarray,
    height_m: np.ndarray,
    pacific_northwest: bool | np.ndarray = False,
    equation: str = FASTFUELS_2026,
) -> np.ndarray:
    """Per-tree crown base height (m) from the compacted crown ratio.

    The height of the lowest live foliage, ``height_m * (1 - ucr)`` with
    ``ucr`` from ``estimate_uncompacted_crown_ratio``, which documents the
    parameters, ``equation`` included. A compacted ratio of zero gives the
    full height whenever the height is valid; NaN wherever the uncompacted
    ratio is NaN or the height is missing or non-positive, and the same
    ValueErrors apply.
    """
    ht_m = np.atleast_1d(np.asarray(height_m, dtype=np.float64))
    ucr = estimate_uncompacted_crown_ratio(
        species_code,
        compacted_crown_ratio,
        dbh_cm,
        height_m,
        pacific_northwest,
        equation=equation,
    )
    return np.where(ht_m > 0, ht_m * (1.0 - ucr), np.nan)
