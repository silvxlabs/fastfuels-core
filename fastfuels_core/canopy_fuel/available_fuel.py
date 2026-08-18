"""Per-tree available canopy fuel (kg), and the crown-class multiplier.

Available canopy fuel is the part of a crown that burns in the flaming
front: foliage plus the fine end of the branchwood. What varies between
configurations is where the two weights come from and whether they are
scaled by the tree's position in the canopy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.allometry import brown, nsvb
from fastfuels_core.canopy_fuel.ref_data import (
    fuelcalc_crown_class_factors,
    fuelcalc_species,
)
from fastfuels_core.units import conversion_factor

VALID_EQUATIONS = ("nsvb", "brown_1978")
NO_CROWN_CLASS_ADJUSTMENT = "none"
REINHARDT_CROWN_CLASS_ADJUSTMENT = "reinhardt_2006"
VALID_CROWN_CLASS_ADJUSTMENTS = (
    NO_CROWN_CLASS_ADJUSTMENT,
    REINHARDT_CROWN_CLASS_ADJUSTMENT,
)


# The three crown class codes CC_Adj folds onto another class before
# the lookup. Codes are FuelCalc's (guide p. 17, tscc_def.h:19-28);
# anything not listed here or in _CROWN_CLASS_INDEX, "N" included,
# takes the Other/none column. Public so callers carrying a different
# coding -- FIA's CCLCD, a field protocol, a model's output -- can see
# which folds this layer already performs and map onto the letters
# rather than duplicating the logic.
CROWN_CLASS_REMAP = {"SC": "I", "O": "C", "E": "D"}
_CROWN_CLASS_COLUMNS = [
    "DOMINANT",
    "CODOMINANT",
    "INTERMEDIATE",
    "SUPPRESSED",
    "OTHER_NONE",
]
_CROWN_CLASS_INDEX = {"D": 0, "C": 1, "I": 2, "S": 3}


def crown_class_factor(
    species_code: np.ndarray, crown_class: np.ndarray | None = None
) -> np.ndarray:
    """FuelCalc's per-tree crown-class biomass multiplier.

    Reinhardt, Scott, Gray & Keane (2006) fitted a multiplier per
    species and crown class as the value minimising the average
    difference between observed crown biomass and the biomass predicted
    by Brown's equations. FuelCalc applies it to every crown component
    (``PTL_SetBioMass``), so available fuel scales by the same factor.
    Species resolve to a factor row through the species table's
    ``CROWN_REDUC_CODE``.

    Because the multipliers were fitted against *Brown's* predictions,
    they carry that correction and nothing else. Applying them on top of
    a different biomass model is a defensible thing to want and is not
    prevented here, but it is no longer the published adjustment.

    Parameters
    ----------
    species_code : numpy.ndarray
        FIA species codes.
    crown_class : numpy.ndarray, optional
        FuelCalc crown class codes per tree — ``D`` dominant, ``C``
        codominant, ``I`` intermediate, ``S`` suppressed, ``O`` open
        grown, ``E`` emergent, ``SC`` subcanopy, ``N`` none. ``O``,
        ``E`` and ``SC`` fold onto ``C``, ``D`` and ``I`` respectively,
        as ``CC_Adj`` does; anything unrecognised takes the Other/none
        column. Omitted, every tree takes Other/none — the factor for a
        species then no longer varies with position in the canopy,
        which is the whole point of the adjustment, so pass the codes
        whenever the inventory has them. ``CROWN_CLASS_REMAP`` is public
        so a caller translating some other coding onto these can see
        exactly which folds are already applied.

    Returns
    -------
    numpy.ndarray
        Multiplier per tree.

    Raises
    ------
    ValueError
        For species outside the FuelCalc species table.
    """
    spcd = np.asarray(species_code)
    species = fuelcalc_species()
    unknown = np.setdiff1d(spcd, species.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FuelCalc "
            f"species table, so they have no crown-class factors."
        )
    factors = fuelcalc_crown_class_factors()
    matrix = factors[_CROWN_CLASS_COLUMNS].to_numpy(dtype=np.float64)
    rows = factors.index.get_indexer(species["CROWN_REDUC_CODE"].loc[spcd])

    if crown_class is None:
        columns = np.full(spcd.shape, len(_CROWN_CLASS_COLUMNS) - 1)
    else:
        codes = np.asarray(crown_class, dtype=object)
        columns = np.array(
            [
                _CROWN_CLASS_INDEX.get(
                    CROWN_CLASS_REMAP.get(
                        str(c).strip().upper(), str(c).strip().upper()
                    ),
                    len(_CROWN_CLASS_COLUMNS) - 1,
                )
                for c in codes
            ]
        )
    return matrix[rows, columns]


def _normalize_crown_class_adjustment(crown_class_adjustment: str | None) -> str:
    if crown_class_adjustment is None:
        return NO_CROWN_CLASS_ADJUSTMENT
    return crown_class_adjustment


def _validate_crown_class_options(
    crown_class_adjustment: str, crown_class_column: str | None
) -> None:
    if crown_class_adjustment not in VALID_CROWN_CLASS_ADJUSTMENTS:
        raise ValueError(
            f"Unknown crown_class_adjustment {crown_class_adjustment!r}; "
            f"expected 'none', None, or 'reinhardt_2006'."
        )
    # The two arguments are one decision: an adjustment needs the data
    # it adjusts by, and the data is pointless without the adjustment.
    # Requiring both keeps either half from failing quietly.
    if (
        crown_class_column is not None
        and crown_class_adjustment == NO_CROWN_CLASS_ADJUSTMENT
    ):
        raise ValueError(
            f"crown_class_column={crown_class_column!r} was given but "
            f"crown_class_adjustment is 'none', so the column would be "
            f"ignored. Set crown_class_adjustment='reinhardt_2006' to "
            f"use it, or drop the column argument."
        )
    if (
        crown_class_adjustment == REINHARDT_CROWN_CLASS_ADJUSTMENT
        and crown_class_column is None
    ):
        raise ValueError(
            "crown_class_adjustment='reinhardt_2006' needs "
            "crown_class_column to say where crown position comes from. "
            "Without it every tree would take the table's Other/none "
            "column, which is 0.5 for 50 of the 54 species — a silent "
            "halving rather than an adjustment. If that is genuinely "
            "what you want, pass a column of 'N', which is what "
            "FuelCalc does with a blank crown class field."
        )


def _validate_equations(equations: str) -> None:
    if equations not in VALID_EQUATIONS:
        raise ValueError(
            f"Unknown equations {equations!r}; expected 'nsvb' or " f"'brown_1978'."
        )


def _fuelcalc_species_for_trees(species_code: np.ndarray) -> pd.DataFrame:
    species = fuelcalc_species()
    unknown = np.setdiff1d(species_code, species.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FuelCalc "
            f"species table. FuelCalc errors on species without assigned "
            f"equations; supply per-tree fuel via fuel_column to bypass "
            f"allometry for these species."
        )
    return species


def _brown_1978_crown_components(
    species: pd.DataFrame,
    species_code: np.ndarray,
    foliage_equation_id: np.ndarray,
    dia_in: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Brown 1978 Table 1 crown weight, split into foliage and branchwood.

    One crown weight, split by P1. Brown's proportions are of the whole
    crown, so branchwood is what P1 leaves behind and the fine share
    applies to it exactly as it does under NSVB.
    """
    crown_kg = brown.crown_weight(
        species["TOTAL"].loc[species_code].to_numpy(), dia_in
    ) * conversion_factor("lb", "kg")
    p1 = brown.foliage_fraction(foliage_equation_id, dia_in)
    return crown_kg * p1, crown_kg * (1.0 - p1)


def _nsvb_crown_components(
    trees: pd.DataFrame, species_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Foliage and branch dry weight from NSVB (Westfall et al. 2024)."""
    return (
        nsvb.foliage_biomass(species_code, trees["dbh"], trees["height"]),
        nsvb.branch_biomass(species_code, trees["dbh"], trees["height"]),
    )


def _apply_crown_class_adjustment(
    fuel: np.ndarray,
    trees: pd.DataFrame,
    species_code: np.ndarray,
    crown_class_adjustment: str,
    crown_class_column: str | None,
) -> np.ndarray:
    if crown_class_adjustment != REINHARDT_CROWN_CLASS_ADJUSTMENT:
        return fuel
    if crown_class_column not in trees.columns:
        raise ValueError(
            f"crown_class_column={crown_class_column!r} is not a "
            f"column of the tree frame. Available columns: "
            f"{sorted(trees.columns)}."
        )
    crown_classes = trees[crown_class_column].to_numpy()
    return fuel * crown_class_factor(species_code, crown_classes)


def available_canopy_fuel(
    trees: pd.DataFrame,
    *,
    fuel_column: str | None = None,
    equations: str = "nsvb",
    crown_class_adjustment: str = NO_CROWN_CLASS_ADJUSTMENT,
    crown_class_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
) -> np.ndarray:
    """Per-tree available canopy fuel (kg).

    With ``fuel_column`` set, that column is returned as-is (precomputed
    fuel, e.g. from LiDAR regression); everything else is ignored.
    Otherwise fuel is ``foliage_fraction * foliage + branchwood_fraction
    * fine branchwood``, where fine branchwood is the total branch
    weight times the Brown (1978) / Snell & Little (1983) fine share
    ``(P2 - P1) / (1 - P1)`` (see
    :mod:`fastfuels_core.allometry.brown`).

    ``equations`` selects where the two weights come from:

    ``"nsvb"`` (default)
        Foliage and branch dry weight from NSVB (Westfall et al. 2024),
        the national estimator FastFuels uses elsewhere. Needs ``dbh``,
        ``height`` and ``fia_species_code``.
    ``"brown_1978"``
        Total live crown weight from Brown 1978 Table 1, split into
        foliage and branchwood by ``P1``. This is the crown weight model
        FuelCalc uses, so it is the arm to select when reproducing
        FuelCalc. Diameter-only, so ``height`` is not read, and it is
        scoped to Brown's eleven Rocky Mountain conifers -- hardwoods
        and everything else raise.

    ``crown_class_adjustment`` scales the result per tree:

    ``"none"`` (default; ``None`` is accepted for it)
        No adjustment.
    ``"reinhardt_2006"``
        FuelCalc's crown-class multipliers, via
        :func:`crown_class_factor`. Requires ``crown_class_column``,
        naming the per-tree column that holds crown position —
        measured, imputed, or modelled.

    The two arguments are one decision and are validated together:
    asking for the adjustment without the column raises, and so does
    naming the column without the adjustment. Neither half is useful
    alone, and both failures are silent if allowed through — the
    column would be discarded, or every tree would take the table's
    Other/none factor, which is 0.5 for 50 of the 54 species and so
    halves nearly every tree instead of varying it by species and
    canopy position. To get that uniform behaviour deliberately, pass
    a column of ``"N"``; that is what FuelCalc does with a blank crown
    class field, and it says so at the call site.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(trees),)``, kg per tree.

    Raises
    ------
    ValueError
        For an unknown ``equations`` choice, or for species the chosen
        equations do not cover — codes outside the FuelCalc species
        table, or whose equations are not implemented (pinyon/juniper,
        the eastern newer species, and under ``"brown_1978"`` the
        hardwoods too). Mirrors FuelCalc's treelist error;
        ``fuel_column`` bypasses allometry entirely.
    """
    if fuel_column is not None:
        return trees[fuel_column].to_numpy(dtype=np.float64)

    crown_class_adjustment = _normalize_crown_class_adjustment(crown_class_adjustment)
    _validate_crown_class_options(crown_class_adjustment, crown_class_column)
    _validate_equations(equations)

    if len(trees) == 0:
        return np.zeros(0, dtype=np.float64)

    spcd = trees["fia_species_code"].to_numpy()
    species = _fuelcalc_species_for_trees(spcd)

    # Brown/SL-83 proportions work in inches; the nsvb wrapper is metric.
    dia_in = trees["dbh"].to_numpy() * conversion_factor("cm", "inch")
    fol_id = species["FOL"].loc[spcd].to_numpy()
    twig_id = species["TWIG"].loc[spcd].to_numpy()
    fine_share = brown.fine_branchwood_share(fol_id, twig_id, dia_in)

    if equations == "brown_1978":
        foliage_kg, branch_kg = _brown_1978_crown_components(
            species, spcd, fol_id, dia_in
        )
    else:
        foliage_kg, branch_kg = _nsvb_crown_components(trees, spcd)

    fuel = foliage_fraction * foliage_kg + branchwood_fraction * fine_share * branch_kg
    return _apply_crown_class_adjustment(
        fuel, trees, spcd, crown_class_adjustment, crown_class_column
    )
