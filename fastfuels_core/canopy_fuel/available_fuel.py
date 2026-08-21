"""Per-tree available canopy fuel (kg), and the crown-class multiplier.

Available canopy fuel is the part of a crown that burns in the flaming
front: foliage plus the fine end of the branchwood. What varies between
configurations is where the two weights come from and whether they are
scaled by the tree's position in the canopy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.allometry import brown, jenkins, nsvb
from fastfuels_core.ref_data import REF_SPECIES
from fastfuels_core.canopy_fuel.ref_data import (
    fuelcalc_crown_class_factors,
    fuelcalc_small_tree_biomass,
    fuelcalc_species,
)
from fastfuels_core.units import CM_TO_IN, LB_TO_KG, M_TO_FT

VALID_EQUATIONS = ("nsvb", "brown_1978", "jenkins")
# Equation families that report only total branchwood, with no size class:
# a branchwood size partition must be applied to reach the fine class.
TOTAL_BRANCHWOOD_EQUATIONS = ("nsvb", "jenkins")
BROWN_PROPORTIONS_PARTITION = "brown_proportions"
NO_PARTITION = "none"
VALID_BRANCHWOOD_SIZE_PARTITIONS = (BROWN_PROPORTIONS_PARTITION, NO_PARTITION)
# Above this dbh the crown-weight equations apply; at or below it the
# small-tree table does.
SMALL_TREE_MAX_DIA_IN = 1.0
NO_CROWN_CLASS_ADJUSTMENT = "none"
FUELCALC_CROWN_CLASS_ADJUSTMENT = "fuelcalc_table"
VALID_CROWN_CLASS_ADJUSTMENTS = (
    NO_CROWN_CLASS_ADJUSTMENT,
    FUELCALC_CROWN_CLASS_ADJUSTMENT,
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

    The table (FuelCalc 1.7 User Guide, Appendix D) has two sources,
    recorded in its ``SOURCE`` column. The six ``KG`` rows — WF, PP, PS,
    IC, DF, LP — are Kathy Gray's fits for Reinhardt, Scott, Gray &
    Keane (2006): per species and crown class, the multiplier that
    minimises the average difference between observed crown biomass and
    Brown's prediction. The ``BK`` rows — WL, WP, WC — are Keane's
    FIRESUM / Fire-BGC factors (Keane, Arno & Brown 1989; Keane, Morgan
    & Running 1996). FuelCalc applies the factor to every crown
    component (``PTL_SetBioMass``), so available fuel scales by it.
    Species resolve to a row through the species table's
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


def _validate_crown_class_options(
    crown_class_adjustment: str, crown_class_column: str | None
) -> None:
    if crown_class_adjustment not in VALID_CROWN_CLASS_ADJUSTMENTS:
        raise ValueError(
            f"Unknown crown_class_adjustment {crown_class_adjustment!r}; "
            f"expected 'none' or 'fuelcalc_table'."
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
            f"ignored. Set crown_class_adjustment='fuelcalc_table' to "
            f"use it, or drop the column argument."
        )
    if (
        crown_class_adjustment == FUELCALC_CROWN_CLASS_ADJUSTMENT
        and crown_class_column is None
    ):
        raise ValueError(
            "crown_class_adjustment='fuelcalc_table' needs "
            "crown_class_column to say where crown position comes from. "
            "Without it every tree would take the table's Other/none "
            "column, which is 0.5 for 51 of the 55 species — a silent "
            "halving rather than an adjustment. If that is genuinely "
            "what you want, pass a column of 'N', which is what "
            "FuelCalc does with a blank crown class field."
        )


def _validate_equations(equations: str) -> None:
    if equations not in VALID_EQUATIONS:
        raise ValueError(
            f"Unknown equations {equations!r}; expected one of "
            f"{list(VALID_EQUATIONS)}."
        )


def _validate_branchwood_size_partition(branchwood_size_partition: str) -> None:
    if branchwood_size_partition not in VALID_BRANCHWOOD_SIZE_PARTITIONS:
        raise ValueError(
            f"Unknown branchwood_size_partition {branchwood_size_partition!r}; "
            f"expected 'brown_proportions' or 'none'."
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
    crown_kg = (
        brown.crown_weight(species["TOTAL"].loc[species_code].to_numpy(), dia_in)
        * LB_TO_KG
    )
    p1 = brown.foliage_fraction(foliage_equation_id, dia_in)
    return crown_kg * p1, crown_kg * (1.0 - p1)


def small_tree_crown_components(
    total_equation_code: np.ndarray, dia_in: np.ndarray, height_ft: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulated foliage and fine branchwood (kg) for saplings.

    Brown (1978) fitted Table 1 to dominant and codominant trees over
    one inch dbh, so below that the crown-weight equations are outside
    the range of the data behind them. Several are additive in
    diameter squared, and their intercepts dominate at sapling size:
    subalpine fir predicts 7.7 lb of crown at 0.5 in against 10.2 lb
    at 1.5 in, so crown weight is nearly flat across the whole range
    where the equations no longer apply.

    Under one inch, this reads measured component weights from
    :func:`~fastfuels_core.canopy_fuel.ref_data.fuelcalc_small_tree_biomass`
    instead, keyed by the species' crown-weight equation code and the
    tree's height. The table gives foliage and 1-hour weights directly,
    so the accumulative proportions that split a fitted crown weight do
    not enter.

    Returns
    -------
    tuple of numpy.ndarray
        Foliage and fine branchwood, kg per tree. Entries for trees
        over one inch are zero; callers select which trees to take
        from here.

    Raises
    ------
    ValueError
        For an equation code the table does not carry. Every Id
        :func:`~fastfuels_core.allometry.brown.crown_weight` accepts is
        in it, so this only fires for a code that has no crown-weight
        equation either.
    """
    table = fuelcalc_small_tree_biomass()
    codes = np.asarray(total_equation_code, dtype=object)
    covered = table.index.get_level_values("CODE").unique()
    unknown = sorted(set(codes) - set(covered))
    if unknown:
        raise ValueError(
            f"No small-tree biomass row for FuelCalc equation Id(s) {unknown}."
        )
    # h <= 1 is class 1, h <= 2 class 2, ..., anything over 9 class 10.
    # Heights reach here through a metre-to-foot conversion, so a tree
    # measured at a whole number of feet lands a few ulp either side of
    # it; round before the ceiling or half of them take the class above.
    height_ft = np.round(np.asarray(height_ft, dtype=np.float64), 6)
    ht_class = np.clip(np.ceil(height_ft), 1, 10).astype(int)
    keys = pd.MultiIndex.from_arrays([codes, ht_class])
    rows = table.reindex(keys)
    in_range = np.asarray(dia_in, dtype=np.float64) <= SMALL_TREE_MAX_DIA_IN
    foliage = np.where(in_range, rows["FOL_LB"].to_numpy(), 0.0) * LB_TO_KG
    twig = np.where(in_range, rows["ONE_HR_LB"].to_numpy(), 0.0) * LB_TO_KG
    return foliage, twig


# NSVB has no crown model for the woodland species group (Jenkins group
# 10: junipers, pinyon, oak, mesquite) and would raise for it. Those trees
# are priced by Jenkins instead, completing the Jenkins-group fallback NSVB
# is built on -- the same substitution the Tree biomass model makes.
WOODLAND_JENKINS_GROUP = 10


def _nsvb_crown_components(
    trees: pd.DataFrame, species_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Foliage and branch dry weight from NSVB, Jenkins for the woodland group.

    NSVB prices most species directly and falls back to the Jenkins
    species-group equations for the rest, but carries no fallback row for
    the woodland group (Jenkins group 10), so it cannot price those. Those
    trees are priced by Jenkins here (see
    :mod:`fastfuels_core.allometry.jenkins`); every other tree by NSVB.
    """
    groups = REF_SPECIES["JENKINS_SPGRPCD"].reindex(species_code).to_numpy()
    use_jenkins = groups == WOODLAND_JENKINS_GROUP
    foliage = np.empty(len(trees), dtype=np.float64)
    branch = np.empty(len(trees), dtype=np.float64)
    if use_jenkins.any():
        dbh_cm = trees["dbh"].to_numpy(dtype=np.float64)[use_jenkins]
        foliage[use_jenkins] = jenkins.foliage_biomass(
            species_code[use_jenkins], dbh_cm
        )
        branch[use_jenkins] = jenkins.branch_biomass(species_code[use_jenkins], dbh_cm)
    if (~use_jenkins).any():
        other = trees[~use_jenkins]
        foliage[~use_jenkins] = nsvb.foliage_biomass(
            species_code[~use_jenkins], other["dbh"], other["height"]
        )
        branch[~use_jenkins] = nsvb.branch_biomass(
            species_code[~use_jenkins], other["dbh"], other["height"]
        )
    return foliage, branch


def _jenkins_crown_components(
    trees: pd.DataFrame, species_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Foliage and branch dry weight from Jenkins et al. (2003).

    Branch is the aboveground residual; see
    :mod:`fastfuels_core.allometry.jenkins`. Diameter is in centimeters,
    the units the Jenkins equations use.
    """
    dbh_cm = trees["dbh"].to_numpy(dtype=np.float64)
    return (
        jenkins.foliage_biomass(species_code, dbh_cm),
        jenkins.branch_biomass(species_code, dbh_cm),
    )


def _national_crown_components(
    equations: str, trees: pd.DataFrame, species_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Foliage and total branch weight from a national estimator (NSVB/Jenkins).

    Both report total branchwood only; the size partition reduces it to the
    fine class separately.
    """
    if equations == "jenkins":
        return _jenkins_crown_components(trees, species_code)
    return _nsvb_crown_components(trees, species_code)


def _apply_crown_class_adjustment(
    fuel: np.ndarray,
    trees: pd.DataFrame,
    species_code: np.ndarray,
    crown_class_adjustment: str,
    crown_class_column: str | None,
) -> np.ndarray:
    if crown_class_adjustment != FUELCALC_CROWN_CLASS_ADJUSTMENT:
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
    equations: str = "brown_1978",
    crown_class_adjustment: str = FUELCALC_CROWN_CLASS_ADJUSTMENT,
    crown_class_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
    branchwood_size_partition: str = BROWN_PROPORTIONS_PARTITION,
) -> np.ndarray:
    """Per-tree available canopy fuel (kg).

    With ``fuel_column`` set, that column is returned as-is (precomputed
    fuel, e.g. from LiDAR regression); everything else is ignored.
    Otherwise fuel is ``foliage_fraction * foliage + branchwood_fraction
    * branchwood``, with ``branchwood_size_partition`` saying which
    branchwood the fraction applies to:

    ``"brown_proportions"`` (default)
        The fine (0-1/4 in) class: total branch weight times the Brown
        (1978) / Snell & Little (1983) fine share ``(P2 - P1) / (1 -
        P1)`` (see :mod:`fastfuels_core.allometry.brown`). This is
        FuelCalc's size line. The proportions are keyed by FuelCalc
        equation Id, so it resolves every species through the FuelCalc
        species table and raises for species outside it or without
        proportion equations — under either ``equations`` arm.
    ``"none"``
        Total branchwood, unpartitioned, so ``branchwood_fraction``
        expresses the fine share and the consumed share as one number.
        Needs no crosswalk, which under ``"nsvb"`` makes every species
        NSVB prices usable.

    Both arms need ``dbh``, ``height`` and ``fia_species_code``.
    ``equations`` selects where the foliage and branch weights come
    from:

    ``"brown_1978"`` (default)
        Total live crown weight from Brown 1978 Table 1, split into
        foliage and branchwood by ``P1``; trees of one inch dbh and
        under take the tabulated small-tree weights instead (see
        :func:`small_tree_crown_components`). This is FuelCalc's crown
        weight model. It is scoped to Brown's eleven Rocky Mountain
        conifers -- hardwoods and everything else raise.
    ``"nsvb"``
        Foliage and branch dry weight from NSVB (Westfall et al. 2024),
        the national estimator FastFuels uses elsewhere. Woodland species
        (Jenkins group 10), which NSVB has no crown model for, fall back to
        Jenkins. Its species reach is the partition's: national under
        ``"none"``, FuelCalc's table under ``"brown_proportions"``.
    ``"jenkins"``
        Foliage and branch dry weight from Jenkins et al. (2003), by
        species group; branch is the aboveground residual (see
        :mod:`fastfuels_core.allometry.jenkins`). Like ``"nsvb"`` it
        reports only total branchwood, so its species reach is the
        partition's — national under ``"none"``, FuelCalc's table under
        ``"brown_proportions"``.

    ``crown_class_adjustment`` scales the result per tree:

    ``"fuelcalc_table"`` (default)
        FuelCalc's crown-class multipliers, via
        :func:`crown_class_factor`. Requires ``crown_class_column``,
        naming the per-tree column that holds crown position —
        measured, imputed, or modelled.
    ``"none"`` (``None`` is accepted for it)
        No adjustment.

    The two arguments are one decision and are validated together:
    asking for the adjustment without the column raises, and so does
    naming the column without the adjustment. Neither half is useful
    alone, and both failures are silent if allowed through — the
    column would be discarded, or every tree would take the table's
    Other/none factor, which is 0.5 for 51 of the 55 species and so
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
        For an unknown ``equations`` or ``branchwood_size_partition``
        choice, or for species the chosen combination does not cover —
        codes outside the FuelCalc species table or without proportion
        equations (pinyon/juniper, the eastern newer species) wherever
        the table is consulted, and under ``"brown_1978"`` the hardwoods
        too. Mirrors FuelCalc's treelist error; ``fuel_column`` bypasses
        allometry entirely.
    """
    if fuel_column is not None:
        return trees[fuel_column].to_numpy(dtype=np.float64)

    if crown_class_adjustment is None:
        crown_class_adjustment = NO_CROWN_CLASS_ADJUSTMENT
    _validate_crown_class_options(crown_class_adjustment, crown_class_column)
    _validate_equations(equations)
    _validate_branchwood_size_partition(branchwood_size_partition)

    if len(trees) == 0:
        return np.zeros(0, dtype=np.float64)

    spcd = trees["fia_species_code"].to_numpy()
    # Brown/SL-83 work in inches; the nsvb wrapper is metric.
    dia_in = trees["dbh"].to_numpy() * CM_TO_IN

    if (
        equations in TOTAL_BRANCHWOOD_EQUATIONS
        and branchwood_size_partition == NO_PARTITION
    ):
        # The combinations that need no FuelCalc crosswalk: a national
        # estimator's total branchwood, taken whole.
        foliage_kg, branch_kg = _national_crown_components(equations, trees, spcd)
        fuel = foliage_fraction * foliage_kg + branchwood_fraction * branch_kg
        return _apply_crown_class_adjustment(
            fuel, trees, spcd, crown_class_adjustment, crown_class_column
        )

    species = _fuelcalc_species_for_trees(spcd)
    proportion_id = species["FOL"].loc[spcd].to_numpy()
    if equations == "brown_1978":
        foliage_kg, branch_kg = _brown_1978_crown_components(
            species, spcd, proportion_id, dia_in
        )
    else:
        foliage_kg, branch_kg = _national_crown_components(equations, trees, spcd)
    if branchwood_size_partition == BROWN_PROPORTIONS_PARTITION:
        fine_branch_kg = brown.fine_branchwood_share(proportion_id, dia_in) * branch_kg
    else:
        fine_branch_kg = branch_kg

    # Brown's crown weights are out of range below an inch; take the
    # tabulated small-tree weights there instead. The table reports the
    # fine class directly, so the partition does not enter for saplings.
    if equations == "brown_1978":
        sapling = dia_in <= SMALL_TREE_MAX_DIA_IN
        if sapling.any():
            height_ft = trees["height"].to_numpy() * M_TO_FT
            small_fol, small_twig = small_tree_crown_components(
                species["TOTAL"].loc[spcd].to_numpy(), dia_in, height_ft
            )
            foliage_kg = np.where(sapling, small_fol, foliage_kg)
            fine_branch_kg = np.where(sapling, small_twig, fine_branch_kg)

    fuel = foliage_fraction * foliage_kg + branchwood_fraction * fine_branch_kg
    return _apply_crown_class_adjustment(
        fuel, trees, spcd, crown_class_adjustment, crown_class_column
    )
