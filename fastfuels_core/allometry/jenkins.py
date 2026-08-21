"""National-scale biomass estimators from Jenkins et al. (2003).

Jenkins, J.C., Chojnacky, D.C., Heath, L.S., Birdsey, R.A. 2003.
National-scale biomass estimators for United States tree species.
*Forest Science* 49(1): 12-35.

Total aboveground biomass for 10 species groups (their Eq. 1, Table 4)::

    agb = exp(b0 + b1 * ln(dbh))        dbh in cm, agb in kg

fitted for trees >= 2.5 cm dbh. A per-group sapling adjustment scales the
estimate for dbh <= 12.7 cm (5 in), where the equation overpredicts.

Component biomass uses the ratio equations (their Eq. 2, Table 6)::

    ratio = exp(b0 + b1 / dbh)

each giving a component's share of aboveground biomass. Jenkins published
ratios for foliage, stem bark, and stem wood (and, belowground, coarse
roots). Branches are the remaining aboveground mass, so the branch share
is the residual of the three published aboveground components::

    branch_ratio = 1 - foliage_ratio - stem_bark_ratio - stem_wood_ratio

Species resolve to a Jenkins group through ``REF_SPECIES``'
``JENKINS_SPGRPCD``; the coefficients are keyed by group in
``REF_JENKINS``. Diameters are in centimeters and weights in kilograms,
the FastFuels metric convention, matching the equations' own units.
"""

from __future__ import annotations

import numpy as np

from fastfuels_core.ref_data import REF_JENKINS, REF_SPECIES

# Below this dbh (cm) the per-group sapling adjustment applies (Jenkins et
# al. 2003, 5 in).
SAPLING_MAX_DIA_CM = 12.7


def _species_groups(species_code: np.ndarray) -> np.ndarray:
    """Jenkins species-group code per tree, from ``REF_SPECIES``.

    Raises ValueError for species codes outside the FIA reference table,
    which carry no group and so cannot be priced.
    """
    spcd = np.asarray(species_code)
    unknown = np.setdiff1d(spcd, REF_SPECIES.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FIA reference "
            f"table, so they have no Jenkins species group."
        )
    return REF_SPECIES["JENKINS_SPGRPCD"].reindex(spcd).to_numpy()


def _group_param(groups: np.ndarray, column: str) -> np.ndarray:
    return REF_JENKINS[column].reindex(groups).to_numpy(dtype=np.float64)


def above_ground_biomass(species_code: np.ndarray, dbh_cm: np.ndarray) -> np.ndarray:
    """Per-tree total aboveground biomass (kg), Jenkins Eq. 1 + sapling adj."""
    dbh = np.asarray(dbh_cm, dtype=np.float64)
    groups = _species_groups(species_code)
    b0 = _group_param(groups, "JENKINS_TOTAL_B1")
    b1 = _group_param(groups, "JENKINS_TOTAL_B2")
    agb = np.exp(b0 + b1 * np.log(dbh))
    sapling_adjustment = _group_param(groups, "JENKINS_SAPLING_ADJUSTMENT")
    small = (dbh <= SAPLING_MAX_DIA_CM) & (sapling_adjustment > 0)
    return np.where(small, agb * sapling_adjustment, agb)


def _component_ratio(
    groups: np.ndarray, dbh_cm: np.ndarray, b0_column: str, b1_column: str
) -> np.ndarray:
    """A component's share of aboveground biomass, Jenkins Eq. 2."""
    b0 = _group_param(groups, b0_column)
    b1 = _group_param(groups, b1_column)
    return np.exp(b0 + b1 / np.asarray(dbh_cm, dtype=np.float64))


def foliage_biomass(species_code: np.ndarray, dbh_cm: np.ndarray) -> np.ndarray:
    """Per-tree foliage dry weight (kg): aboveground biomass x foliage ratio."""
    groups = _species_groups(species_code)
    ratio = _component_ratio(
        groups, dbh_cm, "JENKINS_FOLIAGE_RATIO_B1", "JENKINS_FOLIAGE_RATIO_B2"
    )
    return above_ground_biomass(species_code, dbh_cm) * ratio


def branch_biomass(species_code: np.ndarray, dbh_cm: np.ndarray) -> np.ndarray:
    """Per-tree branch dry weight (kg): the aboveground residual.

    Branch mass is aboveground biomass less the foliage, stem-bark and
    stem-wood components, which is how Jenkins' published component ratios
    account for branchwood (no branch ratio is fit directly).
    """
    groups = _species_groups(species_code)
    foliage_ratio = _component_ratio(
        groups, dbh_cm, "JENKINS_FOLIAGE_RATIO_B1", "JENKINS_FOLIAGE_RATIO_B2"
    )
    bark_ratio = _component_ratio(
        groups, dbh_cm, "JENKINS_STEM_BARK_RATIO_B1", "JENKINS_STEM_BARK_RATIO_B2"
    )
    wood_ratio = _component_ratio(
        groups, dbh_cm, "JENKINS_STEM_WOOD_RATIO_B1", "JENKINS_STEM_WOOD_RATIO_B2"
    )
    branch_ratio = 1.0 - foliage_ratio - bark_ratio - wood_ratio
    return above_ground_biomass(species_code, dbh_cm) * branch_ratio
