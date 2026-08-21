"""Metric-unit wrappers over the NSVB biomass estimators.

The National Scale Volume and Biomass system (Westfall et al. 2024,
USDA GTR WO-104; the ``nsvb`` package) works in inches, feet, and
pounds. These wrappers take the FastFuels metric convention (cm, m) and
return kilograms.
"""

from __future__ import annotations

import numpy as np
from nsvb import estimators

from fastfuels_core.units import CM_TO_IN, LB_TO_KG, M_TO_FT


def foliage_biomass(
    species_code: np.ndarray, dbh_cm: np.ndarray, height_m: np.ndarray
) -> np.ndarray:
    """Per-tree NSVB total foliage dry weight (kg)."""
    foliage_lb = estimators.total_foliage_dry_weight(
        np.asarray(species_code),
        np.asarray(dbh_cm, dtype=np.float64) * CM_TO_IN,
        np.asarray(height_m, dtype=np.float64) * M_TO_FT,
    )
    return foliage_lb * LB_TO_KG


def branch_biomass(
    species_code: np.ndarray, dbh_cm: np.ndarray, height_m: np.ndarray
) -> np.ndarray:
    """Per-tree NSVB total branch dry weight (kg)."""
    branch_lb = estimators.total_branch_weight(
        np.asarray(species_code),
        np.asarray(dbh_cm, dtype=np.float64) * CM_TO_IN,
        np.asarray(height_m, dtype=np.float64) * M_TO_FT,
    )
    return branch_lb * LB_TO_KG
