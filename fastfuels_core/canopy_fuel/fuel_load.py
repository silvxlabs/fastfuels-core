"""Canopy fuel load (kg/m**2), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FUELCALC_LAYER_DEPTH


def canopy_fuel_load(
    profile: np.ndarray, *, layer_depth: float = FUELCALC_LAYER_DEPTH
) -> np.ndarray:
    """Canopy fuel load: vertical integral of the bulk-density profile.

    Returns
    -------
    numpy.ndarray
        CFL (kg/m**2), shape ``(ny, nx)``.
    """
    return profile.sum(axis=0) * layer_depth
