"""Canopy fuel load (kg/m**2), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FT_TO_M


def canopy_fuel_load(
    profile: np.ndarray, *, layer_depth: float = FT_TO_M
) -> np.ndarray:
    """Canopy fuel load: vertical integral of the bulk-density profile.

    Returns
    -------
    numpy.ndarray
        CFL (kg/m**2), shape ``(ny, nx)``.
    """
    return profile.sum(axis=0) * layer_depth
