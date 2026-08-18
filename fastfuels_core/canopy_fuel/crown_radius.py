"""Per-tree maximum crown radius (m).

The radius is the horizontal extent every disk-based stage works from:
the crown-projected profile and all three canopy cover methods. It is
chosen independently of those stages so a run can vary the radius
source and the stage that consumes it separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.allometry import fvs
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.units import conversion_factor

VALID_EQUATIONS = ("purves", "crookston_stage")


def max_crown_radius(
    trees: pd.DataFrame,
    *,
    crown_radius_column: str | None = None,
    equations: str = "purves",
) -> np.ndarray:
    """Per-tree maximum crown radius (m).

    Reads ``crown_radius_column`` when given (e.g. LiDAR-measured
    radii), which overrides ``equations``. Otherwise ``equations``
    selects an allometry:

    ``"purves"`` (default)
        Purves et al. (2007), from ``fia_species_code``, ``dbh``,
        ``height`` and ``crown_ratio``.
    ``"crookston_stage"``
        Half the crown width of Crookston & Stage (1999), reached
        through FVS (see :mod:`fastfuels_core.allometry.fvs`) — diameter
        alone above breast height, coefficients fitted regionally. This
        is the width behind FuelCalc's canopy cover.

    The radius source is independent of how canopy cover treats
    overlap, so the two can be varied separately: a run can compare
    overlap treatments on one radius, or radii under one overlap
    treatment, without confounding the two.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(trees),)``, meters.

    Raises
    ------
    ValueError
        For an unknown ``equations`` choice.
    """
    if crown_radius_column is not None:
        return trees[crown_radius_column].to_numpy(dtype=np.float64)
    if equations not in VALID_EQUATIONS:
        raise ValueError(
            f"Unknown crown radius equations {equations!r}; expected "
            f"'purves' or 'crookston_stage'."
        )
    if equations == "crookston_stage":
        width_ft = fvs.crown_width_for_species(
            trees["fia_species_code"].to_numpy(),
            trees["dbh"].to_numpy(dtype=np.float64) * conversion_factor("cm", "inch"),
            trees["height"].to_numpy(dtype=np.float64) * conversion_factor("m", "foot"),
        )
        return 0.5 * width_ft * conversion_factor("foot", "m")
    model = PurvesCrownProfile(
        trees["fia_species_code"],
        trees["dbh"],
        trees["height"],
        trees["crown_ratio"],
    )
    return np.atleast_1d(np.asarray(model.get_max_radius(), dtype=np.float64))
