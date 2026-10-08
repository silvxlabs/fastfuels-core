"""Post-disturbance fuel models from LANDFIRE's Master_Rulesets.

Replicates the rule-based crosswalk in LANDFIRE's Total Fuel Change Tool
(LFTFC): each disturbed cell's map zone, vegetation, disturbance, cover,
height and biophysical setting select one Master_Rulesets row, which gives
its new fuel model (e.g. FBFM13, FBFM40, FCCS). Every other cell keeps last
year's.

=============================== ============================================
:mod:`fdist_builder`            the FDist raster the rules use, per mode
:mod:`disturbed_fuel_update`    last year's fuel model -> this year's
:mod:`ruleset_lookup`           match each cell to one Master_Rulesets row
=============================== ============================================

A caller builds the FDist raster with :func:`build_fdist_raster` (its
``disturbance`` argument is one of ``DISTURBANCE_MODES``), then passes it,
the map zones and the Master_Rulesets table to :func:`update_fuel_models`.
The LDist attribute table, map zones and Master_Rulesets table are all
supplied by the caller.
"""

from fastfuels_core.fuel_models.disturbed_fuel_update import update_fuel_models
from fastfuels_core.fuel_models.fdist_builder import (
    DISTURBANCE_MODES,
    build_fdist_raster,
)

__all__ = [
    "build_fdist_raster",
    "DISTURBANCE_MODES",
    "update_fuel_models",
]
