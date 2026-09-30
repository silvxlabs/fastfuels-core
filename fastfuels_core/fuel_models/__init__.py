"""Post-disturbance fuel models from LANDFIRE's Master_Rulesets.

Replicates the rule-based crosswalk in LANDFIRE's Total Fuel Change Tool
(LFTFC): each disturbed cell's map zone, vegetation, disturbance, cover,
height and biophysical setting select one Master_Rulesets row, which gives
its new fuel model (e.g. FBFM13, FBFM40, FCCS). Every other cell keeps last
year's.

=============================== ============================================
:mod:`disturbed_fuel_update`    last year's fuel model -> this year's
:mod:`fdist_builder`            the FDist raster the rules use, per mode
:mod:`lf_zone_lookup`           LANDFIRE map zone per cell
:mod:`ruleset_lookup`           match each cell to one Master_Rulesets row
=============================== ============================================

:func:`update_fuel_models` chains the other three and is the one entry
point a caller needs, with :func:`build_ruleset_index` to prepare the rules
once. Its ``disturbance`` argument is one of ``DISTURBANCE_MODES``. The
individual stages are imported from their own modules.
"""

from fastfuels_core.fuel_models.fdist_builder import DISTURBANCE_MODES
from fastfuels_core.fuel_models.disturbed_fuel_update import (
    FuelModelUpdate,
    update_fuel_models,
)
from fastfuels_core.fuel_models.ruleset_lookup import (
    RulesetIndex,
    build_ruleset_index,
)

__all__ = [
    "update_fuel_models",
    "FuelModelUpdate",
    "DISTURBANCE_MODES",
    "build_ruleset_index",
    "RulesetIndex",
]
