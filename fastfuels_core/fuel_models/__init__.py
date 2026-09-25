"""Post-disturbance fuel models from LANDFIRE's Master_Rulesets.

Replicates the rule-based crosswalk in LANDFIRE's Total Fuel Change Tool
(LFTFC): each disturbed cell's map zone, vegetation, disturbance, cover,
height and biophysical setting select one Master_Rulesets row, which gives
its new fuel model (e.g. FBFM13, FBFM40, FCCS). Every other cell keeps last
year's.

=============================== ============================================
:mod:`post_disturbance`         last year's fuel model -> this year's
:mod:`lf_zone_lookup`           LANDFIRE map zone per cell
:mod:`disturbance_crosswalk`    LDist codes to the FDist codes rules expect
:mod:`ruleset_lookup`           match each cell to one Master_Rulesets row
=============================== ============================================

:func:`update_fuel_models` chains the other three and is the one entry
point a caller needs, with :func:`build_ruleset_index` to prepare the rules
once. The individual stages are imported from their own modules.
"""

from fastfuels_core.fuel_models.post_disturbance import (
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
    "build_ruleset_index",
    "RulesetIndex",
]
