"""Traditional canopy fuel metrics (CBD, CBH, CH, CC, CFL) from tree inventories.

One module per quantity produced, chained by
:func:`compute_canopy_metrics`:

=========================== =========================================
:mod:`crown_radius`         max crown radius (m)
:mod:`available_fuel`       available canopy fuel (kg/tree)
:mod:`profile`              vertical bulk-density profile (kg/m**3)
:mod:`bulk_density`         CBD (kg/m**3)
:mod:`canopy_height`        CBH and canopy height (m)
:mod:`fuel_load`            CFL (kg/m**2)
:mod:`cover`                CC (%)
=========================== =========================================

Every stage is public and usable on its own, so a caller can run one,
swap one, or test one in isolation.
"""

from fastfuels_core.canopy_fuel.available_fuel import (
    CROWN_CLASS_REMAP,
    available_canopy_fuel,
    crown_class_factor,
)
from fastfuels_core.canopy_fuel.bulk_density import cbd_running_mean
from fastfuels_core.canopy_fuel.cover import canopy_cover
from fastfuels_core.canopy_fuel.canopy_height import profile_threshold_heights
from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.canopy_fuel.fuel_load import canopy_fuel_load
from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area
from fastfuels_core.canopy_fuel.metrics import compute_canopy_metrics
from fastfuels_core.canopy_fuel.profile import (
    FT_TO_M,
    cumulative_fuel_fraction,
    vertical_profile,
)

__all__ = [
    "CROWN_CLASS_REMAP",
    "FT_TO_M",
    "available_canopy_fuel",
    "canopy_cover",
    "canopy_fuel_load",
    "cbd_running_mean",
    "compute_canopy_metrics",
    "crown_class_factor",
    "cumulative_fuel_fraction",
    "disk_rect_overlap_area",
    "max_crown_radius",
    "profile_threshold_heights",
    "vertical_profile",
]
