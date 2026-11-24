"""Silvicultural treatments that thin a tree inventory.

One module per treatment method. Each treatment is a class whose
``apply`` takes a DataFrame of trees and returns the trees it keeps:

======================= =============================================
:mod:`base`             thinning direction and the treatment protocol
:mod:`diameter_limit`   thin to a diameter limit
:mod:`basal_area`       thin to a target stand basal area
:mod:`tree_density`     thin to a target number of trees
:mod:`crown_spacing`    thin to a minimum crown spacing
======================= =============================================

Every treatment class and the shared types in :mod:`base` are
re-exported here, so callers import them from the package.
"""

from fastfuels_core.treatments.base import ThinningDirection, TreatmentProtocol
from fastfuels_core.treatments.basal_area import (
    DirectionalThinToStandBasalArea,
    ProportionalThinToBasalArea,
)
from fastfuels_core.treatments.crown_spacing import ThinToCrownSpacing
from fastfuels_core.treatments.diameter_limit import DirectionalThinToDiameterLimit
from fastfuels_core.treatments.tree_density import DirectionalThinToTreeDensity

__all__ = [
    "DirectionalThinToDiameterLimit",
    "DirectionalThinToStandBasalArea",
    "DirectionalThinToTreeDensity",
    "ProportionalThinToBasalArea",
    "ThinToCrownSpacing",
    "ThinningDirection",
    "TreatmentProtocol",
]
