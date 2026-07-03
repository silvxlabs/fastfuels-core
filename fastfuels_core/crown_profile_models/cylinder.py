# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# External imports
import numpy as np
from numpy.typing import NDArray


class CylinderCrownProfile(CrownProfileModel):
    """
    Cylinder crown profile.

    A right circular cylinder of uniform radius between the crown base and the
    tree top:

        R(z) = R     for Hb <= z <= Ht

    where Hb is the crown base height, Ht the total tree height, and R the
    (constant) crown radius. A max crown diameter height does not apply.

    Parameters
    ----------
    crown_base_height : float or NDArray[np.float64]
        Height at which the live crown starts (m).
    height : float or NDArray[np.float64]
        Total height of the tree (m).
    max_crown_radius : float or NDArray[np.float64]
        Crown radius (m), uniform over the crown.

    Notes
    -----
    All parameters are stored as 2D arrays with shape [n_trees, 1] to enable
    natural broadcasting with 1D height inputs, matching PurvesCrownProfile and
    BetaCrownProfile.
    """

    crown_base_height: NDArray[np.float64]
    height: NDArray[np.float64]
    max_crown_radius: NDArray[np.float64]

    def __init__(
        self,
        crown_base_height: float | NDArray[np.float64],
        height: float | NDArray[np.float64],
        max_crown_radius: float | NDArray[np.float64],
    ):
        self.crown_base_height = np.atleast_2d(crown_base_height).T
        self.height = np.atleast_2d(height).T
        self.max_crown_radius = np.atleast_2d(max_crown_radius).T

    def get_radius_at_height(self, height) -> float | np.ndarray:
        """
        Returns the crown radius at the given height(s).

        Returns a scalar for a single tree/height, a 1D array for a single tree
        with multiple heights, and a 2D array [n_trees, n_heights] for multiple
        trees.
        """
        z = np.asarray(height)
        inside = (z >= self.crown_base_height) & (z <= self.height)
        result = np.where(inside, self.max_crown_radius, 0.0)

        if result.size == 1:
            return result.item()
        elif result.shape[0] == 1:
            return result.squeeze()
        else:
            return result

    def get_max_radius(self) -> float | np.ndarray:
        """Returns the (uniform) crown radius."""
        r = self.max_crown_radius
        return r.item() if r.size == 1 else r.reshape(-1)

    def get_max_radius_height(self) -> float | np.ndarray:
        """
        Returns a representative height (m) of maximum crown radius. The radius
        is uniform over [crown_base_height, height], so the maximum is attained
        at every height in the crown and there is no unique maximum; the crown
        midpoint is returned as a representative value.
        """
        result = (self.crown_base_height + self.height) / 2.0
        return result.item() if result.size == 1 else result.reshape(-1)
