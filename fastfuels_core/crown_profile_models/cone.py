# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# External imports
import numpy as np
from numpy.typing import NDArray


class ConeCrownProfile(CrownProfileModel):
    """
    Cone crown profile.

    A single right circular cone that is widest at the crown base and tapers
    linearly to a point at the tree top:

        R(z) = R * (Ht - z) / (Ht - Hb)     for Hb <= z <= Ht

    where Hb is the crown base height, Ht the total tree height, and R the
    maximum crown radius (attained at the crown base). This is a single cone,
    not a bicone. A max crown diameter height does not apply.

    Parameters
    ----------
    crown_base_height : float or NDArray[np.float64]
        Height at which the live crown starts (m).
    height : float or NDArray[np.float64]
        Total height of the tree (m).
    max_crown_radius : float or NDArray[np.float64]
        Maximum crown radius (m), attained at the crown base.

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
        crown_length = self.height - self.crown_base_height
        # Guard against a zero-length crown (crown_ratio == 0); the crown mask
        # below zeroes out every cell in that case anyway.
        safe_length = np.where(crown_length == 0, 1.0, crown_length)

        radius = self.max_crown_radius * (self.height - z) / safe_length
        inside = (z >= self.crown_base_height) & (z <= self.height)
        result = np.where(inside, radius, 0.0)

        if result.size == 1:
            return result.item()
        elif result.shape[0] == 1:
            return result.squeeze()
        else:
            return result

    def get_max_radius(self) -> float | np.ndarray:
        """Returns the maximum crown radius (attained at the crown base)."""
        r = self.max_crown_radius
        return r.item() if r.size == 1 else r.reshape(-1)

    def get_max_radius_height(self) -> float | np.ndarray:
        """
        Returns the height (m) of maximum crown radius. The cone tapers upward
        from the crown base, so the maximum is attained at the crown base.
        """
        result = self.crown_base_height
        return result.item() if result.size == 1 else result.reshape(-1)
