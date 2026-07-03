# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# External imports
import numpy as np
from numpy.typing import NDArray


class EllipsoidCrownProfile(CrownProfileModel):
    """
    Double half-ellipsoid crown profile.

    Two half-ellipsoids of revolution joined at the height of maximum crown
    diameter, Hd, peaking there at the maximum crown radius R and tapering to
    zero at the crown base Hb and the tree top Ht:

        lower (Hb <= z <= Hd): R(z) = R * sqrt(1 - ((Hd - z) / (Hd - Hb))**2)
        upper (Hd <= z <= Ht): R(z) = R * sqrt(1 - ((z - Hd) / (Ht - Hd))**2)

    When Hd is the crown midpoint the two lobes have equal vertical semi-axes
    and the solid is a symmetric spheroid. A single half-ellipsoid is the
    limiting case of the dual form: pass ``max_crown_diameter_height == Hb`` for
    a crown widest at the base, or ``== Ht`` for one widest at the top; any
    interior value gives the two-lobed peak.

    Parameters
    ----------
    crown_base_height : float or NDArray[np.float64]
        Height at which the live crown starts (m).
    height : float or NDArray[np.float64]
        Total height of the tree (m).
    max_crown_radius : float or NDArray[np.float64]
        Maximum crown radius (m), attained at max_crown_diameter_height.
    max_crown_diameter_height : float or NDArray[np.float64], optional
        Height of maximum crown diameter (m). Must lie within
        [crown_base_height, height]. Defaults to the crown midpoint
        (crown_base_height + height) / 2 when not provided.

    Notes
    -----
    All parameters are stored as 2D arrays with shape [n_trees, 1] to enable
    natural broadcasting with 1D height inputs, matching PurvesCrownProfile and
    BetaCrownProfile.
    """

    crown_base_height: NDArray[np.float64]
    height: NDArray[np.float64]
    max_crown_radius: NDArray[np.float64]
    max_crown_diameter_height: NDArray[np.float64]

    def __init__(
        self,
        crown_base_height: float | NDArray[np.float64],
        height: float | NDArray[np.float64],
        max_crown_radius: float | NDArray[np.float64],
        max_crown_diameter_height: float | NDArray[np.float64] | None = None,
    ):
        self.crown_base_height = np.atleast_2d(crown_base_height).T
        self.height = np.atleast_2d(height).T
        self.max_crown_radius = np.atleast_2d(max_crown_radius).T
        if max_crown_diameter_height is None:
            self.max_crown_diameter_height = (
                self.crown_base_height + self.height
            ) / 2.0
        else:
            self.max_crown_diameter_height = np.atleast_2d(max_crown_diameter_height).T
        self._validate_diameter_height()

    def _validate_diameter_height(self):
        hd = self.max_crown_diameter_height
        if np.any(hd < self.crown_base_height) or np.any(hd > self.height):
            raise ValueError(
                "max_crown_diameter_height must lie within "
                "[crown_base_height, height] for every tree."
            )

    def get_radius_at_height(self, height) -> float | np.ndarray:
        """
        Returns the crown radius at the given height(s).

        Returns a scalar for a single tree/height, a 1D array for a single tree
        with multiple heights, and a 2D array [n_trees, n_heights] for multiple
        trees.
        """
        z = np.asarray(height)
        hb = self.crown_base_height
        ht = self.height
        hd = self.max_crown_diameter_height
        r = self.max_crown_radius

        lower_len = hd - hb
        upper_len = ht - hd
        # Guard degenerate (zero-length) lobes for the single-ellipsoid limiting
        # cases Hd == Hb or Hd == Ht. The denominators are made safe here and the
        # lobe-selection / crown mask below ensure the empty lobe is never used.
        safe_lower = np.where(lower_len == 0, 1.0, lower_len)
        safe_upper = np.where(upper_len == 0, 1.0, upper_len)

        # Clip the sqrt argument to [0, 1]: np.where evaluates both branches, so
        # out-of-lobe cells would otherwise take a sqrt of a negative number.
        lower = r * np.sqrt(np.clip(1.0 - ((hd - z) / safe_lower) ** 2, 0.0, 1.0))
        upper = r * np.sqrt(np.clip(1.0 - ((z - hd) / safe_upper) ** 2, 0.0, 1.0))

        # Use the lower lobe only where it is non-degenerate; when Hd == Hb the
        # upper lobe covers the whole crown (both give R at the shared peak).
        use_lower = (z <= hd) & (lower_len > 0)
        radius = np.where(use_lower, lower, upper)
        inside = (z >= hb) & (z <= ht)
        result = np.where(inside, radius, 0.0)

        if result.size == 1:
            return result.item()
        elif result.shape[0] == 1:
            return result.squeeze()
        else:
            return result

    def get_max_radius(self) -> float | np.ndarray:
        """Returns the maximum crown radius (attained at Hd)."""
        r = self.max_crown_radius
        return r.item() if r.size == 1 else r.reshape(-1)

    def get_max_radius_height(self) -> float | np.ndarray:
        """Returns the height (m) of maximum crown radius (Hd)."""
        result = self.max_crown_diameter_height
        return result.item() if result.size == 1 else result.reshape(-1)
