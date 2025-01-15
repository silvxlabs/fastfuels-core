# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.ref_data import REF_SPECIES, REF_JENKINS
from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# External imports
import numpy as np
from numpy.typing import NDArray


class BetaCrownProfile(CrownProfileModel):
    """
    A crown profile model based on a beta distribution.
    """

    species_code: NDArray[np.int64]
    crown_base_height: NDArray[np.float64]
    crown_length: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    beta_norm: NDArray[np.float64]

    def __init__(
        self,
        species_code: int | NDArray[np.int64],
        crown_base_height: float | NDArray[np.float64],
        crown_length: float | NDArray[np.float64],
    ):
        """
        Initializes a BetaCrownProfile instance.
        We initialize data as np arrays expecting those to be passed in
        If only a float (scalar) is passed in, then the functions return results as scalars
        """
        self.species_code = np.asarray(species_code)
        self.crown_base_height = np.asarray(crown_base_height)
        self.crown_length = np.asarray(crown_length)

        jenkins_species_group = np.array(
            REF_SPECIES.loc[self.species_code]["JENKINS_SPGRPCD"]
        )
        self.a = np.asarray(REF_JENKINS.loc[jenkins_species_group]["BETA_CANOPY_a"])
        self.b = np.asarray(REF_JENKINS.loc[jenkins_species_group]["BETA_CANOPY_b"])
        self.c = np.asarray(REF_JENKINS.loc[jenkins_species_group]["BETA_CANOPY_c"])

        self.beta_norm = np.asarray(
            REF_JENKINS.loc[jenkins_species_group]["BETA_CANOPY_NORM"]
        )

    def get_max_radius(self) -> float | np.ndarray:
        """
        Returns the maximum radius of the crown. This function finds the mode
        of the beta distribution, which is the value of z at which the beta
        distribution is at its max, and then calculates the radius at that
        height.

        Returns a scalar with scalar input
        Returns a vector with vector input
        """
        # Find the mode of the beta distribution. This is the value of z at
        # which the beta distribution is at its max.
        z_max = (self.a - 1) / (self.a + self.b - 2)
        normalized_max_radius = self._get_radius_at_normalized_height(z_max)
        return (
            normalized_max_radius * self.crown_length
            if isinstance(normalized_max_radius * self.crown_length, float)
            else np.asarray(normalized_max_radius * self.crown_length)
        )

    def get_radius_at_height(self, height):
        """
        Returns the radius of the crown at a given height using the beta
        distribution crown model described in equation (3) in Ferrarese et
        al. (2015). Equation (3) gives the radius of the crown at a given
        height as a proportion of the crown length scaled between 0 and 1. To
        get a crown radius in meters, the result is multiplied by the crown
        length.
        """
        normalized_height = self._get_normalized_height(height)
        radius_at_normalized_height = self._get_radius_at_normalized_height(
            normalized_height
        )
        return radius_at_normalized_height * self.crown_length

    def _get_normalized_height(self, height):
        """
        Converts a height (in meters) of the tree crown to a unitless height
        between 0 and 1 representing the proportion of the crown length.
        """
        return (height - self.crown_base_height) / self.crown_length

    def _get_radius_at_normalized_height(self, z):
        """
        Returns the unitless radius of the crown at a given normalized height.
        The radius is scaled between 0 and 1 and represents a proportion of
        the crown length of the tree.

        The radius is calculated using the beta distribution probability
        density function (PDF) at the given normalized height. The PDF of the
        beta distribution uses an additional scaling factor, 'c', for the
        application to crown profiles and is described by equation (3) in
        Ferrarese et al. (2015).

        """
        z = np.asarray(z)
        mask = (z >= 0) & (z <= 1)

        result = np.where(
            mask,
            ((self.c * z ** (self.a - 1)) * ((1 - z) ** (self.b - 1))) / self.beta_norm,
            0.0,
        )
        result = np.nan_to_num(result, nan=0.0)

        if result.size == 1:
            return result.item()  # Return as a scalar
        else:
            return result  # Return as an array
