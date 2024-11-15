import json
from importlib.resources import files
import numpy as np

from fastfuels_core.crown_profile_models.abc import CrownProfileModel
from scipy.special import beta

DATA_PATH = files("fastfuels_core.data")
with open(DATA_PATH / "spgrp_parameters.json", "r") as f:
    SPGRP_PARAMS = json.load(f)
with open(DATA_PATH / "spcd_parameters.json", "r") as f:
    SPCD_PARAMS = json.load(f)
with open(DATA_PATH / "class_parameters.json", "r") as f:
    CLASS_PARAMS = json.load(f)

class BetaCrownProfile(CrownProfileModel):
    """
    A crown profile model based on a beta distribution.
    """

    a: float
    b: float
    c: float
    crown_length: float
    crown_base_height: float

    def __init__(
        self, species_code: int, crown_base_height: float, crown_length: float
    ):
        """
        Initializes a BetaCrownProfile instance.
        """
        self.species_code = int(species_code)
        self.crown_base_height = crown_base_height
        self.crown_length = crown_length

        species_group = SPCD_PARAMS[str(self.species_code)]["SPGRP"]
        self.a = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_a"]
        self.b = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_b"]
        self.c = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_c"]
        self.beta = beta(self.a, self.b)

    def get_max_radius(self) -> float:
        """
        Returns the maximum radius of the crown. This function finds the mode
        of the beta distribution, which is the value of z at which the beta
        distribution is at its max, and then calculates the radius at that
        height.
        """
        # Find the mode of the beta distribution. This is the value of z at
        # which the beta distribution is at its max.
        z_max = (self.a - 1) / (self.a + self.b - 2)
        normalized_max_radius = self._get_radius_at_normalized_height(z_max)
        return normalized_max_radius * self.crown_length

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

        result = np.zeros_like(z)
        result[mask] = (
            self.c * z[mask] ** (self.a - 1) * (1 - z[mask]) ** (self.b - 1) / self.beta
        )

        if result.size == 1:
            return result.item()  # Return as a scalar
        else:
            return result  # Return as an array
