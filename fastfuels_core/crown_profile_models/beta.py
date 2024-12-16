import json
from importlib.resources import files
import numpy as np

from fastfuels_core.crown_profile_models.abc import CrownProfileModel
from scipy.special import beta
from numpy.typing import NDArray

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
        We initialize data as np arrays expecting those to be passed in
        If only a float (scalar) is passed in, then the functions return results as scalars
        """
        self.species_code = np.asarray(species_code)
        self.crown_base_height = np.asarray(crown_base_height)
        self.crown_length = np.asarray(crown_length)

        # species_group = SPCD_PARAMS[str(self.species_code)]["SPGRP"]
        species_group = vectorized_species_code_lookup(self.species_code, "SPGRP")
        # self.a = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_a"]
        self.a = vectorized_species_group_lookup(species_group, "BETA_CANOPY_a")
        # self.b = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_b"]
        self.b = vectorized_species_group_lookup(species_group, "BETA_CANOPY_b")
        # self.c = SPGRP_PARAMS[str(species_group)]["BETA_CANOPY_c"]
        self.c = vectorized_species_group_lookup(species_group, "BETA_CANOPY_c")
        self.beta = beta(self.a, self.b)

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
            self.c * z ** (self.a - 1) * (1 - z) ** (self.b - 1) / self.beta,
            0.0,
        )
        result = np.nan_to_num(result, nan=0.0)

        if result.size == 1:
            return result.item()  # Return as a scalar
        else:
            return result  # Return as an array

    def get_beta_radius(self, z, height, crown_base, a, b, c, beta):
        """
        Get radius at an array of z heights using the beta crown profile model.

        Parameters
        ----------
        z : NDArray
            Array of z coordinates of float64 type.
        height : float
            Tree height in meters.
        crown_base : float
            Crown base in meters.
        a : float
            Beta distribution parameter.
        b : float
            Beta distribution parameter.
        c : float
            Beta distribution parameter.
        beta : float
            Beta distribution parameter.

        Returns
        -------
        r : NDarray
            Radius of tree evaluated at z heights.
        """

        if np.all(z < crown_base):
            return 0.0
        if np.all(z > height):
            return 0.0

        # Normalize
        crown_length = height - crown_base
        z = (z - crown_base) / crown_length

        r = (c * z ** (a - 1) * (1 - z) ** (b - 1)) / beta
        r = r * crown_length
        return r

    def get_beta_max_crown_radius(self, height, crown_base, a, b, c, beta):
        """
        Gets the maximum radius of a tree for the beta crown profile model.

        Parameters
        ----------
        height : NDArray
            Array of tree heights in meters.
        crown_base : NDArray
            Array of crown base heights in meters.
        crown_base : NDArray
            Crown base heights in meters.
        a : NDArray
            Array of beta distribution parameters.
        b : NDArray
            Array of beta distribution parameters.
        c : NDArray
            Array of beta distribution parameters.
        beta : NDArray
            Array of beta distribution parameters.

        Returns
        -------
        r_max : NDarray
            Maximum radius of each tree in meters.
        """

        # Normalized height of max radius
        z_max = (a - 1) / (a + b - 2)
        # Un-normalized height of max radius
        z_max = crown_base + z_max * (height - crown_base)
        r_max = self.get_beta_radius(z_max, height, crown_base, a, b, c, beta)
        return r_max


# useful function to look up species code. The function is vectorized for convenience.
def _species_code_lookup(species_code: str | NDArray, parameter: str):
    return SPCD_PARAMS[str(species_code)][parameter]


vectorized_species_code_lookup = np.vectorize(_species_code_lookup)


# useful function to look up species groups. The function is vectorized for convenience.
def _species_group_lookup(species_group: str | NDArray, beta_canopy: str):
    return SPGRP_PARAMS[str(species_group)][beta_canopy]


vectorized_species_group_lookup = np.vectorize(_species_group_lookup)
