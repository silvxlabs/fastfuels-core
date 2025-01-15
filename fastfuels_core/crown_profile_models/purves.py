# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.ref_data import REF_SPECIES, REF_JENKINS
from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# External imports
import numpy as np
from numpy.typing import NDArray


# See Purves et al. (2007) Table S2 in Supporting Information
C0_R0 = 0.503
C1_R0 = 3.126
C0_R40 = 0.5
C1_R40 = 10.0
C0_B = 0.196
C1_B = 0.511


class PurvesCrownProfile(CrownProfileModel):
    """
    Purves Crown Profile Model.

    This class computes the crown profile for a given tree based on the Purves et al. (2007) model.
    It uses species-specific parameters to calculate the crown radius at different heights; parameters
    can be found in fastfuels_core/data/spcd_parameters.json.

    Parameters
    ----------
    species_code : int
        The species code of the tree.
    dbh : float
        Diameter at breast height (cm).
    height : float
        Total height of the tree (m).
    crown_ratio : float
        Ratio of the crown length to the total height of the tree.

    Attributes
    ----------
    height : float
        Total height of the tree (m).
    crown_ratio : float
        Ratio of the crown length to the total height of the tree.
    crown_base_height : float
        Height at which the crown starts (m).
    max_theoretical_crown_radius : float
        Maximum radius of the crown (m).
    shape_parameter : float
        Shape parameter for the crown profile.

    Methods
    -------
    get_radius_at_height(height)
        Computes the crown radius at a given height.
    get_max_radius()
        Returns the maximum crown radius.
    """

    def __init__(
        self, species_code: int, dbh: float, height: float, crown_ratio: float
    ):
        """
        Initializes the PurvesCrownProfile model with the given parameters.

        Parameters
        ----------
        species_code : int
            The species code of the tree.
        dbh : float
            Diameter at breast height (cm).
        height : float
            Total height of the tree (m).
        crown_ratio : float
            Ratio of the crown length to the total height of the tree.

        Storing all variables as a [x,1] array allows versatile calculations no matter what the data type
        The calculations do not care if there is one item (scalar) or a large array (vector)
        By storing in a 2D array ( array[x,1], we are able to perform fast calculations without conditionals or for loop

        We utilize np.atleast_2d().T to ensure scalars and one dimensional arrays are transposed into a 2D array with shape [x,1].
        This allows the outputs of functions such as get_radius_at_height() to return data in the proper format [tree, data_from_inputs]
        """

        self.height = np.atleast_2d(height).T
        self.crown_ratio = np.atleast_2d(crown_ratio).T
        self.crown_base_height = np.atleast_2d(height - height * crown_ratio).T
        self.species_code = np.atleast_2d(species_code).T
        self.dbh = np.atleast_2d(dbh).T
        self.trait_score = np.atleast_2d(self._get_purves_trait_score(species_code)).T

        # Compute parameters with 2D arrays
        self.max_theoretical_crown_radius = (
            self._get_purves_max_theoretical_crown_radius()
        )
        self.shape_parameter = self._get_purves_shape_param()

    def get_radius_at_height(self, height: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the crown radius at a given height.

        We expect inputs to be either a single value (scalar)
        Or the input should be a numpy array (vector)

        The function is able to perform calculations on both types of inputs
        and will return the same type as the input

        Parameters
        ----------
        height : float or np.ndarray
            Height(s) at which to compute the crown radius (m).
            Input can be a single value (scalar) or a numpy array (vector ndarray)
        Returns
        -------
        float or np.ndarray
            Crown radius (m) at the given height(s).
            Returns a float if input height is a scalar,
            otherwise returns a numpy array if input height is a vector.

            If height is a vector [x,], and instance variable (ie crown_base_height) is a vector [y,1]
            Then output vector will be [y,x]
        """
        height = np.asarray(height)

        height_mask = np.logical_or(height < 0, height > self.height)

        radius = np.where(
            height_mask,
            0.0,
            self.max_theoretical_crown_radius
            * (np.minimum((self.height - height) / self.height, 0.95) / 0.95)
            ** self.shape_parameter,
        )
        radius = np.nan_to_num(radius, nan=0.0)

        return radius if radius.size > 1 else radius.item()

    def get_max_radius(self) -> float | np.ndarray:
        """
        Returns the maximum crown radius.

        Returns
        -------
        float or np.ndarray
            Maximum crown radius (m).
        """
        return self.get_radius_at_height(self.crown_base_height)

    def _get_purves_shape_param(self):
        """
        Get shape parameters for each tree for the Purves model.

        Returns
        -------
        shape_parameter : NDarray
            Shape parameters for each tree.
        """
        shape_parameter = (1.0 - self.trait_score) * C0_B + self.trait_score * C1_B
        return shape_parameter

    def _get_purves_radius(
        self, z, height, crown_base, max_crown_radius, shape_parameter
    ):
        """
        Get radius at an array of z heights using the Purves crown profile model.

        Parameters
        ----------
        z : NDarray
            Array of z coordinates of float64 type.
        height : float
            Tree height in meters.
        crown_base : float
            Crown base in meters.
        max_crown_radius : float
            Maximum radius of the tree.
        shape_parameter : float
            Purves shape parameter.

        Returns
        -------
        r : NDarray
            Radius of tree evaluated at z heights.
        """

        if z < crown_base:
            return 0.0
        if z > height:
            return 0.0

        return max_crown_radius * ((height - z) / height) ** shape_parameter

    def _get_purves_max_theoretical_crown_radius(self):
        """
        Gets the maximum radius of a tree for the Purves crown profile model.

        Returns
        -------
        NDAarray
            Maximum possible (theoretical) radius of a tree in the Purves model.
        """
        r0j = (1 - self.trait_score) * C0_R0 + self.trait_score * C1_R0
        r40j = (1 - self.trait_score) * C0_R40 + self.trait_score * C1_R40
        max_crown_radius = r0j + (r40j - r0j) * (self.dbh / 40.0)

        return max_crown_radius

    @staticmethod
    def _get_purves_trait_score(spcd):
        """
        Get the trait score for a given species code from the REF_SPECIES table.
        """
        return REF_SPECIES.loc[spcd]["PURVES_TRAIT_SCORE"]
