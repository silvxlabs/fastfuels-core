import json
from importlib.resources import files
import numpy as np

from fastfuels_core.crown_profile_models.abc import CrownProfileModel
from numpy.typing import NDArray

DATA_PATH = files("fastfuels_core.data")
with open(DATA_PATH / "spgrp_parameters.json", "r") as f:
    SPGRP_PARAMS = json.load(f)
with open(DATA_PATH / "spcd_parameters.json", "r") as f:
    SPCD_PARAMS = json.load(f)
with open(DATA_PATH / "class_parameters.json", "r") as f:
    CLASS_PARAMS = json.load(f)

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
    max_crown_radius : float
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
        """
        self.height = height
        self.crown_ratio = crown_ratio
        self.crown_base_height = height - height * crown_ratio
        self.species_code = species_code
        self.dbh = dbh

        #lookup trait score
        trait_score = vectorized_trait_score_lookup(species_code)

        # Compute maximum crown radius
       # r0j = (1 - trait_score) * C0_R0 + trait_score * C1_R0
       #r40j = (1 - trait_score) * C0_R40 + trait_score * C1_R40
        self.max_crown_radius = _get_purves_max_crown_radius(str(species_code), dbh)

        # Compute crown shape parameter
        self.shape_parameter = (1 - trait_score) * C0_B + trait_score * C1_B

    def get_radius_at_height(self, height: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the crown radius at a given height.

        Parameters
        ----------
        height : float or np.ndarray
            Height(s) at which to compute the crown radius (m).

        Returns
        -------
        float or np.ndarray
            Crown radius (m) at the given height(s). Returns a float if input height is a scalar,
            otherwise returns a numpy array.
        """
        height = np.asarray(height)
        radius = np.where(
            height < self.crown_base_height,
            0.0,
            self.max_crown_radius
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
        #should we change this to return the max crown radius parameter we've already calculated?
        return self.get_radius_at_height(self.crown_base_height)


#useful functions outside of class
def _trait_score_lookup(species_code: str | NDArray):
    return SPCD_PARAMS[str(species_code)]["PURVES_TRAIT_SCORE"]

vectorized_trait_score_lookup = np.vectorize(_trait_score_lookup)




def get_purves_shape_param(species_code: str | NDArray):
    """
    Get shape parameters for each tree for the Purves model.

    Parameters
    ----------
    trait_score : NDArray
        Trait scores for each tree.

    Returns
    -------
    shape_parameter : NDarray
        Shape parameters for each tree.
    """
    trait_score = vectorized_trait_score_lookup(species_code)

    C0_B = 0.196
    C1_B = 0.511
    shape_parameter = (1.0 - trait_score) * C0_B + trait_score * C1_B
    return shape_parameter


def _get_purves_max_crown_radius(species_code: str | NDArray, dbh: float | NDArray):
    """
    Gets the maximum radius of a tree for the Purves crown profile model.

    Parameters
    ----------
    trait_score : NDArray
        Trait scores for each tree.
    dbh : NDArray
        Diameter at breast height for each tree in cm.

    Returns
    -------
    r_max : NDarray
        Maximum radius of each tree in meters.
    """
    trait_score = vectorized_trait_score_lookup(species_code)


    r0j = (1 - trait_score) * C0_R0 + trait_score * C1_R0
    r40j = (1 - trait_score) * C0_R40 + trait_score * C1_R40
    max_crown_radius = r0j + (r40j - r0j) * (dbh / 40.0)
    return max_crown_radius

def get_purves_radius(z, height, crown_base, max_crown_radius, shape_parameter):
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