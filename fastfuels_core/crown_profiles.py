# Core imports
import json
import importlib.resources

# Internal imports
from fastfuels_core.trees import TreePopulation

# External imports
import numpy as np
from scipy.special import beta
from numba import vectorize, float64
from numpy.typing import NDArray


def add_crown_profile_params(
    trees: TreePopulation, crown_profile_model="purves"
) -> TreePopulation:
    """
    Adds crown profile model parameters to a tree population. Parameters vary depending
    on the profile model used.

    Parameters
    ----------
    trees : TreePopulation
        The tree population.
    crown_profile_model : str
        The crown profile model to use.

    Returns
    -------
    TreePopulation
        The tree population with added geometric parameters that can be used to evaluate
        a crown profile model.
    """

    spcd_path = importlib.resources.files("fastfuels_core.data").joinpath(
        "spcd_parameters.json"
    )
    spcd_params = json.loads(spcd_path.read_text())

    spgrp_path = importlib.resources.files("fastfuels_core.data").joinpath(
        "spgrp_parameters.json"
    )
    spgrp_params = json.loads(spgrp_path.read_text())

    species_code = trees["SPCD"].to_numpy().astype(str)

    if crown_profile_model == "beta":
        # Species group
        species_group = np.array(
            [spcd_params[s]["SPGRP"] for s in species_code]
        ).astype(str)
        # Beta profile params
        a = np.array([spgrp_params[g]["BETA_CANOPY_a"] for g in species_group])
        b = np.array([spgrp_params[g]["BETA_CANOPY_b"] for g in species_group])
        c = np.array([spgrp_params[g]["BETA_CANOPY_c"] for g in species_group])
        beta_param = beta(a, b)
        trees["A"] = a
        trees["B"] = b
        trees["C"] = c
        trees["BETA"] = beta_param
        h = trees["HT"].to_numpy()
        cr = trees["CR"].to_numpy()
        cb = h - cr * h
        max_radius = get_beta_max_crown_radius(h, cb, a, b, c, beta_param)
        trees["MAX_RADIUS"] = max_radius
    else:
        # Trait score
        trait_score = np.array(
            [spcd_params[s]["PURVES_TRAIT_SCORE"] for s in species_code]
        )
        trees["TRAIT_SCORE"] = trait_score
        # Shape parameter
        shape_param = get_purves_shape_param(trait_score)
        trees["SHAPE_PARAM"] = shape_param
        # Max crown radius
        dbh = trees["DIA"]
        max_radius = get_purves_max_crown_radius(trait_score, dbh)
        trees["MAX_RADIUS"] = max_radius

    return trees


def get_purves_shape_param(trait_score: NDArray):
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

    C0_B = 0.196
    C1_B = 0.511
    shape_parameter = (1.0 - trait_score) * C0_B + trait_score * C1_B
    return shape_parameter


def get_purves_max_crown_radius(trait_score: NDArray, dbh: NDArray):
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

    C0_R0 = 0.503
    C1_R0 = 3.126
    C0_R40 = 0.5
    C1_R40 = 10.0
    r0j = (1 - trait_score) * C0_R0 + trait_score * C1_R0
    r40j = (1 - trait_score) * C0_R40 + trait_score * C1_R40
    max_crown_radius = r0j + (r40j - r0j) * (dbh / 40.0)
    return max_crown_radius


@vectorize([float64(float64, float64, float64, float64, float64)])
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


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def get_beta_radius(z, height, crown_base, a, b, c, beta):
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

    if z < crown_base:
        return 0.0
    if z > height:
        return 0.0

    # Normalize
    crown_length = height - crown_base
    z = (z - crown_base) / crown_length

    r = (c * z ** (a - 1) * (1 - z) ** (b - 1)) / beta
    r = r * crown_length
    return r


@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def get_beta_max_crown_radius(height, crown_base, a, b, c, beta):
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
    r_max = get_beta_radius(z_max, height, crown_base, a, b, c, beta)
    return r_max
