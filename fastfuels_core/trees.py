# Core imports
from __future__ import annotations
import json
from abc import ABC, abstractmethod

import numpy as np
from importlib_resources import files

# Internal imports
from fastfuels_core.base import ObjectIterableDataFrame
from fastfuels_core.point_process import run_point_process

# External Imports
from numpy import ndarray
from scipy.special import beta
from pandera import DataFrameSchema, Column, Check, Index


DATA_PATH = files("fastfuels_core.data")
with open(DATA_PATH / "spgrp_parameters.json", "r") as f:
    SPGRP_PARAMS = json.load(f)
with open(DATA_PATH / "spcd_parameters.json", "r") as f:
    SPCD_PARAMS = json.load(f)
with open(DATA_PATH / "class_parameters.json", "r") as f:
    CLASS_PARAMS = json.load(f)


TREE_SCHEMA_COLS = {
    "TREE_ID": Column(int),
    "SPCD": Column(
        int, title="Species Code", description="An FIA integer species code"
    ),
    "STATUSCD": Column(
        int,
        checks=Check.isin([0, 1, 2, 3]),
        title="Status Code",
        description="0 = No status, 1 = Live, 2 = Dead, 3 = Missing",
    ),
    "DIA": Column(
        float,
        checks=[
            Check.gt(0),
            Check.lt(1200),
        ],
        nullable=True,
        title="Diameter at breast height (cm)",
        description="Diameter of the tree measured at breast height (1.37 m)",
    ),
    "HT": Column(
        float,
        checks=[
            Check.gt(0),
            Check.le(116),
        ],
        nullable=True,
        title="Height (m)",
        description="Height of the tree measured from the ground",
    ),
    "CR": Column(
        float,
        checks=Check.in_range(min_value=0, max_value=1),
        nullable=True,
        title="Crown Ratio",
        description="Ratio of the crown length to the total tree height",
    ),
}


class TreeSample(ObjectIterableDataFrame):
    schema = DataFrameSchema(
        columns={
            **TREE_SCHEMA_COLS,
            **{
                "PLOT_ID": Column(int, required=False),
                "TPA": Column(
                    dtype=float,
                    nullable=True,
                    title="Trees per Area (1/m^2)",
                    description="The number of trees per unit area that the sample "
                    "tree represents based on the plot design",
                ),
            },
        },
        coerce=True,
        index=Index(int, unique=True),
    )

    def expand_to_roi(self, process, roi, **kwargs):
        """
        Expands the trees to the Region of Interest (ROI) using a specified
        point process.

        This method generates a new set of trees that fit within the ROI
        based on the characteristics of the specified point process. It
        utilizes the 'run_point_process' function from the 'point_process'
        module to create an instance of the PointProcess class, generate tree
        locations using the specified point process, and assigns these
        locations to trees.

        Parameters
        ----------
        process : str
            The type of point process to use for expanding the trees.
            Currently, only 'inhomogeneous_poisson' is supported.
        roi : GeoDataFrame
            The Region of Interest to which the trees should be expanded. It
            should be a GeoDataFrame containing the spatial boundaries of the
            region.
        **kwargs : dict
            Additional parameters to pass to the point process. See the
            documentation for the 'point_process' module for more information.

        Returns
        -------
        Trees
            A new Trees instance where the trees have been expanded to fit
            within the ROI based on the specified point process.

        Example
        -------
        # TODO: Add example
        """
        return TreePopulation(run_point_process(process, roi, self, **kwargs))


class TreePopulation(ObjectIterableDataFrame):
    schema = DataFrameSchema(
        columns={
            **TREE_SCHEMA_COLS,
            **{
                "X": Column(
                    float,
                    nullable=True,
                    required=False,
                    title="X Coordinate (m)",
                    description="X coordinate of the tree in a projected coordinate system",
                ),
                "Y": Column(
                    float,
                    nullable=True,
                    required=False,
                    title="Y Coordinate (m)",
                    description="Y coordinate of the tree in a projected coordinate system",
                ),
            },
        },
        coerce=True,
        add_missing_columns=True,
        index=Index(int, unique=True),
    )


class Tree:
    """
    An object representing an individual tree. The tree has required attributes
    that represent measurements of the tree, provide estimates of the tree's
    biomass and crown profile, and the tree's location in a projected
    coordinate system.

    Attributes
    ----------
    id : int
        A unique identifier for the tree.
    species_code : int
        An FIA integer species code.
    status_code : int
        1 = Live, 2 = Dead, 3 = Missing
    diameter : float
        Diameter (cm) of the tree measured at breast height (1.37 m). Units: cm.
    height : float
        Height (m) of the tree measured from the ground. Units: m.
    crown_ratio : float
        Ratio (-) of the crown length to the total tree height. Units: none.
    x : float
        X coordinate of the tree in a projected coordinate system. Units: m.
    y : float
        Y coordinate of the tree in a projected coordinate system. Units: m.
    """

    species_code: int
    status_code: int
    diameter: float
    height: float
    crown_ratio: float
    x: float
    y: float

    def __init__(
        self,
        species_code: int,
        status_code: int,
        diameter: float,
        height: float,
        crown_ratio: float,
        x=0,
        y=0,
        crown_profile_model_type="beta",
    ):
        # TODO: Species code needs to be valid
        self.species_code = species_code

        # TODO: Status code needs to be valid
        self.status_code = status_code

        # TODO: Diameter and height need to be greater than 0
        self.diameter = diameter
        self.height = height

        # TODO: Crown ratio needs to be between 0 and 1
        self.crown_ratio = crown_ratio

        self.x = x
        self.y = y

        if crown_profile_model_type not in ["beta"]:
            raise ValueError(
                "The crown profile model must be one of the following: 'beta'"
            )
        self._crown_profile_model_type = crown_profile_model_type
        self._crown_profile_model = None

    @property
    def crown_length(self) -> float:
        """
        Returns the length of the tree's crown.
        """
        return self.height * self.crown_ratio

    @property
    def crown_base_height(self) -> float:
        """
        Returns the height at which the live crown starts.
        """
        return self.height - self.crown_length

    @property
    def species_group(self) -> int:
        """
        Returns the species group of the tree based on the species code.
        """
        return SPCD_PARAMS[str(self.species_code)]["SPGRP"]

    @property
    def is_live(self):
        """
        Returns True if the tree is alive, False otherwise. The Tree is alive if
        the status code is 1.
        """
        return self.status_code == 1

    @property
    def crown_profile_model(self) -> CrownProfileModel:
        # Initialize the beta crown profile model if it is not already initialized
        if self._crown_profile_model_type == "beta":
            if (
                self._crown_profile_model is None
                or self._crown_profile_model.crown_base_height != self.crown_base_height
                or self._crown_profile_model.crown_length != self.crown_length
                or self._crown_profile_model.species_group != self.species_group
            ):
                self._crown_profile_model = BetaCrownProfile(
                    self.species_group, self.crown_base_height, self.crown_length
                )
        return self._crown_profile_model

    @property
    def max_crown_radius(self) -> float:
        """
        Returns the maximum radius of the tree's crown.
        """
        return self.crown_profile_model.get_max_radius()

    def get_crown_radius_at_height(self, height: float | ndarray) -> float | ndarray:
        """
        Uses the crown profile model to get the radius of the crown at a given
        height in meters.
        """
        return self.crown_profile_model.get_radius_at_height(height)


class CrownProfileModel(ABC):
    """
    Abstract base class representing a tree crown profile model.
    The crown profile model is a rotational solid that can be queried at any
    height to get a radius, or crown width, at that height.

    This abstract class provides methods to get the radius of the crown at a
    given height and to get the maximum radius of the crown.
    """

    @abstractmethod
    def get_radius_at_height(self, height):
        """
        Abstract method to get the radius of the crown at a given height.
        """
        raise NotImplementedError(
            "The get_radius method must be implemented in a subclass."
        )

    @abstractmethod
    def get_max_radius(self) -> float:
        """
        Abstract method to get the maximum radius of the crown.
        """
        raise NotImplementedError(
            "The get_max_radius method must be implemented in a subclass."
        )


class BetaCrownProfile(CrownProfileModel):
    """
    A crown profile model based on a beta distribution.
    """

    a: float
    b: float
    c: float
    species_group: int
    crown_length: float
    crown_base_height: float

    def __init__(
        self, species_group: int, crown_base_height: float, crown_length: float
    ):
        """
        Initializes a BetaCrownProfile instance.
        """
        self.species_group = species_group
        self.crown_base_height = crown_base_height
        self.crown_length = crown_length
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
            z[mask] ** (self.a - 1) * (1 - z[mask]) ** (self.b - 1) / self.beta
        )

        if result.size == 1:
            return result.item()  # Return as a scalar
        else:
            return result  # Return as an array
