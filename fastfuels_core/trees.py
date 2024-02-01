# Core imports
import json
from abc import ABC, abstractmethod
from importlib_resources import files

# Internal imports
from fastfuels_core.base import ObjectIterableDataFrame
from fastfuels_core.point_process import run_point_process

# External Imports
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
        crown_profile_model="beta",
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

    def is_live(self):
        """
        Returns True if the tree is alive, False otherwise. Tree is alive if
        the status code is 1.
        """
        return self.status_code == 1
