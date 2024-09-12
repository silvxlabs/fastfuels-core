# Core imports
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from importlib.resources import files

# Internal imports
from fastfuels_core.base import ObjectIterableDataFrame
from fastfuels_core.point_process import run_point_process
from fastfuels_core.voxelization import VoxelizedTree, voxelize_tree
from fastfuels_core.treatments import TreatmentProtocol

# External Imports
import numpy as np
from numpy import ndarray
from scipy.special import beta
from nsvb.estimators import total_foliage_dry_weight
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

    def _row_to_object(self, row) -> Tree:
        """
        Convert a row of the dataframe to an object.
        """
        return Tree(
            species_code=row.SPCD,
            status_code=row.STATUSCD,
            diameter=row.DIA,
            height=row.HT,
            crown_ratio=row.CR,
            x=row.X,
            y=row.Y,
        )

    def apply_treatment(
        self, treatment: TreatmentProtocol | list[TreatmentProtocol]
    ) -> TreePopulation:
        """
        Applies one or more preconfigured silvicultural treatments to the tree population.

        Parameters
        ----------
        treatment : TreatmentProtocol | list[TreatmentProtocol]
            A single treatment or a list of treatments to apply.

        Returns
        -------
        TreePopulation
            A new TreePopulation object post treatment.
        """
        if isinstance(treatment, list):
            df = self.data.copy()
            for t in treatment:
                df = t.apply(df)
            return TreePopulation(df)
        else:
            return TreePopulation(treatment.apply(self.data))


class Tree:
    """
    An object representing an individual tree. The tree has required attributes
    that represent measurements of the tree, provide estimates of the tree's
    biomass and crown profile, and the tree's location in a projected
    coordinate system.

    Attributes
    ----------
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
        crown_profile_model_type="purves",
        biomass_allometry_model_type="NSVB",
    ):
        # TODO: Species code needs to be valid
        self.species_code = int(species_code)

        # TODO: Status code needs to be valid
        self.status_code = status_code

        # TODO: Diameter and height need to be greater than 0
        self.diameter = diameter
        self.height = height

        # TODO: Crown ratio needs to be between 0 and 1
        self.crown_ratio = crown_ratio

        self.x = x
        self.y = y

        if crown_profile_model_type not in ["beta", "purves"]:
            raise ValueError(
                "The crown profile model must be one of the following: 'beta' or 'purves'"
            )
        self._crown_profile_model_type = crown_profile_model_type

        available_biomass_allometry_models = ["NSVB", "jenkins"]
        if biomass_allometry_model_type == "NSVB" and self.species_group == 10:
            biomass_allometry_model_type = "jenkins"
        if biomass_allometry_model_type not in available_biomass_allometry_models:
            raise ValueError(
                f"Selected biomass allometry model: {biomass_allometry_model_type}. The biomass allometry model must be one of the following: {available_biomass_allometry_models}"
            )
        self._biomass_allometry_model_type = biomass_allometry_model_type

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
        if self._crown_profile_model_type == "beta":
            return BetaCrownProfile(
                self.species_code, self.crown_base_height, self.crown_length
            )
        elif self._crown_profile_model_type == "purves":
            return PurvesCrownProfile(
                self.species_code, self.diameter, self.height, self.crown_ratio
            )

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

    @property
    def biomass_allometry_model(self) -> BiomassAllometryModel:
        if self._biomass_allometry_model_type == "jenkins":
            return JenkinsBiomassEquations(self.species_code, self.diameter)
        elif self._biomass_allometry_model_type == "NSVB":
            return NSVBEquations(
                self.species_code,
                self.diameter,
                self.height,
            )

    @property
    def foliage_biomass(self) -> float:
        """
        Returns the estimated foliage biomass of the tree
        """
        return self.biomass_allometry_model.estimate_foliage_biomass()

    def voxelize(
        self, horizontal_resolution: float, vertical_resolution: float, **kwargs
    ) -> VoxelizedTree:
        return voxelize_tree(self, horizontal_resolution, vertical_resolution, **kwargs)

    @classmethod
    def from_row(cls, row):
        return cls(
            species_code=int(row.SPCD),
            status_code=int(row.STATUSCD),
            diameter=row.DIA,
            height=row.HT,
            crown_ratio=row.CR,
            x=row.X,
            y=row.Y,
        )


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

    # See Purves et al. (2007) Table S2 in Supporting Information
    C0_R0 = 0.503
    C1_R0 = 3.126
    C0_R40 = 0.5
    C1_R40 = 10.0
    C0_B = 0.196
    C1_B = 0.511

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

        trait_score = SPCD_PARAMS[str(species_code)]["PURVES_TRAIT_SCORE"]

        # Compute maximum crown radius
        r0j = (1 - trait_score) * self.C0_R0 + trait_score * self.C1_R0
        r40j = (1 - trait_score) * self.C0_R40 + trait_score * self.C1_R40
        self.max_crown_radius = r0j + (r40j - r0j) * (dbh / 40.0)

        # Compute crown shape parameter
        self.shape_parameter = (1 - trait_score) * self.C0_B + trait_score * self.C1_B

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
        return self.get_radius_at_height(self.crown_base_height)


class BiomassAllometryModel(ABC):
    """
    Abstract base class representing a tree biomass allometry model.
    The biomass allometry model is used to estimate the biomass of a tree
    based on one or more independent variables, such as diameter and height.
    """

    @abstractmethod
    def estimate_foliage_biomass(self) -> float:
        """
        Abstract method to estimate the biomass of a tree based on its diameter
        and height.
        """


class NSVBEquations(BiomassAllometryModel):
    """
    This class implements the National Scale Volume and Biomass Estimators
    modeling system.

    Information on the NSVB system can be found in the following publication:
    https://www.fs.usda.gov/research/treesearch/66998
    """

    def __init__(
        self,
        species_code: int,
        diameter: float,
        height: float,
        division: str = "",
        cull: int = 0,
        decay_code: int = 0,
        actual_height: float = None,
    ):
        if not _is_valid_spcd(species_code):
            raise ValueError(f"Species code {species_code} is not valid.")

        if not _is_valid_diameter(diameter):
            raise ValueError(f"Diameter {diameter} must be greater than 0.")

        self.species_code = species_code
        self.diameter = diameter
        self.height = height
        self.division = division
        self.cull = cull
        self.decay_code = decay_code
        self.actual_height = actual_height if actual_height else height

    def estimate_foliage_biomass(self):
        """
        Foliage weight is estimated by combining foliage weight coefficients
        and the appropriate model in Table S9a of the NSVB GTR.
        """
        dia_ft = self.diameter / 2.54
        height_ft = self.height * 3.28084
        dry_foliage_lb = total_foliage_dry_weight(
            spcd=self.species_code,
            dia=dia_ft,
            ht=height_ft,
            division=self.division,
        )
        return dry_foliage_lb * 0.453592  # Convert to kg


class JenkinsBiomassEquations(BiomassAllometryModel):
    """
    This class implements the National-Scale Biomass Estimators developed by
    Jenkins et al. (2003). These equations are used to estimate above ground and
    component biomass for 10 species groups in the United States.
    """

    def __init__(self, species_code, diameter):
        if not _is_valid_spcd(species_code):
            raise ValueError(f"Species code {species_code} is not valid.")

        if not _is_valid_diameter(diameter):
            raise ValueError(f"Diameter {diameter} must be greater than 0.")

        self.species_code = species_code
        self.diameter = diameter

        self._species_group = SPCD_PARAMS[str(species_code)]["SPGRP"]
        self._is_softwood = SPCD_PARAMS[str(species_code)]["CLASS"]
        self._sapling_adjustment = SPCD_PARAMS[str(species_code)]["SAPADJ"]

    def _estimate_above_ground_biomass(self):
        """
        Uses Equation 1 and parameters in Table 4 of Jenkins et al. 2003 to
        estimate the above ground biomass of a tree based on species group and
        diameter at breast height.

        Biomass equation: bm = Exp(β_0 + (β_1 * ln(dbh))) Where bm is above
        ground biomass (kg) for trees 2.5cm dbh and larger, dbh is the
        diameter at breast height (cm), and β_0 and β_1 are parameters
        estimated from the data.

        NOTE: This method also applies a sapling adjustment factor (sapadj) for
        trees with dbh <= 12.7cm (5 inches).
        """
        # Get the parameters for the species group
        beta_0 = SPGRP_PARAMS[str(self._species_group)]["JENKINS_AGB_b0"]
        beta_1 = SPGRP_PARAMS[str(self._species_group)]["JENKINS_AGB_b1"]

        # Estimate the above ground biomass
        biomass = np.exp(beta_0 + (beta_1 * np.log(self.diameter)))

        # Apply the sapling adjustment factor
        if self.diameter <= 12.7 and self._sapling_adjustment > 0:
            biomass *= self._sapling_adjustment

        return biomass

    def estimate_foliage_biomass(self):
        """
        Uses Equation 2 and parameters in Table 6 of Jenkins et al. 2003 to
        estimate component ratio and foliage biomass of a tree based on species
        group, hardwood classification, and diameter at breast height.

        Biomass ratio equation:
        r = Exp(β_0 + (β_1 / dbh))
        Where r is the ratio of component to total aboveground biomass for
        trees 2.5cm dbh and larger, dbh is the diameter at breast height in cm,
        and β_0 and β_1 are parameters estimated from the data.

        r is multiplied by the above ground biomass to estimate the
        foliage biomass in kg.
        """
        # Get the parameters for the hardwood class
        beta_0 = CLASS_PARAMS[str(self._is_softwood)]["foliage_b0"]
        beta_1 = CLASS_PARAMS[str(self._is_softwood)]["foliage_b1"]

        # Estimate the foliage component ratio
        ratio = np.exp(beta_0 + (beta_1 / self.diameter))

        # Estimate the foliage biomass
        foliage_biomass = self._estimate_above_ground_biomass() * ratio

        return foliage_biomass


def _is_valid_spcd(spcd: int) -> bool:
    return str(int(spcd)) in SPCD_PARAMS


def _is_valid_diameter(diameter: float) -> bool:
    return diameter > 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create subplots for different crown ratios
    crown_ratios = [0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, len(crown_ratios), figsize=(20, 10), sharey=True)

    for ax, crown_ratio in zip(axes, crown_ratios):
        tree_beta = Tree(
            814, 1, 15.0, 5.0, crown_ratio, 0, 0, crown_profile_model_type="beta"
        )
        tree_purves = Tree(
            814, 1, 15.0, 5.0, crown_ratio, 0, 0, crown_profile_model_type="purves"
        )

        heights = np.linspace(0, tree_beta.height, 100)
        beta_p = tree_beta.get_crown_radius_at_height(heights)
        purves_p = tree_purves.get_crown_radius_at_height(heights)

        ax.plot(beta_p, heights, "-r", label="Beta Crown Profile")
        ax.plot(purves_p, heights, "-k", label="Purves Crown Profile")
        ax.set_xlabel("Crown Radius (m)")
        ax.set_title(f"Crown Ratio = {crown_ratio}")
        ax.set_aspect("equal")
        ax.legend()

    axes[0].set_ylabel("Height (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
