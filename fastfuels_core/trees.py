# Core imports
from __future__ import annotations
from abc import ABC, abstractmethod

# Internal imports
from fastfuels_core.base import ObjectIterableDataFrame
from fastfuels_core.point_process import run_point_process
from fastfuels_core.voxelization import VoxelizedTree, voxelize_tree
from fastfuels_core.treatments import TreatmentProtocol
from fastfuels_core.crown_profile_models.abc import CrownProfileModel
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.ref_data import REF_SPECIES, REF_JENKINS

# External Imports
import numpy as np
from numpy import ndarray
from nsvb.estimators import total_foliage_dry_weight
from pandera import DataFrameSchema, Column, Check, Index


TREE_SCHEMA_COLS = {
    "TREE_ID": Column(int),
    "SPCD": Column(
        int,
        title="Species Code",
        description="An FIA integer species code",
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
        jenkins_species_group: int = None,
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

        # Optional parameters
        self._jenkins_species_group = jenkins_species_group

        if crown_profile_model_type not in ["beta", "purves"]:
            raise ValueError(
                "The crown profile model must be one of the following: 'beta' or 'purves'"
            )
        self._crown_profile_model_type = crown_profile_model_type

        available_biomass_allometry_models = ["NSVB", "jenkins"]
        if biomass_allometry_model_type == "NSVB" and self.jenkins_species_group == 10:
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
    def jenkins_species_group(self) -> int:
        """
        Returns the species group of the tree based on the species code.
        """
        if self._jenkins_species_group:
            return self._jenkins_species_group
        return REF_SPECIES.loc[self.species_code]["JENKINS_SPGRPCD"]

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

        # Assign attributes
        self.species_code = species_code
        self.diameter = diameter

        # Read data from reference tables
        self._species_group = REF_SPECIES.loc[species_code]["JENKINS_SPGRPCD"]
        self._sapling_adjustment = REF_JENKINS.loc[self._species_group][
            "JENKINS_SAPLING_ADJUSTMENT"
        ]

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
        # Read beta parameters from reference table.
        # NOTE: in the reference table the parameters are named b1 and b2.
        # BUT in the paper they are named b0 and b1 :(.
        beta_0 = REF_JENKINS.loc[self._species_group]["JENKINS_TOTAL_B1"]
        beta_1 = REF_JENKINS.loc[self._species_group]["JENKINS_TOTAL_B2"]

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
        # Read beta parameters from reference table.
        # NOTE: in the reference table the parameters are named b1 and b2.
        # BUT in the paper they are named b0 and b1 :(.
        beta_0 = REF_JENKINS.loc[self._species_group]["JENKINS_FOLIAGE_RATIO_B1"]
        beta_1 = REF_JENKINS.loc[self._species_group]["JENKINS_FOLIAGE_RATIO_B2"]

        # Estimate the foliage component ratio
        ratio = np.exp(beta_0 + (beta_1 / self.diameter))

        # Estimate the foliage biomass
        foliage_biomass = self._estimate_above_ground_biomass() * ratio

        return foliage_biomass


def _is_valid_spcd(spcd: int) -> bool:
    return spcd in REF_SPECIES.index


def _is_valid_diameter(diameter: float) -> bool:
    return diameter > 0
