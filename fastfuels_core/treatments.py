# Core imports
from __future__ import annotations
import warnings
from enum import Enum
from typing import Protocol

# External imports
import numpy as np
from pandas import DataFrame


class ThinningDirection(str, Enum):
    """
    Enumeration of thinning directions. Thinning from below is the same as low
    thinning and thinning from above is the same as crown thinning.

    Attributes
    ----------
    BELOW : str
        Thinning from below (low thinning).
    ABOVE : str
        Thinning from above (crown thinning).
    """

    BELOW: str = "below"
    ABOVE: str = "above"


class TreatmentProtocol(Protocol):
    """
    Protocol for all treatment implementations.

    Methods
    -------
    apply(trees: DataFrame) -> DataFrame
        Apply the treatment to the DataFrame of trees.
    """

    def apply(self, trees: DataFrame) -> DataFrame:
        pass


class DirectionalThinToDiameterLimit:
    """
    Thinning treatment to limit trees based on their diameter.

    Parameters
    ----------
    limit : float
        Diameter limit for thinning, in centimeters (cm).
    direction : ThinningDirection
        Direction of thinning, either 'below' or 'above'. By default, 'below'.

    Methods
    -------
    apply(trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame
        Apply the diameter limit thinning to the DataFrame of trees.
    """

    def __init__(
        self, limit: float, direction: ThinningDirection = ThinningDirection.BELOW
    ) -> None:
        self.limit = limit
        self.direction = direction

    def apply(self, trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame:
        """
        Apply the diameter limit thinning to the DataFrame of trees.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data.
        dia_column_name : str, optional
            Name of the diameter column in the DataFrame, by default "DIA".
            The diameter values should be in centimeters (cm).

        Returns
        -------
        DataFrame
            DataFrame after applying the diameter limit thinning.

        Raises
        ------
        ValueError
            If the thinning direction is invalid.
        """
        df = trees.copy()

        if self.direction == ThinningDirection.BELOW:
            df = df[df[dia_column_name] >= self.limit]
        elif self.direction == ThinningDirection.ABOVE:
            df = df[df[dia_column_name] < self.limit]
        else:
            raise ValueError("Invalid thinning direction. Use 'below' or 'above'.")

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df


class DirectionalThinToStandBasalArea:
    """
    Thinning treatment to limit the stand basal area.

    Parameters
    ----------
    target : float
        Target basal area for thinning, in square meters (m²).
    direction : ThinningDirection
        Direction of thinning, either 'below' or 'above'. By default, 'below'.

    Methods
    -------
    apply(trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame
        Apply the basal area limit thinning to the DataFrame of trees.
    """

    def __init__(
        self, target: float, direction: ThinningDirection = ThinningDirection.BELOW
    ) -> None:
        self.target = target
        self.direction = direction

    def apply(self, trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame:
        """
        Apply the basal area limit thinning to the DataFrame of trees.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data.
        dia_column_name : str, optional
            Name of the diameter column in the DataFrame, by default "DIA".
            The diameter values should be in centimeters (cm).

        Returns
        -------
        DataFrame
            DataFrame after applying the basal area limit thinning.

        Raises
        ------
        ValueError
            If the thinning direction is invalid.
        """
        df = trees.copy()

        # Calculate basal area for each tree in square meters
        df["BA"] = df[dia_column_name] ** 2 * (np.pi / 40_000)

        if df["BA"].sum() <= self.target:
            return df

        if self.direction == ThinningDirection.BELOW:
            df.sort_values(by=dia_column_name, ascending=False, inplace=True)
        elif self.direction == ThinningDirection.ABOVE:
            df.sort_values(by=dia_column_name, ascending=True, inplace=True)
        else:
            raise ValueError("Invalid thinning direction. Use 'below' or 'above'.")

        df["BA_CUMSUM"] = df["BA"].cumsum()
        df = df[df["BA_CUMSUM"] <= self.target]

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df.drop(columns=["BA", "BA_CUMSUM"])


class ProportionalThinToBasalArea:
    """
    Proportional thinning treatment to reach a target basal area.

    Parameters
    ----------
    target : float
        Target basal area for thinning, in square meters (m²).

    Methods
    -------
    apply(trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame
        Apply the proportional thinning to reach the target basal area.
    """

    def __init__(self, target: float) -> None:
        self.target = target

    def apply(self, trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame:
        """
        Apply the proportional thinning to reach the target basal area.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data.
        dia_column_name : str, optional
            Name of the diameter column in the DataFrame, by default "DIA".
            The diameter values should be in centimeters (cm).

        Returns
        -------
        DataFrame
            DataFrame after applying the proportional thinning to the target basal area.

        Warns
        -----
        RuntimeWarning
            If the resulting basal area is still above the target after thinning.
        """
        df = trees.copy()

        # Calculate basal area for each tree in square meters
        df["BA"] = df[dia_column_name] ** 2 * (np.pi / 40_000)

        total_basal_area = df["BA"].sum()

        if total_basal_area <= self.target:
            return df

        proportion_to_remove = 1 - (self.target / total_basal_area)
        df["remove"] = np.random.rand(len(df)) < proportion_to_remove
        df = df[~df["remove"]]

        result_ba = df["BA"].sum()
        if result_ba > self.target:
            warnings.warn(
                f"Resulting basal area ({result_ba:.4f} m²) is still above the target ({self.target:.4f} m²) "
                f"after proportional thinning. Difference: {result_ba - self.target:.4f} m²",
                RuntimeWarning,
            )

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df.drop(columns=["BA", "remove"])
