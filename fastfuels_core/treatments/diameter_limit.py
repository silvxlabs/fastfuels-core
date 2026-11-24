# Core imports
from __future__ import annotations

# Internal imports
from fastfuels_core.treatments.base import ThinningDirection

# External imports
from pandas import DataFrame


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
