# Core imports
from __future__ import annotations
import warnings

# Internal imports
from fastfuels_core.treatments.base import ThinningDirection

# External imports
import numpy as np
from pandas import DataFrame


class DirectionalThinToTreeDensity:
    """
    Thinning treatment to reach a target number of trees.

    Eligible trees are walked in diameter order and removed at a steady
    rate of ``cut_efficiency`` until the stand is down to ``target``, so
    a partial cut leaves survivors spread evenly through the sizes it
    passed rather than clearing them from one end. Stopping at the
    target leaves everything beyond it untouched.

    The steady rate is what makes a stand of identical trees behave the
    way a stand table does. Where an inventory repeats a tree -- a plot
    expanded by its trees-per-acre, say -- those repeats are a
    contiguous run in diameter order, and a steady rate takes
    ``cut_efficiency`` of each run. It degrades correctly the other way
    too: on a stem list where every tree is distinct, it removes that
    fraction of the trees.

    ``min_diameter`` and ``max_diameter`` bound which trees may be cut
    at all. A treatment that removes ladder fuels without entering the
    overstory is a maximum diameter with thinning from above: the
    largest trees under the limit go first, and everything over it is
    never eligible.

    Parameters
    ----------
    target : int
        Number of trees to leave.
    direction : ThinningDirection
        Direction of thinning, either 'below' or 'above'. By default,
        'below', which removes the smallest eligible trees first.
    min_diameter, max_diameter : float, optional
        Diameter bounds on eligibility, in centimeters (cm), exclusive
        at both ends. By default every tree is eligible.
    cut_efficiency : float, optional
        Fraction of the eligible trees that may be removed, in [0, 1].
        By default 1.0, which allows every eligible tree to be cut.

    Methods
    -------
    apply(trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame
        Apply the tree density thinning to the DataFrame of trees.
    """

    def __init__(
        self,
        target: int,
        direction: ThinningDirection = ThinningDirection.BELOW,
        min_diameter: float = -np.inf,
        max_diameter: float = np.inf,
        cut_efficiency: float = 1.0,
    ) -> None:
        if not 0.0 <= cut_efficiency <= 1.0:
            raise ValueError(f"cut_efficiency must be in [0, 1]; got {cut_efficiency}.")
        if direction not in tuple(ThinningDirection):
            raise ValueError("Invalid thinning direction. Use 'below' or 'above'.")
        self.target = target
        self.direction = direction
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.cut_efficiency = cut_efficiency

    def apply(self, trees: DataFrame, dia_column_name: str = "DIA") -> DataFrame:
        """
        Apply the tree density thinning to the DataFrame of trees.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data.
        dia_column_name : str, optional
            Name of the diameter column in the DataFrame, by default
            "DIA". The diameter values should be in centimeters (cm).

        Returns
        -------
        DataFrame
            DataFrame after applying the tree density thinning, with the
            surviving trees in their original order.

        Warns
        -----
        RuntimeWarning
            If the target could not be reached, because too few trees
            fall inside the diameter bounds or because
            ``cut_efficiency`` held trees back.
        """
        df = trees.copy()

        surplus = len(df) - self.target
        if surplus <= 0:
            return df

        # Work positionally: a tree frame may carry duplicate index
        # labels, and dropping by label would take every tree sharing one.
        work = df.reset_index(drop=True)
        eligible = work[
            (work[dia_column_name] > self.min_diameter)
            & (work[dia_column_name] < self.max_diameter)
        ]
        # Thinning from below takes the smallest eligible trees first.
        ascending = self.direction == ThinningDirection.BELOW
        eligible = eligible.sort_values(
            by=dia_column_name, ascending=ascending, kind="stable"
        )

        # Remove at a steady rate along the eligible trees: the nth is
        # cut when the running quota n * cut_efficiency crosses an
        # integer. Over any run of equal trees this takes the same
        # fraction of that run, and over a stem list of distinct trees
        # it takes the same fraction of the trees.
        quota = np.arange(1, len(eligible) + 1) * self.cut_efficiency
        crossings = np.floor(quota) > np.floor(quota - self.cut_efficiency)
        cut = eligible.index.to_numpy()[crossings][:surplus]

        if len(cut) < surplus:
            warnings.warn(
                f"Thinning left {len(df) - len(cut)} trees, above the target "
                f"of {self.target}: only {len(eligible)} trees fall within "
                f"the diameter bounds, and cut_efficiency caps the cut at "
                f"{self.cut_efficiency:.0%} of them.",
                RuntimeWarning,
            )

        keep = np.ones(len(work), dtype=bool)
        keep[cut] = False
        df = df[keep]

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df
