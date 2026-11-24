# Core imports
from __future__ import annotations
from enum import Enum
from typing import Protocol

# External imports
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
