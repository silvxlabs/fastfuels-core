# Core imports
from typing import TYPE_CHECKING

# Internal imports
if TYPE_CHECKING:
    from fastfuels_core.trees import TreePopulation

# External imports
import rioxarray  # noqa F401
from xarray import DataArray


def run_poisson_disc_sampling(trees: TreePopulation) -> TreePopulation:
    """
    Run a Poisson-disc sampling algorithm on a tree population.

    Parameters
    ----------
    trees : TreePopulation
        A tree population to sample from.

    Returns
    -------
    TreePopulation
        A tree population with trees sampled from the input tree population using a Poisson-disc sampling algorithm.
    """


def run_poisson_disc_sampling_with_height_grid(
    trees: TreePopulation, height_grid: DataArray
) -> TreePopulation:
    """
    Run a Poisson-disc sampling algorithm on a tree population with a height grid.

    Parameters
    ----------
    trees : TreePopulation
        A tree population to sample from.
    height_grid : DataArray
        A height grid to sample from.

    Returns
    -------
    TreePopulation
        A tree population with the sampled trees.
    """
