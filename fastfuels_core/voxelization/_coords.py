# Core imports
from __future__ import annotations
from typing import Literal

# External imports
import numpy as np
from numpy import ndarray

# Type definitions
CenteringMode = Literal["cell", "vertex"]


def _get_vertical_tree_coords(step, tree_height, crown_base_height):
    """
    Returns a grid of coordinates for a tree of height, height, with a spacing
    step. The grid is returned as a 1D array.
    """
    grid = np.arange(crown_base_height, tree_height + step, step)
    return grid


def _get_horizontal_tree_coords(
    step, radius, pos=0.0, centering: CenteringMode = "cell"
):
    """
    Discretizes a stem position and crown radius into a 1D array of coordinates.

    Parameters
    ----------
    step : float
        Grid cell size (resolution)
    radius : float
        Crown radius to cover
    pos : float
        Position of tree stem (default 0.0)
    centering : {"cell", "vertex"}
        - "cell": Grid has odd number of cells, stem at center of middle cell
        - "vertex": Grid has even number of cells, stem at vertex between cells
    """
    cells_per_side = int(np.floor(np.abs(radius / step))) + 1

    if centering == "cell":
        # Odd grid: stem at cell center
        lower_bound = pos - cells_per_side * step
        upper_bound = pos + cells_per_side * step
        grid = np.linspace(lower_bound, upper_bound, 2 * cells_per_side + 1)
    else:  # vertex
        # Even grid: stem at vertex (cell centers offset by half-step)
        lower_bound = pos - (cells_per_side - 0.5) * step
        upper_bound = pos + (cells_per_side - 0.5) * step
        grid = np.linspace(lower_bound, upper_bound, 2 * cells_per_side)

    return grid


def _resample_coords_grid_to_subgrid(
    grid: ndarray, grid_spacing: float, new_spacing: float
) -> ndarray:
    """
    Resamples grid with spacing grid_spacing to a subgrid with spacing
    new_spacing.
    """
    subgrid = np.arange(
        grid[0] - grid_spacing / 2 + new_spacing / 2,
        grid[-1] + grid_spacing / 2,
        new_spacing,
    )

    return subgrid
