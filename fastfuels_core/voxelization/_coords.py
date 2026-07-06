# Core imports
from __future__ import annotations
from typing import Literal

# External imports
import numpy as np
from numpy import ndarray

# Type definitions
CenteringMode = Literal["cell", "vertex"]


def _get_vertical_tree_coords(step, tree_height, crown_base_height, z_origin=None):
    """
    Returns the z cell-center coordinates spanning the crown, spacing ``step``.

    ``z_origin`` controls where the grid is anchored:

    * ``None`` (default): the crown base sits at the *center* of the first cell
      (cells at ``crown_base_height + k*step``). Self-anchored per tree.
    * a float: cell *boundaries* align to ``z_origin + k*step`` (centers at
      ``z_origin + (k + 0.5)*step``), so the crown base falls wherever it lands
      inside a cell. Use this to register the crown to a shared domain grid
      (e.g. ``z_origin=ground_elevation``).
    """
    if z_origin is None:
        return np.arange(crown_base_height, tree_height + step, step)
    k_lo = int(np.floor((crown_base_height - z_origin) / step))
    k_hi = int(np.floor((tree_height - z_origin) / step))
    return z_origin + (np.arange(k_lo, k_hi + 1) + 0.5) * step


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
    grid: ndarray, grid_spacing: float, n_subgrid: int
) -> ndarray:
    """
    Split each cell (centered on a point in ``grid``, of width ``grid_spacing``)
    into ``n_subgrid`` equal subcells and return their centers.

    Exact by construction: the result always has ``len(grid) * n_subgrid``
    points, so it is robust to counts whose implied spacing has no clean float
    representation (e.g. 3 subcells of 1/3 m).
    """
    grid = np.asarray(grid)
    offsets = (np.arange(n_subgrid) + 0.5) / n_subgrid * grid_spacing - grid_spacing / 2
    return (grid[:, None] + offsets[None, :]).ravel()
