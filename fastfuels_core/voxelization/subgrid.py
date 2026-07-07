"""Subgrid voxelization.

A different method for building the crown volume-fraction grid than
:mod:`marching_squares`: instead of computing the exact circle-cell intersection
area, each output voxel is split into ``n_subgrid`` subcells per axis and the
volume fraction is the fraction of subcell centers that fall inside the crown
envelope. This is LANL Trees' own rule (``n_subgrid=10`` reproduces its
10x10x10 sampling).

Compared to marching squares:

* marching squares is exact in the horizontal and only subdivides the vertical;
  subgrid subdivides all three axes, so it is approximate (converging as
  ``n_subgrid`` grows) but needs no rotational-solid area math.
* the "inside" test is a single predicate (``r <= R(z)``), so this method
  extends to any geometry for which that predicate can be written -- the natural
  home for non-rotational crowns a circle-area kernel can't express.

Both methods expose ``discretize_crown_profile`` with the same output -- a
``(nz, ny, nx)`` volume-fraction grid on the same axes -- so they are
interchangeable inputs to
``VoxelizedTree(tree, grid, hr, vr).distribute_biomass(density_field=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from fastfuels_core.voxelization._coords import (
    CenteringMode,
    _get_horizontal_tree_coords,
    _get_vertical_tree_coords,
    _resample_coords_grid_to_subgrid,
)

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree


def discretize_crown_profile(
    tree: "Tree",
    hr: float,
    vr: float,
    centering: CenteringMode = "cell",
    n_subgrid: int = 10,
    z_origin: float = None,
) -> ndarray:
    """Voxelize a crown to a volume-fraction grid by subgrid sampling.

    Sibling to :func:`marching_squares.discretize_crown_profile` -- same output,
    different method. Each output voxel is divided into ``n_subgrid`` subcells
    along each axis (``n_subgrid**3`` total); the volume fraction is the fraction
    of those subcell centers inside the crown envelope ``r <= R(z)``.

    Parameters
    ----------
    tree : Tree
    hr, vr : float
        Horizontal and vertical output resolution (m).
    centering : {"cell", "vertex"}
        Grid centering, matching the marching-squares backend.
    n_subgrid : int
        Subcells per axis per voxel (>= 1). ``n_subgrid=10`` matches LANL Trees;
        ``n_subgrid=1`` samples each voxel once, at its center.
    z_origin : float, optional
        Reference height the z-cells align to (see
        ``_get_vertical_tree_coords``). ``None`` keeps the crown-base-centered
        grid; a float aligns cell boundaries to ``z_origin + k*vr`` (e.g. the
        ground elevation) for registration to a shared domain grid.

    Returns
    -------
    ndarray, shape (nz, ny, nx)
        Volume fraction in [0, 1] on the same axes as the marching-squares
        backend.
    """
    if n_subgrid < 1:
        raise ValueError(f"n_subgrid must be a positive integer, got {n_subgrid}")
    n_subgrid = int(n_subgrid)

    xy_pts = _get_horizontal_tree_coords(hr, tree.max_crown_radius, centering=centering)
    z_pts = _get_vertical_tree_coords(vr, tree.height, tree.crown_base_height, z_origin)
    nz, nxy = len(z_pts), len(xy_pts)

    # Subcell centers along each axis (exact: len * n_subgrid points).
    xy_sub = _resample_coords_grid_to_subgrid(xy_pts, hr, n_subgrid)
    z_sub = _resample_coords_grid_to_subgrid(z_pts, vr, n_subgrid)

    # Inside-crown test per subcell: r <= R(z). get_crown_radius_at_height
    # returns 0 outside [crown_base, height], so those subcells fall outside.
    r_squared = xy_sub[None, :] ** 2 + xy_sub[:, None] ** 2  # (ny_sub, nx_sub)
    radius = np.asarray(tree.get_crown_radius_at_height(z_sub))  # (nz_sub,)
    inside = r_squared[None, :, :] <= (radius**2)[:, None, None]

    # Average the subcells of each voxel -> volume fraction.
    return inside.reshape(nz, n_subgrid, nxy, n_subgrid, nxy, n_subgrid).mean(
        axis=(1, 3, 5)
    )
