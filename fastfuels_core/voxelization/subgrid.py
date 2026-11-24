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

Subgrid can also weight each subcell by a ``DensityField`` and place the stem
anywhere within its cell. With both, a ``"paraboloid"`` crown,
:class:`~fastfuels_core.voxelization.LinearHeightQuadraticRadialDensity`,
``n_subgrid=10`` and ``z_origin=0.0`` it reproduces LANL Trees' crown fuel:
the tree's bulk density times the grid is LANL's bulk-density grid.
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
    from fastfuels_core.voxelization.mass_distribution import DensityField


def discretize_crown_profile(
    tree: "Tree",
    hr: float,
    vr: float,
    centering: CenteringMode = "cell",
    n_subgrid: int = 10,
    z_origin: float = None,
    stem_offset: tuple[float, float] = (0.0, 0.0),
    density_field: "DensityField" = None,
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
    stem_offset : (float, float)
        The stem's position minus the point the grid is centered on (the
        middle cell's center, or the central vertex), in meters along x and y.
        Moves the crown within the grid without changing the grid's shape.
        Raises if the crown would cross the grid's edge, which cannot happen on
        a cell-centered grid for offsets within half a cell.
    density_field : DensityField, optional
        Count each inside subcell as the field's
        :meth:`~DensityField.weight` there, divided by its
        :meth:`~DensityField.crown_mean`, instead of as 1. Each voxel then
        holds the weight integrated over the part of it inside the crown, and
        the weight averages 1 over the crown. Pass the result to
        ``VoxelizedTree.distribute_biomass`` without a density field: passing
        the field again applies it twice.

    Returns
    -------
    ndarray, shape (nz, ny, nx)
        Volume fraction in [0, 1] on the same axes as the marching-squares
        backend, or the weighted volume fraction with ``density_field``. Rows
        run along +y and columns along +x. A crown bulk density times the grid
        is a bulk-density grid with that crown mean, holding the bulk density
        times ``grid.sum() * hr * hr * vr`` kilograms.
    """
    if n_subgrid < 1:
        raise ValueError(f"n_subgrid must be a positive integer, got {n_subgrid}")
    n_subgrid = int(n_subgrid)
    offset_x, offset_y = (float(v) for v in stem_offset)

    xy_pts = _get_horizontal_tree_coords(hr, tree.max_crown_radius, centering=centering)
    z_pts = _get_vertical_tree_coords(vr, tree.height, tree.crown_base_height, z_origin)
    nz, nxy = len(z_pts), len(xy_pts)

    half_width = float(xy_pts[-1]) + hr / 2.0
    if tree.max_crown_radius + max(abs(offset_x), abs(offset_y)) > half_width:
        raise ValueError(
            f"stem_offset {tuple(stem_offset)} moves the crown past the edge of "
            "the grid; keep it within half a cell."
        )

    # Subcell centers along each axis (exact: len * n_subgrid points), relative
    # to the stem.
    xy_sub = _resample_coords_grid_to_subgrid(xy_pts, hr, n_subgrid)
    x_sub = xy_sub - offset_x
    y_sub = xy_sub - offset_y
    r_squared = x_sub[None, :] ** 2 + y_sub[:, None] ** 2  # (ny_sub, nx_sub)
    if density_field is not None:
        radial_distance = np.sqrt(r_squared)[None, :, :]
        scale = 1.0 / density_field.crown_mean(tree)

    # One voxel layer at a time, so memory stays bounded for large crowns.
    grid = np.empty((nz, nxy, nxy))
    for k in range(nz):
        z_sub = _resample_coords_grid_to_subgrid(z_pts[k : k + 1], vr, n_subgrid)
        radius = np.atleast_1d(np.asarray(tree.get_crown_radius_at_height(z_sub)))
        # Inside-crown test per subcell: r <= R(z) between the crown base and
        # the top. R(z) is 0 outside them, but a subcell on the stem (r = 0)
        # would still pass r <= R, so bound z explicitly.
        in_crown = (z_sub >= tree.crown_base_height) & (z_sub <= tree.height)
        inside = (r_squared[None, :, :] <= (radius**2)[:, None, None]) & in_crown[
            :, None, None
        ]
        if density_field is None:
            layer = inside
        else:
            weight = density_field.weight(radial_distance, z_sub[:, None, None], tree)
            layer = inside * (np.asarray(weight, dtype=float) * scale)
        # Average the subcells of each voxel.
        grid[k] = layer.reshape(n_subgrid, nxy, n_subgrid, nxy, n_subgrid).mean(
            axis=(0, 2, 4)
        )
    return grid
