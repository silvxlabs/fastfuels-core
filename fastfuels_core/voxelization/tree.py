# Core imports
from __future__ import annotations
from typing import TYPE_CHECKING

# Internal imports
from fastfuels_core.voxelization._coords import (
    CenteringMode,
    _get_horizontal_tree_coords,
    _get_vertical_tree_coords,
)
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile
from fastfuels_core.voxelization.mass_distribution import DensityField, UniformDensity
from fastfuels_core.voxelization.sampling import sample_occupied_cells

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree

# External imports
import numpy as np
from numpy import ndarray


class VoxelizedTree:
    def __init__(
        self,
        tree: "Tree",
        grid: ndarray,
        hr,
        vr,
        centering: CenteringMode = "cell",
        z_origin: float = None,
    ):
        self.tree = tree
        self.grid = grid
        self.hr = hr
        self.vr = vr
        self.centering = centering
        # Must match the z_origin the occupancy grid was built with so the
        # density field's per-voxel heights line up with the grid.
        self.z_origin = z_origin

    def distribute_biomass(self, density_field: DensityField = None) -> ndarray:
        """Distribute the tree's crown mass across occupied voxels.

        The mass-distribution step is independent of how the occupancy grid was
        produced. ``density_field`` selects the distribution; it defaults to
        :class:`UniformDensity`, reproducing the original constant-density
        behavior.
        """
        if density_field is None:
            density_field = UniformDensity()
        z, r = self._voxel_coordinates()
        cell_volume = self.hr * self.hr * self.vr
        return density_field.apply(self.grid, self.tree, r, z, cell_volume)

    def _voxel_coordinates(self) -> tuple[ndarray, ndarray]:
        """Per-voxel height ``z`` (nz,1,1) and radial distance from the stem
        ``r`` (1,ny,nx), matching ``self.grid``'s axes."""
        z = _get_vertical_tree_coords(
            self.vr, self.tree.height, self.tree.crown_base_height, self.z_origin
        )
        xy = _get_horizontal_tree_coords(
            self.hr, self.tree.max_crown_radius, centering=self.centering
        )
        r = np.sqrt(xy[None, :] ** 2 + xy[:, None] ** 2)
        return z[:, None, None], r[None, :, :]


def voxelize_tree(
    tree: "Tree",
    horizontal_resolution: float,
    vertical_resolution: float,
    centering: CenteringMode = "cell",
    **kwargs,
) -> VoxelizedTree:

    n_vertical_subgrid = kwargs.get("n_vertical_subgrid", 10)
    z_origin = kwargs.get("z_origin", None)
    crown_profile_mask = discretize_crown_profile(
        tree,
        horizontal_resolution,
        vertical_resolution,
        centering=centering,
        n_vertical_subgrid=n_vertical_subgrid,
        z_origin=z_origin,
    )

    alpha = kwargs.get("alpha", 0.5)
    beta = kwargs.get("beta", 0.5)
    rho = kwargs.get("rho", None)
    seed = kwargs.get("seed", None)

    sampled_crown_mask = sample_occupied_cells(
        crown_profile_mask, alpha, beta, rho, seed
    )
    return VoxelizedTree(
        tree,
        sampled_crown_mask,
        horizontal_resolution,
        vertical_resolution,
        centering,
        z_origin=z_origin,
    )
