# Core imports
from __future__ import annotations
from typing import TYPE_CHECKING

# Internal imports
from fastfuels_core.voxelization._coords import CenteringMode
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile
from fastfuels_core.voxelization.sampling import sample_occupied_cells

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree

# External imports
import numpy as np
from numpy import ndarray


class VoxelizedTree:
    def __init__(
        self, tree: "Tree", grid: ndarray, hr, vr, centering: CenteringMode = "cell"
    ):
        self.tree = tree
        self.grid = grid
        self.hr = hr
        self.vr = vr
        self.centering = centering

    def distribute_biomass(self):
        volume = np.sum(self.grid) * self.hr * self.hr * self.vr
        foliage_mpv = self.tree.foliage_biomass / volume
        biomass_grid = self.grid.copy() * foliage_mpv
        return biomass_grid


def voxelize_tree(
    tree: "Tree",
    horizontal_resolution: float,
    vertical_resolution: float,
    centering: CenteringMode = "cell",
    **kwargs,
) -> VoxelizedTree:

    vr_subgrid = kwargs.get("vr_subgrid", 0.1)
    crown_profile_mask = discretize_crown_profile(
        tree,
        horizontal_resolution,
        vertical_resolution,
        centering=centering,
        vr_subgrid=vr_subgrid,
    )

    alpha = kwargs.get("alpha", 0.5)
    beta = kwargs.get("beta", 0.5)
    rho = kwargs.get("rho", None)
    seed = kwargs.get("seed", None)

    sampled_crown_mask = sample_occupied_cells(
        crown_profile_mask, alpha, beta, rho, seed
    )
    return VoxelizedTree(
        tree, sampled_crown_mask, horizontal_resolution, vertical_resolution, centering
    )
