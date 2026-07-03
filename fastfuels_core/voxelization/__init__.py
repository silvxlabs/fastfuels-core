"""Voxelization of tree crowns into volume-fraction / occupancy grids.

Public API is re-exported here; the implementation is split across
``_coords`` (grid coordinates), ``marching_squares`` (volume-fraction
discretization), ``sampling`` (stochastic occupancy realization), and
``tree`` (the ``VoxelizedTree`` / ``voxelize_tree`` orchestration).
"""

from fastfuels_core.voxelization._coords import CenteringMode
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile
from fastfuels_core.voxelization.tree import VoxelizedTree, voxelize_tree

__all__ = [
    "VoxelizedTree",
    "voxelize_tree",
    "CenteringMode",
    "discretize_crown_profile",
]
