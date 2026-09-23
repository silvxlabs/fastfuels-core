"""Voxelization of tree crowns into volume-fraction / occupancy grids.

Public API is re-exported here; the implementation is split across
``_coords`` (grid coordinates), ``marching_squares`` (volume-fraction
discretization), ``measured_crown`` (volume fraction of a crown measured from
a CHM), ``sampling`` (stochastic occupancy realization), and
``tree`` (the ``VoxelizedTree`` / ``voxelize_tree`` orchestration).
"""

from fastfuels_core.voxelization._coords import CenteringMode
from fastfuels_core.voxelization.marching_squares import discretize_crown_profile
from fastfuels_core.voxelization.measured_crown import (
    MeasuredCrown,
    voxelize_measured_crown,
)
from fastfuels_core.voxelization.mass_distribution import (
    DensityField,
    GradientDensity,
    LinearHeightQuadraticRadialDensity,
    UniformDensity,
)
from fastfuels_core.voxelization.sampling import (
    compute_crown_probability_field,
    sample_occupancy,
    sample_occupied_cells,
)
from fastfuels_core.voxelization.tree import VoxelizedTree, voxelize_tree

__all__ = [
    "VoxelizedTree",
    "voxelize_tree",
    "CenteringMode",
    "discretize_crown_profile",
    "MeasuredCrown",
    "voxelize_measured_crown",
    "compute_crown_probability_field",
    "sample_occupancy",
    "sample_occupied_cells",
    "DensityField",
    "UniformDensity",
    "GradientDensity",
    "LinearHeightQuadraticRadialDensity",
]
