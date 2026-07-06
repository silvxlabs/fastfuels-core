"""Mass-distribution models: spread a tree's crown mass across occupied voxels.

This is the step *after* voxelization. The occupancy step (marching squares or
subgrid sampling) produces a volume-fraction grid; a ``DensityField`` then turns
that occupancy into a bulk-density grid (kg/m^3) by deciding *how much* of the
crown mass sits in each voxel.

Two flavors ship here:

* :class:`UniformDensity` -- one constant bulk density through the crown
  (mass / occupied volume). This is the original behavior.
* :class:`GradientDensity` -- distribute mass with an arbitrary weight function
  ``w(r, z, tree)``; the result is renormalized to conserve the crown mass, so a
  constant weight recovers :class:`UniformDensity`. Named subclasses (e.g.
  :class:`LinearHeightQuadraticRadialDensity`) pin a specific weight.

The occupancy step and the mass-distribution step are deliberately independent:
any occupancy backend composes with any ``DensityField``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree

# A weight function maps per-voxel radial distance ``r`` and height ``z`` (plus
# the tree, for geometry) to non-negative relative weights over the voxel grid.
WeightFunction = Callable[[ndarray, ndarray, "Tree"], ndarray]


class DensityField(ABC):
    """Turns an occupancy grid into a bulk-density grid (kg/m^3).

    Implementations receive the ``occupancy`` grid (volume fraction per voxel),
    the ``tree``, per-voxel radial distance ``r`` and height ``z`` (broadcastable
    to the grid), and the ``cell_volume`` (m^3), and return a bulk-density grid
    of the same shape as ``occupancy``.
    """

    @abstractmethod
    def apply(
        self,
        occupancy: ndarray,
        tree: "Tree",
        r: ndarray,
        z: ndarray,
        cell_volume: float,
    ) -> ndarray:
        raise NotImplementedError


class UniformDensity(DensityField):
    """Constant bulk density through the crown: ``mass / occupied volume``.

    Equivalent to the original ``VoxelizedTree.distribute_biomass``: the whole
    crown mass is spread at a single bulk density over the occupied volume.
    """

    def apply(self, occupancy, tree, r, z, cell_volume):
        volume = float(occupancy.sum()) * cell_volume
        if volume <= 0.0:
            return np.zeros_like(occupancy, dtype=float)
        return occupancy * (tree.foliage_biomass / volume)


class GradientDensity(DensityField):
    """Distribute crown mass with an arbitrary weight function ``w(r, z, tree)``.

    The weight sets the *shape* of the distribution; the result is renormalized
    so the integrated mass equals the tree's crown mass. Because of that
    renormalization a constant weight reproduces :class:`UniformDensity` exactly.

    Parameters
    ----------
    weight_fn : callable
        ``weight_fn(r, z, tree) -> ndarray`` returning non-negative relative
        weights broadcast over the voxel grid. ``r`` is the radial distance from
        the stem and ``z`` the height; both are broadcastable to the occupancy
        grid. Any new gradient is just a new weight function.
    """

    def __init__(self, weight_fn: WeightFunction):
        self.weight_fn = weight_fn

    def apply(self, occupancy, tree, r, z, cell_volume):
        weight = np.asarray(self.weight_fn(r, z, tree), dtype=float)
        raw = weight * occupancy
        total = float(raw.sum()) * cell_volume
        if total <= 0.0:
            return np.zeros_like(occupancy, dtype=float)
        return raw * (tree.foliage_biomass / total)


def _linear_height_quadratic_radial(r: ndarray, z: ndarray, tree: "Tree") -> ndarray:
    """Weight that is linear in height and quadratic in radius (see class below)."""
    hb = tree.crown_base_height
    ht = tree.height
    hd = tree.crown_profile_model.get_max_radius_height()
    d = 2.0 * tree.max_crown_radius
    return ((z - hb) + 4.0 * (ht - hd) * r**2 / d**2) / (ht - hb)


class LinearHeightQuadraticRadialDensity(GradientDensity):
    """Crown density that rises linearly with height and quadratically with radius.

    .. math::

        w(r, z) = \\frac{(z - H_b) + 4 (H_t - H_d)\\, r^2 / D^2}{H_t - H_b}

    where :math:`H_b` is the crown base, :math:`H_t` the tree top, :math:`H_d`
    the height of maximum crown diameter (taken from the crown profile's peak,
    ``crown_profile_model.get_max_radius_height()``), and :math:`D = 2\\cdot`
    ``max_crown_radius``. The weight is zero at the crown-base center and peaks
    at the top outer rim.

    This reproduces the vertical + radial fuel gradient of **LANL Trees**
    (``treesMACA``, ``fuels_create.F90``). Pair it with a crown envelope that
    peaks at :math:`H_d` -- i.e. ``crown_profile_model_type="paraboloid"`` -- to
    match LANL's crown shape as well as its mass gradient.

    Note that :class:`GradientDensity` renormalizes to conserve the tree's crown
    mass, so this yields LANL's *shape* with mass conserved; LANL's own absolute
    bulk density is ~0.75x that (its canopy bulk density is defined over an
    ellipsoid crown volume rather than the paraboloid it fills).
    """

    def __init__(self):
        super().__init__(_linear_height_quadratic_radial)
