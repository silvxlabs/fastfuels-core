"""Mass-distribution models: spread a tree's crown mass across occupied voxels.

This is the step *after* voxelization. Voxelization (marching squares or
subgrid) produces a volume-fraction grid, which occupancy sampling may thin; a
``DensityField`` then turns that grid into a bulk-density grid (kg/m^3) by
deciding *how much* of the crown mass sits in each voxel.

A ``DensityField`` is a **crown weight function**: it supplies the relative
weight of each occupied voxel via :meth:`DensityField.crown_weight`. The
mass-conserving normalization -- scale the weighted grid so its integral
equals the tree's crown mass -- is shared and lives in
:meth:`VoxelizedTree.distribute_biomass`, so subclasses only choose the *shape*
of the distribution.

Two flavors ship here:

* :class:`UniformDensity` -- a constant weight, giving one bulk density through
  the crown (mass / occupied volume). This is the original behavior, and it
  reads no crown geometry.
* :class:`GradientDensity` -- an arbitrary weight function ``w(r, z, tree)``.
  Because of the shared renormalization a constant weight recovers
  :class:`UniformDensity`. Named subclasses (e.g.
  :class:`LinearHeightQuadraticRadialDensity`) pin a specific weight.

Voxelization, occupancy sampling and mass distribution are deliberately
independent: any voxelization backend composes with any ``DensityField``.

A field can also be integrated inside voxelization:
``subgrid.discretize_crown_profile(..., density_field=...)`` weights each
subcell by :meth:`DensityField.weight`, scaled by :meth:`DensityField.crown_mean`
so the weight averages 1 over the crown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree
    from fastfuels_core.voxelization.tree import VoxelizedTree

# A weight function maps per-voxel radial distance ``r`` and height ``z`` (plus
# the tree, for geometry) to non-negative relative weights over the voxel grid.
WeightFunction = Callable[[ndarray, ndarray, "Tree"], ndarray]


class DensityField(ABC):
    """A crown weight function for distributing crown mass over occupied voxels.

    Implementations return the relative weight of each occupied voxel; the
    shared, mass-conserving normalization in
    :meth:`VoxelizedTree.distribute_biomass` turns those weights into a
    bulk-density grid (kg/m^3). A constant weight yields uniform density.
    """

    @abstractmethod
    def crown_weight(self, vt: "VoxelizedTree") -> ndarray | float:
        """Return the relative weight of each occupied voxel.

        Parameters
        ----------
        vt : VoxelizedTree
            The voxelized tree being distributed. Weights that vary with crown
            geometry read ``vt.voxel_height`` and ``vt.radial_distance``, which
            are computed lazily -- a constant weight never triggers them.

        Returns
        -------
        ndarray or float
            Per-voxel relative weights broadcastable to ``vt.grid``, or a scalar
            for a spatially constant weight. Need not be normalized;
            ``distribute_biomass`` renormalizes to conserve the crown mass.
        """
        raise NotImplementedError

    def weight(self, r: ndarray, z: ndarray, tree: "Tree") -> ndarray | float:
        """Return the relative weight at radial distance ``r`` and height ``z``.

        ``r`` and ``z`` broadcast against each other. Used where the weight is
        integrated over subcells rather than read at voxel centers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a point weight."
        )

    def crown_mean(self, tree: "Tree") -> float:
        """Return the mean of :meth:`weight` over the tree's continuous crown.

        Dividing the weight by this mean gives a weight that averages 1 over
        the crown, so a crown bulk density times that weight has the bulk
        density as its crown mean.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a crown mean."
        )


class UniformDensity(DensityField):
    """Constant bulk density through the crown: ``mass / occupied volume``.

    Equivalent to the original ``VoxelizedTree.distribute_biomass``: the whole
    crown mass is spread at a single bulk density over the occupied volume. The
    weight is a spatially constant ``1.0``, so it reads no crown geometry.
    """

    def crown_weight(self, vt):
        return 1.0

    def weight(self, r, z, tree):
        return 1.0

    def crown_mean(self, tree):
        return 1.0


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
        the stem (``vt.radial_distance``) and ``z`` the height
        (``vt.voxel_height``); both are broadcastable to the voxel grid. Any
        new gradient is just a new weight function.
    """

    def __init__(self, weight_fn: WeightFunction):
        self.weight_fn = weight_fn

    def crown_weight(self, vt):
        return self.weight(vt.radial_distance, vt.voxel_height, vt.tree)

    def weight(self, r, z, tree):
        return np.asarray(self.weight_fn(r, z, tree), dtype=float)

    def crown_mean(self, tree):
        return _crown_mean(self.weight, tree)


# Gauss-Legendre nodes per crown lobe (in height) and along the radius. A weight
# that is polynomial in r and z over a paraboloid or ellipsoid lobe -- such as
# LANL's gradient -- integrates exactly; other profiles converge quickly.
_CROWN_MEAN_HEIGHT_NODES = 64
_CROWN_MEAN_RADIUS_NODES = 32


def _crown_mean(weight: WeightFunction, tree: "Tree") -> float:
    """Volume-weighted mean of ``weight`` over the tree's continuous crown."""
    hb = float(tree.crown_base_height)
    ht = float(tree.height)
    hd = float(tree.crown_profile_model.get_max_radius_height())
    xz, wz = np.polynomial.legendre.leggauss(_CROWN_MEAN_HEIGHT_NODES)
    xr, wr = np.polynomial.legendre.leggauss(_CROWN_MEAN_RADIUS_NODES)
    u, wu = (xr + 1.0) / 2.0, wr / 2.0  # nodes on [0, 1]

    weighted = volume = 0.0
    # Integrate each side of the widest height separately: the radius has a
    # kink there.
    for lo, hi in ((hb, hd), (hd, ht)):
        if hi <= lo:
            continue
        z = lo + (xz + 1.0) / 2.0 * (hi - lo)
        dz = wz / 2.0 * (hi - lo)
        radius = np.asarray(tree.get_crown_radius_at_height(z), dtype=float)
        r = radius[:, None] * u[None, :]
        w = np.broadcast_to(weight(r, z[:, None], tree), r.shape)
        # Integral over the disc at height z: sum of w * 2*pi*r * dr, dr = R du.
        disc = (w * 2.0 * np.pi * r * radius[:, None] * wu[None, :]).sum(axis=1)
        weighted += float((dz * disc).sum())
        volume += float((dz * np.pi * radius**2).sum())
    if volume <= 0.0:
        raise ValueError("The crown has no volume.")
    return weighted / volume


def _linear_height_quadratic_radial(r: ndarray, z: ndarray, tree: "Tree") -> ndarray:
    """Weight that is linear in height and quadratic in radius (see class below)."""
    hb = tree.crown_base_height
    ht = tree.height
    hd = tree.crown_profile_model.get_max_radius_height()
    d = 2.0 * tree.max_crown_radius
    weight = ((z - hb) + 4.0 * (ht - hd) * r**2 / d**2) / (ht - hb)
    # Clamp to non-negative. When ``z_origin`` anchors the grid, the first cell
    # center can land just below the crown base (``z < hb``); the linear term is
    # then negative and, near the stem where the radial term is small, drives the
    # weight below zero -- which would give that base voxel a negative bulk
    # density. The weight is a density and must be non-negative by construction.
    return np.maximum(weight, 0.0)


class LinearHeightQuadraticRadialDensity(GradientDensity):
    """Crown density that rises linearly with height and quadratically with radius.

    .. math::

        w(r, z) = \\frac{(z - H_b) + 4 (H_t - H_d)\\, r^2 / D^2}{H_t - H_b}

    where :math:`H_b` is the crown base, :math:`H_t` the tree top, :math:`H_d`
    the height of maximum crown diameter (taken from the crown profile's peak,
    ``crown_profile_model.get_max_radius_height()``), and :math:`D = 2\\cdot`
    ``max_crown_radius``. The weight is zero at the crown-base center and peaks
    at the top outer rim.

    This is the fuel gradient of **LANL Trees** (``fuels_create.F90``). Over a
    dual-paraboloid crown it averages 2/3 for any :math:`H_d`, which is why
    LANL scales it by 3/2. :meth:`crown_weight` reads it once at each voxel
    center, while LANL averages it over the parts of each voxel inside the
    crown. To reproduce LANL, integrate it over subcells with
    ``subgrid.discretize_crown_profile(..., density_field=...)`` on a
    ``"paraboloid"`` crown.
    """

    def __init__(self):
        super().__init__(_linear_height_quadratic_radial)
