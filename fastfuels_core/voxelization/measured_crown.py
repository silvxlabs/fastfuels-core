"""Voxelization of a measured crown: a CHM footprint, its heights, and a profile.

A measured crown is a set of CHM cells (the footprint) whose heights are the
crown top. Each cell contributes a vertical column from a bottom surface up to
its top; the bottom is shaped by a crown profile, so the crown is flat-bottomed
for profiles widest at the crown base and rises toward the edge for profiles
widest higher up. The columns are integrated exactly onto the output voxel
lattice, giving a volume-fraction grid that feeds the same occupancy sampling
and mass distribution as parametric crowns.

Both lattices are north-up: row 0 is the north edge and rows increase
southward. Heights are metres above ground, with ``z = 0`` at the ground.
"""

# Core imports
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

# External imports
import numpy as np
from numpy import ndarray
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from fastfuels_core.crown_profile_models.abc import CrownProfileModel

# Samples used to tabulate a profile's lower branch.
_N_PROFILE_SAMPLES = 2049

# Overlaps shorter than this fraction of a cell are floating-point slivers from
# edges that should coincide; they are dropped so they do not mark a voxel as
# crown.
_SLIVER_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MeasuredCrown:
    """One measured crown on the output voxel lattice.

    Attributes
    ----------
    volume_fraction : ndarray
        ``(nz, ny, nx)`` fraction of each voxel's volume inside the crown, in
        ``[0, 1]``.
    offset : tuple[int, int, int]
        ``(z, row, col)`` of ``volume_fraction[0, 0, 0]`` on the output
        lattice. ``z`` counts voxels up from the ground; ``row`` and ``col``
        may be negative.
    bottom : ndarray
        ``(rows, cols)`` bottom height (m) of each cell's column, on the
        source lattice. NaN where the cell contributes no volume.
    """

    volume_fraction: ndarray
    offset: tuple[int, int, int]
    bottom: ndarray


def voxelize_measured_crown(
    footprint: ndarray,
    top: ndarray,
    source_origin: tuple[float, float],
    source_resolution: float,
    stem_xy: tuple[float, float],
    crown_base_height: float,
    profile: "CrownProfileModel",
    output_origin: tuple[float, float],
    output_resolution: tuple[float, float],
    min_thickness: float | None = None,
) -> MeasuredCrown | None:
    """Voxelize a crown measured as a footprint of CHM cells.

    Each footprint cell contributes the column ``[bottom, top]``, where
    ``bottom = max(min(profile_bottom, top - min_thickness), crown_base_height)``.
    ``profile_bottom`` is the lowest height in
    ``[crown_base_height, profile.get_max_radius_height()]`` at which the
    profile's relative radius reaches the cell's position ``rho = d / (d + e)``,
    with ``d`` the distance from the cell centre to the stem and ``e`` the
    distance from the cell centre to the footprint's edge. A cell whose top is
    at or below the crown base contributes nothing.

    The columns are integrated exactly onto the output lattice, so the grid's
    volume equals the sum of the column volumes for any pair of resolutions and
    lattice alignment.

    Parameters
    ----------
    footprint : ndarray
        ``(rows, cols)`` bool, the crown's cells on the source (CHM) lattice.
    top : ndarray
        ``(rows, cols)`` float, CHM height above ground (m). Must be finite
        inside the footprint.
    source_origin : tuple[float, float]
        ``(x_west, y_north)`` of ``footprint[0, 0]`` (m).
    source_resolution : float
        Source cell size (m). Cells are square.
    stem_xy : tuple[float, float]
        Stem position (m), same CRS as ``source_origin``. May lie outside the
        footprint.
    crown_base_height : float
        Crown base height above ground (m).
    profile : CrownProfileModel
        A single-tree crown profile providing ``get_radius_at_height``,
        ``get_max_radius`` and ``get_max_radius_height``.
    output_origin : tuple[float, float]
        ``(x_west, y_north)`` of the output lattice (m).
    output_resolution : tuple[float, float]
        ``(horizontal, vertical)`` output voxel size (m).
    min_thickness : float, optional
        Minimum column thickness (m). Defaults to one vertical voxel.

    Returns
    -------
    MeasuredCrown or None
        The crown's volume-fraction grid, or None when it has no volume.
    """
    footprint = np.asarray(footprint)
    top = np.asarray(top, dtype=float)
    if footprint.ndim != 2 or top.ndim != 2:
        raise ValueError("footprint and top must be 2D arrays.")
    if footprint.shape != top.shape:
        raise ValueError(
            f"footprint {footprint.shape} and top {top.shape} must have the same shape."
        )
    footprint = footprint.astype(bool)

    if np.ndim(source_resolution) != 0:
        res = np.asarray(source_resolution, dtype=float).reshape(-1)
        if res.size != 2 or res[0] != res[1]:
            raise ValueError("source cells must be square.")
        source_resolution = res[0]
    source_resolution = float(source_resolution)
    hr, vr = (float(v) for v in output_resolution)
    if min_thickness is None:
        min_thickness = vr
    min_thickness = float(min_thickness)
    for name, value in (
        ("source_resolution", source_resolution),
        ("output horizontal resolution", hr),
        ("output vertical resolution", vr),
        ("min_thickness", min_thickness),
    ):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive, got {value}.")

    crown_base_height = float(crown_base_height)
    if not (np.isfinite(crown_base_height) and crown_base_height >= 0):
        raise ValueError(
            f"crown_base_height must be non-negative, got {crown_base_height}."
        )
    if not np.all(np.isfinite(top[footprint])):
        raise ValueError("top must be finite inside the footprint.")

    bottom = np.full(top.shape, np.nan)
    active = footprint & (top > crown_base_height)
    if not active.any():
        return None

    # Position of each cell between the stem (0) and the footprint edge (-> 1).
    rows, cols = np.indices(top.shape)
    x_centre = source_origin[0] + (cols + 0.5) * source_resolution
    y_centre = source_origin[1] - (rows + 0.5) * source_resolution
    d = np.hypot(x_centre - stem_xy[0], y_centre - stem_xy[1])
    # Pad so cells on the array border see the background beyond it.
    edt = distance_transform_edt(np.pad(footprint, 1))[1:-1, 1:-1]
    e = (edt - 0.5) * source_resolution
    rho = d[active] / (d[active] + e[active])

    profile_bottom = _profile_bottom(profile, crown_base_height, rho)
    cell_top = top[active]
    cell_bottom = np.maximum(
        np.minimum(profile_bottom, cell_top - min_thickness), crown_base_height
    )
    bottom[active] = cell_bottom

    # Restrict the integration to the bounding box of contributing cells.
    active_rows = np.flatnonzero(active.any(axis=1))
    active_cols = np.flatnonzero(active.any(axis=0))
    r0, r1 = active_rows[0], active_rows[-1] + 1
    c0, c1 = active_cols[0], active_cols[-1] + 1
    box_bottom = bottom[r0:r1, c0:c1]
    box_top = np.where(active[r0:r1, c0:c1], top[r0:r1, c0:c1], np.nan)

    # Source cell edges, measured east (x) and south (y) from the output origin.
    x_edges = (
        source_origin[0] - output_origin[0] + np.arange(c0, c1 + 1) * source_resolution
    )
    y_edges = (
        output_origin[1] - source_origin[1] + np.arange(r0, r1 + 1) * source_resolution
    )
    col_start, overlap_x = _axis_overlap(x_edges[:-1], x_edges[1:], hr)
    row_start, overlap_y = _axis_overlap(y_edges[:-1], y_edges[1:], hr)
    z_start, overlap_z = _axis_overlap(
        np.nan_to_num(box_bottom, nan=0.0).ravel(),
        np.nan_to_num(box_top, nan=0.0).ravel(),
        vr,
    )
    overlap_z = overlap_z.reshape(-1, *box_bottom.shape)

    volume = np.einsum("ir,jc,zrc->zij", overlap_y, overlap_x, overlap_z, optimize=True)
    volume_fraction = np.clip(volume / (hr * hr * vr), 0.0, 1.0)
    if not volume_fraction.any():
        return None

    return MeasuredCrown(
        volume_fraction=volume_fraction,
        offset=(z_start, row_start, col_start),
        bottom=bottom,
    )


def _profile_bottom(
    profile: "CrownProfileModel", crown_base_height: float, rho: ndarray
) -> ndarray:
    """Lowest height at which the profile's relative radius reaches ``rho``.

    The lower branch, from the crown base to the height of maximum radius, is
    tabulated, made monotone with a running maximum, and inverted by linear
    interpolation between the bracketing samples.
    """
    z_max = max(float(profile.get_max_radius_height()), crown_base_height)
    r_max = float(profile.get_max_radius())
    if z_max <= crown_base_height or not (np.isfinite(r_max) and r_max > 0):
        return np.full(rho.shape, crown_base_height)

    z = np.linspace(crown_base_height, z_max, _N_PROFILE_SAMPLES)
    radius = np.asarray(profile.get_radius_at_height(z), dtype=float).reshape(-1)
    ratio = np.maximum.accumulate(radius / r_max)

    idx = np.searchsorted(ratio, rho, side="left")
    result = np.full(rho.shape, z_max)
    at_base = idx == 0
    result[at_base] = crown_base_height
    inside = (idx > 0) & (idx < ratio.size)
    hi = idx[inside]
    lo = hi - 1
    t = (rho[inside] - ratio[lo]) / (ratio[hi] - ratio[lo])
    result[inside] = z[lo] + t * (z[hi] - z[lo])
    return result


def _axis_overlap(lower: ndarray, upper: ndarray, step: float) -> tuple[int, ndarray]:
    """Overlap lengths of intervals with the cells of a lattice along one axis.

    Lattice cell ``k`` spans ``[k * step, (k + 1) * step]``. Returns the index
    of the first cell covered by any interval and a ``(n_cells, n_intervals)``
    matrix of overlap lengths. Empty intervals (``upper <= lower``) contribute
    nothing.
    """
    valid = upper > lower
    start = int(np.floor(lower[valid].min() / step))
    stop = int(np.ceil(upper[valid].max() / step))
    cell_lower = np.arange(start, stop)[:, None] * step
    overlap = np.minimum(upper[None, :], cell_lower + step) - np.maximum(
        lower[None, :], cell_lower
    )
    overlap[(overlap < _SLIVER_TOLERANCE * step) | ~valid[None, :]] = 0.0

    # Drop leading/trailing cells left empty by dropped slivers.
    used = np.flatnonzero(overlap.any(axis=1))
    if used.size == 0:
        return start, overlap[:0]
    return start + int(used[0]), overlap[used[0] : used[-1] + 1]
