"""Per-cell vertical bulk-density profile (kg/m**3 by layer).

The profile is the intermediate every stand-level fuel metric reduces:
each tree's available canopy fuel is spread vertically over its crown
into fixed-depth layers and horizontally over the cells its crown
covers, then accumulated per cell. CBD, CBH/CH and CFL are three ways
of reading the same array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.canopy_fuel.geometry import disk_rect_overlap_area
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species, fuelcalc_vdist

# Default profile layer depth: FuelCalc's 1-ft layers, in meters.
FT_TO_M = 0.3048

VALID_VERTICAL_DISTRIBUTIONS = ("reinhardt_2006", "uniform")
VALID_HORIZONTAL_DISTRIBUTIONS = ("crown_projected", "stem")

# Cap on the transient (n_trees, n_boundaries) arrays built while
# accumulating the vertical profile; larger stands are processed in tree
# batches (scatter-adds commute, so batching does not change the result).
_PROFILE_BATCH_BYTES = 512 * 2**20


def cumulative_fuel_fraction(species_code: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Cumulative fuel fraction at fractional crown height, by species.

    Evaluates the Reinhardt et al. (2006) cubic ``pw(ph) = B1*ph +
    B2*ph**2 + B3*ph**3`` for each (tree, height) pair, resolving each
    FIA species code to its FuelCalc vertical-distribution equation.
    Species absent from the FuelCalc table use the uniform distribution
    (``pw = ph``), as do hardwoods.

    Parameters
    ----------
    species_code : numpy.ndarray
        FIA species codes, shape ``(n_trees,)``.
    ph : numpy.ndarray
        Fractional heights within the crown, clipped to [0, 1]. Shape
        ``(n_trees,)`` or ``(n_trees, n_levels)``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``ph``.
    """
    spcd = np.asarray(species_code)
    ph = np.clip(np.asarray(ph, dtype=np.float64), 0.0, 1.0)

    species = fuelcalc_species()
    vdist_codes = species["VDIST_CODE"].reindex(spcd).fillna("UN").to_numpy()
    coefficients = fuelcalc_vdist().loc[vdist_codes, ["B1", "B2", "B3"]].to_numpy()
    b1, b2, b3 = coefficients[:, 0], coefficients[:, 1], coefficients[:, 2]
    if ph.ndim == 2:
        b1, b2, b3 = b1[:, None], b2[:, None], b3[:, None]
    return b1 * ph + b2 * ph**2 + b3 * ph**3


def _validate_distributions(
    vertical_distribution: str, horizontal_distribution: str
) -> None:
    if horizontal_distribution not in VALID_HORIZONTAL_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown horizontal_distribution {horizontal_distribution!r}; "
            f"expected 'crown_projected' or 'stem'."
        )
    if vertical_distribution not in VALID_VERTICAL_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown vertical_distribution {vertical_distribution!r}; "
            f"expected 'reinhardt_2006' or 'uniform'."
        )


def _stem_cells(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of each tree's stem, checked against the lattice."""
    a, _, c, _, e, f = transform
    ny, nx = shape
    col = np.floor((trees["x"].to_numpy(dtype=np.float64) - c) / a).astype(np.int64)
    row = np.floor((trees["y"].to_numpy(dtype=np.float64) - f) / e).astype(np.int64)
    out = (col < 0) | (col >= nx) | (row < 0) | (row >= ny)
    if out.any():
        raise ValueError(
            f"{int(out.sum())} tree stem(s) fall outside the lattice. "
            f"Inventories are domain-bounded and the lattice covers the "
            f"domain, so this indicates a mismatched lattice or CRS."
        )
    return row, col


def _crown_projected_contributions(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    crown_radius_column: str | None,
    crown_radius_equations: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(cell index, weight) pairs splitting each crown over the cells it covers.

    One pair per offset in the neighbourhood the largest crown reaches;
    weights are the exact crown-disk / cell intersection area as a
    fraction of crown area, so a crown fully inside the lattice has
    weights summing to 1.
    """
    a, _, c, _, e, f = transform
    ny, nx = shape
    x = trees["x"].to_numpy(dtype=np.float64)
    y = trees["y"].to_numpy(dtype=np.float64)
    radius = np.maximum(
        max_crown_radius(
            trees,
            crown_radius_column=crown_radius_column,
            equations=crown_radius_equations,
        ),
        1e-6,
    )
    col_lo = np.floor((x - radius - c) / a).astype(np.int64)
    col_hi = np.floor((x + radius - c) / a).astype(np.int64)
    row_lo = np.floor((y + radius - f) / e).astype(np.int64)  # e < 0
    row_hi = np.floor((y - radius - f) / e).astype(np.int64)
    inv_crown_area = 1.0 / (np.pi * radius * radius)

    contributions = []
    for row_offset in range(int((row_hi - row_lo).max()) + 1):
        rows = row_lo + row_offset
        y_hi = f + rows * e  # north edge; e < 0 makes y_hi > y_lo
        y_lo = y_hi + e
        for col_offset in range(int((col_hi - col_lo).max()) + 1):
            cols = col_lo + col_offset
            x_lo = c + cols * a
            area = disk_rect_overlap_area(x, y, radius, x_lo, x_lo + a, y_lo, y_hi)
            weight = area * inv_crown_area
            in_bounds = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
            weight = np.where(in_bounds, weight, 0.0)
            if not weight.any():
                continue
            cell = np.where(in_bounds, rows * nx + cols, 0)
            contributions.append((cell, weight))
    return contributions


def _layer_weights(
    species_code: np.ndarray | None,
    crown_base: np.ndarray,
    crown_length: np.ndarray,
    boundaries: np.ndarray,
    vertical_distribution: str,
) -> np.ndarray:
    """Fraction of a tree's fuel in each layer, shape (n_trees, n_layers).

    Weights are differences of the cumulative fuel fraction at layer
    boundaries expressed as fractional crown height, which is FuelCalc's
    CLA formula.
    """
    ph = (boundaries[None, :] - crown_base[:, None]) / crown_length[:, None]
    if vertical_distribution == "reinhardt_2006":
        pw = cumulative_fuel_fraction(species_code, ph)
        # Close the crown at the top. FuelCalc's VD_Calc handles the
        # layer containing the crown top with ``pcWT = 1 - pw(layer
        # bottom)`` rather than a difference of cumulatives, so a cubic
        # whose coefficients do not sum to exactly 1 still distributes
        # exactly the tree's fuel. PS is such a cubic: Reinhardt et al.
        # (2006) Table 4 rounds it to 1.0001, which would otherwise hand
        # every Arizona pine 100.01% of its fuel. The cumulative
        # fraction at the crown top is 1 by definition, so this is inert
        # for every other species.
        pw = np.where(ph >= 1.0, 1.0, pw)
    else:
        pw = np.clip(ph, 0.0, 1.0)
    return np.diff(pw, axis=1)


def vertical_profile(
    trees: pd.DataFrame,
    fuel: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    n_layers: int | None = None,
    layer_depth: float = FT_TO_M,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "crown_projected",
    crown_radius_column: str | None = None,
    crown_radius_equations: str = "purves",
) -> np.ndarray:
    """Accumulate per-tree fuel into a per-cell vertical profile grid.

    Each tree's fuel is split across profile layers spanning its crown
    (``vertical_distribution``: ``"reinhardt_2006"`` species cubics or
    ``"uniform"``) and across the cells its crown covers
    (``horizontal_distribution``: ``"crown_projected"`` splits by exact
    crown-disk / cell intersection area; ``"stem"`` assigns everything
    to the stem cell). Contributions scatter-add, so any batch of trees
    in any order accumulates to the same result.

    Parameters
    ----------
    trees : pandas.DataFrame
        Tree records in the grid's CRS.
    fuel : numpy.ndarray
        Per-tree available canopy fuel (kg), from
        :mod:`available_fuel`.
    transform : tuple
        Rasterio-style affine (a, b, c, d, e, f) of the north-up output
        lattice.
    shape : tuple
        Output lattice shape ``(ny, nx)``.
    n_layers : int, optional
        Number of profile layers; defaults to covering the tallest tree.
        A smaller value truncates fuel above the top layer.
    crown_radius_column : str, optional
        Per-tree max crown radius (m) column, overriding
        ``crown_radius_equations``. Only read by ``crown_projected``.
    crown_radius_equations : str, optional
        Allometry the crown radius comes from when no column is given;
        see :func:`~fastfuels_core.canopy_fuel.crown_radius.max_crown_radius`.
        Only read by ``crown_projected``.

    Returns
    -------
    numpy.ndarray
        Bulk density profile (kg/m**3), shape ``(n_layers, ny, nx)``.

    Raises
    ------
    ValueError
        If the transform is rotated, or any stem falls outside the
        lattice — inventories are domain-bounded and the lattice covers
        the domain, so an out-of-bounds stem means a caller error
        (mismatched lattice or wrong CRS) and must not scatter silently.

    Notes
    -----
    Trees with zero crown length contribute their fuel as a point mass
    in the layer containing the crown base. Layer weights are
    differences of :func:`cumulative_fuel_fraction` at layer boundaries
    clipped to the crown interval (FuelCalc's CLA formula), with the
    cumulative fraction pinned to 1 at the crown top as VD_Calc does,
    so each tree's weights sum to 1 and total mass is conserved. Under
    ``crown_projected``, the slice of a crown disk overhanging the
    lattice boundary has no cell and its share of the tree's fuel is
    dropped — mass is conserved for every crown fully inside.
    """
    _validate_distributions(vertical_distribution, horizontal_distribution)
    a, b_rot, c, d_rot, e, f = transform
    if b_rot != 0.0 or d_rot != 0.0:
        raise ValueError("Rotated transforms are not supported.")
    ny, nx = shape

    height = trees["height"].to_numpy(dtype=np.float64)
    if n_layers is None:
        n_layers = max(1, int(np.ceil(height.max() / layer_depth))) if len(trees) else 1
    profile_flat = np.zeros(n_layers * ny * nx, dtype=np.float64)
    if len(trees) == 0:
        return profile_flat.reshape(n_layers, ny, nx)

    row, col = _stem_cells(trees, transform, shape)
    crown_length = height * trees["crown_ratio"].to_numpy(dtype=np.float64)
    crown_base = height - crown_length
    # Zero-length crowns become a point mass at the crown base: a tiny
    # denominator turns the clipped ph into a step function there.
    safe_length = np.maximum(crown_length, 1e-9)
    fuel = np.asarray(fuel, dtype=np.float64)
    spcd = (
        trees["fia_species_code"].to_numpy()
        if vertical_distribution == "reinhardt_2006"
        else None
    )

    if horizontal_distribution == "stem":
        contributions = [(row * nx + col, np.ones(len(trees)))]
    else:
        contributions = _crown_projected_contributions(
            trees, transform, shape, crown_radius_column, crown_radius_equations
        )

    boundaries = np.arange(n_layers + 1, dtype=np.float64) * layer_depth
    layer_offsets = np.arange(n_layers, dtype=np.int64) * (ny * nx)
    n_trees = len(trees)
    batch = max(1, int(_PROFILE_BATCH_BYTES // ((n_layers + 1) * 8 * 4)))
    for start in range(0, n_trees, batch):
        sl = slice(start, min(start + batch, n_trees))
        vertical_weights = _layer_weights(
            spcd[sl] if spcd is not None else None,
            crown_base[sl],
            safe_length[sl],
            boundaries,
            vertical_distribution,
        )
        for cell, horizontal_weight in contributions:
            scattered = vertical_weights * (fuel[sl] * horizontal_weight[sl])[:, None]
            flat = layer_offsets[None, :] + cell[sl, None]
            profile_flat += np.bincount(
                flat.ravel(), weights=scattered.ravel(), minlength=profile_flat.size
            )

    cell_volume = abs(a * e) * layer_depth
    return (profile_flat / cell_volume).reshape(n_layers, ny, nx)
