"""Canopy fuel metrics (CBD, CBH, CH, CC, CFL) from a tree inventory.

The computation follows the FuelCalc profile method applied per output
cell instead of per plot: each tree's available canopy fuel is
distributed vertically over its crown into fixed-depth layers,
attributed horizontally to the cells its crown covers, and accumulated
into a per-cell vertical profile that is reduced to the requested bands.

Each stage is a public function so callers can run, test, or compose
them individually; :func:`compute_canopy_metrics` chains them to fill a
caller-provided georeferenced Dataset. The caller owns the output
lattice — nothing here creates grids or chooses resolutions.

All per-tree math is vectorized; trees are never iterated in Python.

Expected tree columns follow the v2 FastFuels inventory convention:
``x``, ``y``, ``height``, ``crown_ratio``, and, depending on options,
``dbh``, ``fia_species_code``, and caller-named biomass / crown radius
columns. Units: meters, centimeters (dbh), kilograms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers the .rio accessor
import shapely
import xarray as xr
from affine import Affine
from rasterio.features import rasterize

from fastfuels_core.allometry import brown, nsvb
from fastfuels_core.canopy_fuel.ref_data import fuelcalc_species, fuelcalc_vdist
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.units import conversion_factor

# Default profile layer depth: FuelCalc's 1-ft layers, in meters.
FT_TO_M = 0.3048

# Cap on the transient (n_trees, n_boundaries) arrays built while
# accumulating the vertical profile; larger stands are processed in tree
# batches (scatter-adds commute, so batching does not change the result).
_PROFILE_BATCH_BYTES = 512 * 2**20

# Cap on each supersampled canopy-cover strip mask (uint8 bytes).
_COVER_STRIP_BYTES = 64 * 2**20


def _unit_disk_corner_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Area of the unit disk within the corner region {X <= x, Y <= y}.

    Closed form: integrate the clipped chord length from -1 to x. The
    integrand is ``clip(y, -sqrt(1-t^2), sqrt(1-t^2)) + sqrt(1-t^2)``,
    which is the full chord ``2*sqrt(1-t^2)`` where the chord lies
    entirely below y (for y >= 0), zero where it lies entirely above
    (y < 0), and ``y + sqrt(1-t^2)`` in between; the pieces meet at
    ``t = +/-sqrt(1-y^2)``.
    """
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    t_star = np.sqrt(np.clip(1.0 - y * y, 0.0, None))

    def antiderivative(t):
        # Integral of sqrt(1-t^2).
        return 0.5 * (
            t * np.sqrt(np.clip(1.0 - t * t, 0.0, None))
            + np.arcsin(np.clip(t, -1.0, 1.0))
        )

    outer = np.where(y >= 0.0, 2.0, 0.0)
    left_hi = np.minimum(x, -t_star)
    left = outer * (antiderivative(left_hi) - antiderivative(-1.0))
    mid_hi = np.clip(x, -t_star, t_star)
    middle = (y * mid_hi + antiderivative(mid_hi)) - (
        y * -t_star + antiderivative(-t_star)
    )
    right_hi = np.maximum(x, t_star)
    right = outer * (antiderivative(right_hi) - antiderivative(t_star))
    return left + middle + right


def disk_rect_overlap_area(
    cx: np.ndarray,
    cy: np.ndarray,
    radius: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Exact intersection area of disks and axis-aligned rectangles.

    All arguments broadcast; rectangles are ``[x0, x1] x [y0, y1]`` with
    ``x0 < x1`` and ``y0 < y1``. Evaluated by inclusion-exclusion over
    the four corner integrals of the unit disk.
    """
    r = np.asarray(radius, dtype=np.float64)
    u0, u1 = (x0 - cx) / r, (x1 - cx) / r
    v0, v1 = (y0 - cy) / r, (y1 - cy) / r
    area_unit = (
        _unit_disk_corner_area(u1, v1)
        - _unit_disk_corner_area(u0, v1)
        - _unit_disk_corner_area(u1, v0)
        + _unit_disk_corner_area(u0, v0)
    )
    return np.clip(area_unit, 0.0, np.pi) * r * r


def max_crown_radius(
    trees: pd.DataFrame, *, crown_radius_column: str | None = None
) -> np.ndarray:
    """Per-tree maximum crown radius (m).

    Reads ``crown_radius_column`` when given (e.g. LiDAR-measured radii);
    otherwise computes the Purves et al. (2007) allometric radius from
    ``fia_species_code``, ``dbh``, ``height``, and ``crown_ratio``.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(trees),)``, meters.
    """
    if crown_radius_column is not None:
        return trees[crown_radius_column].to_numpy(dtype=np.float64)
    model = PurvesCrownProfile(
        trees["fia_species_code"],
        trees["dbh"],
        trees["height"],
        trees["crown_ratio"],
    )
    return np.atleast_1d(np.asarray(model.get_max_radius(), dtype=np.float64))


def available_canopy_fuel(
    trees: pd.DataFrame,
    *,
    fuel_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
) -> np.ndarray:
    """Per-tree available canopy fuel (kg).

    With ``fuel_column`` set, that column is returned as-is (precomputed
    fuel, e.g. from LiDAR regression); the fractions are ignored.
    Otherwise fuel is ``foliage_fraction * NSVB foliage +
    branchwood_fraction * fine branchwood``, where fine branchwood is
    the NSVB total branch weight times the Brown (1978) / Snell & Little
    (1983) fine share ``(P2 - P1) / (1 - P1)`` (see
    :mod:`fastfuels_core.allometry.brown`). Requires ``dbh``, ``height``,
    and ``fia_species_code`` columns.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(trees),)``, kg per tree.

    Raises
    ------
    ValueError
        For species FuelCalc has no equations for — codes outside the
        FuelCalc species table, or whose equations are not implemented
        (pinyon/juniper, the eastern newer species). Mirrors FuelCalc's
        treelist error; ``fuel_column`` bypasses allometry entirely.
    """
    if fuel_column is not None:
        return trees[fuel_column].to_numpy(dtype=np.float64)

    spcd = trees["fia_species_code"].to_numpy()
    species = fuelcalc_species()
    unknown = np.setdiff1d(spcd, species.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FuelCalc "
            f"species table. FuelCalc errors on species without assigned "
            f"equations; supply per-tree fuel via fuel_column to bypass "
            f"allometry for these species."
        )

    # Brown/SL-83 proportions work in inches; the nsvb wrapper is metric.
    dia_in = trees["dbh"].to_numpy() * conversion_factor("cm", "inch")
    fol_id = species["FOL"].loc[spcd].to_numpy()
    twig_id = species["TWIG"].loc[spcd].to_numpy()
    fine_share = brown.fine_branchwood_share(fol_id, twig_id, dia_in)

    foliage_kg = nsvb.foliage_biomass(spcd, trees["dbh"], trees["height"])
    branch_kg = nsvb.branch_biomass(spcd, trees["dbh"], trees["height"])
    return foliage_fraction * foliage_kg + branchwood_fraction * fine_share * branch_kg


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
        :func:`available_canopy_fuel`.
    transform : tuple
        Rasterio-style affine (a, b, c, d, e, f) of the north-up output
        lattice.
    shape : tuple
        Output lattice shape ``(ny, nx)``.
    n_layers : int, optional
        Number of profile layers; defaults to covering the tallest tree.
        A smaller value truncates fuel above the top layer.
    crown_radius_column : str, optional
        Per-tree max crown radius (m) column; defaults to the Purves
        allometric radius. Only read by ``crown_projected``.

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
    clipped to the crown interval (FuelCalc's CLA formula), so each
    tree's weights sum to 1 and total mass is conserved. Under
    ``crown_projected``, the slice of a crown disk overhanging the
    lattice boundary has no cell and its share of the tree's fuel is
    dropped — mass is conserved for every crown fully inside.
    """
    if horizontal_distribution not in ("crown_projected", "stem"):
        raise ValueError(
            f"Unknown horizontal_distribution {horizontal_distribution!r}; "
            f"expected 'crown_projected' or 'stem'."
        )
    if vertical_distribution not in ("reinhardt_2006", "uniform"):
        raise ValueError(
            f"Unknown vertical_distribution {vertical_distribution!r}; "
            f"expected 'reinhardt_2006' or 'uniform'."
        )
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

    col = np.floor((trees["x"].to_numpy(dtype=np.float64) - c) / a).astype(np.int64)
    row = np.floor((trees["y"].to_numpy(dtype=np.float64) - f) / e).astype(np.int64)
    out = (col < 0) | (col >= nx) | (row < 0) | (row >= ny)
    if out.any():
        raise ValueError(
            f"{int(out.sum())} tree stem(s) fall outside the lattice. "
            f"Inventories are domain-bounded and the lattice covers the "
            f"domain, so this indicates a mismatched lattice or CRS."
        )

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

    boundaries = np.arange(n_layers + 1, dtype=np.float64) * layer_depth
    layer_offsets = np.arange(n_layers, dtype=np.int64) * (ny * nx)
    cell_volume = abs(a * e) * layer_depth
    n_trees = len(trees)

    # Horizontal attribution: per-tree (cell index, weight) contribution
    # lists. Stem puts everything in the stem cell; crown_projected
    # splits by exact crown-disk / cell intersection area over the
    # neighborhood of cells the largest crown can reach.
    if horizontal_distribution == "stem":
        contributions = [(row * nx + col, np.ones(n_trees))]
    else:
        x = trees["x"].to_numpy(dtype=np.float64)
        y = trees["y"].to_numpy(dtype=np.float64)
        radius = np.maximum(
            max_crown_radius(trees, crown_radius_column=crown_radius_column), 1e-6
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

    batch = max(1, int(_PROFILE_BATCH_BYTES // ((n_layers + 1) * 8 * 4)))
    for start in range(0, n_trees, batch):
        stop = min(start + batch, n_trees)
        sl = slice(start, stop)
        ph = (boundaries[None, :] - crown_base[sl, None]) / safe_length[sl, None]
        if vertical_distribution == "reinhardt_2006":
            pw = cumulative_fuel_fraction(spcd[sl], ph)
        else:
            pw = np.clip(ph, 0.0, 1.0)
        vertical_weights = np.diff(pw, axis=1)
        for cell, horizontal_weight in contributions:
            scattered = vertical_weights * (fuel[sl] * horizontal_weight[sl])[:, None]
            flat = layer_offsets[None, :] + cell[sl, None]
            profile_flat += np.bincount(
                flat.ravel(), weights=scattered.ravel(), minlength=profile_flat.size
            )

    return (profile_flat / cell_volume).reshape(n_layers, ny, nx)


def cbd_running_mean(
    profile: np.ndarray,
    *,
    layer_depth: float = FT_TO_M,
    window: float | None = 3.0,
) -> np.ndarray:
    """Canopy bulk density: per-cell maximum running mean of the profile.

    ``window`` is the running-mean depth in meters (Reinhardt et al.
    2006 use 3.0 m; FuelCalc's guide states 5 ft in one place and no
    smoothing in another). ``window=None`` skips smoothing and returns
    the maximum single layer. A profile shallower than the window is
    zero-padded above — density above the canopy is zero, and the
    effective CBD of a shallow canopy is diluted over the full window,
    matching Reinhardt et al.'s "any 3-m deep layer".

    Returns
    -------
    numpy.ndarray
        CBD (kg/m**3), shape ``(ny, nx)``.
    """
    if window is None:
        return profile.max(axis=0)
    w = max(1, int(round(window / layer_depth)))
    n_layers = profile.shape[0]
    if n_layers < w:
        pad = np.zeros((w - n_layers, *profile.shape[1:]), dtype=profile.dtype)
        profile = np.concatenate([profile, pad], axis=0)
    cumsum = np.concatenate(
        [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
    )
    means = (cumsum[w:] - cumsum[:-w]) / w
    return means.max(axis=0)


def profile_threshold_heights(
    profile: np.ndarray,
    *,
    layer_depth: float = FT_TO_M,
    threshold: float = 0.012,
    relative_fraction: float | None = 0.1,
    smoothing_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Canopy base height and canopy height from a bulk-density threshold.

    Per cell, the effective threshold is ``min(relative_fraction *
    profile_max, threshold)`` (FuelCalc's rule; ``relative_fraction=None``
    uses the flat threshold alone). CBH is the bottom of the lowest
    layer at or above threshold; canopy height is the top of the highest.
    ``smoothing_window`` (m) optionally smooths the profile first with a
    centered running mean truncated at the profile ends (FFE-FVS uses
    0.9144 m). Cells with no layer at or above threshold — including
    empty cells — are NaN.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cbh, chm)`` in meters, each shape ``(ny, nx)``.
    """
    profile = np.asarray(profile, dtype=np.float64)
    n_layers = profile.shape[0]

    if smoothing_window is not None:
        w = max(1, int(round(smoothing_window / layer_depth)))
        cumsum = np.concatenate(
            [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
        )
        k = np.arange(n_layers)
        lo = np.clip(k - (w - 1) // 2, 0, n_layers)
        hi = np.clip(k + w // 2 + 1, 0, n_layers)
        profile = (cumsum[hi] - cumsum[lo]) / (hi - lo).reshape(
            -1, *([1] * (profile.ndim - 1))
        )

    profile_max = profile.max(axis=0)
    if relative_fraction is not None:
        effective = np.minimum(relative_fraction * profile_max, threshold)
    else:
        effective = np.full_like(profile_max, threshold)
    # An all-zero cell has effective threshold 0 under the relative rule;
    # require positive density so empty cells read as no-canopy.
    qualifies = (profile >= effective) & (profile > 0.0)

    any_qualifies = qualifies.any(axis=0)
    lowest = qualifies.argmax(axis=0)
    highest = n_layers - 1 - qualifies[::-1].argmax(axis=0)
    cbh = np.where(any_qualifies, lowest * layer_depth, np.nan)
    chm = np.where(any_qualifies, (highest + 1) * layer_depth, np.nan)
    return cbh, chm


def canopy_fuel_load(
    profile: np.ndarray, *, layer_depth: float = FT_TO_M
) -> np.ndarray:
    """Canopy fuel load: vertical integral of the bulk-density profile.

    Returns
    -------
    numpy.ndarray
        CFL (kg/m**2), shape ``(ny, nx)``.
    """
    return profile.sum(axis=0) * layer_depth


def canopy_cover(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    crown_radius_column: str | None = None,
    supersample: int | None = None,
) -> np.ndarray:
    """Per-cell projected canopy cover (%) from the crown-disk union.

    Overlapping crowns count once (a true union, not a sum of crown
    areas): crown disks are rasterized as a binary union on a fine grid
    of ``supersample x supersample`` pixels per output cell, then
    block-reduced to covered fractions. ``supersample`` defaults to a
    fine pixel of at most 0.5 m so crown edges are resolved at any
    output resolution. Crown radii come from ``crown_radius_column`` or
    the Purves allometric radius.

    The union internals are an implementation detail (fastfuels-core#95
    keeps them swappable against a kd-tree analytic candidate pending
    profiling); only the covered fractions are contract.

    Returns
    -------
    numpy.ndarray
        Cover (%), shape ``(ny, nx)``.
    """
    a, b_rot, c, d_rot, e, f = transform
    if b_rot != 0.0 or d_rot != 0.0:
        raise ValueError("Rotated transforms are not supported.")
    ny, nx = shape
    if len(trees) == 0:
        return np.zeros(shape)
    if supersample is None:
        supersample = max(2, int(np.ceil(abs(a) / 0.5)))

    x = trees["x"].to_numpy(dtype=np.float64)
    y = trees["y"].to_numpy(dtype=np.float64)
    radius = max_crown_radius(trees, crown_radius_column=crown_radius_column)
    crowns = shapely.buffer(shapely.points(x, y), radius, quad_segs=16)
    max_radius = float(radius.max())

    # Fine masks are built in horizontal strips so peak memory stays
    # bounded regardless of lattice size and supersample factor.
    rows_per_strip = max(1, int(_COVER_STRIP_BYTES // (nx * supersample**2)))
    cover = np.empty(shape, dtype=np.float64)
    for row_start in range(0, ny, rows_per_strip):
        row_stop = min(row_start + rows_per_strip, ny)
        y_north = f + row_start * e  # e < 0
        y_south = f + row_stop * e
        in_strip = (y >= y_south - max_radius) & (y <= y_north + max_radius)
        strip_shape = ((row_stop - row_start) * supersample, nx * supersample)
        if in_strip.any():
            mask = rasterize(
                crowns[in_strip],
                out_shape=strip_shape,
                transform=Affine(
                    a / supersample, 0.0, c, 0.0, e / supersample, y_north
                ),
                fill=0,
                default_value=1,
                all_touched=False,
                dtype="uint8",
            )
            block = mask.reshape(
                row_stop - row_start, supersample, nx, supersample
            ).mean(axis=(1, 3))
        else:
            block = 0.0
        cover[row_start:row_stop] = block
    return cover * 100.0


def compute_canopy_metrics(
    trees: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    fuel_column: str | None = None,
    crown_radius_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
    min_tree_height: float = 0.0,
    layer_depth: float = FT_TO_M,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "crown_projected",
    cbd_window: float | None = 3.0,
    cbh_threshold: float = 0.012,
    cbh_relative_fraction: float | None = 0.1,
    threshold_smoothing_window: float | None = None,
) -> xr.Dataset:
    """Fill a georeferenced Dataset with canopy fuel metrics.

    The caller provides ``dataset`` with one variable per requested band
    — any of ``cbd``, ``cbh``, ``chm``, ``cc``, ``cfl`` — on a north-up
    lattice with CRS and transform set via rioxarray. Only the variables
    present are computed. Tree coordinates must be in the dataset's CRS;
    only live trees should be passed. Trees shorter than
    ``min_tree_height`` (m) are excluded.

    Chains the public stages: :func:`available_canopy_fuel` →
    :func:`vertical_profile` → :func:`cbd_running_mean` /
    :func:`profile_threshold_heights` / :func:`canopy_fuel_load`, plus
    :func:`canopy_cover`. See each stage for parameter semantics.

    Cells with no canopy come back as 0 for ``cbd``, ``cfl``, and ``cc``
    (zero density is physical) and NaN for ``cbh`` and ``chm`` (no
    canopy has no base or top); masking is the caller's choice.

    Returns
    -------
    xarray.Dataset
        ``dataset`` with its band variables filled in place.
    """
    known_bands = {"cbd", "cbh", "chm", "cc", "cfl"}
    bands = set(dataset.data_vars)
    unknown = bands - known_bands
    if unknown:
        raise ValueError(
            f"Unknown dataset variable(s) {sorted(unknown)}; expected a "
            f"subset of {sorted(known_bands)}."
        )

    t = dataset.rio.transform()
    transform = (t.a, t.b, t.c, t.d, t.e, t.f)
    shape = (dataset.sizes["y"], dataset.sizes["x"])

    trees = trees[trees["height"].to_numpy() >= min_tree_height]

    if bands & {"cbd", "cbh", "chm", "cfl"}:
        fuel = available_canopy_fuel(
            trees,
            fuel_column=fuel_column,
            foliage_fraction=foliage_fraction,
            branchwood_fraction=branchwood_fraction,
        )
        profile = vertical_profile(
            trees,
            fuel,
            transform,
            shape,
            layer_depth=layer_depth,
            vertical_distribution=vertical_distribution,
            horizontal_distribution=horizontal_distribution,
            crown_radius_column=crown_radius_column,
        )
        if "cbd" in bands:
            dataset["cbd"].data[...] = cbd_running_mean(
                profile, layer_depth=layer_depth, window=cbd_window
            )
        if bands & {"cbh", "chm"}:
            cbh, chm = profile_threshold_heights(
                profile,
                layer_depth=layer_depth,
                threshold=cbh_threshold,
                relative_fraction=cbh_relative_fraction,
                smoothing_window=threshold_smoothing_window,
            )
            if "cbh" in bands:
                dataset["cbh"].data[...] = cbh
            if "chm" in bands:
                dataset["chm"].data[...] = chm
        if "cfl" in bands:
            dataset["cfl"].data[...] = canopy_fuel_load(
                profile, layer_depth=layer_depth
            )

    if "cc" in bands:
        dataset["cc"].data[...] = canopy_cover(
            trees, transform, shape, crown_radius_column=crown_radius_column
        )

    return dataset
