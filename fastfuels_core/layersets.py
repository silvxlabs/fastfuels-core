"""
layersets.py
============
Rasterize a GeoDataFrame of fuel polygons into a single rioxarray Dataset —
one named variable per fuel type, five bands per variable — following the
spatial distribution logic in CAPSIS/StandFire's pipeline.

INPUT COLUMNS
-------------
Required for all rows
    fuel_type           str    — e.g. "litter", "shrub", "herb"
    fuel_loading        float  — kg/m²
    fuel_height         float  — m above surface
    percent_cover       float  — 0–100
    distribution        str    — see DISTRIBUTION MODES below

Required for random_clusters
    patch_size          float  — m

Optional
    live_fuel_moisture  float  — %
    dead_fuel_moisture  float  — %
    heat_of_combustion  float  — kJ/kg

DISTRIBUTION MODES
------------------
homogeneous
    Fuel placed in every in-polygon cell; per-cell loading scaled by
    percent_cover / 100.

uniform_random
    Cells inside the polygon are randomly selected until the requested
    cover fraction is reached; selected cells get weight 1.0.

random_clusters
    Circular patches of diameter patch_size are placed at uniformly random
    centers until percent_cover is reached. Patch diameters vary around
    patch_size with a 10% coefficient of variation.

OVERLAP RESOLUTION (same fuel_type, multiple polygons)
-------------------------------------------------------
    loading          — always summed
    all other bands  — np.mean (default), np.min, or np.max

OUTPUT
------
xarray.Dataset with:
  • one DataArray per fuel type, named by fuel_type string
  • dims:  (band, y, x)
  • coords: band = ["loading","height","live_fuel_moisture",
                    "dead_fuel_moisture","heat_of_combustion","patch_size"]
            y, x = cell-center coordinates in the input CRS
  • spatial_ref stored via rioxarray conventions
  • NoData = NaN for optional bands where no value was provided

"""

from __future__ import annotations

import math
from typing import Optional
from dataclasses import dataclass

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIONAL_BANDS = (
    "live_fuel_moisture",
    "dead_fuel_moisture",
    "heat_of_combustion",
    "patch_size",
)
ALL_BANDS = ("loading", "height") + OPTIONAL_BANDS
DISTRIBUTION_MODES = ("homogeneous", "uniform_random", "random_clusters")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def rasterize_layerset(
    gdf: gpd.GeoDataFrame,
    resolution: float = 2.0,
    overlap_method: callable = np.mean,
    seed: Optional[int] = None,
) -> xr.Dataset:
    """
    Rasterize fuel polygons into a single rioxarray Dataset.

    Parameters
    ----------
    gdf : GeoDataFrame
        Fuel polygons.  See module docstring for required/optional columns.
    resolution : float
        Cell size in meters (default 2 m).
    overlap_method : callable
        numpy function (np.mean, np.min, np.max) used to combine
        optional-band values when polygons of the same fuel_type overlap.
        Loading is always summed.
    seed : int, optional
        RNG seed for reproducible random / cluster placement.

    Returns
    -------
    xr.Dataset
        One DataArray per fuel type; dims (band, y, x).
    """
    _validate_gdf(gdf)
    if overlap_method not in (np.mean, np.min, np.max):
        raise ValueError("overlap_method must be np.mean, np.min, or np.max")

    rng = np.random.default_rng(seed)
    grid = _build_raster_grid(gdf, resolution)

    arrays: dict[str, xr.DataArray] = {}
    for fuel_type, group in gdf.groupby("fuel_type"):
        da = _rasterize_fuel_type(
            group.reset_index(drop=True),
            nx=grid.nx,
            ny=grid.ny,
            xs=grid.xs,
            ys=grid.ys,
            transform=grid.transform,
            resolution=resolution,
            overlap_method=overlap_method,
            rng=rng,
        )
        arrays[str(fuel_type)] = da

    ds = xr.Dataset(arrays)
    ds = ds.rio.write_crs(gdf.crs)
    ds = ds.rio.write_transform(grid.transform, inplace=True)
    return ds


# ---------------------------------------------------------------------------
# Raster grid construction
# ---------------------------------------------------------------------------
@dataclass
class RasterGrid:
    """Shared spatial grid for all fuel-type rasters."""

    nx: int
    ny: int
    xs: np.ndarray  # cell-center x coords, shape (nx,)
    ys: np.ndarray  # cell-center y coords, shape (ny,)  top-down
    transform: object  # rasterio Affine


def _build_raster_grid(gdf: gpd.GeoDataFrame, resolution: float) -> RasterGrid:
    """
    Compute a common raster grid that covers the union of all polygon extents,
    with edges snapped to multiples of resolution.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    x0 = math.floor(minx / resolution) * resolution
    y0 = math.floor(miny / resolution) * resolution
    x1 = math.ceil(maxx / resolution) * resolution
    y1 = math.ceil(maxy / resolution) * resolution

    nx = max(1, round((x1 - x0) / resolution))
    ny = max(1, round((y1 - y0) / resolution))

    xs = x0 + (np.arange(nx) + 0.5) * resolution  # cell centers, left→right
    ys = y1 - (np.arange(ny) + 0.5) * resolution  # cell centers, top→bottom

    transform = from_origin(x0, y1, resolution, resolution)
    return RasterGrid(nx=nx, ny=ny, xs=xs, ys=ys, transform=transform)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_gdf(gdf: gpd.GeoDataFrame) -> None:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set one with gdf.set_crs().")
    if gdf.crs.is_geographic:
        raise ValueError(
            f"GeoDataFrame CRS is geographic ({gdf.crs.to_string()}). "
            "Reproject to a projected CRS (e.g. a UTM zone) so that "
            "resolution is in meters, not degrees."
        )

    always_required = {
        "fuel_type",
        "fuel_loading",
        "fuel_height",
        "distribution",
        "percent_cover",
    }
    missing = always_required - set(gdf.columns)
    if missing:
        raise ValueError(f"GeoDataFrame is missing required columns: {missing}")

    for col in ("fuel_loading", "fuel_height", "percent_cover"):
        missing = gdf[col].isna()
        if missing.any():
            raise ValueError(f"{missing.sum()} row(s) are missing a value for '{col}'.")

    if gdf["distribution"].isna().any():
        raise ValueError(
            f"{gdf['distribution'].isna().sum()} row(s) are missing a distribution value."
        )

    bad_modes = set(gdf["distribution"].unique()) - set(DISTRIBUTION_MODES)
    if bad_modes:
        raise ValueError(
            f"Unknown distribution mode(s): {bad_modes}. "
            f"Valid modes: {DISTRIBUTION_MODES}"
        )

    cluster_mask = gdf["distribution"] == "random_clusters"
    if cluster_mask.any():
        if "patch_size" not in gdf.columns:
            raise ValueError("'patch_size' is required for random_clusters rows.")
        missing_patch = cluster_mask & gdf["patch_size"].isna()
        if missing_patch.any():
            raise ValueError(
                f"{missing_patch.sum()} random_clusters row(s) are missing 'patch_size'."
            )


# ---------------------------------------------------------------------------
# Per-fuel-type rasterization
# ---------------------------------------------------------------------------


def _rasterize_fuel_type(
    group: gpd.GeoDataFrame,
    *,
    nx: int,
    ny: int,
    xs: np.ndarray,
    ys: np.ndarray,
    transform,
    resolution: float,
    overlap_method: callable,
    rng: np.random.Generator,
) -> xr.DataArray:
    """Build a (band, y, x) DataArray for one fuel_type."""

    loading_acc = np.zeros((ny, nx), dtype=np.float64)

    # accumulate all non-loading bands; only track what overlap_method needs
    opt: dict[str, dict[str, np.ndarray]] = {}
    for col in ("height",) + OPTIONAL_BANDS:
        d = {}
        if overlap_method is np.mean:
            d["sum"] = np.zeros((ny, nx), dtype=np.float64)
            d["count"] = np.zeros((ny, nx), dtype=np.int32)
        if overlap_method is np.min:
            d["val"] = np.full((ny, nx), np.inf, dtype=np.float64)
        if overlap_method is np.max:
            d["val"] = np.full((ny, nx), -np.inf, dtype=np.float64)
        opt[col] = d

    for _, row in group.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        cover_frac = row.percent_cover / 100.0
        distribution = row.distribution
        patch_size = (
            row.patch_size
            if "patch_size" in row.index and not np.isnan(row.patch_size)
            else None
        )

        in_poly = _poly_mask(geom, transform, ny, nx)
        if not np.any(in_poly):
            continue

        hd = _build_horizontal_distribution(
            distribution=distribution,
            cover_frac=cover_frac,
            patch_size=patch_size,
            in_poly=in_poly,
            resolution=resolution,
            rng=rng,
        )

        fuel_loading = row.fuel_loading

        loading_acc += fuel_loading * hd

        fuel_present = hd > 0
        band_vals = {
            col: row.get(col, np.nan) if col in row.index else np.nan
            for col in OPTIONAL_BANDS
        }
        band_vals["height"] = row.fuel_height

        for col, val in band_vals.items():
            if np.isnan(val):
                continue
            a = opt[col]
            if overlap_method is np.mean:
                a["sum"] = np.where(fuel_present, a["sum"] + val, a["sum"])
                a["count"] = np.where(fuel_present, a["count"] + 1, a["count"])
            elif overlap_method is np.min:
                a["val"] = np.where(fuel_present, np.minimum(a["val"], val), a["val"])
            elif overlap_method is np.max:
                a["val"] = np.where(fuel_present, np.maximum(a["val"], val), a["val"])

    any_fuel = loading_acc > 0

    def _resolve(a: dict) -> np.ndarray:
        if overlap_method is np.mean:
            has = a["count"] > 0
            v = np.where(has, a["sum"] / np.maximum(a["count"], 1), np.nan)
        else:
            v = a["val"].copy()
            v[~np.isfinite(v)] = np.nan
        return np.where(any_fuel, v, np.nan).astype(np.float32)

    band_arrays = np.stack(
        [np.where(any_fuel, loading_acc, np.nan).astype(np.float32)]
        + [_resolve(opt[c]) for c in ("height",) + OPTIONAL_BANDS],
        axis=0,
    )  # (6, ny, nx)

    da = xr.DataArray(
        band_arrays,
        dims=["band", "y", "x"],
        coords={"band": list(ALL_BANDS), "y": ys, "x": xs},  # 6 bands
    )
    da.rio.write_nodata(np.nan, inplace=True)
    return da


# ---------------------------------------------------------------------------
# Raster mask
# ---------------------------------------------------------------------------


def _poly_mask(geom, transform, ny: int, nx: int) -> np.ndarray:
    """Boolean (ny, nx) — True where cell center lies inside geom."""
    return rio_rasterize(
        [(geom, 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)


# ---------------------------------------------------------------------------
# Horizontal distribution — the core rasterization logic
# ---------------------------------------------------------------------------


def _build_horizontal_distribution(
    distribution: str,
    cover_frac: Optional[float],
    patch_size: Optional[float],
    in_poly: np.ndarray,
    resolution: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return float64 weight array (ny, nx) in [0, 1] encoding per-cell
    fractional fuel coverage.  Mirrors HorizontalDistribution in STANDFIRE.
    """
    hd = np.zeros(in_poly.shape, dtype=np.float64)
    cells_in_poly = int(np.count_nonzero(in_poly))
    if cells_in_poly == 0:
        return hd

    if distribution == "homogeneous":
        return _dist_homogeneous(hd, in_poly, cover_frac)
    if distribution == "uniform_random":
        return _dist_uniform_random(hd, in_poly, cover_frac, rng)
    if distribution == "random_clusters":
        return _dist_random_clusters(
            hd, in_poly, cover_frac, patch_size, resolution, rng
        )
    raise ValueError(f"Unknown distribution: {distribution!r}")


def _dist_homogeneous(
    hd: np.ndarray, in_poly: np.ndarray, cover_frac: float
) -> np.ndarray:
    # Every in-polygon cell gets weight = cover_frac.
    hd[in_poly] = cover_frac
    return hd


def _dist_uniform_random(
    hd: np.ndarray,
    in_poly: np.ndarray,
    cover_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Each in-polygon cell is independently selected with probability cover_frac
    cells_in_poly = int(np.count_nonzero(in_poly))
    selected = rng.random(cells_in_poly) < cover_frac
    hd[in_poly] = selected.astype(float)
    return hd


def _dist_random_clusters(
    hd: np.ndarray,
    in_poly: np.ndarray,
    cover_frac: float,
    patch_size: float,
    resolution: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Place circular patches inside the polygon until cover_frac is reached.
    Patch diameters vary around patch_size with a 10% coefficient of variation.
    Centers are drawn uniformly at random, matching the original STANDFIRE behavior.
    """
    ny, nx = hd.shape
    cells_in_poly = int(np.count_nonzero(in_poly))
    occupied = np.zeros((ny, nx), dtype=bool)
    current_cells = 0
    target_cells = max(1, min(cells_in_poly, round(cover_frac * cells_in_poly)))

    def _stamp(cx: float, cy: float) -> bool:
        nonlocal current_cells
        # draw a patch radius with 10% CV around the nominal patch size
        actual_patch_size = max(resolution, rng.normal(patch_size, patch_size * 0.1))
        ray = 0.5 * actual_patch_size / resolution

        # bounding box of the circle in cell indices
        imin = max(0, int(math.floor(cx - ray)))
        imax = min(nx, int(math.ceil(cx + ray)))
        jmin = max(0, int(math.floor(cy - ray)))
        jmax = min(ny, int(math.ceil(cy + ray)))

        # loop through cell indices
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                # pass if cell is not in poly or is already filled
                if not in_poly[j, i] or occupied[j, i]:
                    continue
                # pass if cell center is outside circle
                if math.sqrt((i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2) >= ray:
                    continue
                # otherwise fill cell
                occupied[j, i] = True
                hd[j, i] = 1.0
                current_cells += 1
                # exit loop if enough cells are filled
                if current_cells >= target_cells:
                    return True
        return False

    # randomly select patch center locations and
    # fill patches until percent cover met
    while current_cells < target_cells:
        cx = rng.uniform(0, nx)
        cy = rng.uniform(0, ny)
        if _stamp(cx, cy):
            break

    return hd
