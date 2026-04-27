from __future__ import annotations

import re
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers the .rio accessor on xarray objects
import geopandas as gpd
from rasterio.transform import Affine, from_origin
from rasterio.features import rasterize as rio_rasterize


def _sample_polygon_cells(
    polygon_mask: np.ndarray,
    cover_fraction: float,
    rng: np.random.Generator,
    fill_value: float,
) -> np.ndarray:
    # Randomly select cover_fraction of the cells inside the polygon and assign fill_value
    n_pixels = np.count_nonzero(polygon_mask)
    selected = rng.random(n_pixels) < cover_fraction
    output = np.zeros(polygon_mask.shape, dtype=float)
    output[polygon_mask] = selected.astype(float) * fill_value
    return output


def _resolve_strata_names(
    raw_strata: list[str],
    combine: bool,
) -> dict[str, str]:
    # Map each raw strata name (e.g. "Shrub1", "Shrub2") to its output name.
    # With combine=True both collapse to "Shrub"; with combine=False they become
    # "Shrub_primary" and "Shrub_secondary".
    mapping = {}
    for name in raw_strata:
        match = re.search(r"\d+$", name)
        if not match:
            mapping[name] = name
        else:
            base = name[: match.start()]
            number = int(match.group())
            if combine:
                mapping[name] = base
            else:
                if number == 1:
                    mapping[name] = f"{base}_primary"
                elif number == 2:
                    mapping[name] = f"{base}_secondary"
                else:
                    warnings.warn(
                        f"Strata '{name}' has a trailing number > 2. Only _primary (1) "
                        f"and _secondary (2) are supported when "
                        f"combine_primary_secondary_strata=False. Mapping to "
                        f"'{base}_{number}'.",
                        stacklevel=3,
                    )
                    mapping[name] = f"{base}_{number}"
    return mapping


def create_layerset(
    gdf: gpd.GeoDataFrame,
    horizontal_resolution: Optional[float] = None,
    seed: Optional[int] = None,
    combine_primary_secondary_strata: bool = True,
    transform: Optional[Affine] = None,
    shape: Optional[Tuple[int, int]] = None,
    height_func: Callable = np.mean,
) -> xr.Dataset:
    """
    Create a 2D layerset xarray Dataset from a surface fuels GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Surface fuels data with columns: strata, loading, height, spatial_pattern,
        percent_cover, and a geometry column. Must be in a projected CRS.
    horizontal_resolution : float, optional
        Output grid cell size in CRS units (applied to both x and y). Required
        when ``transform`` and ``shape`` are not provided.
    seed : int, optional
        Seed for the random number generator used in percent-cover sampling.
    combine_primary_secondary_strata : bool
        If True, strata names with trailing digits are merged (e.g., Shrub1 and
        Shrub2 both become Shrub). If False, digit 1 maps to _primary and digit
        2 maps to _secondary.
    transform : Affine, optional
        Rasterio affine transform defining the output grid origin and resolution.
        Must be provided together with ``shape``; if omitted the grid is derived
        from the GeoDataFrame extent and ``horizontal_resolution``.
    shape : tuple of (int, int), optional
        Output grid dimensions as ``(n_rows, n_cols)``. Must be provided together
        with ``transform``.
    height_func : callable
        Reduction applied when multiple polygons contribute height to the same
        cell. Supported values are ``np.mean`` (default), ``np.min``, and
        ``np.max`` (and their ``nan``-aware equivalents).

    Returns
    -------
    xr.Dataset
        Dataset with variables `loading` and `height`, each with dimensions
        (strata, y, x). CRS and affine transform are written via rioxarray.
    """
    if gdf.crs is None:
        raise ValueError("GDF must have a CRS.")
    if not gdf.crs.is_projected:
        raise ValueError(
            f"GDF must be in a projected CRS; got geographic CRS: {gdf.crs}. "
            "Reproject with gdf.to_crs() before calling create_layerset()."
        )

    for col in ("loading", "height", "spatial_pattern", "percent_cover"):
        if col not in gdf.columns:
            raise ValueError(
                f"GDF is missing required column '{col}'. "
                "Ensure the GeoDataFrame was built from a valid surface fuels GeoJSON."
            )

    # Exclude rows with zero/missing fuel, non-uniform spatial pattern, or empty geometry
    mask = (
        (gdf["loading"] > 0)
        & (gdf["height"] > 0)
        & (gdf["spatial_pattern"].str.lower() == "uniform")
        & (gdf["percent_cover"] > 0)
    )
    gdf_filtered = gdf[mask & ~gdf.geometry.is_empty].copy()

    if gdf_filtered.empty:
        raise ValueError(
            "No rows remain after filtering for loading > 0, height > 0, "
            "spatial_pattern == 'uniform', percent_cover > 0, and non-empty geometry."
        )

    # Map raw strata names to output names based on combine flag
    raw_strata = gdf_filtered["strata"].unique().tolist()
    strata_mapping = _resolve_strata_names(raw_strata, combine_primary_secondary_strata)
    output_strata = list(dict.fromkeys(strata_mapping.values()))

    # Build grid dimensions from external transform+shape or from the GDF extent
    if (transform is None) != (shape is None):
        raise ValueError(
            "'transform' and 'shape' must be provided together or not at all."
        )

    if transform is not None:
        n_rows, n_cols = shape
    else:
        if horizontal_resolution is None:
            raise ValueError(
                "horizontal_resolution is required when transform and shape are not provided."
            )
        minx, miny, maxx, maxy = gdf_filtered.total_bounds
        n_cols = int(np.ceil((maxx - minx) / horizontal_resolution))
        n_rows = int(np.ceil((maxy - miny) / horizontal_resolution))
        transform = from_origin(
            minx, maxy, horizontal_resolution, horizontal_resolution
        )

    # Cell-center coordinates derived uniformly from the affine transform
    x_coords = transform.c + (np.arange(n_cols) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(n_rows) + 0.5) * transform.e

    loading_grids = {s: np.zeros((n_rows, n_cols), dtype=float) for s in output_strata}
    height_grids = {s: np.zeros((n_rows, n_cols), dtype=float) for s in output_strata}
    hit_counts = {s: np.zeros((n_rows, n_cols), dtype=int) for s in output_strata}

    _is_mean = height_func is np.mean

    rng = np.random.default_rng(seed)

    for _, row in gdf_filtered.iterrows():
        out_name = strata_mapping[row.strata]
        cover_fraction = float(row.percent_cover) / 100.0
        loading_val = float(row.loading)
        height_val = float(row.height)

        # Rasterize this polygon to a boolean cell mask
        geom = row.geometry
        polygon_mask = rio_rasterize(
            [(geom.__geo_interface__, 1)],
            out_shape=(n_rows, n_cols),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        if not polygon_mask.any():
            warnings.warn(
                f"Polygon for strata '{row['strata']}' (fuelbed_num={row.get('fuelbed_num', 'N/A')}) "
                "does not intersect any grid cells at the given resolution and will be skipped.",
                stacklevel=2,
            )
            continue

        # Randomly assign loading to cover_fraction of cells within the polygon
        loading_contribution = _sample_polygon_cells(
            polygon_mask, cover_fraction, rng, loading_val
        )
        loading_grids[out_name] += loading_contribution

        # Update height in-place
        filled_cells = loading_contribution > 0
        hit_counts[out_name][filled_cells] += 1

        if _is_mean:
            # Accumulate sum; divide by hit_counts after the loop for exact mean
            height_grids[out_name][filled_cells] += height_val
        else:
            # First hit sets the value; subsequent hits reduce pairwise with height_func
            first_hit = filled_cells & (hit_counts[out_name] == 1)
            multi_hit = filled_cells & (hit_counts[out_name] > 1)
            height_grids[out_name][first_hit] = height_val
            if multi_hit.any():
                height_grids[out_name][multi_hit] = height_func(
                    [
                        height_grids[out_name][multi_hit],
                        np.full(multi_hit.sum(), height_val),
                    ],
                    axis=0,
                )

    if _is_mean:
        for s in output_strata:
            multi = hit_counts[s] > 1
            height_grids[s][multi] /= hit_counts[s][multi]

    coords = {"strata": output_strata, "y": y_coords, "x": x_coords}
    ds = xr.Dataset(
        {
            "loading": xr.DataArray(
                np.stack([loading_grids[s] for s in output_strata]),
                dims=["strata", "y", "x"],
                coords=coords,
            ),
            "height": xr.DataArray(
                np.stack([height_grids[s] for s in output_strata]),
                dims=["strata", "y", "x"],
                coords=coords,
            ),
        }
    )
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs(gdf_filtered.crs)
    ds = ds.rio.write_transform(transform)

    return ds
