from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
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
    horizontal_resolution: float,
    seed: Optional[int] = None,
    combine_primary_secondary_strata: bool = True,
) -> xr.Dataset:
    """
    Create a 2D layerset xarray Dataset from a surface fuels GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Surface fuels data with columns: strata, loading, height, spatial_pattern,
        percent_cover, and a geometry column. Must be in a projected CRS.
    horizontal_resolution : float
        Output grid cell size in CRS units (applied to both x and y).
    seed : int, optional
        Seed for the random number generator used in percent-cover sampling.
    combine_primary_secondary_strata : bool
        If True, strata names with trailing digits are merged (e.g., Shrub1 and
        Shrub2 both become Shrub). If False, digit 1 maps to _primary and digit
        2 maps to _secondary.

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

    # Build grid dimensions from the extent of filtered data
    minx, miny, maxx, maxy = gdf_filtered.total_bounds
    n_cols = int(np.ceil((maxx - minx) / horizontal_resolution))
    n_rows = int(np.ceil((maxy - miny) / horizontal_resolution))

    transform = from_origin(minx, maxy, horizontal_resolution, horizontal_resolution)

    x_coords = minx + (np.arange(n_cols) + 0.5) * horizontal_resolution
    y_coords = maxy - (np.arange(n_rows) + 0.5) * horizontal_resolution

    loading_grids = {s: np.zeros((n_rows, n_cols), dtype=float) for s in output_strata}
    height_grids = {s: np.zeros((n_rows, n_cols), dtype=float) for s in output_strata}
    # Track how many polygons contributed to each cell for averaging
    hit_counts = {s: np.zeros((n_rows, n_cols), dtype=int) for s in output_strata}

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
                f"Polygon for strata '{row['strata']}' (fccs_id={row.get('fccs_id', 'N/A')}) "
                "does not intersect any grid cells at the given resolution and will be skipped.",
                stacklevel=2,
            )
            continue

        # Randomly assign loading to cover_fraction of cells within the polygon
        loading_contribution = _sample_polygon_cells(
            polygon_mask, cover_fraction, rng, loading_val
        )
        loading_grids[out_name] += loading_contribution

        # Accumulate height on filled cells; divide at the end to get per-cell average
        filled_cells = loading_contribution > 0
        height_grids[out_name][filled_cells] += height_val
        hit_counts[out_name][filled_cells] += 1

    # Finalize height as average across all polygons that filled each cell
    for s in output_strata:
        nonzero = hit_counts[s] > 0
        height_grids[s][nonzero] /= hit_counts[s][nonzero]

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
