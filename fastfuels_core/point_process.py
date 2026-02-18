"""
Point process module for expanding trees to a region of interest (ROI) and
generating random tree locations based on a specified point process.
"""

# Core imports
from __future__ import annotations

# External imports
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from geopandas import GeoDataFrame
from scipy.interpolate import griddata


def run_point_process(process_type: str, roi, trees, **kwargs):
    if process_type == "inhomogeneous_poisson":
        return inhomogeneous_poisson_process(roi, trees, **kwargs)
    raise ValueError(f"Invalid point process type: {process_type}")


def inhomogeneous_poisson_process(
    roi: GeoDataFrame,
    trees: DataFrame,
    plots: GeoDataFrame,
    intensity_resolution: int = 15,
    intensity_interpolation_method: str = "linear",
    seed: int | None = None,
    chunk_size: float | None = None,
) -> dd.DataFrame:
    """
    Generates random tree locations based on an inhomogeneous Poisson
    point process. Returns a dask DataFrame for lazy, memory-efficient
    computation.

    Parameters
    ----------
    roi : GeoDataFrame
        Region of interest polygon.
    trees : DataFrame
        Tree sample data with PLOT_ID, TPA, and tree attributes.
    plots : GeoDataFrame
        Plot locations with geometry and PLOT_ID.
    intensity_resolution : int
        Grid cell size in metres for the density grid.
    intensity_interpolation_method : str
        Interpolation method for density grid ("linear" or "cubic").
    seed : int or None
        Random seed for reproducibility.
    chunk_size : float or None
        Tile size in metres for spatial chunking. None = single tile.

    Returns
    -------
    dask.dataframe.DataFrame
        Lazy DataFrame of generated trees with X, Y coordinates.
    """
    bounds = tuple(roi.total_bounds)

    # Eager: build global grids
    grid_x, grid_y = _create_structured_coords_grid(bounds, intensity_resolution)
    density_grid = _interpolate_tree_density_to_grid(
        trees,
        plots,
        grid_x,
        grid_y,
        intensity_resolution,
        intensity_interpolation_method,
    )
    plot_id_grid = _interpolate_plot_id_to_grid(trees, plots, grid_x, grid_y)

    # Eager: compute tile indices and per-tile seeds
    tile_indices = _generate_tile_indices(
        grid_x.shape, chunk_size, intensity_resolution
    )
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(tile_indices))

    # Build output meta
    meta = _build_output_meta(trees)

    # Lazy: build dask partitions
    partitions = []
    for (row_sl, col_sl), child_seed in zip(tile_indices, child_seeds):
        delayed_df = dask.delayed(_process_single_tile)(
            tile_grid_x=grid_x[row_sl, col_sl],
            tile_grid_y=grid_y[row_sl, col_sl],
            tile_density=density_grid[row_sl, col_sl],
            tile_plot_ids=plot_id_grid[row_sl, col_sl],
            intensity_resolution=intensity_resolution,
            roi_bounds=bounds,
            trees=trees,
            child_seed=child_seed,
        )
        partitions.append(dd.from_delayed(delayed_df, meta=meta))

    return dd.concat(partitions)


def _build_output_meta(trees: DataFrame) -> DataFrame:
    """Returns a zero-row DataFrame with the expected output schema."""
    cols = {col: trees[col].dtype for col in trees.columns}
    cols["X"] = np.float64
    cols["Y"] = np.float64
    meta = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in cols.items()})
    return meta


def _create_structured_coords_grid(
    bounds: tuple, resolution: float
) -> tuple[ndarray, ndarray]:
    """
    Creates a structured grid of cell-centered coordinates for the given bounds.
    """
    west, south, east, north = bounds
    x = np.arange(west + resolution / 2, east, resolution)
    y = np.arange(north - resolution / 2, south, -resolution)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def _interpolate_tree_density_to_grid(
    trees,
    plots,
    grid_x,
    grid_y,
    cell_resolution,
    interpolation_method="linear",
) -> ndarray:
    """
    Interpolates tree density to a structured grid of cells.
    """
    plots_with_data_col = _calculate_all_plots_tree_density(plots, trees)
    data_to_interpolate = plots_with_data_col["TPA"] * (cell_resolution**2)
    interpolated_plot_data = _interpolate_data_to_grid(
        plots_with_data_col,
        data_to_interpolate,
        grid_x,
        grid_y,
        interpolation_method,
    )
    return interpolated_plot_data


def _interpolate_plot_id_to_grid(trees, plots, grid_x, grid_y) -> ndarray:
    """
    Interpolates the plot IDs of plots containing trees to a structured
    grid of cells (nearest-neighbor Voronoi assignment).
    """
    plots_with_data_col = _calculate_occupied_plots_tree_density(plots, trees)
    data_to_interpolate = plots_with_data_col["PLOT_ID"]
    interpolated_plot_data = _interpolate_data_to_grid(
        plots_with_data_col, data_to_interpolate, grid_x, grid_y, "nearest"
    )
    return interpolated_plot_data


def _calculate_all_plots_tree_density(plots, trees) -> GeoDataFrame:
    """Calculates tree density (TPA) for all plots, including empty ones."""
    return _calculate_per_plot_tree_density(plots, trees, "left")


def _calculate_occupied_plots_tree_density(plots, trees) -> GeoDataFrame:
    """Calculates tree density (TPA) only for plots containing trees."""
    return _calculate_per_plot_tree_density(plots, trees, "right")


def _calculate_per_plot_tree_density(plots, trees, merge_type) -> GeoDataFrame:
    """
    Calculate the tree density for a specific grouping of plots based on
    the merge type.
    """
    sum_by_plot = trees.groupby("PLOT_ID")["TPA"].sum().reset_index()
    merged_plots = plots.merge(sum_by_plot, on="PLOT_ID", how=merge_type)
    return merged_plots.fillna(0)


def _interpolate_data_to_grid(plots, data, grid_x, grid_y, method) -> ndarray:
    """Interpolate unstructured plot data to a structured grid."""
    if len(data) == 0:
        return np.zeros(grid_x.shape)

    interpolated_grid = griddata(
        (plots.geometry.x, plots.geometry.y),
        data,
        (grid_x, grid_y),
        method=method,
    )
    interpolated_grid = np.nan_to_num(interpolated_grid, nan=0)
    interpolated_grid[interpolated_grid < 0] = 0

    return interpolated_grid


def _generate_tile_indices(
    grid_shape: tuple[int, int],
    chunk_size: float | None,
    intensity_resolution: float,
) -> list[tuple[slice, slice]]:
    """
    Converts chunk_size (metres) to grid-cell slices over the global grid.
    Returns list of (row_slice, col_slice) for indexing into global arrays.
    chunk_size=None returns a single tile covering the full grid.
    """
    nrows, ncols = grid_shape
    if chunk_size is None:
        return [(slice(0, nrows), slice(0, ncols))]

    chunk_cells = max(1, int(chunk_size / intensity_resolution))
    tiles = []
    for r0 in range(0, nrows, chunk_cells):
        for c0 in range(0, ncols, chunk_cells):
            row_sl = slice(r0, min(r0 + chunk_cells, nrows))
            col_sl = slice(c0, min(c0 + chunk_cells, ncols))
            tiles.append((row_sl, col_sl))
    return tiles


def _generate_tree_counts(intensity_grid: ndarray, rng: np.random.Generator) -> ndarray:
    """
    Draws a random number of trees from a Poisson distribution for each
    cell in the intensity grid.
    """
    return rng.poisson(intensity_grid)


def _get_flattened_tree_indices(count_grid: ndarray) -> ndarray:
    """
    Generate a list of indices representing the locations of trees in a
    flattened grid.
    """
    return np.repeat(np.arange(count_grid.size), count_grid.ravel())


def _get_grid_cell_indices(tree_indices, count_grid) -> tuple[ndarray, ndarray]:
    """
    Converts the flattened tree indices into 2D cell indices.
    """
    return np.unravel_index(tree_indices, count_grid.shape)


def _calculate_tree_coordinates(
    cell_i, cell_j, grid_resolution, grid_x, grid_y, rng: np.random.Generator
) -> tuple[ndarray, ndarray]:
    """
    Calculates the x and y coordinates of trees within each cell,
    adding a uniform random offset within the cell bounds.
    """
    random_offsets_x, random_offsets_y = _generate_random_offsets(
        len(cell_i), grid_resolution, rng
    )
    x_coords = grid_x[cell_i, cell_j] + random_offsets_x
    y_coords = grid_y[cell_i, cell_j] + random_offsets_y
    return x_coords, y_coords


def _generate_random_offsets(
    num_indices: int, grid_resolution: float, rng: np.random.Generator
) -> tuple[ndarray, ndarray]:
    """
    Generates a random offset for each tree centered at 0 within
    [-grid_resolution/2, grid_resolution/2].
    """
    return (
        rng.uniform(-grid_resolution / 2, grid_resolution / 2, num_indices),
        rng.uniform(-grid_resolution / 2, grid_resolution / 2, num_indices),
    )


def _count_trees_per_plot(trees, tree_locations):
    """Count the number of trees per plot."""
    tree_counts = tree_locations.value_counts("PLOT_ID", dropna=False)
    all_plot_ids = trees["PLOT_ID"].unique()
    return tree_counts.reindex(all_plot_ids, fill_value=0)


def _sample_trees_within_plots(trees, num_trees_per_plot, rng: np.random.Generator):
    """Sample trees based on plot intensity using the provided RNG."""

    def sample_trees_for_plot(plot):
        n = num_trees_per_plot.loc[plot.name]
        if n == 0:
            return plot.iloc[:0]
        weights = np.nan_to_num(plot["TPA"].values.astype(float), nan=0.0)
        total = weights.sum()
        if total <= 0:
            return plot.iloc[:0]
        probs = weights / total
        indices = rng.choice(len(plot), size=n, replace=True, p=probs)
        return plot.iloc[indices]

    sampled_trees = trees.groupby("PLOT_ID").apply(sample_trees_for_plot)

    if "PLOT_ID" not in sampled_trees.columns:
        sampled_trees = sampled_trees.reset_index(level=0)
    return sampled_trees.reset_index(drop=True)


def _assign_tree_locations(sampled_trees, tree_locations):
    """Assign tree locations to sampled trees."""
    tree_locations = tree_locations.sort_values("PLOT_ID")
    sampled_trees = sampled_trees.sort_values("PLOT_ID")
    sampled_trees["X"] = tree_locations["X"].values
    sampled_trees["Y"] = tree_locations["Y"].values
    return sampled_trees


def _sample_trees_by_location(trees, tree_locations, rng: np.random.Generator):
    """Sample trees and assign generated locations."""
    num_trees_per_plot = _count_trees_per_plot(trees, tree_locations)
    sampled_trees = _sample_trees_within_plots(trees, num_trees_per_plot, rng)
    sampled_trees = _assign_tree_locations(sampled_trees, tree_locations)
    return sampled_trees


def _drop_trees_outside_roi_bounds(df: DataFrame, roi_bounds: tuple) -> DataFrame:
    """Drop trees that are outside the ROI bounds."""
    minx, miny, maxx, maxy = roi_bounds
    df = df.copy()
    df.drop(df[(df["Y"] < miny) | (df["Y"] > maxy)].index, inplace=True)
    df.drop(df[(df["X"] < minx) | (df["X"] > maxx)].index, inplace=True)
    return df


def _process_single_tile(
    tile_grid_x: ndarray,
    tile_grid_y: ndarray,
    tile_density: ndarray,
    tile_plot_ids: ndarray,
    intensity_resolution: float,
    roi_bounds: tuple,
    trees: DataFrame,
    child_seed: np.random.SeedSequence,
) -> DataFrame:
    """
    Process a single spatial tile: generate tree counts, locations, and
    sample trees. Returns a plain pandas DataFrame.
    """
    rng = np.random.default_rng(child_seed)
    meta = _build_output_meta(trees)

    tree_count_grid = _generate_tree_counts(tile_density, rng)

    if tree_count_grid.sum() == 0:
        return meta.copy()

    flat_tree_indices = _get_flattened_tree_indices(tree_count_grid)
    cell_i, cell_j = _get_grid_cell_indices(flat_tree_indices, tree_count_grid)
    x_coords, y_coords = _calculate_tree_coordinates(
        cell_i, cell_j, intensity_resolution, tile_grid_x, tile_grid_y, rng
    )
    tree_locations = pd.DataFrame(
        {
            "X": x_coords,
            "Y": y_coords,
            "PLOT_ID": tile_plot_ids.ravel()[flat_tree_indices],
        }
    )

    sampled_trees = _sample_trees_by_location(trees, tree_locations, rng)
    sampled_trees = _drop_trees_outside_roi_bounds(sampled_trees, roi_bounds)
    # Ensure column order matches meta (tree columns + X, Y)
    expected_cols = list(trees.columns) + ["X", "Y"]
    sampled_trees = sampled_trees[expected_cols]
    return sampled_trees.reset_index(drop=True)
