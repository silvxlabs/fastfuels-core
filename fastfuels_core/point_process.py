"""
Point process module for expanding trees to a region of interest (ROI) and
generating random tree locations based on a specified point process.
"""

# Core imports
from __future__ import annotations

# External imports
import numpy as np
import pandas as pd
import geopandas as gpd
from numpy import ndarray
from pandas import DataFrame
from geopandas import GeoDataFrame
from scipy.interpolate import griddata


class InhomogeneousPoissonProcess:
    def simulate(
        self,
        roi: GeoDataFrame,
        trees: DataFrame,
        plots: GeoDataFrame,
        intensity_resolution: int = 15,
        intensity_interpolation_method: str = "linear",
        seed: int = None,
    ) -> GeoDataFrame:
        """
        Generates random tree locations based on an inhomogeneous Poisson
        point process.
        """
        self._set_seed(seed)
        grid_x, grid_y = self._create_structured_coords_grid(roi, intensity_resolution)
        tree_density_grid = self._interpolate_tree_density_to_grid(
            trees,
            plots,
            grid_x,
            grid_y,
            intensity_resolution,
            intensity_interpolation_method,
        )
        plot_id_grid = self._interpolate_plot_id_to_grid(trees, plots, grid_x, grid_y)
        tree_count_grid = self._generate_tree_counts(tree_density_grid)
        tree_locations = self._generate_tree_locations_in_grid(
            tree_count_grid,
            plot_id_grid,
            intensity_resolution,
            grid_x,
            grid_y,
        )
        sampled_trees = self._sample_trees_by_location(trees, tree_locations)
        trees_in_bounds = self._drop_trees_outside_roi_bounds(sampled_trees, roi)
        return self._convert_to_geodataframe(trees_in_bounds, roi.crs)

    @staticmethod
    def _set_seed(seed):
        """Sets the seed for the random number generator."""
        seed = np.random.randint(0, 1000000) if seed is None else seed
        np.random.seed(seed)

    @staticmethod
    def _create_structured_coords_grid(roi, resolution) -> tuple[ndarray, ndarray]:
        """
        Creates a structured grid of cell-centered coordinates for the ROI.
        The resolution parameter specifies the size of each cell in the grid.
        """
        west, south, east, north = roi.total_bounds
        x = np.arange(west + resolution / 2, east, resolution)
        y = np.arange(north - resolution / 2, south, -resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def _interpolate_tree_density_to_grid(
        self,
        trees,
        plots,
        grid_x,
        grid_y,
        cell_resolution,
        interpolation_method="linear",
    ) -> ndarray:
        """
        Interpolates tree density to a structured grid of cells. Tree density
        is calculated for each plot and then interpolated to the grid using
        the specified interpolation method, typically "linear" or "cubic".
        """
        plots_with_data_col = self._calculate_all_plots_tree_density(plots, trees)
        data_to_interpolate = plots_with_data_col["TPA"] * (cell_resolution**2)
        interpolated_plot_data = self._interpolate_data_to_grid(
            plots_with_data_col,
            data_to_interpolate,
            grid_x,
            grid_y,
            interpolation_method,
        )
        return interpolated_plot_data

    def _interpolate_plot_id_to_grid(self, trees, plots, grid_x, grid_y) -> ndarray:
        """
        Interpolates the plot IDs of plots containing trees to a structured
        grid of cells. The plot ID is a categorical value that is used to
        assign trees to plots. This function effectively creates a
        discretized Veroni diagram of the plots containing trees.
        """
        plots_with_data_col = self._calculate_occupied_plots_tree_density(plots, trees)
        data_to_interpolate = plots_with_data_col["PLOT_ID"]
        interpolated_plot_data = self._interpolate_data_to_grid(
            plots_with_data_col, data_to_interpolate, grid_x, grid_y, "nearest"
        )
        return interpolated_plot_data

    def _calculate_all_plots_tree_density(self, plots, trees) -> GeoDataFrame:
        """
        Calculates the tree density, stored in the TPA column, for all
        plots in the ROI. Returns a DataFrame of all plots with a TPA
        column containing the tree density for each plot.
        """
        return self._calculate_per_plot_tree_density(plots, trees, "left")

    def _calculate_occupied_plots_tree_density(self, plots, trees) -> GeoDataFrame:
        """
        Calculates the tree density, stored in the TPA column, for plots
        containing trees in the ROI. Returns a DataFrame of plots containing
        trees with a TPA column containing the tree density for each
        plot.
        """
        return self._calculate_per_plot_tree_density(plots, trees, "right")

    @staticmethod
    def _calculate_per_plot_tree_density(plots, trees, merge_type) -> GeoDataFrame:
        """
        Calculate the tree density for a specific grouping of plots based on
        the merge type. if the merge type is "left", the tree density is
        calculated for all plots in the ROI. If the merge type is "right", the
        tree density is calculated only for plots containing trees in the ROI.
        Returns a DataFrame of plots with a TPA column containing the
        tree density for each plot.
        """
        sum_by_plot = trees.groupby("PLOT_ID")["TPA"].sum().reset_index()
        merged_plots = plots.merge(sum_by_plot, on="PLOT_ID", how=merge_type)
        return merged_plots.fillna(0)

    @staticmethod
    def _interpolate_data_to_grid(plots, data, grid_x, grid_y, method) -> ndarray:
        """
        Interpolate unstructured plot data to a structured grid.
        """
        # If there is no data to interpolate, return a grid of zeros.
        # This can happen if there are no trees in the ROI.
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

    @staticmethod
    def _generate_tree_counts(intensity_grid) -> ndarray:
        """
        Draws a random number of trees from a Poisson distribution for each
        cell in the intensity grid. The intensity grid is a 2D array where
        each cell represents a plot and the value in the cell represents the
        tree density of the plot.
        """
        return np.random.poisson(intensity_grid)

    def _generate_tree_locations_in_grid(
        self, count_grid, plot_grid, grid_resolution, grid_x, grid_y
    ):
        """
        Generates tree locations in a grid of plots.
        """
        flat_tree_indices = self._get_flattened_tree_indices(count_grid)
        cell_i, cell_j = self._get_grid_cell_indices(flat_tree_indices, count_grid)
        x_coords, y_coords = self._calculate_tree_coordinates(
            cell_i, cell_j, grid_resolution, grid_x, grid_y
        )
        return pd.DataFrame(
            {
                "X": x_coords,
                "Y": y_coords,
                "PLOT_ID": plot_grid.ravel()[flat_tree_indices],
            }
        )

    @staticmethod
    def _get_flattened_tree_indices(count_grid: ndarray) -> ndarray:
        """
        Generate a list of indices representing the locations of trees in a
        grid.

        This function takes a 2D grid (a numpy array) as input, where each
        cell in the grid represents a plot of land and the value in the cell
        represents the number of trees in that plot. The function uses the
        `numpy.repeat` function to repeat each index in the flattened grid
        according to the number of trees in the corresponding plot.

        The output of this function is a 1D numpy array where each element is
        an index of a cell in the flattened grid, and each index is repeated
        a number of times equal to the number of trees in the corresponding
        plot. This array of indices can then be used to generate the
        locations of individual trees within each plot.
        """
        return np.repeat(np.arange(count_grid.size), count_grid.ravel())

    @staticmethod
    def _get_grid_cell_indices(tree_indices, count_grid) -> tuple[ndarray, ndarray]:
        """
        Converts the flattened tree indices into 2D cell indices. Returns two
        1D arrays containing the row and column indices of each tree.

        This method takes a 1D array of tree indices and the 2D count grid as
        input. It uses the `numpy.unravel_index` function to convert the
        flattened tree indices into 2D cell indices that correspond to their
        original locations in the count grid.
        """
        return np.unravel_index(tree_indices, count_grid.shape)

    def _calculate_tree_coordinates(
        self, cell_i, cell_j, grid_resolution, grid_x, grid_y
    ) -> tuple[ndarray, ndarray]:
        """
        Calculates the x and y coordinates of trees within each plot in the
        grid. Returns two 1D arrays containing the x and y coordinates of each
        tree.

        This function takes the row and column indices of each tree's
        location in the grid, the resolution of the grid, and the x and y
        coordinates of the grid as inputs. It generates random offsets for
        each tree within its respective plot, ensuring that the tree's
        location is within the plot's boundaries. The offsets are generated
        uniformly at random within the range of [-grid_resolution / 2,
        grid_resolution / 2]. These offsets are then added to the x and y
        coordinates of the plot's center to get the final coordinates of each
        tree.
        """
        random_offsets_x, random_offsets_y = self._generate_random_offsets(
            len(cell_i), grid_resolution
        )
        x_coords = grid_x[cell_i, cell_j] + random_offsets_x
        y_coords = grid_y[cell_i, cell_j] + random_offsets_y
        return x_coords, y_coords

    @staticmethod
    def _generate_random_offsets(num_indices, grid_resolution):
        """
        Generates a random offset for each tree centered at 0. The offsets are
        generated uniformly at random within the range of
        [-grid_resolution / 2, grid_resolution / 2].
        """
        return (
            np.random.uniform(-grid_resolution / 2, grid_resolution / 2, num_indices),
            np.random.uniform(-grid_resolution / 2, grid_resolution / 2, num_indices),
        )

    def _sample_trees_by_location(self, trees, tree_locations):
        """
        This function first checks if the trees belong to plots. If they do,
        it counts the number of trees per plot, samples trees based on plot
        intensity, and sorts the tree locations by PlotID.

        If the trees do not belong to plots, it samples the trees without
        considering plots.

        The function then assigns the generated tree locations to the sampled
        trees. Finally, it converts the sampled trees to a GeoDataFrame with
        spatial coordinates and returns it.
        """
        num_trees_per_plot = self._count_trees_per_plot(trees, tree_locations)
        sampled_trees = self._sample_trees_within_plots(trees, num_trees_per_plot)
        sampled_trees = self._assign_tree_locations(sampled_trees, tree_locations)
        return sampled_trees

    @staticmethod
    def _count_trees_per_plot(trees, tree_locations):
        """
        Count the number of trees per plot.
        """
        tree_counts = tree_locations.value_counts("PLOT_ID", dropna=False)
        all_plot_ids = trees["PLOT_ID"].unique()
        return tree_counts.reindex(all_plot_ids, fill_value=0)

    @staticmethod
    def _sample_trees_within_plots(trees, num_trees_per_plot):
        """
        Sample trees based on plot intensity, with handling for PLOT_IDs not in num_trees_per_plot.
        """

        # Define the sampling function
        def sample_trees_for_plot(plot):
            return plot.sample(
                n=num_trees_per_plot.loc[plot.name], weights=plot["TPA"], replace=True
            )

        # Apply the sampling function to each plot
        sampled_trees = trees.groupby("PLOT_ID").apply(sample_trees_for_plot)

        return sampled_trees.reset_index(drop=True)

    @staticmethod
    def _assign_tree_locations(sampled_trees, tree_locations):
        """
        Assign tree locations to sampled trees.
        """
        # Align sampled_trees and tree_locations by sorting by PLOT_ID
        tree_locations = tree_locations.sort_values("PLOT_ID")
        sampled_trees = sampled_trees.sort_values("PLOT_ID")

        # Assign tree locations to sampled trees
        sampled_trees["X"] = tree_locations["X"].values
        sampled_trees["Y"] = tree_locations["Y"].values

        return sampled_trees

    @staticmethod
    def _drop_trees_outside_roi_bounds(df, roi):
        """
        Drop trees that are outside the total bounds of the ROI.
        """
        minx, miny, maxx, maxy = roi.total_bounds
        df.drop(df[(df["Y"] < miny) | (df["Y"] > maxy)].index, inplace=True)
        df.drop(df[(df["X"] < minx) | (df["X"] > maxx)].index, inplace=True)
        return df

    @staticmethod
    def _convert_to_geodataframe(sampled_trees, crs):
        """
        Convert sampled trees to a GeoDataFrame with spatial coordinates.
        """
        return gpd.GeoDataFrame(
            sampled_trees,
            geometry=gpd.points_from_xy(sampled_trees["X"], sampled_trees["Y"]),
            crs=crs,
        )


def run_point_process(process_type: str, roi, trees, **kwargs):
    if process_type == "inhomogeneous_poisson":
        point_process = InhomogeneousPoissonProcess()
    else:
        raise ValueError(f"Invalid point process type: {process_type}")
    return point_process.simulate(roi, trees, **kwargs)
