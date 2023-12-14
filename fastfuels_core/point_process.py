"""
Point process module for expanding trees to a region of interest (ROI) and
generating random tree locations based on a specified point process.
"""

from __future__ import annotations

# External imports
import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import DataFrame
from geopandas import GeoDataFrame
from scipy.interpolate import griddata


class PointProcess:
    def __init__(self, process_type):
        self.process_type = process_type

        if process_type == "inhomogeneous_poisson":
            self.process = InhomogeneousPoissonProcess
        else:
            raise NotImplementedError(
                f"Point process type {process_type} is "
                f"not implemented. Currently only "
                f"'inhomogeneous_poisson' is "
                f"implemented."
            )

    def run(self, roi, trees, **kwargs):
        try:
            return self.process.generate_tree_locations(
                self.process, roi, trees, **kwargs
            )
        except AttributeError:
            raise NotImplementedError(
                f"Point process type {self.process_type} does not have a "
                f"valid generate_tree_locations method."
            )


class InhomogeneousPoissonProcess(PointProcess):
    def generate_tree_locations(
        self,
        roi: GeoDataFrame,
        trees: DataFrame,
        plots: GeoDataFrame,
        intensity_resolution: int = 15,
        seed: int = None,
    ) -> GeneratedTreeLocations:
        """
        Generates random tree locations based on an inhomogeneous Poisson
        point process.
        """
        seed = self._set_seed(seed)
        grids = self._create_structured_coords_grid(roi, intensity_resolution)
        structured_grid_x, structured_grid_y = grids

        tree_density_grid = self._interpolate_tree_density_to_grid(
            trees,
            plots,
            structured_grid_x,
            structured_grid_y,
            intensity_resolution,
        )
        plot_id_grid = self._interpolate_plot_id_to_grid(
            trees,
            plots,
            structured_grid_x,
            structured_grid_y,
            intensity_resolution,
        )

        tree_count_grid = self._generate_tree_counts(tree_density_grid)

        tree_locations = self._generate_tree_locations_in_grid(
            tree_count_grid,
            plot_id_grid,
            intensity_resolution,
            structured_grid_x,
            structured_grid_y,
        )

        return GeneratedTreeLocations(trees, tree_locations)

    @staticmethod
    def _set_seed(seed):
        seed = np.random.randint(0, 1000000) if seed is None else seed
        np.random.seed(seed)
        return seed

    @staticmethod
    def _create_structured_coords_grid(roi, resolution):
        """
        Creates a structured grid of cell-center plot coordinates from a list
        of plots.
        """
        west, south, east, north = roi.total_bounds
        x = np.arange(west + resolution / 2, east, resolution)
        # TODO: Flip y axis so that it is oriented correctly
        y = np.arange(south + resolution / 2, north, resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def _interpolate_tree_density_to_grid(
        self, trees, plots, grid_x, grid_y, cell_resolution
    ):
        plots_with_data_col = self._calculate_all_plots_tree_density(plots, trees)
        data_to_interpolate = plots_with_data_col["TPA_UNADJ"] * cell_resolution**2
        interpolated_plot_data = self._interpolate_data_to_grid(
            plots_with_data_col, data_to_interpolate, grid_x, grid_y, "cubic"
        )
        return interpolated_plot_data

    def _interpolate_plot_id_to_grid(
        self, trees, plots, grid_x, grid_y, cell_resolution
    ):
        plots_with_data_col = self._calculate_occupied_plots_tree_density(plots, trees)
        data_to_interpolate = plots_with_data_col["TM_CN"]
        interpolated_plot_data = self._interpolate_data_to_grid(
            plots_with_data_col, data_to_interpolate, grid_x, grid_y, "nearest"
        )
        return interpolated_plot_data

    def _calculate_all_plots_tree_density(self, plots, trees):
        return self._calculate_per_plot_tree_density(plots, trees, "left")

    def _calculate_occupied_plots_tree_density(self, plots, trees):
        return self._calculate_per_plot_tree_density(plots, trees, "right")

    @staticmethod
    def _calculate_per_plot_tree_density(plots, trees, merge_type):
        """
        Calculate the tree density for plots based on merge type (all or occupied). Returns a
        DataFrame of plots with a TPA_UNADJ column containing the tree density for each plot.
        """
        sum_by_plot = trees.groupby("TM_CN")["TPA_UNADJ"].sum().reset_index()
        merged_plots = plots.merge(sum_by_plot, on="TM_CN", how=merge_type)
        return merged_plots.fillna(0)

    @staticmethod
    def _interpolate_data_to_grid(plots, data, grid_x, grid_y, method):
        """
        Interpolate unstructured plot data to a structured grid.
        """
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
    def _generate_tree_counts(density_grid):
        return np.random.poisson(density_grid)

    @staticmethod
    def _generate_tree_locations_in_grid(
        count_grid, plot_grid, grid_resolution, grid_x, grid_y
    ):
        """
        Generates tree locations in a grid of plots.
        """
        # Calculate indices for each tree
        tree_indices = np.repeat(np.arange(count_grid.size), count_grid.ravel())

        # Calculate the cell indices for each tree
        cell_i, cell_j = np.unravel_index(tree_indices, count_grid.shape)

        # Get the plot ID for each tree
        plot_id_total = plot_grid.ravel()[tree_indices]

        # Generate random offsets within the cell for each tree
        num_tree_indices = len(tree_indices)
        random_offsets_x = np.random.uniform(
            -grid_resolution / 2, grid_resolution / 2, num_tree_indices
        )
        random_offsets_y = np.random.uniform(
            -grid_resolution / 2, grid_resolution / 2, num_tree_indices
        )

        # Calculate the actual coordinates of each tree
        x_coords = grid_x[cell_i, cell_j] + random_offsets_x
        y_coords = grid_y[cell_i, cell_j] + random_offsets_y

        # Create DataFrame
        tree_locations = pd.DataFrame(
            {"x": x_coords, "y": y_coords, "PlotID": plot_id_total}
        )

        return tree_locations


class GeneratedTreeLocations:
    def __init__(self, trees, tree_locations):
        self.trees = trees
        self.tree_locations = tree_locations

    def _count_trees_per_plot(self):
        """
        Count the number of trees per plot.
        """
        return self.tree_locations.value_counts("PlotID", dropna=False)

    def _sample_trees_within_plots(self, num_trees_per_plot):
        """
        Sample trees based on plot intensity.
        """
        sampled_trees = self.trees.groupby("PlotID").apply(
            lambda plot: plot.sample(
                n=num_trees_per_plot[plot.name], weights=plot["EXP_tph"], replace=True
            )
        )
        return sampled_trees.reset_index(drop=True)

    def _sample_trees_without_plots(self):
        """
        Sample trees when trees are not assigned to plots.
        """
        number_trees = len(self.tree_locations)
        sampled_trees = self.trees.sample(
            n=number_trees, weights=self.trees["EXP_tph"], replace=True
        )
        return sampled_trees.reset_index(drop=True)

    def _assign_tree_locations(self, sampled_trees):
        """
        Assign tree locations to sampled trees.
        """
        sampled_trees = sampled_trees.copy()
        sampled_trees["x"] = self.tree_locations["x"].values
        sampled_trees["y"] = self.tree_locations["y"].values
        return sampled_trees

    def _convert_to_geodataframe(self, sampled_trees):
        """
        Convert sampled trees to a GeoDataFrame with spatial coordinates.
        """
        return gpd.GeoDataFrame(
            sampled_trees,
            geometry=gpd.points_from_xy(sampled_trees.x, sampled_trees.y),
            crs=self.trees.crs,
        )

    def sample_trees_by_location(self):
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

        if "PlotID" in self.trees.columns:
            num_trees_per_plot = self._count_trees_per_plot()
            sampled_trees = self._sample_trees_within_plots(num_trees_per_plot)
            self.tree_locations = self.tree_locations.sort_values("PlotID")
        else:
            sampled_trees = self._sample_trees_without_plots()

        sampled_trees = sampled_trees.sort_values("PlotID")
        sampled_trees = self._assign_tree_locations(sampled_trees)
        return self._convert_to_geodataframe(sampled_trees)


def run_point_process(process, roi, trees, **kwargs):
    point_process = PointProcess(process)
    tree_locations = point_process.run(roi, trees, **kwargs)
    sampled_trees = tree_locations.sample_trees_by_location()
    return sampled_trees
