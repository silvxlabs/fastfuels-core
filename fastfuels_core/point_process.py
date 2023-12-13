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
    ) -> GeneratedTreeLocations:
        """
        Generates random tree locations based on an inhomogeneous Poisson
        point process.
        """
        (
            structured_grid_x,
            structured_grid_y,
        ) = self._create_structured_coords_grid(roi, intensity_resolution)

        all_plots_density_grid = self._calculate_and_interpolate_plot_data(
            trees, plots, structured_grid_x, structured_grid_y, plot_type="all"
        )
        occupied_plots_id_grid = self._calculate_and_interpolate_plot_data(
            trees, plots, structured_grid_x, structured_grid_y, plot_type="occupied"
        )

        # # Get the tree density for ALL plots and occupied plots in the grid
        # density = self._sum_by_plot(
        #     trees.assign(EXP_tpm=exp_tpm), "PlotID", "TPA_UNADJ"
        # )
        # all_plots = self._get_tree_density_all_plots(plots, density)
        # occupied_plots = self._get_tree_density_occupied_plots(all_plots, density)
        #
        # # Create a structured grid of plot coordinates
        # (
        #     structured_grid_x,
        #     structured_grid_y,
        # ) = self._create_structured_coords_grid_from_roi(roi, intensity_resolution)
        #
        # # Interpolate the tree density for all plots in the grid
        # grid_cell_area = intensity_resolution * intensity_resolution
        # density_grid = self._interpolate_unstructured_plot_data_to_structured_grid(
        #     all_plots,
        #     all_plots["EXP_tpm"] * grid_cell_area,
        #     structured_grid_x,
        #     structured_grid_y,
        #     method="linear",
        # )
        # plot_id_grid = self._interpolate_unstructured_plot_data_to_structured_grid(
        #     occupied_plots,
        #     occupied_plots["PlotID"],
        #     structured_grid_x,
        #     structured_grid_y,
        # )
        #
        # # Sample the density grid with a Poisson distribution to get the number of trees per plot
        # tree_count_grid = np.random.poisson(density_grid)
        #
        # # Generate tree locations for each plot in the tree count grid
        # tree_locations = self._generate_tree_locations_in_grid(
        #     tree_count_grid,
        #     plot_id_grid,
        #     intensity_resolution,
        #     structured_grid_x,
        #     structured_grid_y,
        # )

        # return GeneratedTreeLocations(trees, tree_locations)

    @staticmethod
    def _create_structured_coords_grid(roi, resolution):
        """
        Creates a structured grid of cell-center plot coordinates from a list
        of plots.
        """
        west, south, east, north = roi.total_bounds
        x = np.arange(west + resolution / 2, east, resolution)
        y = np.arange(south + resolution / 2, north, resolution)
        # x = np.arange(west, east, resolution)
        # y = np.arange(south, north, resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

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
    def _interpolate_plot_data_to_grid(plots, col_name, grid_x, grid_y, method):
        """
        Interpolate unstructured plot data to a structured grid.
        """
        interpolated_grid = griddata(
            (plots.geometry.x, plots.geometry.y),
            plots[col_name],
            (grid_x, grid_y),
            method=method,
        )
        interpolated_grid = np.nan_to_num(interpolated_grid, nan=0)
        interpolated_grid[interpolated_grid < 0] = 0

        return interpolated_grid

    def _calculate_and_interpolate_plot_data(
        self, trees, plots, grid_x, grid_y, plot_type
    ):
        """
        Calculate and interpolate tree density based on plot type (all or occupied).
        """
        if plot_type == "all":
            merge_type = "left"
            method = "linear"
            column_to_interpolate = "TPA_UNADJ"
        elif plot_type == "occupied":
            merge_type = "right"
            method = "nearest"
            column_to_interpolate = "TM_CN"
        else:
            raise ValueError("Invalid plot type. Must be 'all' or 'occupied'.")

        plots_with_data_col = self._calculate_per_plot_tree_density(
            plots, trees, merge_type
        )
        interpolated_plot_data = self._interpolate_plot_data_to_grid(
            plots_with_data_col, column_to_interpolate, grid_x, grid_y, method
        )
        return interpolated_plot_data

    # @staticmethod
    # def _interpolate_unstructured_plot_data_to_structured_grid(
    #     plots, data, grid_x, grid_y, method="nearest"
    # ):
    #     """
    #     Interpolated unstructured plot data to a structured grid.
    #     """
    #     grid = griddata(
    #         (plots.geometry.x, plots.geometry.y), data, (grid_x, grid_y), method=method
    #     )
    #     if method != "nearest":
    #         grid[grid < 0] = 0
    #         grid = np.nan_to_num(grid)
    #     return grid
    #
    # @staticmethod
    # def _generate_tree_locations_in_grid(
    #     count_grid, plot_grid, grid_resolution, grid_x, grid_y
    # ):
    #     """
    #     Generates tree locations in a grid of plots.
    #     """
    #     # Calculate indices for each tree
    #     tree_indices = np.repeat(np.arange(count_grid.size), count_grid.ravel())
    #
    #     # Calculate the cell indices for each tree
    #     cell_i, cell_j = np.unravel_index(tree_indices, count_grid.shape)
    #
    #     # Get the plot ID for each tree
    #     plot_id_total = plot_grid.ravel()[tree_indices]
    #
    #     # Generate random offsets within the cell for each tree
    #     num_tree_indices = len(tree_indices)
    #     random_offsets_x = np.random.uniform(
    #         -grid_resolution / 2, grid_resolution / 2, num_tree_indices
    #     )
    #     random_offsets_y = np.random.uniform(
    #         -grid_resolution / 2, grid_resolution / 2, num_tree_indices
    #     )
    #
    #     # Calculate the actual coordinates of each tree
    #     x_coords = grid_x[cell_i, cell_j] + random_offsets_x
    #     y_coords = grid_y[cell_i, cell_j] + random_offsets_y
    #
    #     # Create DataFrame
    #     tree_locations = pd.DataFrame(
    #         {"x": x_coords, "y": y_coords, "PlotID": plot_id_total}
    #     )
    #
    #     return tree_locations


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
