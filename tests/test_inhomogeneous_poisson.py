"""
Tests for the InhomogeneousPoissonProcess class.
"""

# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.point_process import InhomogeneousPoissonProcess

# External imports
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"
TEST_FIGS_PATH = TEST_PATH / "figs"
TEST_FIGS_PATH.mkdir(exist_ok=True)


class TestGenerateTreeLocations:
    def test_basic_case(self):
        roi_gdf = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
        plots_gdf = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
        trees_df = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")

        roi_gdf = roi_gdf.to_crs(roi_gdf.estimate_utm_crs())
        plots_gdf = plots_gdf.to_crs(plots_gdf.estimate_utm_crs())

        point_process = InhomogeneousPoissonProcess("inhomogeneous_poisson")
        point_process.generate_tree_locations(roi_gdf, trees_df, plots_gdf)


class TestGetStructuredCoordsGrid:
    def test_play_data_10m_resolution(self):
        polygon = Polygon(
            [
                (0, 0),
                (0, 100),
                (100, 100),
                (100, 0),
            ]
        )
        roi = gpd.GeoDataFrame(geometry=[polygon])
        x, y = InhomogeneousPoissonProcess._create_structured_coords_grid(roi, 10)

        assert x.shape == (10, 10)
        assert y.shape == (10, 10)
        assert x[0, 0] == 5
        assert y[0, 0] == 5
        assert x[9, 9] == 95
        assert y[9, 9] == 95

    def test_play_data_1m_resolution(self):
        polygon = Polygon(
            [
                (0, 0),
                (0, 100),
                (100, 100),
                (100, 0),
            ]
        )
        roi = gpd.GeoDataFrame(geometry=[polygon])
        x, y = InhomogeneousPoissonProcess._create_structured_coords_grid(roi, 1)

        assert x.shape == (100, 100)
        assert y.shape == (100, 100)
        assert x[0, 0] == 0.5
        assert y[0, 0] == 0.5
        assert x[99, 99] == 99.5
        assert y[99, 99] == 99.5


class TestCalculateTreeDensityByPlot:
    tree_data = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
    plots_data = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "TM_CN"

    def test_all_plots(self):
        """
        This test case calculates the tree density for all plots in the plots
        data. The sum of the tree density for each plot in the plots data should
        equal the sum of the TPA_UNADJ for each tree in the tree data that
        belongs to that plot.

        This test case accounts for ALL PLOTS, even those with 0 density.
        """
        all_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_per_plot_tree_density(
                self.plots_data, self.tree_data, merge_type="left"
            )
        )

        # Make sure that all plots are in the output
        assert len(all_plots_with_tree_density) == len(self.plots_data)

        # Sum the TPA_UNADJ for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            tpa_unadj = tree["TPA_UNADJ"] if tree["TPA_UNADJ"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + tpa_unadj

        # Compare the calculated tree density to the expected tree density
        for _, plot in all_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            if plot_id not in plot_tpa_sum:
                plot_tpa_sum[plot_id] = 0
            assert np.allclose(plot["TPA_UNADJ"], plot_tpa_sum[plot_id])

    def test_occupied_plots(self):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA_UNADJ
        for each tree in the tree data that belongs to that plot.

        This test case only accounts for OCCUPIED PLOTS, i.e. those with a
        tree density > 0. Plots without trees should not be in the output.
        """
        occupied_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_per_plot_tree_density(
                self.plots_data, self.tree_data, merge_type="right"
            )
        )

        # Not all plots are occupied, so the output should be less than the
        # number of plots in the plots data
        assert 0 < len(occupied_plots_with_tree_density) < len(self.plots_data)

        # Sum the TPA_UNADJ for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            tpa_unadj = tree["TPA_UNADJ"] if tree["TPA_UNADJ"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + tpa_unadj

        # Compare the calculated tree density to the expected tree density
        for _, plot in occupied_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            assert np.allclose(plot["TPA_UNADJ"], plot_tpa_sum[plot_id])

        # Make sure that plots with 0 density are not in the output, plots with
        # tree density > 0 should be in the output
        for _, plot in self.plots_data.iterrows():
            plot_id = plot[self.plot_id_col]
            if plot_id not in plot_tpa_sum:
                assert (
                    plot_id
                    not in occupied_plots_with_tree_density[self.plot_id_col].values
                )
            else:
                assert (
                    plot_id in occupied_plots_with_tree_density[self.plot_id_col].values
                )


class TestCalculateAllPlotsTreeDensity:
    tree_data = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
    plots_data = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "TM_CN"

    def test_all_plots(self):
        """
        This test case calculates the tree density for all plots in the plots
        data. The sum of the tree density for each plot in the plots data should
        equal the sum of the TPA_UNADJ for each tree in the tree data that
        belongs to that plot.

        This test case accounts for ALL PLOTS, even those with 0 density.
        """
        all_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_all_plots_tree_density(
                InhomogeneousPoissonProcess, self.plots_data, self.tree_data
            )
        )

        # Make sure that all plots are in the output
        assert len(all_plots_with_tree_density) == len(self.plots_data)

        # Sum the TPA_UNADJ for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            tpa_unadj = tree["TPA_UNADJ"] if tree["TPA_UNADJ"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + tpa_unadj

        # Compare the calculated tree density to the expected tree density
        for _, plot in all_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            if plot_id not in plot_tpa_sum:
                plot_tpa_sum[plot_id] = 0
            assert np.allclose(plot["TPA_UNADJ"], plot_tpa_sum[plot_id])


class TestCalculateOccupiedPlotsTreeDensity:
    tree_data = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
    plots_data = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "TM_CN"

    def test_occupied_plots(self):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA_UNADJ
        for each tree in the tree data that belongs to that plot.

        This test case only accounts for OCCUPIED PLOTS, i.e. those with a
        tree density > 0. Plots without trees should not be in the output.
        """
        occupied_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_occupied_plots_tree_density(
                InhomogeneousPoissonProcess, self.plots_data, self.tree_data
            )
        )

        # Not all plots are occupied, so the output should be less than the
        # number of plots in the plots data
        assert 0 < len(occupied_plots_with_tree_density) < len(self.plots_data)

        # Sum the TPA_UNADJ for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            tpa_unadj = tree["TPA_UNADJ"] if tree["TPA_UNADJ"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + tpa_unadj

        # Compare the calculated tree density to the expected tree density
        for _, plot in occupied_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            assert np.allclose(plot["TPA_UNADJ"], plot_tpa_sum[plot_id])

        # Make sure that plots with 0 density are not in the output, plots with
        # tree density > 0 should be in the output
        for _, plot in self.plots_data.iterrows():
            plot_id = plot[self.plot_id_col]
            if plot_id not in plot_tpa_sum:
                assert (
                    plot_id
                    not in occupied_plots_with_tree_density[self.plot_id_col].values
                )


class TestInterpolateDataToGrid:
    def test_nearest_play_data(self):
        # Create a 4x4 grid
        x_coords = np.array([0, 1, 2, 3])
        y_coords = np.array([0, 1, 2, 3])
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Create a GeoDataFrame with 4 plots. Each plot is a corner of the grid
        upper_left = Point(0, 0)
        upper_right = Point(3, 0)
        lower_left = Point(0, 3)
        lower_right = Point(3, 3)
        plots = gpd.GeoDataFrame(
            geometry=[upper_left, upper_right, lower_left, lower_right]
        )
        plots["PlotID"] = [1, 2, 3, 4]

        # Use nearest neighbor to interpolate the plotIDs to the grid
        data_to_interpolate = plots["PlotID"]
        interpolated_grid = InhomogeneousPoissonProcess._interpolate_data_to_grid(
            plots, data_to_interpolate, x_grid, y_grid, method="nearest"
        )

        # Assert that each quadrant has the correct plotID
        assert interpolated_grid.shape == (4, 4)
        assert np.allclose(interpolated_grid[:2, :2], 1)
        assert np.allclose(interpolated_grid[:2, 2:], 2)
        assert np.allclose(interpolated_grid[2:, :2], 3)
        assert np.allclose(interpolated_grid[2:, 2:], 4)


class TestInterpolateTreeDensityToGrid:
    roi = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
    roi = roi.to_crs(roi.estimate_utm_crs())
    tree_data = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
    plots_data = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "TM_CN"

    grid_resolution = 5
    grid_cell_area = grid_resolution**2

    create_visualization = False
    show_visualization = False

    def test_interpolate_tree_density(self):
        """
        This test case calculates the tree density for all plots in the plots
        and interpolates tree density to a structured grid.
        """
        # Build the structured grid for interpolation
        grid_x, grid_y = InhomogeneousPoissonProcess._create_structured_coords_grid(
            self.roi, self.grid_resolution
        )
        process = InhomogeneousPoissonProcess("inhomogeneous_poisson")
        tree_density_grid = (
            InhomogeneousPoissonProcess._interpolate_tree_density_to_grid(
                process,
                self.tree_data,
                self.plots_data,
                grid_x,
                grid_y,
                self.grid_resolution,
            )
        )

        # Check that the output is a 2D array with the same shape as the grid
        assert len(tree_density_grid.shape) == 2
        assert tree_density_grid.shape == grid_x.shape
        assert tree_density_grid.shape == grid_y.shape

        # Do this for a list of grid cell resolutions. All density sums should
        # be close to eachother. I.e. the density sum should not change much
        # when the grid cell resolution changes.
        for resolution in [0.5, 1, 2, 5, 10, 15, 30]:
            grid_x, grid_y = InhomogeneousPoissonProcess._create_structured_coords_grid(
                self.roi, resolution
            )
            interpolated_grid = (
                InhomogeneousPoissonProcess._interpolate_tree_density_to_grid(
                    process,
                    self.tree_data,
                    self.plots_data,
                    grid_x,
                    grid_y,
                    resolution,
                )
            )
            assert np.allclose(
                np.sum(interpolated_grid),
                0.22,
                atol=0.02,
            )

        if self.create_visualization or self.show_visualization:
            self.visualize_data(grid_x, grid_y, tree_density_grid)

    def visualize_data(self, grid_x, grid_y, all_plots_grid):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(
            all_plots_grid,
            origin="lower",
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        )
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Tree Intensity Interpolated to Grid")

        if self.create_visualization:
            plt.savefig(TEST_FIGS_PATH / "tree_intensity_grid.png")
        if self.show_visualization:
            plt.show()


class TestInterpolatePlotIdToGrid:
    roi = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
    roi = roi.to_crs(roi.estimate_utm_crs())
    tree_data = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
    plots_data = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "TM_CN"

    grid_resolution = 5
    grid_cell_area = grid_resolution**2

    create_visualization = False
    show_visualization = False

    def test_interpolate_plot_id(self):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA_UNADJ
        for each tree in the tree data that belongs to that plot.

        This test case only accounts for OCCUPIED PLOTS, i.e. those with a
        tree density > 0. Plots without trees should not be in the output.
        """
        # Build the structured grid for interpolation
        grid_x, grid_y = InhomogeneousPoissonProcess._create_structured_coords_grid(
            self.roi, self.grid_resolution
        )
        process = InhomogeneousPoissonProcess("inhomogeneous_poisson")
        plot_id_grid = InhomogeneousPoissonProcess._interpolate_plot_id_to_grid(
            process,
            self.tree_data,
            self.plots_data,
            grid_x,
            grid_y,
            self.grid_resolution,
        )

        # Check that the output is a 2D array with the same shape as the grid
        assert len(plot_id_grid.shape) == 2
        assert plot_id_grid.shape == grid_x.shape
        assert plot_id_grid.shape == grid_y.shape

        # Get the occupied plots from the plots data
        occupied_plots_df = (
            InhomogeneousPoissonProcess._calculate_per_plot_tree_density(
                self.plots_data, self.tree_data, merge_type="right"
            )
        )
        occupied_plots_df = occupied_plots_df.to_crs(self.roi.crs)
        plots_inside_roi = gpd.sjoin(
            occupied_plots_df, self.roi, how="inner", predicate="within"
        )
        occupied_plots_set = set(plot_id_grid.flatten().tolist())

        # Check that all plots in the output are within the ROI
        for _, plot in plots_inside_roi.iterrows():
            plot_id = plot[self.plot_id_col]
            assert plot_id in occupied_plots_set
        for plot_id in occupied_plots_set:
            assert plot_id in occupied_plots_df[self.plot_id_col].values

        if self.create_visualization or self.show_visualization:
            self.visualize_data(grid_x, grid_y, plot_id_grid, occupied_plots_df)

    def visualize_data(self, grid_x, grid_y, occupied_plots_grid, occupied_plots_df):
        # Create a consistent mapping of plot IDs to randomized indices
        plot_ids = occupied_plots_df[self.plot_id_col].unique()
        plot_indices = list(range(len(plot_ids)))
        plot_id_to_index = {
            pid: plot_indices[i] for i, pid in enumerate(sorted(plot_ids))
        }
        num_colors = len(plot_id_to_index)

        # Create a colormap
        colors = plt.cm.get_cmap("Set3", num_colors)
        cmap = mcolors.ListedColormap(colors.colors)

        # Map grid values to indices based on plot_id_to_index
        grid_color_indices = np.vectorize(plot_id_to_index.get)(occupied_plots_grid)
        grid_color_indices[np.isnan(grid_color_indices)] = -1  # Handle NaN values

        # Display the grid
        plt.figure(figsize=(10, 8))
        plt.imshow(
            grid_color_indices,
            cmap=cmap,
            origin="lower",
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        )
        plt.colorbar()

        # Overlay plot locations
        for plot_id, idx in plot_id_to_index.items():
            plot_color = colors.colors[idx]
            plot_subset = occupied_plots_df[
                occupied_plots_df[self.plot_id_col] == plot_id
            ]
            x = plot_subset.geometry.centroid.x
            y = plot_subset.geometry.centroid.y
            plt.scatter(
                x, y, color=plot_color, edgecolor="black", label=f"Plot {plot_id}"
            )

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Plot ID Interpolated to Grid")

        # Create custom legend
        plt.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Plot {pid}",
                    markerfacecolor=colors.colors[plot_id_to_index[pid]],
                    markersize=10,
                )
                for pid in plot_ids
            ]
        )

        if self.create_visualization:
            plt.savefig(TEST_FIGS_PATH / "plot_id_grid.png")
        if self.show_visualization:
            plt.show()
