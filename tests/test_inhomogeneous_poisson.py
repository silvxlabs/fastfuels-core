"""
Tests for the InhomogeneousPoissonProcess class.
"""

# Core imports
from pathlib import Path
from collections import defaultdict

import numpy as np

# Internal imports
from fastfuels_core.point_process import PointProcess, InhomogeneousPoissonProcess

# External imports
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestGenerateTreeLocations:
    def test_basic_case(self):
        roi_gdf = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
        plots_gdf = gpd.read_file(TEST_DATA_PATH / "plots_test_data.geojson")
        trees_df = pd.read_parquet(TEST_DATA_PATH / "trees_test_data.parquet")
        point_process = InhomogeneousPoissonProcess("inhomogeneous_poisson")
        point_process.generate_tree_locations(roi_gdf, trees_df, plots_gdf)
        print()


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
    plots_data["PlotID"] = plots_data.index
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
        assert len(occupied_plots_with_tree_density) < len(self.plots_data)

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
            plot_id = plot["PlotID"]
            if plot_id not in plot_tpa_sum:
                assert (
                    plot_id
                    not in occupied_plots_with_tree_density[self.plot_id_col].values
                )
            else:
                assert (
                    plot_id in occupied_plots_with_tree_density[self.plot_id_col].values
                )


class TestInterpolatePlotDensityToGrid:
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
        interpolated_grid = (
            InhomogeneousPoissonProcess._interpolate_plot_density_to_grid(
                plots, plots["PlotID"], x_grid, y_grid, method="nearest"
            )
        )

        # Assert that each quadrant has the correct plotID
        assert interpolated_grid.shape == (4, 4)
        assert np.allclose(interpolated_grid[:2, :2], 1)
        assert np.allclose(interpolated_grid[:2, 2:], 2)
        assert np.allclose(interpolated_grid[2:, :2], 3)
        assert np.allclose(interpolated_grid[2:, 2:], 4)

class TestInterpolatePlotDensity