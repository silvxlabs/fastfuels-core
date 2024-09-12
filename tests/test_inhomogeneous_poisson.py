"""
Tests for the InhomogeneousPoissonProcess class.
"""

# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.point_process import InhomogeneousPoissonProcess

# External imports
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

# TODO: Revisit tests with additional data


class TestGenerateTreeLocations:
    def test_basic_case(self):
        roi_gdf = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
        plots_gdf = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
        trees_df = pd.read_parquet(
            TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
        )

        roi_gdf = roi_gdf.to_crs(roi_gdf.estimate_utm_crs())
        plots_gdf = plots_gdf.to_crs(plots_gdf.estimate_utm_crs())

        point_process = InhomogeneousPoissonProcess()
        point_process.simulate(roi_gdf, trees_df, plots_gdf)


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
        assert y[0, 0] == 95
        assert x[9, 9] == 95
        assert y[9, 9] == 5

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
        assert y[0, 0] == 99.5
        assert x[99, 99] == 99.5
        assert y[99, 99] == 0.5


class TestCalculateTreeDensityByPlot:
    def test_treemap_2016(self):
        tree_data = pd.read_parquet(
            TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
        )
        plot_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
        plot_data = plot_data.to_crs(plot_data.estimate_utm_crs())

        # Calculate the tree density for all plots
        self._test_all_plots(tree_data, plot_data)

        # Calculate the tree density for occupied plots
        self._test_occupied_plots(tree_data, plot_data)

    @staticmethod
    def _test_all_plots(tree_data, plot_data):
        """
        This test case calculates the tree density for all plots in the plots
        data. The sum of the tree density for each plot in the plots data should
        equal the sum of the TPA for each tree in the tree data that
        belongs to that plot.

        This test case accounts for ALL PLOTS, even those with 0 density.
        """
        all_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_per_plot_tree_density(
                plot_data, tree_data, merge_type="left"
            )
        )

        # Make sure that all plots are in the output
        assert len(all_plots_with_tree_density) == len(plot_data)

        # Sum the TPA for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in tree_data.iterrows():
            plot_id = tree["PLOT_ID"]
            TPA = tree["TPA"] if tree["TPA"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + TPA

        # Compare the calculated tree density to the expected tree density
        for _, plot in all_plots_with_tree_density.iterrows():
            plot_id = plot["PLOT_ID"]
            if plot_id not in plot_tpa_sum:
                plot_tpa_sum[plot_id] = 0
            assert np.allclose(plot["TPA"], plot_tpa_sum[plot_id])

    @staticmethod
    def _test_occupied_plots(tree_data, plot_data):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA
        for each tree in the tree data that belongs to that plot.

        This test case only accounts for OCCUPIED PLOTS, i.e. those with a
        tree density > 0. Plots without trees should not be in the output.
        """
        occupied_plots_with_tree_density = (
            InhomogeneousPoissonProcess._calculate_per_plot_tree_density(
                plot_data, tree_data, merge_type="right"
            )
        )

        # Not all plots are occupied, so the output should be less than the
        # number of plots in the plots data
        assert 0 < len(occupied_plots_with_tree_density) < len(plot_data)

        # Sum the TPA for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in tree_data.iterrows():
            plot_id = tree["PLOT_ID"]
            TPA = tree["TPA"] if tree["TPA"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + TPA

        # Compare the calculated tree density to the expected tree density
        for _, plot in occupied_plots_with_tree_density.iterrows():
            plot_id = plot["PLOT_ID"]
            assert np.allclose(plot["TPA"], plot_tpa_sum[plot_id])

        # Make sure that plots with 0 density are not in the output, plots with
        # tree density > 0 should be in the output
        for _, plot in plot_data.iterrows():
            plot_id = plot["PLOT_ID"]
            if plot_id not in plot_tpa_sum:
                assert plot_id not in occupied_plots_with_tree_density["PLOT_ID"].values
            else:
                assert plot_id in occupied_plots_with_tree_density["PLOT_ID"].values


class TestCalculateAllPlotsTreeDensity:
    tree_data = pd.read_parquet(
        TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
    )
    plots_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "PLOT_ID"

    def test_all_plots(self):
        """
        This test case calculates the tree density for all plots in the plots
        data. The sum of the tree density for each plot in the plots data should
        equal the sum of the TPA for each tree in the tree data that
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

        # Sum the TPA for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            TPA = tree["TPA"] if tree["TPA"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + TPA

        # Compare the calculated tree density to the expected tree density
        for _, plot in all_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            if plot_id not in plot_tpa_sum:
                plot_tpa_sum[plot_id] = 0
            assert np.allclose(plot["TPA"], plot_tpa_sum[plot_id])


class TestCalculateOccupiedPlotsTreeDensity:
    tree_data = pd.read_parquet(
        TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
    )
    plots_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "PLOT_ID"

    def test_occupied_plots(self):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA
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

        # Sum the TPA for each tree belonging to a plot
        plot_tpa_sum = {}
        for _, tree in self.tree_data.iterrows():
            plot_id = tree[self.plot_id_col]
            TPA = tree["TPA"] if tree["TPA"] > 0 else 0
            plot_tpa_sum[plot_id] = plot_tpa_sum.get(plot_id, 0) + TPA

        # Compare the calculated tree density to the expected tree density
        for _, plot in occupied_plots_with_tree_density.iterrows():
            plot_id = plot[self.plot_id_col]
            assert np.allclose(plot["TPA"], plot_tpa_sum[plot_id])

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
    tree_data = pd.read_parquet(
        TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
    )
    plots_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())

    grid_resolution = 15
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
        process = InhomogeneousPoissonProcess()
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
        intensity_sums = []
        for resolution in [0.5, 1, 2, 5, 10, 15, 30]:
            (
                grid_x,
                grid_y,
            ) = InhomogeneousPoissonProcess._create_structured_coords_grid(
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
            intensity_sums.append(interpolated_grid.sum())

        # Assert that the density sums are within a few trees of eachother
        assert np.allclose(intensity_sums, intensity_sums[0], atol=4)

        if self.create_visualization or self.show_visualization:
            self.visualize_data(grid_x, grid_y, tree_density_grid)

    def visualize_data(self, grid_x, grid_y, all_plots_grid):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(
            all_plots_grid,
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
    tree_data = pd.read_parquet(
        TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
    )
    plots_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
    plots_data = plots_data.to_crs(plots_data.estimate_utm_crs())
    plot_id_col = "PLOT_ID"

    grid_resolution = 15
    grid_cell_area = grid_resolution**2

    create_visualization = False
    show_visualization = False

    def test_interpolate_plot_id(self):
        """
        This test case calculates the tree density for occupied plots (i.e.
        those containing trees) in the plots data. The sum of the tree density
        for each plot in the plots data should equal the sum of the TPA
        for each tree in the tree data that belongs to that plot.

        This test case only accounts for OCCUPIED PLOTS, i.e. those with a
        tree density > 0. Plots without trees should not be in the output.
        """
        # Build the structured grid for interpolation
        grid_x, grid_y = InhomogeneousPoissonProcess._create_structured_coords_grid(
            self.roi, self.grid_resolution
        )
        process = InhomogeneousPoissonProcess()
        plot_id_grid = InhomogeneousPoissonProcess._interpolate_plot_id_to_grid(
            process,
            self.tree_data,
            self.plots_data,
            grid_x,
            grid_y,
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
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        )
        # plt.colorbar()

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
        plt.title("PLOT_ID Interpolated to Grid")

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


class TestGenerateTreeCounts:
    def test_null_case(self):
        density_grid = np.zeros((10, 10))
        tree_counts = InhomogeneousPoissonProcess._generate_tree_counts(density_grid)
        assert np.allclose(tree_counts, 0)

    def test_play_data(self):
        density_grid = np.array([0, 1, 2, 3])
        tree_counts = InhomogeneousPoissonProcess._generate_tree_counts(density_grid)
        assert tree_counts[0] == 0
        assert tree_counts[:-1].sum() > 0

    def test_seed(self):
        seed = 42
        np.random.seed(seed)
        density_grid = np.array([100, 100, 100, 100])
        tree_counts = InhomogeneousPoissonProcess._generate_tree_counts(density_grid)
        assert np.allclose(tree_counts, np.array([96, 107, 88, 103]))

    def test_treemap_2016_data(self):
        roi = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
        roi = roi.to_crs(roi.estimate_utm_crs())
        plot_data = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
        plot_data = plot_data.to_crs(plot_data.estimate_utm_crs())
        tree_data = pd.read_parquet(
            TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
        )
        self._test_real_data(roi, 5, tree_data, plot_data)

    def _test_real_data(self, roi, resolution, tree_data, plot_data):
        grid_x, grid_y = InhomogeneousPoissonProcess._create_structured_coords_grid(
            roi, resolution
        )
        process = InhomogeneousPoissonProcess()
        # process._set_seed(124678)
        tree_density_grid = (
            InhomogeneousPoissonProcess._interpolate_tree_density_to_grid(
                process,
                tree_data,
                plot_data,
                grid_x,
                grid_y,
                resolution,
            )
        )

        # Draw tree counts from the tree density grid
        tree_counts_sums = []
        for _ in range(100):
            tree_counts = InhomogeneousPoissonProcess._generate_tree_counts(
                tree_density_grid
            )
            tree_counts_sums.append(tree_counts.sum())

        # Assert that tree density sum is in the range of tree counts sums
        tree_counts_sums = np.array(tree_counts_sums)
        tree_density_sum = tree_density_grid.sum()
        assert tree_counts_sums.min() < tree_density_sum < tree_counts_sums.max()


class TestCalculateTreeIndices:
    """
    Tests for the _get_flattened_tree_indices method of the
    InhomogeneousPoissonProcess class.
    """

    process = InhomogeneousPoissonProcess()

    def test_empty_grid(self):
        """
        Test with an empty grid (all zeros). Should return an empty array.
        """
        count_grid = np.zeros((5, 5), dtype=int)
        expected_indices = np.array([], dtype=int)
        calculated_indices = self.process._get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(calculated_indices, expected_indices)

    def test_uniform_grid(self):
        """
        Test with a uniform grid (same count in each cell). Should return indices
        with equal counts for each cell.
        """
        count = 3
        count_grid = np.full((4, 4), count)
        expected_indices = self._get_expected_indices(count_grid)
        calculated_indices = self.process._get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(calculated_indices, expected_indices)

    def test_varied_grid(self):
        """
        Test with a varied count grid. Should return indices that reflect
        the distribution of tree counts.
        """
        count_grid = np.array([[1, 2], [0, 3]])
        expected_indices = self._get_expected_indices(count_grid)
        calculated_indices = self.process._get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(calculated_indices, expected_indices)

    def test_large_grid(self):
        """
        Test with a larger grid to ensure scalability.
        """
        count_grid = np.random.randint(0, 5, (10, 10))
        expected_indices = self._get_expected_indices(count_grid)
        calculated_indices = self.process._get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(calculated_indices, expected_indices)

    @staticmethod
    def _get_expected_indices(grid):
        """
        This function manually creates the expected indices array for a given
        count grid without the fancy numpy tricks. This is used to test the
        logic of the _get_flattened_tree_indices method.

        The way this works is that for each cell in the grid we get the count
        value at that cell. Then, there should be an entry in that list with
        the index of that cell repeated count times.
        """
        indices_list = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                index = i * grid.shape[1] + j
                count = grid[i, j]
                indices_list.extend([index] * count)
        return np.array(indices_list)


class TestGetCellIndices:
    process = InhomogeneousPoissonProcess()

    def test_get_cell_indices_case_1(self):
        # Mock data
        count_grid = np.array([[1, 3], [2, 4]])
        tree_indices = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3])

        # Expected result
        expected_result = self._get_expected_result(count_grid)

        # Call the method
        result = self.process._get_grid_cell_indices(tree_indices, count_grid)

        # Assert the result
        assert np.array_equal(
            result, expected_result
        ), "The _get_cell_indices method does not return the expected result."

    def test_get_cell_indices_case_2(self):
        # Mock data
        count_grid = np.array([[2, 2], [2, 2]])
        tree_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        # Expected result
        expected_result = self._get_expected_result(count_grid)

        # Call the method
        result = self.process._get_grid_cell_indices(tree_indices, count_grid)

        # Assert the result
        assert np.array_equal(
            result, expected_result
        ), "The _get_cell_indices method does not return the expected result."

    @staticmethod
    def _get_expected_result(count_grid):
        """
        This function manually creates the expected result for a given
        count grid and tree indices without the fancy numpy tricks. This is
        used to test the logic of the _get_cell_indices method.
        """
        x_indices = []
        y_indices = []
        for i in range(count_grid.shape[0]):
            for j in range(count_grid.shape[1]):
                for _ in range(count_grid[i, j]):
                    x_indices.append(i)
                    y_indices.append(j)
        return np.array(x_indices), np.array(y_indices)


class TestCalculateTreeCoordinates:
    process = InhomogeneousPoissonProcess()
    process._set_seed(42)

    def test_basic_case(self):
        # Build mock data.
        # 1x2 grid of plots
        # 1 tree in plot 0, 2 trees in plot 1
        cell_i = np.array([0, 0, 0])
        cell_j = np.array([0, 1, 1])
        grid_resolution = 1
        x = np.array([0.5, 1.5])
        y = np.array([0.5, 0.5])
        grid_x, grid_y = np.meshgrid(x, y)

        # Call the method
        result = self.process._calculate_tree_coordinates(
            cell_i, cell_j, grid_resolution, grid_x, grid_y
        )

        # Make sure that the results make sense
        x_tree_coords = result[0]
        y_tree_coords = result[1]
        assert x_tree_coords.shape == y_tree_coords.shape
        assert x_tree_coords.shape == cell_i.shape
        assert y_tree_coords.shape == cell_j.shape

        # First tree should be in the range of the first plot
        assert 0 <= x_tree_coords[0] <= 1
        assert 0 <= y_tree_coords[0] <= 1

        # Second tree should be in the range of the second plot
        assert 1 <= x_tree_coords[1] <= 2
        assert 0 <= y_tree_coords[1] <= 1

        # Third tree should be in the range of the second plot
        assert 1 <= x_tree_coords[2] <= 2
        assert 0 <= y_tree_coords[2] <= 1


class TestCountTreesPerPlot:
    process = InhomogeneousPoissonProcess()
    plot_id_list = [1, 1, 2, 2, 2, 3]
    num_trees = len(plot_id_list)
    trees = pd.DataFrame({"PLOT_ID": plot_id_list})
    tree_locations = pd.DataFrame(
        {
            "X": np.random.rand(num_trees),
            "Y": np.random.rand(num_trees),
            "PLOT_ID": plot_id_list,
        }
    )

    def test_count_trees_per_plot(self):
        num_trees_per_plot = self.process._count_trees_per_plot(
            self.trees, self.tree_locations
        )
        assert num_trees_per_plot[1] == 2
        assert num_trees_per_plot[2] == 3
        assert num_trees_per_plot[3] == 1

    def test_count_trees_per_plot_empty_trees(self):
        empty_trees = pd.DataFrame(columns=["PLOT_ID"])
        num_trees_per_plot = self.process._count_trees_per_plot(self.trees, empty_trees)
        assert num_trees_per_plot[1] == 0
        assert num_trees_per_plot[2] == 0
        assert num_trees_per_plot[3] == 0


class TestSampleTreesWithinPlots:
    process = InhomogeneousPoissonProcess()
    trees = pd.DataFrame(
        {
            "PLOT_ID": [1, 1, 2, 2, 2, 3],
            "TREE_ID": [1, 2, 3, 4, 5, 6],
            "TPA": [1, 2, 3, 4, 5, 6],
        }
    )
    num_trees_per_plot = pd.Series([2, 3, 1], index=[1, 2, 3])

    def test_sample_trees_within_plots(self):
        sampled_trees = self.process._sample_trees_within_plots(
            self.trees, self.num_trees_per_plot
        )
        assert len(sampled_trees) == self.num_trees_per_plot.sum()

        # Make sure that each sampled tree belongs to the correct plot
        for _, tree in sampled_trees.iterrows():
            sampled_plot_id = tree["PLOT_ID"]
            sampled_tree_id = tree["TREE_ID"]
            original_tree = self.trees[self.trees["TREE_ID"] == sampled_tree_id]
            original_plot_id = original_tree["PLOT_ID"].values[0]
            assert sampled_plot_id == original_plot_id

    def test_sample_trees_within_plots_empty(self):
        empty_trees = pd.DataFrame(columns=["PLOT_ID", "TPA"])
        sampled_trees = self.process._sample_trees_within_plots(
            empty_trees, self.num_trees_per_plot
        )
        assert len(sampled_trees) == 0


class TestAssignTreeLocations:
    process = InhomogeneousPoissonProcess()
    sampled_trees = pd.DataFrame(
        {"PLOT_ID": [1, 1, 2, 2, 2, 3], "TPA": [1, 2, 3, 4, 5, 6]}
    )
    tree_locations = pd.DataFrame(
        {"X": np.random.rand(6), "Y": np.random.rand(6), "PLOT_ID": [1, 1, 2, 2, 2, 3]}
    )
