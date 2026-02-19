"""
Tests for the inhomogeneous Poisson point process functions.
"""

# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.point_process import (
    run_point_process,
    inhomogeneous_poisson_process,
    _create_structured_coords_grid,
    _calculate_per_plot_tree_density,
    _interpolate_data_to_grid,
    _interpolate_tree_density_to_grid,
    _interpolate_plot_id_to_grid,
    _generate_tree_counts,
    _get_flattened_tree_indices,
    _get_grid_cell_indices,
    _calculate_tree_coordinates,
    _count_trees_per_plot,
    _sample_trees_within_plots,
    _assign_tree_locations,
    _drop_trees_outside_roi_bounds,
    _build_output_meta,
    _generate_tile_indices,
    _process_single_tile,
)
from fastfuels_core.trees import TreeSample, TreePopulation

# External imports
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"
FIGURES_PATH = TEST_PATH / "figures"


@pytest.fixture
def real_data():
    """Load the real TreeMap 2016 test data."""
    roi = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
    roi = roi.to_crs(roi.estimate_utm_crs())
    plots = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
    plots = plots.to_crs(plots.estimate_utm_crs())
    trees = pd.read_parquet(
        TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
    )
    return roi, trees, plots


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCreateStructuredCoordsGrid:
    def test_shape_and_cell_centers(self):
        bounds = (0, 0, 100, 100)
        x, y = _create_structured_coords_grid(bounds, 10)

        assert x.shape == (10, 10)
        assert y.shape == (10, 10)
        assert x[0, 0] == 5
        assert y[0, 0] == 95
        assert x[9, 9] == 95
        assert y[9, 9] == 5


class TestCalculatePerPlotTreeDensity:
    def test_treemap_2016_all_plots(self, real_data):
        _, trees, plots = real_data
        result = _calculate_per_plot_tree_density(plots, trees, "left")
        assert len(result) == len(plots)

        plot_tpa_sum = {}
        for _, tree in trees.iterrows():
            pid = tree["PLOT_ID"]
            tpa = tree["TPA"] if tree["TPA"] > 0 else 0
            plot_tpa_sum[pid] = plot_tpa_sum.get(pid, 0) + tpa

        for _, plot in result.iterrows():
            pid = plot["PLOT_ID"]
            expected = plot_tpa_sum.get(pid, 0)
            assert np.allclose(plot["TPA"], expected)

    def test_treemap_2016_occupied_plots(self, real_data):
        _, trees, plots = real_data
        result = _calculate_per_plot_tree_density(plots, trees, "right")
        assert 0 < len(result) < len(plots)


class TestInterpolateDataToGrid:
    def test_nearest_play_data(self):
        x_coords = np.array([0, 1, 2, 3])
        y_coords = np.array([0, 1, 2, 3])
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        upper_left = Point(0, 0)
        upper_right = Point(3, 0)
        lower_left = Point(0, 3)
        lower_right = Point(3, 3)
        plots = gpd.GeoDataFrame(
            geometry=[upper_left, upper_right, lower_left, lower_right]
        )
        plots["PlotID"] = [1, 2, 3, 4]

        data_to_interpolate = plots["PlotID"]
        interpolated_grid = _interpolate_data_to_grid(
            plots, data_to_interpolate, x_grid, y_grid, method="nearest"
        )

        assert interpolated_grid.shape == (4, 4)
        assert np.allclose(interpolated_grid[:2, :2], 1)
        assert np.allclose(interpolated_grid[:2, 2:], 2)
        assert np.allclose(interpolated_grid[2:, :2], 3)
        assert np.allclose(interpolated_grid[2:, 2:], 4)


class TestInterpolateTreeDensityToGrid:
    save_fig = True

    def test_interpolate_tree_density(self, real_data):
        roi, trees, plots = real_data
        resolution = 15
        bounds = tuple(roi.total_bounds)
        grid_x, grid_y = _create_structured_coords_grid(bounds, resolution)
        tree_density_grid = _interpolate_tree_density_to_grid(
            trees, plots, grid_x, grid_y, resolution
        )

        assert len(tree_density_grid.shape) == 2
        assert tree_density_grid.shape == grid_x.shape

        intensity_sums = []
        for res in [0.5, 1, 2, 5, 10, 15, 30]:
            gx, gy = _create_structured_coords_grid(bounds, res)
            grid = _interpolate_tree_density_to_grid(trees, plots, gx, gy, res)
            intensity_sums.append(grid.sum())

        assert np.allclose(intensity_sums, intensity_sums[0], atol=4)

        if self.save_fig:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.pcolormesh(grid_x, grid_y, tree_density_grid, shading="auto")
            ax.scatter(
                plots.geometry.x,
                plots.geometry.y,
                c="red",
                s=20,
                zorder=5,
                label="Plots",
            )
            ax.set_title(f"Density grid ({resolution}m) with plot locations")
            ax.set_aspect("equal")
            ax.legend()
            fig.colorbar(im, ax=ax, label="Expected trees per cell")
            plt.tight_layout()
            FIGURES_PATH.mkdir(exist_ok=True)
            plt.savefig(FIGURES_PATH / "density_grid.png", dpi=150)
            plt.close(fig)


class TestInterpolatePlotIdToGrid:
    def test_interpolate_plot_id(self, real_data):
        roi, trees, plots = real_data
        resolution = 15
        bounds = tuple(roi.total_bounds)
        grid_x, grid_y = _create_structured_coords_grid(bounds, resolution)
        plot_id_grid = _interpolate_plot_id_to_grid(trees, plots, grid_x, grid_y)

        assert len(plot_id_grid.shape) == 2
        assert plot_id_grid.shape == grid_x.shape


class TestGenerateTreeCounts:
    def test_zero_density_produces_zero_counts(self):
        rng = np.random.default_rng(0)
        density_grid = np.zeros((10, 10))
        tree_counts = _generate_tree_counts(density_grid, rng)
        assert np.allclose(tree_counts, 0)

    def test_seed_reproducibility(self):
        density_grid = np.array([100, 100, 100, 100])
        rng1 = np.random.default_rng(42)
        counts1 = _generate_tree_counts(density_grid, rng1)
        rng2 = np.random.default_rng(42)
        counts2 = _generate_tree_counts(density_grid, rng2)
        np.testing.assert_array_equal(counts1, counts2)

    def test_treemap_2016_data(self, real_data):
        roi, trees, plots = real_data
        resolution = 5
        bounds = tuple(roi.total_bounds)
        grid_x, grid_y = _create_structured_coords_grid(bounds, resolution)
        tree_density_grid = _interpolate_tree_density_to_grid(
            trees, plots, grid_x, grid_y, resolution
        )

        tree_counts_sums = []
        for i in range(100):
            rng = np.random.default_rng(i)
            tree_counts = _generate_tree_counts(tree_density_grid, rng)
            tree_counts_sums.append(tree_counts.sum())

        tree_counts_sums = np.array(tree_counts_sums)
        tree_density_sum = tree_density_grid.sum()
        assert tree_counts_sums.min() < tree_density_sum < tree_counts_sums.max()


class TestGetFlattenedTreeIndices:
    def test_empty_grid(self):
        count_grid = np.zeros((5, 5), dtype=int)
        result = _get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(result, np.array([], dtype=int))

    def test_uniform_grid(self):
        count_grid = np.full((4, 4), 3)
        expected = self._get_expected_indices(count_grid)
        result = _get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(result, expected)

    def test_varied_grid(self):
        count_grid = np.array([[1, 2], [0, 3]])
        expected = self._get_expected_indices(count_grid)
        result = _get_flattened_tree_indices(count_grid)
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def _get_expected_indices(grid):
        indices_list = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                index = i * grid.shape[1] + j
                count = grid[i, j]
                indices_list.extend([index] * count)
        return np.array(indices_list)


class TestGetGridCellIndices:
    def test_case_1(self):
        count_grid = np.array([[1, 3], [2, 4]])
        tree_indices = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
        expected = self._get_expected_result(count_grid)
        result = _get_grid_cell_indices(tree_indices, count_grid)
        assert np.array_equal(result, expected)

    def test_case_2(self):
        count_grid = np.array([[2, 2], [2, 2]])
        tree_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        expected = self._get_expected_result(count_grid)
        result = _get_grid_cell_indices(tree_indices, count_grid)
        assert np.array_equal(result, expected)

    @staticmethod
    def _get_expected_result(count_grid):
        x_indices = []
        y_indices = []
        for i in range(count_grid.shape[0]):
            for j in range(count_grid.shape[1]):
                for _ in range(count_grid[i, j]):
                    x_indices.append(i)
                    y_indices.append(j)
        return np.array(x_indices), np.array(y_indices)


class TestCalculateTreeCoordinates:
    def test_basic_case(self):
        rng = np.random.default_rng(42)
        cell_i = np.array([0, 0, 0])
        cell_j = np.array([0, 1, 1])
        grid_resolution = 1
        x = np.array([0.5, 1.5])
        y = np.array([0.5, 0.5])
        grid_x, grid_y = np.meshgrid(x, y)

        x_coords, y_coords = _calculate_tree_coordinates(
            cell_i, cell_j, grid_resolution, grid_x, grid_y, rng
        )

        assert x_coords.shape == y_coords.shape == cell_i.shape
        assert 0 <= x_coords[0] <= 1
        assert 0 <= y_coords[0] <= 1
        assert 1 <= x_coords[1] <= 2
        assert 0 <= y_coords[1] <= 1
        assert 1 <= x_coords[2] <= 2
        assert 0 <= y_coords[2] <= 1


class TestCountTreesPerPlot:
    def test_count_trees_per_plot(self):
        rng = np.random.default_rng(0)
        plot_id_list = [1, 1, 2, 2, 2, 3]
        trees = pd.DataFrame({"PLOT_ID": plot_id_list})
        tree_locations = pd.DataFrame(
            {
                "X": rng.random(len(plot_id_list)),
                "Y": rng.random(len(plot_id_list)),
                "PLOT_ID": plot_id_list,
            }
        )
        result = _count_trees_per_plot(trees, tree_locations)
        assert result[1] == 2
        assert result[2] == 3
        assert result[3] == 1

    def test_count_trees_per_plot_empty(self):
        trees = pd.DataFrame({"PLOT_ID": [1, 2, 3]})
        empty_locs = pd.DataFrame(columns=["PLOT_ID", "X", "Y"])
        result = _count_trees_per_plot(trees, empty_locs)
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == 0


class TestSampleTreesWithinPlots:
    trees = pd.DataFrame(
        {
            "PLOT_ID": [1, 1, 2, 2, 2, 3],
            "TREE_ID": [1, 2, 3, 4, 5, 6],
            "TPA": [1, 2, 3, 4, 5, 6],
        }
    )
    num_trees_per_plot = pd.Series([2, 3, 1], index=[1, 2, 3])

    def test_sample_trees_within_plots(self):
        rng = np.random.default_rng(42)
        sampled = _sample_trees_within_plots(self.trees, self.num_trees_per_plot, rng)
        assert len(sampled) == self.num_trees_per_plot.sum()

        for _, tree in sampled.iterrows():
            sampled_plot_id = tree["PLOT_ID"]
            sampled_tree_id = tree["TREE_ID"]
            original_tree = self.trees[self.trees["TREE_ID"] == sampled_tree_id]
            original_plot_id = original_tree["PLOT_ID"].values[0]
            assert sampled_plot_id == original_plot_id

    def test_sample_trees_within_plots_empty(self):
        rng = np.random.default_rng(42)
        empty_trees = pd.DataFrame(columns=["PLOT_ID", "TPA"])
        sampled = _sample_trees_within_plots(empty_trees, self.num_trees_per_plot, rng)
        assert len(sampled) == 0

    def test_nan_and_zero_tpa(self):
        """Trees with NaN or zero TPA should not cause errors."""
        rng = np.random.default_rng(42)
        trees = pd.DataFrame(
            {
                "PLOT_ID": [1, 1, 1],
                "TREE_ID": [10, 20, 30],
                "TPA": [np.nan, 0.0, 5.0],
            }
        )
        num_per_plot = pd.Series([10], index=[1])
        sampled = _sample_trees_within_plots(trees, num_per_plot, rng)
        assert len(sampled) == 10
        # Only tree 30 has positive TPA, so all samples should be tree 30
        assert (sampled["TREE_ID"] == 30).all()

    def test_all_zero_tpa_returns_empty(self):
        """If all TPA values are zero, no trees should be sampled."""
        rng = np.random.default_rng(42)
        trees = pd.DataFrame(
            {
                "PLOT_ID": [1, 1],
                "TREE_ID": [10, 20],
                "TPA": [0.0, 0.0],
            }
        )
        num_per_plot = pd.Series([5], index=[1])
        sampled = _sample_trees_within_plots(trees, num_per_plot, rng)
        assert len(sampled) == 0


class TestAssignTreeLocations:
    def test_assign_tree_locations(self):
        sampled_trees = pd.DataFrame(
            {"PLOT_ID": [1, 1, 2, 2, 2, 3], "TPA": [1, 2, 3, 4, 5, 6]}
        )
        rng = np.random.default_rng(0)
        tree_locations = pd.DataFrame(
            {
                "X": rng.random(6),
                "Y": rng.random(6),
                "PLOT_ID": [1, 1, 2, 2, 2, 3],
            }
        )
        result = _assign_tree_locations(sampled_trees.copy(), tree_locations.copy())
        assert "X" in result.columns
        assert "Y" in result.columns
        assert len(result) == len(sampled_trees)

    def test_preserves_plot_alignment(self):
        """Trees from plot N must get X/Y values from locations tagged plot N."""
        sampled_trees = pd.DataFrame(
            {"PLOT_ID": [1, 1, 2, 2, 3], "TPA": [1, 2, 3, 4, 5]}
        )
        tree_locations = pd.DataFrame(
            {
                "X": [100.0, 101.0, 200.0, 201.0, 300.0],
                "Y": [100.0, 101.0, 200.0, 201.0, 300.0],
                "PLOT_ID": [1, 1, 2, 2, 3],
            }
        )
        result = _assign_tree_locations(sampled_trees.copy(), tree_locations.copy())
        # Plot 1 trees should have X in [100, 101], plot 2 in [200, 201], etc.
        for _, row in result.iterrows():
            pid = row["PLOT_ID"]
            expected_locs = tree_locations[tree_locations["PLOT_ID"] == pid]
            assert row["X"] in expected_locs["X"].values
            assert row["Y"] in expected_locs["Y"].values


class TestDropTreesOutsideBounds:
    def test_basic_case(self):
        df = pd.DataFrame(
            {
                "X": [0, 5, 10, 15],
                "Y": [0, 5, 10, 15],
            }
        )
        bounds = (0, 0, 10, 10)
        result = _drop_trees_outside_roi_bounds(df, bounds)
        assert len(result) == 3
        assert 15 not in result["X"].values

    def test_boundary_inclusive(self):
        """Trees exactly on the boundary should be kept."""
        df = pd.DataFrame(
            {
                "X": [0.0, 10.0, 5.0],
                "Y": [0.0, 10.0, 5.0],
            }
        )
        bounds = (0, 0, 10, 10)
        result = _drop_trees_outside_roi_bounds(df, bounds)
        assert len(result) == 3

    def test_just_outside_boundary(self):
        """Trees just outside the boundary should be dropped."""
        eps = 1e-10
        df = pd.DataFrame(
            {
                "X": [-eps, 10 + eps, 5.0],
                "Y": [5.0, 5.0, -eps],
            }
        )
        bounds = (0, 0, 10, 10)
        result = _drop_trees_outside_roi_bounds(df, bounds)
        assert len(result) == 0


class TestBuildOutputMeta:
    def test_returns_correct_schema(self):
        trees = pd.DataFrame(
            {
                "PLOT_ID": [1],
                "TREE_ID": [1],
                "TPA": [0.1],
                "SPCD": [122],
            }
        )
        meta = _build_output_meta(trees)
        assert len(meta) == 0
        assert "X" in meta.columns
        assert "Y" in meta.columns
        assert "PLOT_ID" in meta.columns


class TestGenerateTileIndices:
    def test_single_tile_when_none(self):
        tiles = _generate_tile_indices((100, 100), None, 15)
        assert len(tiles) == 1
        row_sl, col_sl = tiles[0]
        assert row_sl == slice(0, 100)
        assert col_sl == slice(0, 100)

    def test_multiple_tiles(self):
        tiles = _generate_tile_indices((20, 20), 30, 15)
        # chunk_cells = max(1, int(30/15)) = 2
        # 20/2 = 10 tiles per dimension -> 100 tiles
        assert len(tiles) == 100

    def test_coverage_and_no_overlap(self):
        shape = (17, 23)
        tiles = _generate_tile_indices(shape, 45, 15)
        count = np.zeros(shape, dtype=int)
        for row_sl, col_sl in tiles:
            count[row_sl, col_sl] += 1
        assert (count == 1).all()

    def test_chunk_size_smaller_than_resolution(self):
        """chunk_size < intensity_resolution should produce 1-cell tiles."""
        tiles = _generate_tile_indices((4, 4), 5, 15)
        # chunk_cells = max(1, int(5/15)) = max(1, 0) = 1
        assert len(tiles) == 16
        for row_sl, col_sl in tiles:
            assert row_sl.stop - row_sl.start == 1
            assert col_sl.stop - col_sl.start == 1


class TestProcessSingleTile:
    def test_returns_dataframe(self, real_data):
        roi, trees, plots = real_data
        bounds = tuple(roi.total_bounds)
        resolution = 15
        grid_x, grid_y = _create_structured_coords_grid(bounds, resolution)
        density = _interpolate_tree_density_to_grid(
            trees, plots, grid_x, grid_y, resolution
        )
        plot_ids = _interpolate_plot_id_to_grid(trees, plots, grid_x, grid_y)

        ss = np.random.SeedSequence(42)
        child_seed = ss.spawn(1)[0]

        result = _process_single_tile(
            tile_grid_x=grid_x,
            tile_grid_y=grid_y,
            tile_density=density,
            tile_plot_ids=plot_ids,
            intensity_resolution=resolution,
            roi_bounds=bounds,
            trees=trees,
            child_seed=child_seed,
        )
        assert isinstance(result, pd.DataFrame)
        assert "X" in result.columns
        assert "Y" in result.columns
        assert len(result) > 0

    def test_empty_density_returns_empty(self):
        trees = pd.DataFrame(
            {
                "PLOT_ID": pd.Series(dtype=int),
                "TPA": pd.Series(dtype=float),
                "TREE_ID": pd.Series(dtype=int),
            }
        )
        grid_x = np.array([[0.5]])
        grid_y = np.array([[0.5]])
        density = np.array([[0.0]])
        plot_ids = np.array([[0]])
        ss = np.random.SeedSequence(0)
        child_seed = ss.spawn(1)[0]

        result = _process_single_tile(
            grid_x, grid_y, density, plot_ids, 1.0, (0, 0, 1, 1), trees, child_seed
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunPointProcess:
    def test_dispatch_inhomogeneous_poisson(self, real_data):
        import dask.dataframe as dd

        roi, trees, plots = real_data
        ddf = run_point_process(
            "inhomogeneous_poisson", roi, trees, plots=plots, seed=42
        )
        assert isinstance(ddf, dd.DataFrame)
        result = ddf.compute()
        assert len(result) > 0

    def test_invalid_process_type_raises(self, real_data):
        roi, trees, plots = real_data
        with pytest.raises(ValueError, match="Invalid point process type"):
            run_point_process("bogus_process", roi, trees, plots=plots)


class TestEndToEnd:
    save_fig = True

    def test_returns_dask_dataframe(self, real_data):
        import dask.dataframe as dd

        roi, trees, plots = real_data
        ddf = inhomogeneous_poisson_process(roi, trees, plots, seed=42)
        assert isinstance(ddf, dd.DataFrame)

        result = ddf.compute()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "X" in result.columns
        assert "Y" in result.columns

    def test_seed_reproducibility(self, real_data):
        roi, trees, plots = real_data
        result1 = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        result2 = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        pd.testing.assert_frame_equal(result1, result2)

    def test_different_seeds_produce_different_results(self, real_data):
        roi, trees, plots = real_data
        result1 = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        result2 = inhomogeneous_poisson_process(roi, trees, plots, seed=43).compute()
        # Different seeds should produce different results (count or values)
        different_count = len(result1) != len(result2)
        n = min(len(result1), len(result2))
        different_values = not np.allclose(
            result1["X"].values[:n], result2["X"].values[:n]
        )
        assert different_count or different_values

    def test_output_coordinates_within_roi_bounds(self, real_data):
        """All generated X/Y must fall within the ROI bounding box."""
        roi, trees, plots = real_data
        result = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        minx, miny, maxx, maxy = roi.total_bounds
        assert (result["X"] >= minx).all(), "Found X below ROI west bound"
        assert (result["X"] <= maxx).all(), "Found X above ROI east bound"
        assert (result["Y"] >= miny).all(), "Found Y below ROI south bound"
        assert (result["Y"] <= maxy).all(), "Found Y above ROI north bound"

        if self.save_fig:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            roi.boundary.plot(ax=ax, color="red", linewidth=2, label="ROI")
            ax.scatter(
                result["X"],
                result["Y"],
                s=1,
                alpha=0.3,
                c="blue",
                label=f"Trees (n={len(result)})",
            )
            ax.set_title("Generated tree locations within ROI")
            ax.set_aspect("equal")
            ax.legend()
            plt.tight_layout()
            FIGURES_PATH.mkdir(exist_ok=True)
            plt.savefig(FIGURES_PATH / "tree_locations.png", dpi=150)
            plt.close(fig)

    def test_output_species_codes_from_input(self, real_data):
        """All SPCD values in output must be a subset of input SPCD values."""
        roi, trees, plots = real_data
        result = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        input_spcds = set(trees["SPCD"].unique())
        output_spcds = set(result["SPCD"].unique())
        assert output_spcds.issubset(
            input_spcds
        ), f"Output contains SPCD values not in input: {output_spcds - input_spcds}"

    def test_output_tree_attributes_realistic(self, real_data):
        """DIA, HT, CR values should all come from input data ranges."""
        roi, trees, plots = real_data
        result = inhomogeneous_poisson_process(roi, trees, plots, seed=42).compute()
        # Every DIA in output must exist in input
        assert set(result["DIA"].dropna().unique()).issubset(
            set(trees["DIA"].dropna().unique())
        )
        # Every HT in output must exist in input
        assert set(result["HT"].dropna().unique()).issubset(
            set(trees["HT"].dropna().unique())
        )

    def test_expected_tree_count_matches_density(self, real_data):
        """End-to-end tree count should be within reasonable bounds of
        the density grid integral (expected value of the Poisson process)."""
        roi, trees, plots = real_data
        bounds = tuple(roi.total_bounds)
        resolution = 15
        grid_x, grid_y = _create_structured_coords_grid(bounds, resolution)
        density = _interpolate_tree_density_to_grid(
            trees, plots, grid_x, grid_y, resolution
        )
        expected_count = density.sum()

        # Run multiple seeds and check the mean is close to expected
        counts = []
        for seed in range(20):
            result = inhomogeneous_poisson_process(
                roi, trees, plots, seed=seed
            ).compute()
            counts.append(len(result))
        mean_count = np.mean(counts)
        # Mean should be within 20% of expected (generous for 20 samples)
        assert abs(mean_count - expected_count) / expected_count < 0.2, (
            f"Mean tree count {mean_count:.0f} deviates >20% from "
            f"expected {expected_count:.0f}"
        )


class TestChunkedProcessing:
    save_fig = True

    def test_multiple_partitions(self, real_data):
        roi, trees, plots = real_data
        ddf = inhomogeneous_poisson_process(roi, trees, plots, seed=42, chunk_size=200)
        assert ddf.npartitions > 1
        result = ddf.compute()
        assert len(result) > 0

    def test_chunked_seed_reproducibility(self, real_data):
        roi, trees, plots = real_data
        r1 = inhomogeneous_poisson_process(
            roi, trees, plots, seed=99, chunk_size=200
        ).compute()
        r2 = inhomogeneous_poisson_process(
            roi, trees, plots, seed=99, chunk_size=200
        ).compute()
        pd.testing.assert_frame_equal(r1, r2)

    def test_chunked_coordinates_within_roi_bounds(self, real_data):
        """Chunked output should also have all coordinates within ROI."""
        roi, trees, plots = real_data
        result = inhomogeneous_poisson_process(
            roi, trees, plots, seed=42, chunk_size=200
        ).compute()
        minx, miny, maxx, maxy = roi.total_bounds
        assert (result["X"] >= minx).all()
        assert (result["X"] <= maxx).all()
        assert (result["Y"] >= miny).all()
        assert (result["Y"] <= maxy).all()

    def test_chunked_vs_unchunked_similar_counts(self, real_data):
        """Chunked and unchunked should produce statistically similar
        tree counts (same density grid, different RNG streams)."""
        roi, trees, plots = real_data
        counts_unchunked = []
        counts_chunked = []
        results_for_plot = {}
        for seed in range(10):
            r1 = inhomogeneous_poisson_process(roi, trees, plots, seed=seed).compute()
            r2 = inhomogeneous_poisson_process(
                roi, trees, plots, seed=seed + 1000, chunk_size=200
            ).compute()
            counts_unchunked.append(len(r1))
            counts_chunked.append(len(r2))
            if seed == 0:
                results_for_plot = {"unchunked": r1, "chunked": r2}
        # Means should be within 15% of each other
        mean_u = np.mean(counts_unchunked)
        mean_c = np.mean(counts_chunked)
        assert abs(mean_u - mean_c) / max(mean_u, mean_c) < 0.15

        if self.save_fig:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
            for ax, (label, r) in zip(axes, results_for_plot.items()):
                roi.boundary.plot(ax=ax, color="red", linewidth=2)
                ax.scatter(r["X"], r["Y"], s=1, alpha=0.3, c="blue")
                ax.set_title(f"{label} (n={len(r)})")
                ax.set_aspect("equal")
            fig.suptitle("Chunked vs unchunked tree distributions (seed=0)")
            plt.tight_layout()
            FIGURES_PATH.mkdir(exist_ok=True)
            plt.savefig(FIGURES_PATH / "chunked_vs_unchunked.png", dpi=150)
            plt.close(fig)


class TestToParquetRoundtrip:
    def test_parquet_roundtrip(self, real_data, tmp_path):
        roi, trees, plots = real_data
        ddf = inhomogeneous_poisson_process(roi, trees, plots, seed=42, chunk_size=200)
        out_path = tmp_path / "output.parquet"
        ddf.to_parquet(str(out_path))

        loaded = pd.read_parquet(str(out_path))
        computed = ddf.compute().reset_index(drop=True)
        assert set(loaded.columns) == set(computed.columns)
        assert len(loaded) == len(computed)


class TestExpandToRoi:
    def test_returns_valid_tree_population(self, real_data):
        """expand_to_roi should return a TreePopulation that passes schema."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        population = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, plots=plots, seed=42
        )
        assert isinstance(population, TreePopulation)
        assert len(population) > 0
        # Schema validation happened in TreePopulation.__init__
        assert "X" in population.data.columns
        assert "Y" in population.data.columns

    def test_lazy_returns_dask_dataframe(self, real_data):
        """lazy=True returns a dask DataFrame without materializing."""
        import dask.dataframe as dd

        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        result = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=True, plots=plots, seed=42
        )
        assert isinstance(result, dd.DataFrame)

    def test_lazy_output_computable(self, real_data):
        """Lazy dask DataFrame can be computed to a valid pandas DataFrame."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        ddf = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=True, plots=plots, seed=42
        )
        result = ddf.compute()
        assert len(result) > 0
        assert "X" in result.columns
        assert "Y" in result.columns

    def test_lazy_to_parquet(self, real_data, tmp_path):
        """Lazy dask DataFrame can be written to parquet without materializing."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        ddf = tree_sample.expand_to_roi(
            "inhomogeneous_poisson",
            roi,
            lazy=True,
            plots=plots,
            seed=42,
            chunk_size=200,
        )
        out_path = tmp_path / "lazy_output.parquet"
        ddf.to_parquet(str(out_path))
        loaded = pd.read_parquet(out_path)
        assert len(loaded) > 0

    def test_lazy_matches_eager(self, real_data):
        """lazy=True and lazy=False produce identical results for same seed."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        eager = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, plots=plots, seed=42
        )
        lazy = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=True, plots=plots, seed=42
        )
        pd.testing.assert_frame_equal(
            eager.data.reset_index(drop=True),
            lazy.compute().reset_index(drop=True),
        )

    def test_lazy_seed_reproducibility(self, real_data):
        """Two lazy calls with the same seed produce identical results."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        r1 = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=True, plots=plots, seed=42
        ).compute()
        r2 = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=True, plots=plots, seed=42
        ).compute()
        pd.testing.assert_frame_equal(r1, r2)

    def test_lazy_with_chunk_size(self, real_data):
        """lazy=True with chunk_size produces multiple partitions."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        ddf = tree_sample.expand_to_roi(
            "inhomogeneous_poisson",
            roi,
            lazy=True,
            plots=plots,
            seed=42,
            chunk_size=200,
        )
        assert ddf.npartitions > 1

    def test_lazy_chunked_matches_eager_chunked(self, real_data):
        """lazy=True and lazy=False with chunk_size produce identical results."""
        roi, trees, plots = real_data
        tree_sample = TreeSample(trees)
        eager = tree_sample.expand_to_roi(
            "inhomogeneous_poisson", roi, plots=plots, seed=42, chunk_size=200
        )
        lazy = tree_sample.expand_to_roi(
            "inhomogeneous_poisson",
            roi,
            lazy=True,
            plots=plots,
            seed=42,
            chunk_size=200,
        )
        pd.testing.assert_frame_equal(
            eager.data.reset_index(drop=True),
            lazy.compute().reset_index(drop=True),
        )
