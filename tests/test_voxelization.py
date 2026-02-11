# Core imports
import math
import random
from pathlib import Path

# Internal imports
from tests.utils import make_random_tree
from fastfuels_core.trees import Tree
from fastfuels_core.voxelization import (
    _get_horizontal_tree_coords,
    _get_vertical_tree_coords,
    _resample_coords_grid_to_subgrid,
    _compute_circle_segment_area,
    _calculate_case_1_area,
    _calculate_case_3_area,
    _calculate_case_9_area,
    _calculate_case_11_area,
    _encode_corners,
    _compute_intersection_area,
    _compute_intersection_area_by_case,
    _sum_area_along_axis,
    _align_quadrants,
    discretize_crown_profile,
    _compute_horizontal_probability,
    _compute_vertical_probability,
    _compute_joint_probability,
    _sample_voxels_from_probability_grid,
    voxelize_tree,
)

# External imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

TEST_PATH = Path(__file__).parent
FIGURES_PATH = TEST_PATH / "figures"


class TestGetHorizontalTreeCoords:
    def test_positive_radius(self):
        step = 1
        radius = 5
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 13
        assert result[0] == -radius - step
        assert result[1] == -radius
        assert result[6] == 0
        assert result[-2] == radius
        assert result[-1] == radius + step

    def test_zero_radius(self):
        step = 1
        radius = 0
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == -1
        assert result[0] + step == -radius
        assert result[1] == 0
        assert result[-1] == 1
        assert result[-1] - step == radius

    def test_subgrid_radius(self):
        step = 1
        radius = 0.1
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == -1
        assert result[1] == 0
        assert result[-1] == 1

    def test_negative_radius(self):
        step = 1
        radius = -5
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 13
        assert result[0] == -6
        assert result[0] + step == radius
        assert result[6] == 0
        assert result[-1] == 6
        assert result[-1] - step == -radius

    def test_fractional_step(self):
        step = 0.5
        radius = 5
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 23
        assert result[0] == -5.5
        assert result[0] + step == -radius
        assert result[11] == 0
        assert result[-1] == 5.5
        assert result[-1] - step == radius

    def test_fine_step(self):
        step = 0.1
        radius = 5
        pos = 0
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 103
        assert np.allclose(result[0], -5 - step)
        assert np.allclose(result[1], -radius)
        assert result[51] == 0
        assert np.allclose(result[-2], radius)
        assert np.allclose(result[-1], radius + step)

    def test_zero_step(self):
        step = 0
        radius = 5
        pos = 0
        with pytest.raises(ZeroDivisionError):
            _get_horizontal_tree_coords(step, radius, pos)

    def test_non_zero_pos(self):
        step = 1
        radius = 5
        pos = 2
        result = _get_horizontal_tree_coords(step, radius, pos)
        assert isinstance(result, np.ndarray)
        assert len(result) == 13
        assert result[0] == -4
        assert result[0] + step == -radius + pos
        assert result[6] == 2
        assert result[-1] == 8
        assert result[-1] - step == radius + pos

    # Centering mode tests
    def test_cell_centered_default_backward_compatible(self):
        """Default should produce identical results to original implementation."""
        step = 1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius)
        assert len(result) == 13  # 2*6 + 1 = 13
        assert result[0] == -6
        assert result[6] == 0
        assert result[-1] == 6

    def test_cell_centered_explicit_matches_default(self):
        """Explicit centering='cell' should match default behavior."""
        step = 1
        radius = 5
        default_result = _get_horizontal_tree_coords(step, radius)
        explicit_result = _get_horizontal_tree_coords(step, radius, centering="cell")
        assert np.array_equal(default_result, explicit_result)

    def test_vertex_centered_basic(self):
        """Vertex centering produces even-sized grid with no cell at origin."""
        step = 1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0  # even
        assert 0.0 not in result  # no cell center at origin
        mid = len(result) // 2
        assert result[mid - 1] < 0 < result[mid]  # origin between middle cells

    def test_vertex_centered_symmetry(self):
        """Vertex-centered grid should be perfectly symmetric about origin."""
        step = 1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        for i in range(len(result) // 2):
            assert np.isclose(result[i], -result[-(i + 1)])

    def test_vertex_centered_cell_spacing(self):
        """Vertex-centered cells should have correct spacing."""
        step = 1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        for i in range(len(result) - 1):
            assert np.isclose(result[i + 1] - result[i], step)

    def test_vertex_centered_covers_crown(self):
        """Vertex-centered grid should cover at least the crown radius."""
        step = 1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        # Grid should extend at least to cover the crown
        assert result[-1] + step / 2 >= radius  # rightmost cell edge covers radius
        assert result[0] - step / 2 <= -radius  # leftmost cell edge covers -radius

    def test_vertex_centered_zero_radius(self):
        """Vertex centering with zero radius."""
        step = 1
        radius = 0
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0
        assert len(result) == 2  # minimum even grid

    def test_vertex_centered_small_radius(self):
        """Vertex centering with radius smaller than step."""
        step = 1
        radius = 0.1
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0
        assert 0.0 not in result

    def test_vertex_centered_fractional_step(self):
        """Vertex centering with fractional step size."""
        step = 0.5
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0
        for i in range(len(result) - 1):
            assert np.isclose(result[i + 1] - result[i], step)

    def test_vertex_centered_fine_step(self):
        """Vertex centering with fine step size."""
        step = 0.1
        radius = 5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0
        assert 0.0 not in result

    def test_vertex_centered_non_zero_pos(self):
        """Vertex centering with non-zero stem position."""
        step = 1
        radius = 5
        pos = 2
        result = _get_horizontal_tree_coords(step, radius, pos, centering="vertex")
        assert len(result) % 2 == 0
        assert pos not in result
        mid = len(result) // 2
        assert result[mid - 1] < pos < result[mid]

    def test_vertex_centered_negative_pos(self):
        """Vertex centering with negative stem position."""
        step = 1
        radius = 5
        pos = -3
        result = _get_horizontal_tree_coords(step, radius, pos, centering="vertex")
        assert len(result) % 2 == 0
        assert pos not in result

    def test_vertex_centered_negative_radius(self):
        """Vertex centering handles negative radius (absolute value)."""
        step = 1
        radius = -5
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0

    def test_vertex_centered_large_radius(self):
        """Vertex centering with large radius."""
        step = 1
        radius = 100
        result = _get_horizontal_tree_coords(step, radius, centering="vertex")
        assert len(result) % 2 == 0
        assert result[-1] + step / 2 >= radius

    def test_both_modes_various_params(self):
        """Both modes work correctly across various parameter combinations."""
        for step in [0.1, 0.5, 1.0, 2.0]:
            for radius in [0.5, 1.0, 5.0, 10.0]:
                cell_result = _get_horizontal_tree_coords(
                    step, radius, centering="cell"
                )
                vertex_result = _get_horizontal_tree_coords(
                    step, radius, centering="vertex"
                )
                assert len(cell_result) % 2 == 1  # cell always odd
                assert len(vertex_result) % 2 == 0  # vertex always even


class TestGetVerticalTreeCoords:
    def test_positive_height(self):
        step = 1
        tree_height = 5
        crown_base_height = 0
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 6
        assert result[0] == crown_base_height
        assert result[-1] == tree_height

    def test_fractional_height(self):
        step = 1
        tree_height = 10.1
        crown_base_height = 5
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 7
        assert result[0] == crown_base_height
        assert result[-1] == math.ceil(tree_height)

    def test_zero_height(self):
        step = 1
        tree_height = 0
        crown_base_height = 0
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0] == tree_height

    def test_negative_height(self):
        step = 1
        tree_height = -5
        crown_base_height = 0
        res = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(res, np.ndarray)
        assert len(res) == 0

    def test_fractional_step(self):
        step = 0.5
        tree_height = 5
        crown_base_height = 0
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 11
        assert result[0] == crown_base_height
        assert result[-1] == tree_height

    def test_fine_step(self):
        step = 0.1
        tree_height = 10.1
        crown_base_height = 5
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 52
        assert result[0] == crown_base_height
        assert np.allclose(result[-1], tree_height)

    def test_zero_step(self):
        step = 0
        tree_height = 5
        crown_base_height = 0
        with pytest.raises(ZeroDivisionError):
            _get_vertical_tree_coords(step, tree_height, crown_base_height)

    def test_fractional_cbh(self):
        step = 1
        tree_height = 20.3643
        crown_base_height = 5.891270987
        result = _get_vertical_tree_coords(step, tree_height, crown_base_height)
        assert isinstance(result, np.ndarray)
        assert len(result) == 16
        assert result[0] == crown_base_height
        assert result[-1] == 20.891270987


class TestResampleCoordsGridToSubgrid:
    def test_resample(self):
        original_pts = np.array([0.5])
        original_spacing = 1
        new_spacing = 0.5
        expected_pts_1 = np.array([0.25, 0.75])
        resampled_pts_1 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_1, expected_pts_1)

        original_pts = np.array([0.5])
        original_spacing = 1
        new_spacing = 0.25
        expected_pts_2 = np.array([0.125, 0.375, 0.625, 0.875])
        resampled_pts_2 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_2, expected_pts_2)

        original_pts = np.array([0.5])
        original_spacing = 1
        new_spacing = 0.1
        expected_pts_3 = np.array(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
        resampled_pts_3 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_3, expected_pts_3)

        original_pts = np.array([0.5, 1.5])
        original_spacing = 1
        new_spacing = 0.5
        expected_pts_4 = np.array([0.25, 0.75, 1.25, 1.75])
        resampled_pts_4 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_4, expected_pts_4)

        original_pts = np.array([0.5, 1.5, 2.5])
        original_spacing = 1
        new_spacing = 0.5
        expected_pts_5 = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        resampled_pts_5 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_5, expected_pts_5)

        original_pts = np.array([0.5, 1.5, 2.5])
        original_spacing = 1
        new_spacing = 0.25
        expected_pts_6 = np.array(
            [
                0.125,
                0.375,
                0.625,
                0.875,
                1.125,
                1.375,
                1.625,
                1.875,
                2.125,
                2.375,
                2.625,
                2.875,
            ]
        )
        resampled_pts_6 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_6, expected_pts_6)

        original_pts = np.array([0.5, 1.5, 2.5])
        original_spacing = 1
        new_spacing = 0.1
        expected_pts_7 = np.array(
            [
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                0.55,
                0.65,
                0.75,
                0.85,
                0.95,
                1.05,
                1.15,
                1.25,
                1.35,
                1.45,
                1.55,
                1.65,
                1.75,
                1.85,
                1.95,
                2.05,
                2.15,
                2.25,
                2.35,
                2.45,
                2.55,
                2.65,
                2.75,
                2.85,
                2.95,
            ]
        )
        resampled_pts_7 = _resample_coords_grid_to_subgrid(
            original_pts, original_spacing, new_spacing
        )
        assert np.allclose(resampled_pts_7, expected_pts_7)

    def test_same_resolution_shape(self):
        # Test 1m resolution
        grid = np.arange(0.5, 100.5, 1)
        grid_spacing = 1
        new_spacing = 1
        resampled_grid = _resample_coords_grid_to_subgrid(
            grid, grid_spacing, new_spacing
        )
        assert resampled_grid.shape == grid.shape

        # Test 0.5m resolution
        grid = np.arange(0.25, 100.25, 0.5)
        grid_spacing = 0.5
        new_spacing = 0.5
        resampled_grid = _resample_coords_grid_to_subgrid(
            grid, grid_spacing, new_spacing
        )
        assert resampled_grid.shape == grid.shape

        # Test 0.25m resolution
        grid = np.arange(0.125, 100.125, 0.25)
        grid_spacing = 0.25
        new_spacing = 0.25
        resampled_grid = _resample_coords_grid_to_subgrid(
            grid, grid_spacing, new_spacing
        )
        assert resampled_grid.shape == grid.shape

        # Test 0.1m resolution
        grid = np.arange(0.05, 100.05, 0.1)
        grid_spacing = 0.1
        new_spacing = 0.1
        resampled_grid = _resample_coords_grid_to_subgrid(
            grid, grid_spacing, new_spacing
        )
        assert resampled_grid.shape == grid.shape


class TestComputeCircleSegmentArea:
    def test_zero_radius(self):
        # Test case where radius is zero
        csa = _compute_circle_segment_area(0, 0.5, 1, 0.5, 0)
        assert csa == 0

    def test_zero_chord_length(self):
        # Test case where chord length is zero (points are the same)
        csa = _compute_circle_segment_area(1, 1, 1, 1, 0.5)
        assert csa == 0


class TestCalculateCase1Area:
    save_fig = False
    show_fig = False

    def test_with_floats(self):
        bottom = 0
        left = 0
        radius = 0.75
        intersection_area = _calculate_case_1_area(left, bottom, radius)
        assert isinstance(intersection_area, float)
        assert 0 < intersection_area < 1

    def test_with_numpy(self):
        bottom = np.array([0])
        left = np.array([0])
        radius = np.array([0.75])
        intersection_area = _calculate_case_1_area(left, bottom, radius)
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert 0 < intersection_area[0] < 1

    def test_compare_with_shapely_exact(self):
        length = 1
        radius = np.array([1e-4, 0.15, 0.25, 0.5, 0.75, 1.0])
        x_left = np.zeros_like(radius)
        y_bottom = np.zeros_like(radius)
        intersection = _calculate_case_1_area(x_left, y_bottom, radius, exact=True)

        bottom_left = Point(0, 0)
        bottom_right = Point(length, 0)
        top_right = Point(length, length)
        top_left = Point(0, length)
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0, 0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)
        assert np.allclose(intersection[4], intersection_shapely[4].area)
        assert np.allclose(intersection[5], intersection_shapely[5].area)

        # plot the cell and the circle
        fig, ax = plt.subplots(3, 2, figsize=(6, 8))
        for row in range(3):
            for col in range(2):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(-radius[i] / 8, radius[i] + radius[i] / 8)
                ax[row, col].set_ylim(-radius[i] / 8, radius[i] + radius[i] / 8)
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")
                # ax[row, col].legend(loc="upper right")
        # plt.legend()
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_1_area.png")
        if self.show_fig:
            plt.show()


class TestCalculateCase3Area:
    save_fig = False
    show_fig = False

    def test_with_floats(self):
        left = -0.5
        right = 0.5
        bottom = 0.5
        length = 1
        radius = 1.25
        intersection_area = _calculate_case_3_area(left, right, bottom, radius, length)
        assert isinstance(intersection_area, float)
        assert 0 < intersection_area < 1

    def test_with_numpy(self):
        left = np.array([-0.5])
        right = np.array([0.5])
        bottom = np.array([0.5])
        length = np.array([1])
        radius = np.array([1.25])
        intersection_area = _calculate_case_3_area(left, right, bottom, radius, length)
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert 0 < intersection_area[0] < 1

    def test_compare_with_shapely_exact(self):
        length = 1
        radius = np.array([np.sqrt(0.5), 0.75, 1.0, 1.15, 1.35, 1.5])
        left = np.zeros_like(radius) - length / 2
        right = np.zeros_like(radius) + length / 2
        bottom = np.zeros_like(radius) + length / 2
        intersection = _calculate_case_3_area(
            left, right, bottom, radius, length, exact=True
        )

        bottom_left = Point(-length / 2, length / 2)
        bottom_right = Point(length / 2, length / 2)
        top_left = Point(-length / 2, 3 * length / 2)
        top_right = Point(length / 2, 3 * length / 2)
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0, 0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)
        assert np.allclose(intersection[4], intersection_shapely[4].area)
        assert np.allclose(intersection[5], intersection_shapely[5].area)

        # plot the cell and the circle
        nrows = radius.size // 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(-length / 2 - 0.01, length / 2 + 0.01)
                ax[row, col].set_ylim(length / 2, 3 * length / 2)
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_3_area.png")
        if self.show_fig:
            plt.show()

    def test_compare_with_shapely_offset_cell_exact(self):
        length = 0.5
        radius = np.array([1.46, 1.5, 1.6, 1.75])
        left = np.zeros_like(radius) + length / 2
        right = np.zeros_like(radius) + 3 * length / 2
        bottom = np.zeros_like(radius) + 5 * length / 2
        top = np.zeros_like(radius) + 7 * length / 2
        intersection = _calculate_case_3_area(
            left, right, bottom, radius, length, exact=True
        )

        bottom_left = Point(left[0], bottom[0])
        bottom_right = Point(right[0], bottom[0])
        top_left = Point(left[0], top[0])
        top_right = Point(right[0], top[0])
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0.0, 0.0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)

        # plot the cell and the circle
        nrows = radius.size // 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(left[0], right[0])
                ax[row, col].set_ylim(bottom[0], top[0])
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_3_area_offset.png")
        if self.show_fig:
            plt.show()


class TestCalculateCase9Area:
    save_fig = False
    show_fig = False

    def test_with_floats(self):
        left = 0.5
        bottom = -0.5
        top = 0.5
        length = 1
        radius = 1.25
        intersection_area = _calculate_case_9_area(top, bottom, left, radius, length)
        assert isinstance(intersection_area, float)
        assert 0 < intersection_area < 1

    def test_with_numpy(self):
        left = np.array([0.5])
        bottom = np.array([-0.5])
        top = np.array([0.5])
        length = np.array([1])
        radius = np.array([1.25])
        intersection_area = _calculate_case_9_area(top, bottom, left, radius, length)
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert 0 < intersection_area[0] < 1

    def test_compare_with_shapely_exact(self):
        length = 1
        radius = np.array([np.sqrt(0.5), 0.75, 1.0, 1.15, 1.35, 1.5])
        top = np.zeros_like(radius) + length / 2
        bottom = np.zeros_like(radius) - length / 2
        left = np.zeros_like(radius) + length / 2
        right = np.zeros_like(radius) + 3 * length / 2
        intersection = _calculate_case_9_area(
            top, bottom, left, radius, length, exact=True
        )

        bottom_left = Point(left[0], bottom[0])
        bottom_right = Point(right[0], bottom[0])
        top_left = Point(left[0], top[0])
        top_right = Point(right[0], top[0])
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0, 0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        # assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)
        assert np.allclose(intersection[4], intersection_shapely[4].area)
        assert np.allclose(intersection[5], intersection_shapely[5].area)

        # plot the cell and the circle
        nrows = radius.size // 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(length / 2 - 0.01, 3 * length / 2 + 0.01)
                ax[row, col].set_ylim(-length / 2 - 0.01, length / 2 + 0.01)
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_9_area.png")
        if self.show_fig:
            plt.show()

    def test_compare_with_shapely_offset_cell_exact(self):
        length = 0.5
        radius = np.array([1.46, 1.5, 1.6, 1.75])
        bottom = np.zeros_like(radius) + length / 2
        top = np.zeros_like(radius) + 3 * length / 2
        left = np.zeros_like(radius) + 5 * length / 2
        right = np.zeros_like(radius) + 7 * length / 2
        intersection = _calculate_case_9_area(
            top, bottom, left, radius, length, exact=True
        )

        bottom_left = Point(left[0], bottom[0])
        bottom_right = Point(right[0], bottom[0])
        top_left = Point(left[0], top[0])
        top_right = Point(right[0], top[0])
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0.0, 0.0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)

        # plot the cell and the circle
        nrows = radius.size // 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(left[0], right[0])
                ax[row, col].set_ylim(bottom[0], top[0])
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_9_area_offset.png")
        if self.show_fig:
            plt.show()


class TestCalculateCase11Area:
    save_fig = False
    show_fig = False

    def test_with_floats(self):
        right = 1.5
        top = 1.5
        radius = 1.75
        length = 1
        intersection_area = _calculate_case_11_area(top, right, radius, length)
        assert isinstance(intersection_area, float)
        assert 0 < intersection_area < 1

    def test_with_numpy(self):
        right = np.array([1.5])
        top = np.array([1.5])
        radius = np.array([1.75])
        length = np.array([1])
        intersection_area = _calculate_case_11_area(top, right, radius, length)
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert 0 < intersection_area[0] < 1

    def test_compare_with_shapely_exact(self):
        length = 1
        radius = np.array([1.59, 1.6, 1.75, 1.9, 2.0, 2.1])
        left = np.zeros_like(radius) + length / 2
        right = np.zeros_like(radius) + 3 * length / 2
        bottom = np.zeros_like(radius) + length / 2
        top = np.zeros_like(radius) + 3 * length / 2
        intersection = _calculate_case_11_area(top, right, radius, length, exact=True)

        bottom_left = Point(left[0], bottom[0])
        bottom_right = Point(right[0], bottom[0])
        top_left = Point(left[0], top[0])
        top_right = Point(right[0], top[0])
        cell = Polygon([bottom_left, bottom_right, top_right, top_left])
        circle = Point(0, 0).buffer(radius, quad_segs=2**16)
        intersection_shapely = cell.intersection(circle)

        assert len(intersection) == len(intersection_shapely)
        assert np.allclose(intersection[0], intersection_shapely[0].area)
        assert np.allclose(intersection[1], intersection_shapely[1].area)
        assert np.allclose(intersection[2], intersection_shapely[2].area)
        assert np.allclose(intersection[3], intersection_shapely[3].area)
        assert np.allclose(intersection[4], intersection_shapely[4].area)
        assert np.allclose(intersection[5], intersection_shapely[5].area)

        # plot the cell and the circle
        nrows = radius.size // 2
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                i = row * 2 + col
                ax[row, col].plot(*cell.exterior.xy, lw=5, label="cell")
                ax[row, col].plot(*circle[i].exterior.xy, lw=3, c="g", label="circle")
                ax[row, col].plot(
                    *intersection_shapely[i].exterior.xy,
                    "--",
                    lw=2,
                    c="r",
                    label="intersection",
                )
                ax[row, col].set_xlim(left[0], right[0])
                ax[row, col].set_ylim(bottom[0], top[0])
                ax[row, col].set_ylabel("y (m)")
                ax[row, col].set_xlabel("x (m)")
                ax[row, col].set_aspect("equal")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_calculate_case_11_area.png")
        if self.show_fig:
            plt.show()


class TestEncodeCorners:
    def test_case_0(self):
        """No corners inside - should return case 0"""
        top_left = np.array([False])
        top_right = np.array([False])
        bottom_left = np.array([False])
        bottom_right = np.array([False])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 0

    def test_case_1(self):
        """Bottom left corner inside - should return case 1"""
        top_left = np.array([False])
        top_right = np.array([False])
        bottom_left = np.array([True])
        bottom_right = np.array([False])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 1

    def test_case_3(self):
        """bottom left and bottom right corners inside - should return case 3"""
        top_left = np.array([False])
        top_right = np.array([False])
        bottom_left = np.array([True])
        bottom_right = np.array([True])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 3

    def test_case_9(self):
        """top left and bottom left corners inside - should return case 9"""
        top_left = np.array([True])
        top_right = np.array([False])
        bottom_left = np.array([True])
        bottom_right = np.array([False])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 9

    def test_case_11(self):
        """top left, bottom left, and bottom right corners inside - should
        return case 11"""
        top_left = np.array([True])
        top_right = np.array([False])
        bottom_left = np.array([True])
        bottom_right = np.array([True])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 11

    def test_case_15(self):
        """All corners inside - should return case 15"""
        top_left = np.array([True])
        top_right = np.array([True])
        bottom_left = np.array([True])
        bottom_right = np.array([True])
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.size == 1
        assert case == 15

    def test_multiple(self):
        """
        Tests a 3x3 grid of the form:

        3   1 0
        15  11 1
        15 15 9
        """
        top_left = np.array(
            [[False, False, False], [True, True, False], [True, True, True]]
        )
        top_right = np.array(
            [[False, False, False], [True, False, False], [True, True, False]]
        )
        bottom_left = np.array(
            [[True, True, False], [True, True, True], [True, True, True]]
        )
        bottom_right = np.array(
            [[True, False, False], [True, True, False], [True, True, False]]
        )
        case = _encode_corners(top_left, top_right, bottom_left, bottom_right)
        assert isinstance(case, np.ndarray)
        assert case.shape == (3, 3)
        assert case[0, 0] == 3
        assert case[0, 1] == 1
        assert case[0, 2] == 0
        assert case[1, 0] == 15
        assert case[1, 1] == 11
        assert case[1, 2] == 1
        assert case[2, 0] == 15
        assert case[2, 1] == 15
        assert case[2, 2] == 9


class TestComputeIntersectionAreaByCase:
    length = 1

    def test_case_0_no_radius(self):
        """No corners inside - should return 0"""
        case = np.array([[[0]]])
        left = np.array([[[0]]])
        right = np.array([[[0]]])
        bottom = np.array([[[0]]])
        top = np.array([[[0]]])
        radius = np.array([[[0]]])
        intersection_area = _compute_intersection_area_by_case(
            case, self.length, left, right, bottom, top, radius
        )
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert intersection_area == 0

    def test_case_0_with_radius(self):
        """No corners inside, but with a nonzero radius - should return area
        of the circle"""
        case = np.array([[[0]]])
        left = np.array([[[0]]])
        right = np.array([[[0]]])
        bottom = np.array([[[0]]])
        top = np.array([[[0]]])
        radius = np.array([[[0.25]]])
        intersection_area = _compute_intersection_area_by_case(
            case, self.length, left, right, bottom, top, radius
        )
        assert isinstance(intersection_area, np.ndarray)
        assert intersection_area.size == 1
        assert np.allclose(intersection_area, np.pi * radius[0, 0, 0] ** 2)


class TestSumAreaAlongAxis:
    def test_1D_1_cell(self):
        data = np.array([1, 1])
        summed = _sum_area_along_axis(data, 0, 2)
        assert summed == 2

    def test_1D_2_cells(self):
        data = np.array([1, 1, 1, 1])
        summed = _sum_area_along_axis(data, 0, 2)
        assert np.allclose(summed, [2, 2])

    def test_1D_10_cells(self):
        data = np.ones(10)
        summed = _sum_area_along_axis(data, 0, 2)
        assert np.allclose(summed, np.ones(5) * 2)

    def test_1D_invalid_axis(self):
        data = np.ones(10)
        with pytest.raises(ValueError):
            _sum_area_along_axis(data, 1, 2)

    def test_2D(self):
        data = np.array([[1, 1], [1, 1]])

        # Sum row-wise
        summed = _sum_area_along_axis(data, 0, 2)
        assert summed.shape == (1, 2)
        assert np.allclose(summed, [2, 2])

        # Sum column-wise
        summed = _sum_area_along_axis(data, 1, 2)
        assert summed.shape == (2, 1)
        assert np.allclose(summed, [2, 2])

    def test_2D_two_sums(self):
        data = np.ones([4, 4])

        # Sum row-wise
        summed = _sum_area_along_axis(data, 0, 2)
        assert summed.shape == (2, 4)
        assert np.allclose(summed, np.ones([2, 4]) * 2)

        # Sum column-wise
        summed_again = _sum_area_along_axis(summed, 1, 2)
        assert summed_again.shape == (2, 2)
        assert np.allclose(summed_again, np.ones([2, 2]) * 4)

    def test_3D(self):
        data = np.ones([10, 10, 10])
        summed = _sum_area_along_axis(data, 0, 10)
        assert summed.shape == (1, 10, 10)
        assert np.allclose(summed, np.ones([1, 10, 10]) * 10)

    def test_3D_2_layers(self):
        data = np.ones([20, 10, 10])
        summed = _sum_area_along_axis(data, 0, 10)
        assert summed.shape == (2, 10, 10)
        assert np.allclose(summed, np.ones([2, 10, 10]) * 10)


class TestAlignQuadrants:
    """Tests for _align_quadrants with centering modes."""

    def test_cell_centered_default_backward_compatible(self):
        """Default should match original behavior (cell-centered)."""
        q = np.ones((2, 3, 3))
        # Original: 3 + 3 - 1 = 5
        result = _align_quadrants(q, q, q, q)
        assert result.shape == (2, 5, 5)

    def test_cell_centered_explicit_matches_default(self):
        """Explicit cell centering matches default."""
        q = np.ones((2, 3, 3))
        default_result = _align_quadrants(q, q, q, q)
        explicit_result = _align_quadrants(q, q, q, q, centering="cell")
        assert np.array_equal(default_result, explicit_result)

    def test_vertex_centered_no_overlap(self):
        """Vertex centering: quadrants meet without overlap."""
        q = np.ones((2, 3, 3))
        # Vertex: 3 + 3 = 6 (no overlap)
        result = _align_quadrants(q, q, q, q, centering="vertex")
        assert result.shape == (2, 6, 6)

    def test_vertex_centered_correct_placement(self):
        """Verify quadrants are placed correctly in vertex mode."""
        # Create distinct quadrants for verification
        q1 = np.full((1, 2, 2), 1)
        q2 = np.full((1, 2, 2), 2)
        q3 = np.full((1, 2, 2), 3)
        q4 = np.full((1, 2, 2), 4)

        result = _align_quadrants(q1, q2, q3, q4, centering="vertex")
        assert result.shape == (1, 4, 4)

        # Verify quadrant placement
        assert np.all(result[:, :2, :2] == 1)  # q1 top-left
        assert np.all(result[:, :2, 2:] == 2)  # q2 top-right
        assert np.all(result[:, 2:, 2:] == 3)  # q3 bottom-right
        assert np.all(result[:, 2:, :2] == 4)  # q4 bottom-left

    def test_cell_centered_overlapping_values(self):
        """Cell centering: center row/column comes from quadrant overlap."""
        # Create quadrants with unique values at corners for verification
        q1 = np.full((1, 3, 3), 1)
        q2 = np.full((1, 3, 3), 2)
        q3 = np.full((1, 3, 3), 3)
        q4 = np.full((1, 3, 3), 4)
        result = _align_quadrants(q1, q2, q3, q4, centering="cell")
        # Result is 5x5, quadrants share the center row/column
        # q1 fills [:, :3, :3], q2 fills [:, :3, 2:], q3 fills [:, 2:, 2:], q4 fills [:, 2:, :3]
        # Non-overlapping corners should have their quadrant values
        assert result[0, 0, 0] == 1  # pure q1
        assert result[0, 0, 4] == 2  # pure q2
        assert result[0, 4, 4] == 3  # pure q3
        assert result[0, 4, 0] == 4  # pure q4

    def test_vertex_centered_even_output_always(self):
        """Vertex centering always produces even dimensions."""
        for size in [2, 3, 4, 5]:
            q = np.ones((1, size, size))
            result = _align_quadrants(q, q, q, q, centering="vertex")
            assert result.shape[1] % 2 == 0
            assert result.shape[2] % 2 == 0

    def test_cell_centered_odd_output_always(self):
        """Cell centering always produces odd dimensions."""
        for size in [2, 3, 4, 5]:
            q = np.ones((1, size, size))
            result = _align_quadrants(q, q, q, q, centering="cell")
            assert result.shape[1] % 2 == 1
            assert result.shape[2] % 2 == 1


class TestDiscretizeCrownProfile:
    def test_known_tree_fine_resolution(self):
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 0.5, 0.5)
        assert np.sum(grid) > 1

    def test_known_tree_mod_resolution(self):
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 1, 1)
        assert np.sum(grid) > 1

    def test_known_tree_coarse_resolution(self):
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 2, 1)
        assert np.sum(grid) > 1

    def test_random_trees(self):
        resolutions = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0))
        for _ in range(1000):
            tree = make_random_tree()
            res_tuple = random.choice(resolutions)
            grid = discretize_crown_profile(tree, *res_tuple)
            assert np.sum(grid) > 0

    # Centering mode tests
    def test_default_matches_original_behavior(self):
        """Default should produce identical results to original implementation."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid_default = discretize_crown_profile(test_tree, 1, 1)
        grid_cell = discretize_crown_profile(test_tree, 1, 1, centering="cell")
        assert np.array_equal(grid_default, grid_cell)

    def test_cell_centered_odd_dimensions(self):
        """Cell-centered grids have odd x and y dimensions."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="cell")
        assert grid.shape[1] % 2 == 1
        assert grid.shape[2] % 2 == 1

    def test_vertex_centered_even_dimensions(self):
        """Vertex-centered grids have even x and y dimensions."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="vertex")
        assert grid.shape[1] % 2 == 0
        assert grid.shape[2] % 2 == 0

    def test_cell_centered_symmetric(self):
        """Cell-centered grid should be symmetric about center."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="cell")
        assert np.allclose(grid, np.flip(np.flip(grid, axis=1), axis=2))

    def test_vertex_centered_symmetric(self):
        """Vertex-centered grid should be symmetric about center vertex."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="vertex")
        assert np.allclose(grid, np.flip(np.flip(grid, axis=1), axis=2))

    def test_centering_modes_preserve_approximate_volume(self):
        """Both centering modes should produce similar total volume."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        hr, vr = 0.5, 0.5

        grid_cell = discretize_crown_profile(test_tree, hr, vr, centering="cell")
        grid_vertex = discretize_crown_profile(test_tree, hr, vr, centering="vertex")

        volume_cell = np.sum(grid_cell) * hr * hr * vr
        volume_vertex = np.sum(grid_vertex) * hr * hr * vr

        assert np.isclose(volume_cell, volume_vertex, atol=1e-3)

    def test_vertex_centered_fine_resolution(self):
        """Vertex-centered works at fine resolution."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 0.5, 0.5, centering="vertex")
        assert grid.shape[1] % 2 == 0
        assert grid.shape[2] % 2 == 0
        assert np.sum(grid) > 0

    def test_vertex_centered_coarse_resolution(self):
        """Vertex-centered works at coarse resolution."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        grid = discretize_crown_profile(test_tree, 2, 1, centering="vertex")
        assert grid.shape[1] % 2 == 0
        assert grid.shape[2] % 2 == 0
        assert np.sum(grid) > 0

    def test_random_trees_cell_centered(self):
        """Random trees work with cell centering."""
        resolutions = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0))
        for _ in range(100):
            tree = make_random_tree()
            res_tuple = random.choice(resolutions)
            grid = discretize_crown_profile(tree, *res_tuple, centering="cell")
            assert np.sum(grid) > 0
            assert grid.shape[1] % 2 == 1
            assert grid.shape[2] % 2 == 1

    def test_random_trees_vertex_centered(self):
        """Random trees work with vertex centering."""
        resolutions = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0))
        for _ in range(100):
            tree = make_random_tree()
            res_tuple = random.choice(resolutions)
            grid = discretize_crown_profile(tree, *res_tuple, centering="vertex")
            assert np.sum(grid) > 0
            assert grid.shape[1] % 2 == 0
            assert grid.shape[2] % 2 == 0

    def test_very_small_tree_vertex_centered(self):
        """Very small tree (small crown) with vertex centering."""
        test_tree = Tree(122, 1, 5, 3.0, 0.1)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="vertex")
        assert grid.shape[1] % 2 == 0
        assert grid.shape[2] % 2 == 0

    def test_large_tree_vertex_centered(self):
        """Large tree with vertex centering."""
        test_tree = Tree(122, 1, 100, 50.0, 0.8)
        grid = discretize_crown_profile(test_tree, 1, 1, centering="vertex")
        assert grid.shape[1] % 2 == 0
        assert grid.shape[2] % 2 == 0
        assert np.sum(grid) > 0


class TestCenteringVisualization:
    """Visualization tests comparing cell-centered vs vertex-centered grids."""

    save_fig = True
    show_fig = False

    def test_visualize_horizontal_coords(self):
        """Visualize the difference in horizontal coordinate grids."""
        step = 1.0
        radius = 3.0

        cell_coords = _get_horizontal_tree_coords(step, radius, centering="cell")
        vertex_coords = _get_horizontal_tree_coords(step, radius, centering="vertex")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Cell-centered plot
        ax = axes[0]
        ax.set_title(f"Cell-Centered (n={len(cell_coords)}, odd)")
        for i, x in enumerate(cell_coords):
            # Draw cell boundaries
            ax.axvline(x - step / 2, color="lightgray", linestyle="-", linewidth=0.5)
            ax.axvline(x + step / 2, color="lightgray", linestyle="-", linewidth=0.5)
            # Draw cell center
            ax.plot(x, 0, "bs", markersize=8, label="Cell center" if i == 0 else "")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Stem position")
        ax.axvline(
            -radius, color="green", linestyle=":", linewidth=1.5, label="Crown radius"
        )
        ax.axvline(radius, color="green", linestyle=":", linewidth=1.5)
        ax.set_xlim(-radius - 2 * step, radius + 2 * step)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("x coordinate (m)")
        ax.legend(loc="upper right")
        ax.set_yticks([])

        # Vertex-centered plot
        ax = axes[1]
        ax.set_title(f"Vertex-Centered (n={len(vertex_coords)}, even)")
        for i, x in enumerate(vertex_coords):
            ax.axvline(x - step / 2, color="lightgray", linestyle="-", linewidth=0.5)
            ax.axvline(x + step / 2, color="lightgray", linestyle="-", linewidth=0.5)
            ax.plot(x, 0, "bs", markersize=8, label="Cell center" if i == 0 else "")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Stem at vertex")
        ax.axvline(
            -radius, color="green", linestyle=":", linewidth=1.5, label="Crown radius"
        )
        ax.axvline(radius, color="green", linestyle=":", linewidth=1.5)
        ax.set_xlim(-radius - 2 * step, radius + 2 * step)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("x coordinate (m)")
        ax.legend(loc="upper right")
        ax.set_yticks([])

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_centering_horizontal_coords.png", dpi=150)
        if self.show_fig:
            plt.show()
        plt.close()

        # Assertions
        assert len(cell_coords) % 2 == 1
        assert len(vertex_coords) % 2 == 0
        assert 0.0 in cell_coords
        assert 0.0 not in vertex_coords

    def test_visualize_crown_profile_slice(self):
        """Visualize a horizontal slice of voxelized crown for both centering modes."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        hr, vr = 1.0, 1.0

        grid_cell = discretize_crown_profile(test_tree, hr, vr, centering="cell")
        grid_vertex = discretize_crown_profile(test_tree, hr, vr, centering="vertex")

        # Get the actual coordinates for each grid
        cell_coords = _get_horizontal_tree_coords(
            hr, test_tree.max_crown_radius, centering="cell"
        )
        vertex_coords = _get_horizontal_tree_coords(
            hr, test_tree.max_crown_radius, centering="vertex"
        )

        # Get a middle z-slice where crown is widest
        z_mid_cell = grid_cell.shape[0] // 2
        z_mid_vertex = grid_vertex.shape[0] // 2

        slice_cell = grid_cell[z_mid_cell, :, :]
        slice_vertex = grid_vertex[z_mid_vertex, :, :]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Cell-centered - use actual spatial coordinates
        ax = axes[0]
        # Calculate extent: [left, right, bottom, top] based on cell edges
        extent_cell = [
            cell_coords[0] - hr / 2,
            cell_coords[-1] + hr / 2,
            cell_coords[0] - hr / 2,
            cell_coords[-1] + hr / 2,
        ]
        im = ax.imshow(
            slice_cell, cmap="YlOrRd", origin="lower", aspect="equal", extent=extent_cell
        )
        ax.set_title(f"Cell-Centered\nGrid: {slice_cell.shape[0]}x{slice_cell.shape[1]} (odd)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        # Draw grid lines at cell boundaries
        for coord in cell_coords:
            ax.axhline(coord - hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axhline(coord + hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axvline(coord - hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axvline(coord + hr / 2, color="gray", linewidth=0.3, alpha=0.5)
        # Mark stem at origin (should be at center of center cell)
        ax.plot(0, 0, "k+", markersize=20, markeredgewidth=3, label="Stem position")
        # Draw crown circle at this z-height
        z_height = test_tree.crown_base_height + z_mid_cell * vr
        crown_r = test_tree.get_crown_radius_at_height(z_height)
        circle = plt.Circle(
            (0, 0),
            crown_r,
            fill=False,
            color="blue",
            linewidth=2,
            linestyle="--",
            label=f"Crown radius ({crown_r:.1f}m)",
        )
        ax.add_patch(circle)
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label="Volume fraction")

        # Vertex-centered - use actual spatial coordinates
        ax = axes[1]
        extent_vertex = [
            vertex_coords[0] - hr / 2,
            vertex_coords[-1] + hr / 2,
            vertex_coords[0] - hr / 2,
            vertex_coords[-1] + hr / 2,
        ]
        im = ax.imshow(
            slice_vertex,
            cmap="YlOrRd",
            origin="lower",
            aspect="equal",
            extent=extent_vertex,
        )
        ax.set_title(
            f"Vertex-Centered\nGrid: {slice_vertex.shape[0]}x{slice_vertex.shape[1]} (even)"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        # Draw grid lines at cell boundaries
        for coord in vertex_coords:
            ax.axhline(coord - hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axhline(coord + hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axvline(coord - hr / 2, color="gray", linewidth=0.3, alpha=0.5)
            ax.axvline(coord + hr / 2, color="gray", linewidth=0.3, alpha=0.5)
        # Mark stem at origin (should be at vertex between 4 cells)
        ax.plot(0, 0, "k+", markersize=20, markeredgewidth=3, label="Stem position")
        # Draw crown circle at this z-height
        circle = plt.Circle(
            (0, 0),
            crown_r,
            fill=False,
            color="blue",
            linewidth=2,
            linestyle="--",
            label=f"Crown radius ({crown_r:.1f}m)",
        )
        ax.add_patch(circle)
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label="Volume fraction")

        plt.suptitle(
            f"Crown Profile Slice at z-index={z_mid_cell} "
            f"(height={z_height:.1f}m, tree height={test_tree.height:.1f}m)"
        )
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_centering_crown_slice.png", dpi=150)
        if self.show_fig:
            plt.show()
        plt.close()

        # Assertions
        assert slice_cell.shape[0] % 2 == 1
        assert slice_cell.shape[1] % 2 == 1
        assert slice_vertex.shape[0] % 2 == 0
        assert slice_vertex.shape[1] % 2 == 0

    def test_visualize_grid_overlay(self):
        """Visualize crown profile with grid overlay showing cell boundaries."""
        test_tree = Tree(122, 1, 24, 20.66443, 0.5)
        hr = 1.0

        cell_coords = _get_horizontal_tree_coords(
            hr, test_tree.max_crown_radius, centering="cell"
        )
        vertex_coords = _get_horizontal_tree_coords(
            hr, test_tree.max_crown_radius, centering="vertex"
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, coords, title, centering in [
            (axes[0], cell_coords, "Cell-Centered", "cell"),
            (axes[1], vertex_coords, "Vertex-Centered", "vertex"),
        ]:
            # Draw grid cells
            for x in coords:
                for y in coords:
                    rect = plt.Rectangle(
                        (x - hr / 2, y - hr / 2),
                        hr,
                        hr,
                        fill=False,
                        edgecolor="lightgray",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)
                    # Mark cell centers
                    ax.plot(x, y, "b.", markersize=3)

            # Draw crown circle
            crown_radius = test_tree.max_crown_radius
            circle = plt.Circle(
                (0, 0),
                crown_radius,
                fill=False,
                color="green",
                linewidth=2,
                label="Crown radius",
            )
            ax.add_patch(circle)

            # Mark stem position
            ax.plot(0, 0, "r+", markersize=15, markeredgewidth=2, label="Stem position")

            ax.set_xlim(-crown_radius - hr, crown_radius + hr)
            ax.set_ylim(-crown_radius - hr, crown_radius + hr)
            ax.set_aspect("equal")
            ax.set_title(f"{title}\nGrid: {len(coords)}x{len(coords)}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(loc="upper right")

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(FIGURES_PATH / "test_centering_grid_overlay.png", dpi=150)
        if self.show_fig:
            plt.show()
        plt.close()

        # Assertions
        assert len(cell_coords) % 2 == 1
        assert len(vertex_coords) % 2 == 0


class TestComputeHorizontalProbability:
    def test_alpha_zero_creates_uniform_distribution(self):
        alpha = 0
        mask = np.ones((10, 10, 10))
        expected = np.ones_like(mask)
        actual = _compute_horizontal_probability(mask, alpha)
        assert np.allclose(actual, expected)

    def test_alpha_zero_with_mask(self):
        alpha = 0
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_horizontal_probability(mask, alpha)

        assert np.all(result[outside_mask] == 0)
        assert np.all(result[inside_mask] == 1)

    def test_zero_mask_results_in_zero_probability(self):
        beta = 1
        mask = np.zeros((10, 10, 10))
        actual = _compute_horizontal_probability(mask, beta)
        assert np.all(actual == 0)

    def test_positive_alpha_pushes_probabilities_to_edges(self):
        alpha = 1
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_horizontal_probability(mask, alpha)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask]) == 0

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask]) > 0

        # The probability is lowest at the center of the mask and increases
        # as we move away from the center (in the xy plane)
        for i in range(45, 69):
            assert np.all(result[:, i, i] <= result[:, i + 1, i + 1])

        # The min and max probability are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

    def test_range_positive_alphas(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        for alpha in np.random.random(100):
            result = _compute_horizontal_probability(mask, alpha)

            # The probability is 0 outside the mask
            assert np.all(result[outside_mask]) == 0

            # The probability is > 0 everywhere inside the mask
            assert np.all(result[inside_mask]) > 0

            # The probability is lowest at the center of the mask and increases
            # as we move away from the center (in the xy plane)
            for i in range(4, 6):
                assert np.all(result[:, i, i] <= result[:, i + 1, i + 1])

            # The min and max probability are 0 and 1
            assert np.min(result) == 0
            assert np.max(result) == 1

    def test_negative_alpha_pushes_probabilities_to_center(self):
        alpha = -1
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_horizontal_probability(mask, alpha)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask]) == 0

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask]) > 0

        # The probability is highest at the center of the mask and decreases
        # as we move away from the center (in the xy plane)
        for i in range(45, 69):
            assert np.all(result[:, i, i] >= result[:, i + 1, i + 1])

        # The minimum and maximum probabilities are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

    def test_range_negative_alphas(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        for alpha in -np.random.random(100):
            result = _compute_horizontal_probability(mask, alpha)

            # The probability is 0 outside the mask
            assert np.all(result[outside_mask]) == 0

            # The probability is > 0 everywhere inside the mask
            assert np.all(result[inside_mask]) > 0

            # The probability is highest at the center of the mask and decreases
            # as we move away from the center (in the xy plane)
            for i in range(4, 6):
                assert np.all(result[:, i, i] >= result[:, i + 1, i + 1])

            # The minimum and maximum probabilities are 0 and 1
            assert np.min(result) == 0
            assert np.max(result) == 1

    def test_higher_alpha_pushes_more_probability_to_edges(self):
        alpha_1 = 0.5
        alpha_2 = 1.0
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        result_1 = _compute_horizontal_probability(mask, alpha_1)
        result_2 = _compute_horizontal_probability(mask, alpha_2)

        # Assert that result_2 pushes more probability to the edges
        for i in range(45, 69):
            assert np.all(result_1[:, i, i] >= result_2[:, i, i])

        # The maximum probability for both alpha values is 1
        assert np.max(result_1) == 1
        assert np.max(result_2) == 1

    # TODO: TEST MASK VALUES LIKE MASK[20:70, 20:70, 20:70] = 0.5


class TestComputeVerticalProbability:
    def test_beta_zero_creates_uniform_distribution(self):
        beta = 0
        mask = np.ones((10, 10, 10))
        expected = np.ones_like(mask)
        actual = _compute_vertical_probability(mask, beta)
        assert np.allclose(actual, expected)

    def test_beta_zero_with_mask(self):
        alpha = 0
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_vertical_probability(mask, alpha)

        assert np.all(result[outside_mask] == 0)
        assert np.all(result[inside_mask] == 1)

    def test_zero_mask_results_in_zero_probability(self):
        beta = 1
        mask = np.zeros((10, 10, 10))
        actual = _compute_vertical_probability(mask, beta)
        assert np.all(actual == 0)

    def test_positive_beta_pushes_probabilities_to_top(self):
        beta = 1
        mask = np.zeros((90, 90, 90))
        mask[:, 20:70, 20:70] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_vertical_probability(mask, beta)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask]) == 0

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask]) > 0

        # The probability is lowest at the bottom of the mask and increases
        # as we move towards the top (along the z-axis)
        for i in range(89):
            assert np.all(result[i, ...] <= result[i + 1, ...])

        # The minimum and maximum probabilities are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

    def test_range_positive_betas(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        for beta in np.random.random(100):
            result = _compute_vertical_probability(mask, beta)
            # The probability is 0 outside the mask
            assert np.all(result[outside_mask]) == 0

            # The probability is > 0 everywhere inside the mask
            assert np.all(result[inside_mask]) > 0

            # The probability is lowest at the bottom of the mask and increases
            # as we move towards the top (along the z-axis)
            for i in range(8):
                assert np.all(result[i, :, :] <= result[i + 1, :, :])

            # The minimum and maximum probabilities are 0 and 1
            assert np.min(result) == 0
            assert np.max(result) == 1

    def test_higher_beta_values_increase_exponentially(self):
        beta_1 = 1
        beta_2 = 2
        mask = np.ones((10, 10, 10))
        actual_1 = _compute_vertical_probability(mask, beta_1)
        actual_2 = _compute_vertical_probability(mask, beta_2)

        # Assert that actual_2 is less than actual_1 for all z, except at the
        # top of the mask, where both are 1
        for z in range(9):
            assert np.all(actual_1[z, :, :] >= actual_2[z, :, :])

        # The maximum probability for both beta values is 1
        assert np.max(actual_1) == 1
        assert np.max(actual_2) == 1


class TestComputeJointProbability:

    def test_uniform_joint_probability(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_joint_probability(mask=mask, alpha=0, beta=0)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask] == 0)

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask] > 0)

        # The minimum and maximum probabilities are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

        # Probability is the same everywhere inside the mask
        assert np.all(result[inside_mask] == result[inside_mask][0])

    def test_positive_alpha_zero_beta(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_joint_probability(mask=mask, alpha=1, beta=0)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask] == 0)

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask] > 0)

        # Minimum and maximum probabilities are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

        # The horizontal probability (xy plane) is lowest at the edges of the
        # mask and increases as we move towards the center
        for i in range(4, 6):
            assert np.all(result[:, i, i] <= result[:, i + 1, i + 1])

        # The vertical probability is uniform along the z-axis
        for i in range(8):
            assert np.all(result[i, ...] == result[i + 1, ...])

    def test_zero_alpha_positive_beta(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        result = _compute_joint_probability(mask=mask, alpha=0, beta=1)

        # The probability is 0 outside the mask
        assert np.all(result[outside_mask] == 0)

        # The probability is > 0 everywhere inside the mask
        assert np.all(result[inside_mask] > 0)

        # Minimum and maximum probabilities are 0 and 1
        assert np.min(result) == 0
        assert np.max(result) == 1

        # The horizontal probability (xy plane) is uniform along the x/y axes
        for i in range(2, 6):
            assert np.all(result[:, i, i] == result[:, i + 1, i + 1])

        # The vertical probability is lowest at the bottom of the mask and
        # increases as we move towards the top (along the z-axis)
        for i in range(8):
            assert np.all(result[i, ...] <= result[i + 1, ...])

    def test_positive_alpha_positive_beta(self):
        mask = np.zeros((9, 9, 9))
        mask[:, 2:7, 2:7] = 1
        inside_mask = np.where(mask == 1)
        outside_mask = np.where(mask == 0)
        for alpha in np.random.random(10):
            for beta in np.random.random(10):
                result = _compute_joint_probability(mask=mask, alpha=alpha, beta=beta)

                # The probability is 0 outside the mask
                assert np.all(result[outside_mask] == 0)

                # The probability is > 0 everywhere inside the mask
                assert np.all(result[inside_mask] > 0)

                # Minimum and maximum probabilities are 0 and 1
                assert np.min(result) == 0
                assert np.max(result) == 1

                # The horizontal probability (xy plane) is lowest at the edges
                # of the mask and increases as we move towards the center
                for i in range(4, 6):
                    assert np.all(result[:, i, i] <= result[:, i + 1, i + 1])

                # The vertical probability is lowest at the bottom of the mask
                # and increases as we move towards the top (along the z-axis)
                for i in range(8):
                    assert np.all(result[i, ...] <= result[i + 1, ...])

    def test_multiple_masks_equal(self):
        for j in range(10):
            alpha = np.random.random()
            beta = np.random.random()
            results_list = []
            for i in range(5):
                mask = np.zeros((9, 9, 9))
                mask[:, 2:7, 2:7] = 1
                result = _compute_joint_probability(mask=mask, alpha=alpha, beta=beta)
                results_list.append(result)

            # Assert that all results are equal
            for i in range(1, len(results_list)):
                assert np.all(results_list[i] == results_list[i - 1])


class TestSampleCrownVoxels:

    def test_basic_functionality(self):
        """Test basic functionality"""
        joint_probability = np.ones((3, 3, 3))
        n = 5
        result = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        assert np.sum(result) > 0
        assert np.count_nonzero(result) == n

    def test_all_zero_probabilities(self):
        """Test with all zero probabilities"""
        joint_probability = np.zeros((3, 3, 3))
        n = 5
        result = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        assert np.sum(result) == 0
        assert np.count_nonzero(result) == 0

    def test_n_greater_than_voxels(self):
        """Test with n greater than the number of voxels"""
        joint_probability = np.ones((3, 3, 3))
        n = joint_probability.size + 1
        with pytest.raises(ValueError):
            _sample_voxels_from_probability_grid(n, joint_probability, 42)

    def test_negative_n(self):
        """Test with negative n"""
        joint_probability = np.ones((3, 3, 3))
        n = -5
        with pytest.raises(ValueError):
            _sample_voxels_from_probability_grid(n, joint_probability, 42)

    def test_n_zero(self):
        """Test with n equals zero"""
        joint_probability = np.ones((3, 3, 3))
        n = 0
        result = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        assert np.sum(result) == 0
        assert np.count_nonzero(result) == 0

    def test_n_range(self):
        """Tests a range of values for n from 1 to the number of voxels"""
        joint_probability = np.ones((3, 3, 3))
        for n in range(1, joint_probability.size + 1):
            result = _sample_voxels_from_probability_grid(n, joint_probability, 42)
            assert np.sum(result) > 0
            assert np.count_nonzero(result) == n

    def test_non_uniform_probabilities(self):
        """Test with non-uniform probabilities"""
        joint_probability = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            ]
        )
        n = joint_probability.size // 2
        result = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        assert np.sum(result) > 0
        assert np.count_nonzero(result) == n

        # Check that cells with higher probabilities are more likely to be selected
        selected = np.where(result > 0, 1, 0)
        selected_cells_prob = np.sum(joint_probability[selected == 1])
        unselected_cells_prob = np.sum(joint_probability[selected == 0])
        assert selected_cells_prob > unselected_cells_prob

    def test_same_seed(self):
        """Test if the random seed works as expected"""
        joint_probability = np.ones((3, 3, 3))
        n = 5

        result1 = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        result2 = _sample_voxels_from_probability_grid(n, joint_probability, 42)

        assert np.array_equal(result1, result2)

    def test_different_seed(self):
        """Test if the random seed works as expected"""
        joint_probability = np.ones((3, 3, 3))
        n = 5

        result1 = _sample_voxels_from_probability_grid(n, joint_probability, 42)
        result2 = _sample_voxels_from_probability_grid(n, joint_probability, 43)

        assert not np.array_equal(result1, result2)


class TestVoxelizedTree:
    def test_distribute_biomass(self):
        for _ in range(1000):
            tree = make_random_tree(status_code=1)
            voxelized_tree = voxelize_tree(tree, 1, 1)
            biomass_array = voxelized_tree.distribute_biomass()
            distributed_biomass = np.sum(biomass_array)
            assert np.allclose(distributed_biomass, tree.foliage_biomass)

    # Centering mode tests
    def test_default_centering_is_cell(self):
        """Default centering is 'cell' for backward compatibility."""
        tree = make_random_tree(status_code=1)
        voxelized_tree = voxelize_tree(tree, 1, 1)
        assert voxelized_tree.centering == "cell"

    def test_centering_attribute_cell(self):
        """VoxelizedTree stores centering='cell'."""
        tree = make_random_tree(status_code=1)
        voxelized_tree = voxelize_tree(tree, 1, 1, centering="cell")
        assert voxelized_tree.centering == "cell"

    def test_centering_attribute_vertex(self):
        """VoxelizedTree stores centering='vertex'."""
        tree = make_random_tree(status_code=1)
        voxelized_tree = voxelize_tree(tree, 1, 1, centering="vertex")
        assert voxelized_tree.centering == "vertex"

    def test_voxelize_tree_cell_odd_dimensions(self):
        """voxelize_tree with cell centering produces odd dimensions."""
        tree = make_random_tree(status_code=1)
        vt = voxelize_tree(tree, 1, 1, centering="cell")
        assert vt.grid.shape[1] % 2 == 1
        assert vt.grid.shape[2] % 2 == 1

    def test_voxelize_tree_vertex_even_dimensions(self):
        """voxelize_tree with vertex centering produces even dimensions."""
        tree = make_random_tree(status_code=1)
        vt = voxelize_tree(tree, 1, 1, centering="vertex")
        assert vt.grid.shape[1] % 2 == 0
        assert vt.grid.shape[2] % 2 == 0

    def test_distribute_biomass_cell_centered(self):
        """distribute_biomass works with cell centering."""
        for _ in range(100):
            tree = make_random_tree(status_code=1)
            vt = voxelize_tree(tree, 1, 1, centering="cell")
            biomass_array = vt.distribute_biomass()
            distributed_biomass = np.sum(biomass_array)
            assert np.allclose(distributed_biomass, tree.foliage_biomass)

    def test_distribute_biomass_vertex_centered(self):
        """distribute_biomass works with vertex centering."""
        for _ in range(100):
            tree = make_random_tree(status_code=1)
            vt = voxelize_tree(tree, 1, 1, centering="vertex")
            biomass_array = vt.distribute_biomass()
            distributed_biomass = np.sum(biomass_array)
            assert np.allclose(distributed_biomass, tree.foliage_biomass)

    def test_voxelize_tree_kwargs_with_centering(self):
        """Other kwargs (alpha, beta, etc.) work with centering parameter."""
        tree = make_random_tree(status_code=1)
        vt = voxelize_tree(tree, 1, 1, centering="vertex", alpha=0.3, beta=0.7, seed=42)
        assert vt.centering == "vertex"
        assert vt.grid.shape[1] % 2 == 0


class TestComputeIntersectionArea:
    """Tests that the optimized _compute_intersection_area produces identical
    results to the original per-cell implementation."""

    @staticmethod
    def _compute_intersection_area_old(
        x_center: np.ndarray,
        y_center: np.ndarray,
        length: float,
        radius: np.ndarray,
        exact=False,
    ):
        """Original implementation of _compute_intersection_area using per-cell edge
        and distance computations. Preserved here as a reference for verifying the
        optimized implementation produces identical results."""
        # Calculate location of cell edges using the cell center and length
        x_right = x_center + length / 2
        x_left = x_center - length / 2
        y_top = y_center + length / 2
        y_bottom = y_center - length / 2

        # Calculate the distance from the origin to each cell's corner points
        top_left_dist = np.sqrt(x_left**2 + y_top**2)
        top_right_dist = np.sqrt(x_right**2 + y_top**2)
        bottom_left_dist = np.sqrt(x_left**2 + y_bottom**2)
        bottom_right_dist = np.sqrt(x_right**2 + y_bottom**2)

        # Check if each corner of the cell is inside the circle
        top_left_in = top_left_dist < radius
        top_right_in = top_right_dist < radius
        bottom_left_in = bottom_left_dist < radius
        bottom_right_in = bottom_right_dist < radius

        case_index = _encode_corners(
            top_left_in, top_right_in, bottom_left_in, bottom_right_in
        )

        area = _compute_intersection_area_by_case(
            case_index, length, x_left, x_right, y_bottom, y_top, radius, exact
        )

        return area

    def test_equivalence_simple(self):
        """Simple case: small grid, multiple radii."""
        x_pts = np.array([0.0, 1.0, 2.0])
        y_pts = np.array([2.0, 1.0, 0.0])
        r_at_height = np.array([1.5, 2.0, 2.5])
        hr = 1.0

        # Old method (meshgrid approach)
        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(x_grid, y_grid, hr, r_grid)

        # New method (pre-computed edges)
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_exact(self):
        """Equivalence with exact circular segment computation."""
        x_pts = np.array([0.0, 1.0, 2.0, 3.0])
        y_pts = np.array([3.0, 2.0, 1.0, 0.0])
        r_at_height = np.array([2.0, 3.0, 3.5, 4.0, 5.0])
        hr = 1.0

        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(
            x_grid, y_grid, hr, r_grid, exact=True
        )
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_fractional_resolution(self):
        """Equivalence with hr=0.5 matching realistic usage."""
        hr = 0.5
        x_pts = np.arange(0, 5.5, hr)
        y_pts = np.flip(x_pts)
        r_at_height = np.linspace(0.5, 4.0, 50)

        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(
            x_grid, y_grid, hr, r_grid, exact=True
        )
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_fine_resolution(self):
        """Equivalence with hr=0.1 for high-resolution grids."""
        hr = 0.1
        x_pts = np.arange(0, 3.1, hr)
        y_pts = np.flip(x_pts)
        r_at_height = np.linspace(0.1, 2.5, 100)

        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(
            x_grid, y_grid, hr, r_grid, exact=True
        )
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_zero_radius(self):
        """Equivalence when some radii are zero."""
        x_pts = np.array([0.0, 1.0, 2.0])
        y_pts = np.array([2.0, 1.0, 0.0])
        r_at_height = np.array([0.0, 1.0, 0.0, 2.0])
        hr = 1.0

        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(x_grid, y_grid, hr, r_grid)
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_single_cell(self):
        """Equivalence with a single cell grid."""
        x_pts = np.array([0.0])
        y_pts = np.array([0.0])
        r_at_height = np.array([0.25, 0.5, 0.75])
        hr = 1.0

        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        area_old = self._compute_intersection_area_old(x_grid, y_grid, hr, r_grid)
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr)

        assert area_old.shape == area_new.shape
        assert np.allclose(area_old, area_new)

    def test_equivalence_random_trees(self):
        """Integration: full discretize_crown_profile pipeline with random trees."""
        resolutions = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0))
        for _ in range(100):
            tree = make_random_tree()
            hr, vr = random.choice(resolutions)
            grid = discretize_crown_profile(tree, hr, vr)
            assert np.sum(grid) > 0


class TestPerformanceBenchmark:
    """Performance comparison between old and new implementations."""

    def test_benchmark_large_grid(self):
        import time

        hr = 0.5
        x_pts = np.arange(0, 10.5, hr)
        y_pts = np.flip(x_pts)
        r_at_height = np.linspace(0.5, 9.0, 300)
        n_iterations = 1000

        # Old method
        r_grid, y_grid, x_grid = np.meshgrid(r_at_height, y_pts, x_pts, indexing="ij")
        # Warmup
        TestComputeIntersectionArea._compute_intersection_area_old(
            x_grid, y_grid, hr, r_grid, exact=True
        )
        start = time.perf_counter()
        for _ in range(n_iterations):
            TestComputeIntersectionArea._compute_intersection_area_old(
                x_grid, y_grid, hr, r_grid, exact=True
            )
        time_old = time.perf_counter() - start

        # New method
        # Warmup
        _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)
        start = time.perf_counter()
        for _ in range(n_iterations):
            _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)
        time_new = time.perf_counter() - start

        speedup = time_old / time_new
        print(
            f"\nBenchmark (hr=0.5, nx=ny=21, nz=300, {n_iterations} iterations):"
            f"\n  Old: {time_old:.4f}s"
            f"\n  New: {time_new:.4f}s"
            f"\n  Speedup: {speedup:.2f}x"
        )

        # Verify results are equivalent
        area_old = TestComputeIntersectionArea._compute_intersection_area_old(
            x_grid, y_grid, hr, r_grid, exact=True
        )
        area_new = _compute_intersection_area(x_pts, y_pts, r_at_height, hr, exact=True)
        assert np.allclose(area_old, area_new)


class TestTreeVoxelizeMethod:
    """Tests for Tree.voxelize() method with centering parameter."""

    def test_default_centering(self):
        """Default centering is 'cell'."""
        tree = Tree(122, 1, 24, 20.66443, 0.5)
        vt = tree.voxelize(1.0, 1.0)
        assert vt.centering == "cell"

    def test_cell_centering_explicit(self):
        """Explicit cell centering."""
        tree = Tree(122, 1, 24, 20.66443, 0.5)
        vt = tree.voxelize(1.0, 1.0, centering="cell")
        assert vt.centering == "cell"
        assert vt.grid.shape[1] % 2 == 1

    def test_vertex_centering(self):
        """Vertex centering via Tree.voxelize()."""
        tree = Tree(122, 1, 24, 20.66443, 0.5)
        vt = tree.voxelize(1.0, 1.0, centering="vertex")
        assert vt.centering == "vertex"
        assert vt.grid.shape[1] % 2 == 0

    def test_kwargs_pass_through(self):
        """Other kwargs pass through correctly."""
        tree = Tree(122, 1, 24, 20.66443, 0.5)
        vt = tree.voxelize(1.0, 1.0, centering="vertex", alpha=0.3, seed=42)
        assert vt.centering == "vertex"

    def test_various_resolutions(self):
        """Tree.voxelize works at various resolutions with both modes."""
        tree = Tree(122, 1, 24, 20.66443, 0.5)
        for hr, vr in [(0.5, 0.5), (1.0, 1.0), (2.0, 1.0)]:
            for centering in ["cell", "vertex"]:
                vt = tree.voxelize(hr, vr, centering=centering)
                assert vt.centering == centering
                if centering == "cell":
                    assert vt.grid.shape[1] % 2 == 1
                else:
                    assert vt.grid.shape[1] % 2 == 0
