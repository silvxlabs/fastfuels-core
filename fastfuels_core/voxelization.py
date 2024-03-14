# Core imports
from __future__ import annotations
from typing import TYPE_CHECKING

# Internal imports
if TYPE_CHECKING:
    from fastfuels_core.trees import Tree

# External imports
import numpy as np
from numpy import ndarray
from scipy.ndimage import distance_transform_edt


class VoxelizedTree:
    def __init__(self, tree: "Tree", grid: ndarray, hr, vr):
        self.tree = tree
        self.grid = grid
        self.hr = hr
        self.vr = vr

    def distribute_biomass(self):
        volume = np.sum(self.grid) * self.hr * self.hr * self.vr
        foliage_mpv = self.tree.foliage_biomass / volume
        biomass_grid = self.grid.copy() * foliage_mpv
        return biomass_grid


def voxelize_tree(
    tree: "Tree",
    horizontal_resolution: float,
    vertical_resolution: float,
    **kwargs,
) -> VoxelizedTree:

    crown_profile_mask = discretize_crown_profile(
        tree, horizontal_resolution, vertical_resolution
    )

    alpha = kwargs.get("alpha", 0.5)
    beta = kwargs.get("beta", 0.5)
    rho = kwargs.get("rho", None)
    seed = kwargs.get("seed", None)

    sampled_crown_mask = sample_occupied_cells(
        crown_profile_mask, alpha, beta, rho, seed
    )
    return VoxelizedTree(
        tree, sampled_crown_mask, horizontal_resolution, vertical_resolution
    )


def discretize_crown_profile(
    tree: "Tree", hr: float, vr: float, full_intersection=True
) -> ndarray:
    # Get the horizontal and vertical coordinates of the tree crown
    horizontal_coords = _get_horizontal_tree_coords(hr, tree.max_crown_radius)
    z_pts = _get_vertical_tree_coords(vr, tree.height, tree.crown_base_height)

    # Slice the horizontal coordinates to get the first quadrant of the xy plane
    q2_slice = slice(len(horizontal_coords) // 2, None)
    x_pts_q2 = horizontal_coords[q2_slice]
    y_pts_q2 = np.flip(x_pts_q2)

    q2_grid = _discretize_crown_profile_quadrant(
        tree, x_pts_q2, y_pts_q2, z_pts, hr, vr, full_intersection
    )

    # Build the other quadrants by flipping the q2 grid about the x and y axes
    # Note that q2 grid has dimensions (z, y, x)
    q1_grid = np.flip(q2_grid, axis=2)
    q3_grid = np.flip(q2_grid, axis=1)
    q4_grid = np.flip(q3_grid, axis=2)

    return _align_quadrants(q1_grid, q2_grid, q3_grid, q4_grid)


def _get_vertical_tree_coords(step, tree_height, crown_base_height):
    """
    Returns a grid of coordinates for a tree of height, height, with a spacing
    step. The grid is returned as a 1D array.
    """
    grid = np.arange(crown_base_height, tree_height + step, step)
    return grid


def _get_horizontal_tree_coords(step, radius, pos=0.0):
    """
    Discretizes a stem position and crown radius into a 1D array of coordinates
    centered at pos, with a spacing step. The grid has an odd number of cells.
    The grid is rounded out to include one cell beyond the crown radius.
    """
    cells_per_side = int(np.floor(np.abs(radius / step))) + 1
    lower_bound = pos - cells_per_side * step
    upper_bound = pos + cells_per_side * step
    grid = np.linspace(lower_bound, upper_bound, 2 * cells_per_side + 1)
    return grid


def _discretize_crown_profile_quadrant(
    tree: "Tree", x_pts, y_pts, z_pts, hr, vr, full_intersection=False
):
    """
    Build a 3D grid of a quadrant of a tree crown represented as a rotational
    solid.
    """
    # Create a subgrid of the z-axis to increase the resolution of the crown
    vr_subgrid = 0.1
    num_subgrid_cells_per_z = int(vr / vr_subgrid)
    z_pts_subgrid = _resample_coords_grid_to_subgrid(z_pts, vr, vr_subgrid)
    r_at_height_z = tree.get_crown_radius_at_height(z_pts_subgrid)

    # Create a 3D grid of the tree crown and compute the area of intersection
    # between the tree crown and each cell of the grid along the z-axis
    r_grid, y_grid, x_grid = np.meshgrid(r_at_height_z, y_pts, x_pts, indexing="ij")
    area = _compute_intersection_area(x_grid, y_grid, hr, r_grid, full_intersection)

    # Convert the area of intersection to a volume fraction by summing the area
    # along the z-axis and dividing by the cell volume
    volume_subgrid = area * vr_subgrid
    volume = _sum_area_along_axis(volume_subgrid, 0, num_subgrid_cells_per_z)
    volume_fraction = np.minimum(volume / (hr * hr * vr), 1.0)

    return volume_fraction


def _resample_coords_grid_to_subgrid(
    grid: ndarray, grid_spacing: float, new_spacing: float
) -> ndarray:
    """
    Resamples grid with spacing grid_spacing to a subgrid with spacing
    new_spacing.
    """
    subgrid = np.arange(
        grid[0] - grid_spacing / 2 + new_spacing / 2,
        grid[-1] + grid_spacing / 2,
        new_spacing,
    )

    return subgrid


def _compute_intersection_area(
    x_center: ndarray, y_center: ndarray, length: float, radius: ndarray, exact=False
):
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


def _encode_corners(
    top_left_inside, top_right_inside, bottom_left_inside, bottom_right_inside
):
    """
    Encode the inside/outside status of the corners of a cell into an index.
    Each array of inside/outside status is a boolean array of the same shape
    as the cell grid. The index is computed as the sum of the inside/outside
    status of each corner in binary representation.
    """
    case_index = (
        top_left_inside * 8
        + top_right_inside * 4
        + bottom_right_inside * 2
        + bottom_left_inside
    )
    return case_index


def _compute_intersection_area_by_case(
    case_index, length, left, right, bottom, top, radius, exact=False
):
    # Initialize an array to hold the computed areas
    areas = np.zeros_like(case_index, dtype=float)

    # Case 0a: No corners inside, no intersection
    areas[case_index == 0] = 0.0

    # Case 0b: No corners inside, but circle is inside cell
    areas[:, -1, 0] = np.pi * radius[:, -1, 0] ** 2

    # Case 1: Bottom left inside
    case_1 = case_index == 1
    areas[case_1] = _calculate_case_1_area(
        left[case_1], bottom[case_1], radius[case_1], exact
    )

    # Case 3: Bottom left and bottom right inside
    case_3 = case_index == 3
    areas[case_3] = _calculate_case_3_area(
        left[case_3],
        right[case_3],
        bottom[case_3],
        radius[case_3],
        length,
        exact,
    )

    # Case 9: Top left and bottom left inside
    case_9 = case_index == 9
    areas[case_9] = _calculate_case_9_area(
        top[case_9],
        bottom[case_9],
        left[case_9],
        radius[case_9],
        length,
        exact,
    )

    # Case 11: Top left, bottom left, and bottom right inside
    case_11 = case_index == 11
    areas[case_11] = _calculate_case_11_area(
        top[case_11], right[case_11], radius[case_11], length, exact
    )

    # Case 15: All corners inside, full cell area
    areas[case_index == 15] = np.square(length)

    return areas


def _calculate_case_1_area(left_edge, bottom_edge, radius, exact=False):
    """
    Calculate the area of intersection between a circle and a cell when the
    bottom left corner of the cell is inside the circle.

    Area is approximated as a triangle whose height is the distance from the
    bottom left corner of the cell to the circle's intersection with the
    left edge of the cell, and whose base is the distance from the bottom
    left corner of the cell to the circle's intersection with the bottom
    edge of the cell.

    For an exact area calculation, the area of the circular segment is added to
    the area of the triangle.
    """
    p_x = left_edge
    p_y = _find_circle_cell_intersection_coord(radius, p_x)
    q_y = bottom_edge
    q_x = _find_circle_cell_intersection_coord(radius, q_y)

    triangle_area = _compute_triangle_area(p_y - bottom_edge, q_x - left_edge)
    if exact:
        circular_segment_area = _compute_circle_segment_area(p_x, p_y, q_x, q_y, radius)
        return triangle_area + circular_segment_area

    return triangle_area


def _calculate_case_3_area(
    left_edge, right_edge, bottom_edge, radius, length, exact=False
):
    """
    Calculate the area of intersection between a circle and a cell when the
    bottom left and bottom right corners of the cell are inside the circle.

    Area is approximated by the areas of a triangle and a rectangle. The
    triangle is given by a polygon whose height is the distance from the
    height of the circle's intersection with the cell on the right edge to
    the height of the circle's intersection with the cell on the left edge,
    and whose base is the width of the cell. The rectangle's height is given by
    the distance from the bottom right corner to the circle's intersection with
    the right edge, and its width is the width of the cell.

    For an exact area calculation, the area of the circular segment is added to
    the area of the triangle and rectangle.
    """
    p_x = left_edge
    p_y = _find_circle_cell_intersection_coord(radius, p_x)
    q_x = right_edge
    q_y = _find_circle_cell_intersection_coord(radius, q_x)

    triangle_area = _compute_triangle_area(p_y - q_y, length)
    rectangle_area = _compute_rectangle_area(q_y - bottom_edge, length)
    if exact:
        circular_segment_area = _compute_circle_segment_area(p_x, p_y, q_x, q_y, radius)
        return triangle_area + rectangle_area + circular_segment_area

    return triangle_area + rectangle_area


def _calculate_case_9_area(
    top_edge, bottom_edge, left_edge, radius, length, exact=False
):
    """
    Calculate the area of intersection between a circle and a cell when the
    top left and bottom left corners of the cell are inside the circle.

    Area is approximated by the areas of a triangle and a rectangle. The
    triangle is given by a polygon whose height is the height of the cell,
    and whose base is the width from the circle's intersection with the top
    edge to the circle's intersection with the bottom edge. The rectangle's
    height is given by the height of the cell, and its width is the distance
    from the bottom left corner to the circle's intersection with the bottom
    edge.

    For an exact area calculation, the area of the circular segment is added to
    the area of the triangle and rectangle.
    """
    p_y = top_edge
    p_x = _find_circle_cell_intersection_coord(radius, p_y)
    q_y = bottom_edge
    q_x = _find_circle_cell_intersection_coord(radius, q_y)

    triangle_area = _compute_triangle_area(q_x - p_x, length)
    rectangle_area = _compute_rectangle_area(p_x - left_edge, length)
    if exact:
        circular_segment_area = _compute_circle_segment_area(p_x, p_y, q_x, q_y, radius)
        return triangle_area + circular_segment_area + rectangle_area

    return triangle_area + rectangle_area


def _calculate_case_11_area(top_edge, right_edge, radius, length, exact=False):
    """
    Calculate the area of intersection between a circle and a cell when the
    top left, bottom left, and bottom right corners of the cell are inside
    the circle.

    Area is approximated by the area of a trapezoid. The area of the trapezoid
    is given by the area of the cell, minus the area of the triangle formed by
    the circle's intersection with the top edge and the circle's intersection
    with the right edge.

    For an exact area calculation, the area of the circular segment is
    subtracted from the area of the triangle.
    """
    p_y = top_edge
    p_x = _find_circle_cell_intersection_coord(radius, p_y)
    q_x = right_edge
    q_y = _find_circle_cell_intersection_coord(radius, q_x)

    triangle_area = _compute_triangle_area(right_edge - p_x, top_edge - q_y)
    if exact:
        circular_segment_area = _compute_circle_segment_area(p_x, p_y, q_x, q_y, radius)
        return length**2 - (triangle_area - circular_segment_area)

    return length**2 - triangle_area


def _find_circle_cell_intersection_coord(radius, known_coord):
    return np.sqrt(radius**2 - known_coord**2)


def _compute_circle_segment_area(p_x, p_y, q_x, q_y, radius):
    # Compute chord length as the distance between the two intersection points
    chord_length = np.sqrt((p_x - q_x) ** 2 + (p_y - q_y) ** 2)

    # Compute the angle subtended by the circular segment
    theta = _compute_central_angle(chord_length, radius)

    # Compute the area of the circular segment
    circular_segment_area = 0.5 * radius**2 * (theta - np.sin(theta))

    return np.nan_to_num(circular_segment_area)


def _compute_central_angle(chord_length, radius):
    return 2 * np.arcsin(chord_length / (2 * radius))


def _compute_triangle_area(base, height):
    return 0.5 * base * height


def _compute_rectangle_area(base, height):
    return base * height


def _sum_area_along_axis(area: ndarray, axis: int, cells_per_axis: int) -> ndarray:
    """Sum the area along a specified axis to the desired grid resolution."""
    try:
        return np.add.reduceat(
            area, np.arange(0, area.shape[axis], cells_per_axis), axis=axis
        )
    except IndexError:
        raise ValueError("Invalid axis index.")


def _align_quadrants(q1, q2, q3, q4):
    """
    Align four quadrants into a single grid. This function assumes that the
    four quadrants share the same origin and that the quadrants are aligned
    along the x and y axes.
    """
    # Create a grid to hold the four quadrants
    num_x = q1.shape[2] + q2.shape[2] - 1
    num_y = q1.shape[1] + q3.shape[1] - 1
    num_z = q1.shape[0]
    grid = np.zeros((num_z, num_y, num_x))

    mid_x = num_x // 2
    mid_y = num_y // 2

    grid[:, : mid_y + 1, : mid_x + 1] = q1
    grid[:, : mid_y + 1, mid_x:] = q2
    grid[:, mid_y:, mid_x:] = q3
    grid[:, mid_y:, : mid_x + 1] = q4

    return grid


def sample_occupied_cells(
    volume_fraction_array: ndarray,
    alpha: float,
    beta: float,
    rho: float = None,
    seed: int = None,
) -> ndarray:
    # Create a probability mask for the crown grid
    mask_bool = np.where(volume_fraction_array > 0.0, 1.0, 0.0)
    mask_prob = _compute_joint_probability(mask_bool, alpha, beta)

    # Choose n voxels from the joint probability grid
    if rho is None:
        rho = _estimate_crown_density(np.sum(mask_bool))
    n = int(np.count_nonzero(mask_bool) * rho)
    sampled = _sample_voxels_from_probability_grid(n, mask_prob, seed)

    # Make non-zero selected voxels 1
    selected = np.where(sampled > 0, 1.0, 0.0)

    return selected * volume_fraction_array


def _compute_joint_probability(mask: ndarray, alpha: float, beta: float) -> ndarray:
    """
    Combines the horizontal and vertical probability spatial to create a joint
    probability grid for the crown mask.
    """
    # Build the horizontal and vertical probability spatial
    horizontal_probability = _compute_horizontal_probability(mask, alpha)
    vertical_probability = _compute_vertical_probability(mask, beta)

    # Combine the probability spatial for joint probability
    joint_probability = horizontal_probability * vertical_probability
    joint_probability /= np.max(joint_probability)

    return joint_probability


def _compute_horizontal_probability(mask: ndarray, alpha: float) -> ndarray:
    """
    Builds a horizontal probability grid from a binary mask using a Euclidean
    Distance Transform (EDT). The function computes the EDT of the input
    mask, inverts the EDT values by subtracting them from the maximum value,
    applies a power function using the alpha parameter, and finally
    normalizes the result. Any NaN values in the final probability grid are
    replaced with 0.
    """
    # Compute the Euclidean Distance Transform (EDT) of the mask
    edt = distance_transform_edt(mask)

    # Inverse the distance transform
    # Add a small value to avoid zero probabilities
    max_dist = np.max(edt) + 1e-6
    edt = max_dist - edt
    edt[edt == max_dist] = 0

    # Apply the alpha parameter
    edt = np.nan_to_num(mask * edt**alpha)

    # Normalize and replace nans with 0
    horizontal_probability = edt / np.max(edt)
    horizontal_probability = np.nan_to_num(horizontal_probability)

    return horizontal_probability


def _compute_vertical_probability(mask: ndarray, beta: float) -> ndarray:
    """
    Builds a vertical probability grid from a binary mask. The function
    computes a 1D grid along the vertical axis (axis=2) of the input mask,
    raises this grid to the power of beta, and then broadcasts this
    transformed grid across the 2D plane of each layer of the mask (along
    axis=0 and axis=1), and multiplies it with the mask. Finally, the result
    is normalized and any NaN values are replaced with 0.
    """
    # Create a grid for the vertical axis (axis=2)
    z_grid = np.arange(mask.shape[0]).astype(float)

    # Compute the power of the vertical axis grid with beta
    z_power_beta = z_grid**beta

    # Add a small value to avoid zero probabilities
    z_power_beta += 0.01

    # Broadcast the vertical probability grid along axis=0 and axis=1
    vertical_probability = mask * z_power_beta[:, np.newaxis, np.newaxis]

    # Normalize and replace nans with 0
    vertical_probability /= np.max(vertical_probability)
    vertical_probability = np.nan_to_num(vertical_probability)

    return vertical_probability


def _estimate_crown_density(volume: float, threshold: float = 16) -> float:
    if volume < threshold:
        return 1
    return 0.5


def _sample_voxels_from_probability_grid(
    n: int, joint_probability: ndarray, seed: int = None
) -> ndarray:
    """
    Samples n voxels from the joint probability grid. Sampling is weighted by
    the joint probability of each voxel, such that voxels with higher joint
    probability are more likely to be sampled. Voxels are sampled without
    replacement.
    """
    if seed:
        np.random.seed(seed)

    # If joint probability is all zeros, return an empty array
    if np.all(joint_probability == 0):
        return joint_probability

    # Flatten and normalize the joint probability to sum to one
    jp_flat = joint_probability.flatten()
    jp_flat = jp_flat / np.sum(jp_flat)

    # If jp_flat contains nan values, return an empty array
    if np.any(np.isnan(jp_flat)):
        return joint_probability

    # Choose n indices from the flattened joint probability
    chosen_indices = np.random.choice(
        np.arange(joint_probability.size), n, replace=False, p=jp_flat
    )

    # Build flat selection array and reshape to original shape
    selected_flat = np.zeros(joint_probability.size)
    selected_flat[chosen_indices] = jp_flat[chosen_indices]
    return selected_flat.reshape(joint_probability.shape)
