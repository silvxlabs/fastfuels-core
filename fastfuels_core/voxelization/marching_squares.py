# Core imports
from __future__ import annotations
from typing import TYPE_CHECKING

# Internal imports
from fastfuels_core.voxelization._coords import (
    CenteringMode,
    _get_horizontal_tree_coords,
    _get_vertical_tree_coords,
    _resample_coords_grid_to_subgrid,
)

if TYPE_CHECKING:
    from fastfuels_core.trees import Tree

# External imports
import numpy as np
from numpy import ndarray


def discretize_crown_profile(
    tree: "Tree",
    hr: float,
    vr: float,
    full_intersection=True,
    centering: CenteringMode = "cell",
    n_vertical_subgrid: int = 10,
    z_origin: float = None,
) -> ndarray:
    # Each output z-cell is evaluated at n_vertical_subgrid sub-heights to
    # capture how the crown radius changes over the cell. Marching squares is
    # exact in the horizontal, so there is no horizontal subgrid; for finer
    # horizontal detail pass a finer hr.
    if n_vertical_subgrid < 1:
        raise ValueError(
            f"n_vertical_subgrid must be a positive integer, got {n_vertical_subgrid}"
        )
    n_vertical_subgrid = int(n_vertical_subgrid)

    # Get the horizontal and vertical coordinates of the tree crown
    horizontal_coords = _get_horizontal_tree_coords(
        hr, tree.max_crown_radius, centering=centering
    )
    z_pts = _get_vertical_tree_coords(vr, tree.height, tree.crown_base_height, z_origin)

    # Slice the horizontal coordinates to get the first quadrant of the xy plane
    q2_slice = slice(len(horizontal_coords) // 2, None)
    x_pts_q2 = horizontal_coords[q2_slice]
    y_pts_q2 = np.flip(x_pts_q2)

    q2_grid = _discretize_crown_profile_quadrant(
        tree, x_pts_q2, y_pts_q2, z_pts, hr, vr, full_intersection, n_vertical_subgrid
    )

    # Build the other quadrants by flipping the q2 grid about the x and y axes
    # Note that q2 grid has dimensions (z, y, x)
    q1_grid = np.flip(q2_grid, axis=2)
    q3_grid = np.flip(q2_grid, axis=1)
    q4_grid = np.flip(q3_grid, axis=2)

    return _align_quadrants(q1_grid, q2_grid, q3_grid, q4_grid, centering=centering)


def _discretize_crown_profile_quadrant(
    tree: "Tree",
    x_pts,
    y_pts,
    z_pts,
    hr,
    vr,
    full_intersection=False,
    n_vertical_subgrid=10,
):
    """
    Build a 3D grid of a quadrant of a tree crown represented as a rotational
    solid.
    """
    # Split each z-cell into n_vertical_subgrid sub-heights to resolve the crown
    vr_subgrid = vr / n_vertical_subgrid
    z_pts_subgrid = _resample_coords_grid_to_subgrid(z_pts, vr, n_vertical_subgrid)
    r_at_height_z = tree.get_crown_radius_at_height(z_pts_subgrid)

    # Compute the area of intersection between the tree crown and each cell
    area = _compute_intersection_area(
        x_pts, y_pts, r_at_height_z, hr, full_intersection
    )

    # Convert the area of intersection to a volume fraction by summing the area
    # along the z-axis and dividing by the cell volume
    volume_subgrid = area * vr_subgrid
    volume = _sum_area_along_axis(volume_subgrid, 0, n_vertical_subgrid)
    volume_fraction = np.minimum(volume / (hr * hr * vr), 1.0)

    return volume_fraction


def _compute_intersection_area(
    x_pts: ndarray,
    y_pts: ndarray,
    r_at_height: ndarray,
    length: float,
    exact: bool = False,
) -> ndarray:
    """Compute the area of intersection between a circle and each cell of a
    regular grid. Pre-computes shared cell edge coordinates and uses squared
    distances to classify corners, then looks up edge values from 1D arrays
    using cell indices to avoid allocating full 3D edge/radius arrays.

    Parameters
    ----------
    x_pts : ndarray, shape (nx,)
        1D array of cell center x-coordinates (increasing).
    y_pts : ndarray, shape (ny,)
        1D array of cell center y-coordinates (decreasing).
    r_at_height : ndarray, shape (nz,)
        1D array of crown radius at each z-level.
    length : float
        Cell side length (horizontal resolution).
    exact : bool
        If True, include circular segment areas for exact computation.

    Returns
    -------
    ndarray, shape (nz, ny, nx)
        Area of intersection between the circle and each cell.
    """
    half = length / 2.0
    nx = len(x_pts)
    ny = len(y_pts)
    nz = len(r_at_height)

    # Compute unique cell edge coordinates as 1D arrays (n+1 edges for n cells).
    # For x (increasing): left edge of cell ix = x_edges[ix], right = x_edges[ix+1]
    # For y (decreasing): top edge of cell iy = y_edges[iy], bottom = y_edges[iy+1]
    x_edges = np.empty(nx + 1)
    x_edges[:-1] = x_pts - half
    x_edges[-1] = x_pts[-1] + half

    y_edges = np.empty(ny + 1)
    y_edges[:-1] = y_pts + half
    y_edges[-1] = y_pts[-1] - half

    # Compute squared distances from the origin to each corner point.
    # Shape: (ny+1, nx+1) - independent of z, computed once.
    corner_dist_sq = x_edges[np.newaxis, :] ** 2 + y_edges[:, np.newaxis] ** 2

    # Determine which corners are inside the circle for each z-level.
    # Uses squared comparison to avoid sqrt entirely.
    # Shape: (nz, ny+1, nx+1)
    r_sq = r_at_height**2
    corners_inside = corner_dist_sq[np.newaxis, :, :] < r_sq[:, np.newaxis, np.newaxis]

    # Extract per-cell corner status by slicing (views, not copies).
    # Cell (iy, ix) corners: top-left=(iy,ix), top-right=(iy,ix+1),
    #                         bottom-left=(iy+1,ix), bottom-right=(iy+1,ix+1)
    top_left_in = corners_inside[:, :ny, :nx]
    top_right_in = corners_inside[:, :ny, 1:]
    bottom_left_in = corners_inside[:, 1:, :nx]
    bottom_right_in = corners_inside[:, 1:, 1:]

    # Classify cells into trivial (all inside / all outside) vs boundary.
    any_inside = top_left_in | top_right_in | bottom_left_in | bottom_right_in
    all_inside = top_left_in & top_right_in & bottom_left_in & bottom_right_in

    areas = np.zeros((nz, ny, nx))

    # Case 0b: circle entirely inside cell (origin cell at index [:, -1, 0])
    areas[:, -1, 0] = np.pi * r_sq

    # Case 15: all corners inside → full cell area
    areas[all_inside] = length**2

    # Boundary cells: at least one corner inside, but not all.
    # Single np.where call to find all boundary indices at once.
    boundary = any_inside & ~all_inside
    z_b, y_b, x_b = np.where(boundary)

    if len(z_b) > 0:
        # Encode only boundary cells (small 1D arrays) to classify cases.
        case_b = _encode_corners(
            top_left_in[z_b, y_b, x_b],
            top_right_in[z_b, y_b, x_b],
            bottom_left_in[z_b, y_b, x_b],
            bottom_right_in[z_b, y_b, x_b],
        )

        # Case 1: only bottom-left corner inside
        m = case_b == 1
        if np.any(m):
            zi, yi, xi = z_b[m], y_b[m], x_b[m]
            areas[zi, yi, xi] = _calculate_case_1_area(
                x_edges[xi], y_edges[yi + 1], r_at_height[zi], exact
            )

        # Case 3: bottom-left and bottom-right inside
        m = case_b == 3
        if np.any(m):
            zi, yi, xi = z_b[m], y_b[m], x_b[m]
            areas[zi, yi, xi] = _calculate_case_3_area(
                x_edges[xi],
                x_edges[xi + 1],
                y_edges[yi + 1],
                r_at_height[zi],
                length,
                exact,
            )

        # Case 9: top-left and bottom-left inside
        m = case_b == 9
        if np.any(m):
            zi, yi, xi = z_b[m], y_b[m], x_b[m]
            areas[zi, yi, xi] = _calculate_case_9_area(
                y_edges[yi],
                y_edges[yi + 1],
                x_edges[xi],
                r_at_height[zi],
                length,
                exact,
            )

        # Case 11: top-left, bottom-left, and bottom-right inside
        m = case_b == 11
        if np.any(m):
            zi, yi, xi = z_b[m], y_b[m], x_b[m]
            areas[zi, yi, xi] = _calculate_case_11_area(
                y_edges[yi], x_edges[xi + 1], r_at_height[zi], length, exact
            )

    return areas


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


def _align_quadrants(q1, q2, q3, q4, centering: CenteringMode = "cell"):
    """
    Align four quadrants into a single grid.

    For cell-centered: quadrants share the center cell (overlap by 1)
    For vertex-centered: quadrants meet at center vertex (no overlap)
    """
    num_z = q1.shape[0]

    if centering == "cell":
        # Overlap at center cell
        num_x = q1.shape[2] + q2.shape[2] - 1
        num_y = q1.shape[1] + q3.shape[1] - 1
        grid = np.zeros((num_z, num_y, num_x))

        mid_x = num_x // 2
        mid_y = num_y // 2

        grid[:, : mid_y + 1, : mid_x + 1] = q1
        grid[:, : mid_y + 1, mid_x:] = q2
        grid[:, mid_y:, mid_x:] = q3
        grid[:, mid_y:, : mid_x + 1] = q4
    else:  # vertex
        # No overlap: quadrants meet at vertex
        num_x = q1.shape[2] + q2.shape[2]
        num_y = q1.shape[1] + q3.shape[1]
        grid = np.zeros((num_z, num_y, num_x))

        mid_x = q1.shape[2]
        mid_y = q1.shape[1]

        grid[:, :mid_y, :mid_x] = q1
        grid[:, :mid_y, mid_x:] = q2
        grid[:, mid_y:, mid_x:] = q3
        grid[:, mid_y:, :mid_x] = q4

    return grid
