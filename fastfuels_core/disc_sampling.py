# Core imports
from math import ceil
from typing import Optional
import numpy.random as random

# Internal imports
from fastfuels_core.trees import TreePopulation
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile

# External imports
import numpy as np
from numba import njit
import dask.array as da
from xarray import DataArray
from numpy.typing import NDArray
from dask_image.ndfilters import uniform_filter


class DiscSampler:
    """
    Poisson disk sampling object that refines the placement of trees based
    on canopy intersection and height distribution.

    Parameters
    ----------
    trees : TreePopulation
        An object representating a tree population.
    height_grid : DataArray
        An estimated canopy height model for the region of interest.
    sample_heights : bool
        Should the height of trees be accounted for when sampling?
    crown_profile_model : str
        The type of crown profile model to use when looking for crown overlap.
    """

    def __init__(
        self,
        trees: TreePopulation,
        height_grid: Optional[DataArray] = None,
        sample_heights=True,
        crown_profile_model="purves",
    ):
        # Get tree properties as an array for use in Numba
        trees = trees.dropna()
        self.trees = TreePopulation(trees)

        # What type of crown profile model to use
        if crown_profile_model not in ["beta", "purves"]:
            raise ValueError(
                "The crown profile model must be one of the following: 'beta' or 'purves'"
            )
        self.crown_profile_model = crown_profile_model

        # Compute max crown radius using the beta crown profile model
        if self.crown_profile_model == "beta":
            self.beta_model = BetaCrownProfile(
                species_code=trees["SPCD"],
                crown_base_height=trees["HT"] - trees["CR"],
                crown_length=trees["HT"] * trees["CR"],
            )
            trees["MAX_RADIUS"] = self.beta_model.get_max_radius()

        # Compute max crown radius using the beta crown profile model
        else:
            self.purves_model = PurvesCrownProfile(
                species_code=trees["SPCD"],
                dbh=trees["DIA"],
                height=trees["HT"],
                crown_ratio=trees["CR"],
            )
            trees["MAX_RADIUS"] = self.purves_model.get_max_radius()

        # Tree coordinates
        x = trees["X"].to_numpy()
        y = trees["Y"].to_numpy()
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()
        self.bounds = (x_min, y_min, x_max, y_max)

        # Default grid size if none is provided
        self.dx = 0.816281605977565

        # Restrict the CHM to the TreeMap
        if not (height_grid is None):
            self.dx = abs(height_grid.x.data[1] - height_grid.x.data[0])
            height_grid = height_grid.rio.clip_box(
                minx=x_min, miny=y_min, maxx=x_max, maxy=y_max
            )

        xi, yi = self.map_to_pixel_coords(x, y)

        # If there isn't a height grid, make a dummy height grid
        if height_grid is None:
            self.chm = np.ones((xi.max() + 1, yi.max() + 1)).astype(np.float32)
        else:
            self.height_grid = height_grid
            self.chm = height_grid.data[0][::-1, :].astype(np.float32)

        # No trees starting on the edge
        xi[xi < 1] = 1
        yi[yi < 1] = 1
        xi[xi > self.chm.shape[0] - 1] = self.chm.shape[0] - 1
        yi[yi > self.chm.shape[1] - 1] = self.chm.shape[1] - 1
        self.tree_coords = np.c_[xi, yi]

        # If local height is used in sampling, then compute a z-score of
        # height in a distribution of its neighbors
        if sample_heights:
            mask = np.zeros_like(self.chm)
            mask[self.chm > 0.1] = 1.0
            self.mask = mask

            # Do the computations in dask for speed
            mask_da = da.from_array(self.mask, chunks=(1000, 1000))
            chm_da = da.from_array(self.chm, chunks=(1000, 1000))

            # Window size
            w = 15
            mask_m = uniform_filter(mask_da, w)
            mask_sum = mask_m.compute() * w**2

            # Local Mean
            chm_m = uniform_filter(chm_da, w)
            chm_sum = chm_m.compute() * w**2
            chm_mu = chm_sum / (mask_sum + 1.0)

            # Local standard deviation
            chm_s = uniform_filter((chm_da - chm_m) ** 2 * mask, w)
            chm_s_sum = chm_s.compute() * w**2
            chm_sigma = np.sqrt(chm_s_sum / (mask_sum + 1.0))
            chm_sigma[chm_sigma < 1.0] = 1.0

            # Now compute z-score
            self.z_score = (self.chm - chm_mu) / chm_sigma
        else:
            # Otherwize fabricate a z score field
            self.z_score = np.ones_like(self.chm) * 100.0

    def map_to_pixel_coords(self, x, y):
        """
        Convert map coordinates to pixel indexes on a grid.

        Parameters
        ----------
        x : NDArray
            x map coordinate.
        y : NDArray
            y map coordinate.

        Returns
        -------
        xi : ndarray
            Row indexes on a grid.
        yi : NDArray
            Column indexes on a grid.
        """

        (x_min, y_min, x_max, y_max) = self.bounds
        dx = self.dx
        xi = y - y_min
        yi = x - x_min
        xi = np.rint(xi / dx).astype(int)
        yi = np.rint(yi / dx).astype(int)
        return xi, yi

    def pixel_to_map_coords(self, xi, yi):
        """
        Convert pixel indexes on a grid to map coordinates.

        Parameters
        ----------
        xi : ndarray
            Row indexes on a grid.
        yi : NDArray
            Column indexes on a grid.

        Returns
        -------
        x : NDArray
            x map coordinate.
        y : NDArray
            y map coordinate.

        """

        dx = self.dx
        (x_min, y_min, x_max, y_max) = self.bounds
        x = yi * dx + x_min
        y = xi * dx + y_min
        return x, y

    def sample(
        self,
        max_dist: Optional[NDArray] = None,
        tree_scale: Optional[NDArray] = None,
        height_error: Optional[NDArray] = None,
    ) -> TreePopulation:
        """
        Performs Poisson disc sampling to refine placement of trees in the tree population.
        The main criteria are:
        1) Place trees on areas where the CHM > 1 m, if provided.
        2) Avoid excessive crown overlap between trees.
        3) Place tallest trees in a plot in places where the CHM is higher.
        If no CHM has been provided, then sampling just tries to reduce crown overlap.

        Parameters
        ----------
        max_dist (optional): NDArray
            Maximum distance that a tree can move from its original position.
            Can change with iteration. If not specified, use a fixed 30 m per iteration.
        tree_scale (optional): NDArray
            On each sampling iteration, the radius of each tree is multiplied by this factor.
        height_error (optional): NDArray
            A tree can be placed on a pixel if it has a smaller z score than the z score of the CHM
            + height_error. Hence, larger height_error means less stringent placement based on height.


        Returns
        -------
        TreePopulation
            The population of trees with refined x, y coordinates.
        """

        if max_dist is None:
            # Number of sampling iterations
            iterations = 1000
            # Maximum distance a tree can be from its original spot per iteration
            max_dist = np.ones(iterations) * 30.0
            # Paramter that reduces the intersection radius of trees with iteratations
            tree_scale = np.linspace(1.0, 0.0, iterations) ** 2
            # Height error per iteration
            height_error = np.linspace(0.0, 5.0, iterations)

        chm = self.chm
        z_score = self.z_score
        tree_grid = np.zeros_like(self.chm, dtype=np.int64)
        tree_coords = self.tree_coords
        dx = self.dx
        tree_props = self.tree_props

        # Do the disk sampling
        tree_grid, trees_to_place = sample_trees(
            chm,
            z_score,
            tree_grid,
            tree_coords,
            tree_props,
            dx,
            tree_scale,
            max_dist,
            height_error,
        )
        # Get the pixel coordinates of trees and convert to map coords
        coords = np.argwhere(tree_grid >= 1)
        x, y = self.pixel_to_map_coords(coords[:, 0], coords[:, 1])

        # Set new coordinates of the tree population
        trees = self.trees
        trees["X"] = x
        trees["Y"] = y

        return trees


@njit
def propose_point(min_dist: float, max_dist: float, dx=1.0):
    """
    Utility function that proposes a point between a minimum and maximum
    distance from the origin.

    Parameters
    ----------
    min_radius : float
        Minimum distance of proposed point from origin.
    max_radius: float
        Maximum distance of proposed point from origin.
    dx : float
        Resolution of grid cell.

    Returns
    -------
    (int, int)
        Relative grid offsets of proposed point.
    """

    r = min_dist + random.random() * (max_dist - min_dist)
    theta = 2.0 * np.pi * random.random()
    di = r * np.cos(theta)
    dj = r * np.sin(theta)
    di = int(di / dx)
    dj = int(dj / dx)

    return (di, dj)


@njit
def propose_index(
    i: int,
    j: int,
    ni: int,
    nj: int,
    dx=1.0,
    min_radius=1.5,
    max_radius=10.0,
):
    """
    Simple utility function that takes in indices on a grid and proposes new indices of a pixel on the grid
    within an annulus. The proposed point is  constrained to be on the grid.

    Parameters
    ----------
    i : int
        Row index of initial point.
    j : int
        Column index of initial point
    ni : int
        Number of rows on grid.
    nj : int
        Number of columns on grid.
    dx : float
        Resolution of grid cell.
    min_radius : float
        Minimum distance of proposed point from original points.
    max_radius: float
        Maximum distance of proposed point from original point.

    Returns
    -------
    (int, int)
        Grid indices of proposed point.
    """

    # Propose new index
    di, dj = propose_point(min_radius, max_radius, dx)
    i1 = i + di
    j1 = j + dj

    # Make sure it's in bounds
    i1 = max(i1, 1)
    i1 = min(i1, ni - 1)
    j1 = max(j1, 1)
    j1 = min(j1, nj - 1)

    return (i1, j1)


@njit
def intersects(
    distance: float,
    props0: NDArray,
    props1: NDArray,
    scale=1.0,
    crown_profile_model="purves",
):
    """
    A generic wrapper function for detecting crown intersection for different crown profile models.

    Parameters
    ----------
    distance : float
        Distance between two trees
    props0 : NDArray
        Geometric parameters of first tree.
    props0 : NDArray
        Geometric parameters of second tree tree.
    scale : float
        The radius of each tree is multiplied by this scaling factor.
    crown_profile_model: str
        The type of crown profile model to use.

    Returns
    -------
    bool
        Do the two trees with radii multiplied by scale intersect?
    """

    if crown_profile_model == "beta":
        intersects = detect_intersection_beta(distance, props0, props1, scale)
    else:
        intersects = detect_intersection_purves(distance, props0, props1, scale)

    return intersects


@njit
def detect_intersection_purves(
    distance: float, props0: NDArray, props1: NDArray, scale=1.0
):
    """
    Detects intersections between trees using the Purves crown profile model.

    Parameters
    ----------
    distance : float
        Distance between two trees
    props0 : NDArray
        Geometric parameters of first tree.
    props0 : NDArray
        Geometric parameters of second tree tree.
    scale : float
        The radius of each tree is multiplied by this scaling factor.
    crown_profile_model: str
        The type of crown profile model to use.

    Returns
    -------
    bool
        Do the two trees with radii multiplied by scale intersect?
    """

    height0 = props0[0]
    crown_ratio0 = props0[1]
    max_radius0 = props0[2]
    shape0 = props0[3]
    crown_base0 = height0 - crown_ratio0 * height0

    height1 = props1[0]
    crown_ratio1 = props1[1]
    max_radius1 = props1[2]
    shape1 = props1[3]
    crown_base1 = height1 - crown_ratio1 * height1

    cb_max = max(crown_base0, crown_base1)
    r0_max = scale * PurvesCrownProfile._get_purves_radius(
        cb_max, height0, crown_base0, max_radius0, shape0
    )
    r1_max = scale * PurvesCrownProfile._get_purves_radius(
        cb_max, height1, crown_base1, max_radius1, shape1
    )

    return r0_max + r1_max >= distance


@njit
def detect_intersection_beta(
    distance: float, props0: NDArray, props1: NDArray, scale=1.0
):
    """
    Detects intersections between trees using the beta crown profile model. Slightly less
    efficient than Purves model implementation.

    Parameters
    ----------
    distance : float
        Distance between two trees
    props0 : NDArray
        Geometric parameters of first tree.
    props0 : NDArray
        Geometric parameters of second tree tree.
    scale : float
        The radius of each tree is multiplied by this scaling factor.
    crown_profile_model: str
        The type of crown profile model to use.

    Returns
    -------
    bool
        Do the two trees with radii multiplied by scale intersect?
    """

    height0 = props0[0]
    crown_ratio0 = props0[1]
    a0 = props0[3]
    b0 = props0[4]
    c0 = props0[5]
    beta0 = props0[6]
    crown_base0 = height0 - crown_ratio0 * height0

    height1 = props1[0]
    crown_ratio1 = props1[1]
    a1 = props1[3]
    b1 = props1[4]
    c1 = props1[5]
    beta1 = props1[6]
    crown_base1 = height1 - crown_ratio1 * height1

    z0 = max(crown_base0, crown_base1)
    z1 = min(height0, height1)
    dz = (z1 - z0) / 8.0
    # print(z0, z1, dz)

    z = z0
    while z <= z1:
        r0 = scale * BetaCrownProfile.get_beta_radius(
            z, height0, crown_base0, a0, b0, c0, beta0
        )
        r1 = scale * BetaCrownProfile.get_beta_radius(
            z, height1, crown_base1, a1, b1, c1, beta1
        )
        if r0 + r1 >= distance:
            return True
        z += dz

    return False


@njit
def accept(
    chm: NDArray,
    z_score: NDArray,
    tree_grid: NDArray,
    tree_props: NDArray,
    tree_index: int,
    i: int,
    j: int,
    dx=1.0,
    min_radius=1.0,
    tree_scale=1.0,
    height_error=0.0,
    crown_profile_model="purves",
):
    """
    Essentially the heart of the disc sampling routine. Determines whether or not
    to accept the proposed placement for a tree on a grid.

    Parameters
    ----------
    chm: NDArray
        Crown height model (CHM).
    z_score: NDArray
        A z_score for each CHM pixel representing its height compared to its neighbors.
    tree_grid: NDArray
        An array of the same size as the CHM used to mark tree locations.
    tree_props: NDArray
        The geometric parameters of each tree in a tree population.
    tree_index : int
        A tree index that can be used to get tree properties int the tree_props array.
    i : int
        Proposed row to place tree.
    j : int
        Proposed column to palace tree.
    min_radius : float
        A strict minimum distance between trees. Usually just the grid size.
    tree_scale : float
        Rescale tree radii when detecting crown overlap.
    height_error : float
        A tree can be placed on a pixel if it has a smaller z score than the z score of the CHM
        + height_error. Hence, larger height_error means less stringent placement based on height.
    crown_profile_model : str
        The crown profile model to use.

    Returns
    -------
    bool
        Returns tree if we accept the proposed tree placement, false otherwise.
    """

    # Only consider a point where the normalized CHM is above a given threshold
    valid = chm[i, j] >= 1.0 and tree_grid[i, j] == 0
    if not valid:
        return valid

    # Get properties of the tree we're placing
    props0 = tree_props[tree_index]

    # Get the z score of the tree we're placing
    z = props0[-1]
    # Compare to the z-score of the given pixel on the grid
    if z > z_score[i, j] + height_error:
        # We only accept z-scores below a given threshold
        return False

    # Tree grid dimensions
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]

    # Get max radius of tree
    r = props0[2] * tree_scale
    # Check a local window around a tree to see if there's too much overlap with other trees
    w = ceil(r / dx) + 1
    for i1 in range(max(1, i - w), min(n0 - 1, i + w)):
        for j1 in range(max(1, j - w), min(n1 - 1, j + w)):
            if tree_grid[i1, j1] > 0:

                # Make sure tree is further than minimum distance
                dist = np.sqrt(((i - i1) * dx) ** 2 + ((j - j1) * dx) ** 2)
                if dist < min_radius:
                    return False

                # Get properties of this nearby tree
                props1 = tree_props[tree_grid[i1, j1] - 1]

                # Check for crown overlap
                collision = intersects(
                    dist,
                    props0,
                    props1,
                    tree_scale,
                    crown_profile_model=crown_profile_model,
                )
                if collision:
                    return False

    return True


@njit
def local_search(tree_grid, i, j, dx=1.0, max_dist=30.0):
    """
    Performs a local search to find an unoccupied pixel near a point.

    Parameters
    ----------
    tree_grid : ndarray
        Array of tree indexes of int64 type. Zero values indicate no tree. Positive values refer to a particular tree index.
    i : int
        Proposed row index for tree
    j : int
        Proposed column index for tree
    dx : float
        Pixel size used to compute distances.
    max_dist : float
        Determines how large of an area to search.

    Returns
    -------
    (int, int)
        Grid indices of free location or attempted indices.
    """

    # Tree grid dimensions
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]

    # Check a local window around a tree to see if there's an open spot
    i2 = i
    j2 = j
    w = int(max_dist / dx) + 1
    for i1 in range(max(1, i - w), min(n0 - 1, i + w)):
        for j1 in range(max(1, j - w), min(n1 - 1, j + w)):
            i2 = i1
            j2 = j1

            if tree_grid[i1, j1] == 0:
                break

    return i2, j2


@njit
def sample_trees(
    chm: NDArray,
    z_score: NDArray,
    tree_grid: NDArray,
    tree_coords: NDArray,
    tree_props: NDArray,
    dx: float,
    tree_scale: NDArray,
    max_dist: NDArray,
    height_error: NDArray,
):
    """
    Perform Poisson disc sampling. On each iteration, try to place any trees not placed on previous iterations.
    Stringency of placement parameters are relaxed per iteration.

    Parameters
    ----------
    chm: NDArray
        Crown height model (CHM).
    z_score: NDArray
        A z_score for each CHM pixel representing its height compared to its neighbors.
    tree_grid: NDArray
        An array of the same size as the CHM used to mark tree locations.
    tree_coords : NDArray
        The initial pixel coordinates of all trees in a tree population.
    tree_props: NDArray
        The geometric parameters of each tree in a tree population.
    dx : float
        Size of a CHM pixel.
    tree_scale : NDArray
        Tree radii are multipled by this factor.
    max_dist: NDArray
        Max distance a tree can move from its original location.
    height_error: NDArray
        Used to control how stringent height based placement is.


    Returns
    -------
    tree_grid : NDArray
        The tree_grid with new tree locations marked.
    trees_to_place : list of ints
        The number of trees left to place per sampling iteration.
    """

    indexes = list(np.arange(len(tree_coords)))

    # This list will track the number of trees left to place at each iteration
    trees_to_place = []

    # Iterate until we've placed all trees
    for i in range(len(tree_scale)):
        new_indexes = []
        trees_to_place.append(len(indexes))

        for j in range(len(indexes)):
            tree_index = indexes[j]

            xi = tree_coords[tree_index, 0]
            xj = tree_coords[tree_index, 1]
            if i > 0:
                xi, xj = propose_index(
                    xi, xj, tree_grid.shape[0], tree_grid.shape[1], dx, dx, max_dist[i]
                )

            valid = accept(
                chm,
                z_score,
                tree_grid,
                tree_props,
                tree_index,
                xi,
                xj,
                dx,
                dx,
                tree_scale[i],
                height_error[i],
            )

            # If the tree's current position is good then place it on the grid
            if valid:
                tree_grid[xi, xj] = tree_index + 1
            else:
                # Otherwise add it to this list of indexes for subsequent passes
                new_indexes.append(tree_index)

        indexes = new_indexes
        if len(indexes) == 0:
            break

    # If there are some trees left over, just do a local search for an open space
    for j in range(len(indexes)):
        tree_index = indexes[j]
        xi = tree_coords[tree_index, 0]
        xj = tree_coords[tree_index, 1]

        xi, xj = local_search(tree_grid, xi, xj, dx=dx, max_dist=30.0)
        tree_grid[xi, xj] = tree_index + 1

    return tree_grid, trees_to_place
