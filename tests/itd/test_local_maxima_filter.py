import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
import rioxarray  # noqa: F401


from fastfuels_core.itd.local_maxima_filter import (
    variable_window_filter,
    fixed_window_filter,
)


def generate_complex_synthetic_chm() -> xr.DataArray:
    """
    Generates a realistic synthetic CHM with three distinct stand structures:
    1. Isolated Dominant Trees
    2. Suppressed Understory (Gap Phase)
    3. Co-Dominant Clusters (Interlocking crowns with distinct peaks)
    """
    pixel_size = 0.5
    width, height = 200, 200  # 100m x 100m plot
    origin_x, origin_y = 500000.0, 4000000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    rng = np.random.default_rng(seed=42)

    # 1. Background noise floor
    chm_array = rng.normal(loc=1.0, scale=0.1, size=(height, width))

    # We will build the canopy surface strictly using np.maximum to simulate top-hits
    canopy_surface = np.zeros((height, width))
    y_grid, x_grid = np.ogrid[:height, :width]

    ground_truth_trees = []
    existing_centers = []

    def plant_tree(row: int, col: int, max_height: float, sigma: float, tree_type: str):
        """Generates a Gaussian tree and merges it into the canopy surface."""
        nonlocal canopy_surface

        dist_sq = (x_grid - col) ** 2 + (y_grid - row) ** 2
        tree_footprint = max_height * np.exp(-dist_sq / (2 * sigma**2))

        # Simulating the LiDAR top-hit surface
        canopy_surface = np.maximum(canopy_surface, tree_footprint)

        x_coord = origin_x + (col * pixel_size) + (pixel_size / 2)
        y_coord = origin_y - (row * pixel_size) - (pixel_size / 2)

        ground_truth_trees.append(
            {
                "type": tree_type,
                "row": row,
                "col": col,
                "x": x_coord,
                "y": y_coord,
                "height": max_height,
            }
        )
        existing_centers.append((row, col))

    def get_valid_location(min_dist: float) -> tuple[int, int]:
        """Finds a coordinate that is at least min_dist away from existing trees."""
        for _ in range(200):
            r = rng.integers(20, height - 20)
            c = rng.integers(20, width - 20)
            if not existing_centers or all(
                np.sqrt((r - er) ** 2 + (c - ec) ** 2) >= min_dist
                for er, ec in existing_centers
            ):
                return r, c
        raise ValueError("Could not find space to plant tree!")

    # 2. Plant 5 Isolated Dominant Trees (Distance > 10m / 20 pixels)
    for _ in range(5):
        r, c = get_valid_location(min_dist=20)
        plant_tree(r, c, max_height=25.0, sigma=4.0, tree_type="dominant")

    # 3. Plant 10 Suppressed Trees in the gaps (Distance > 6m / 12 pixels)
    for _ in range(10):
        r, c = get_valid_location(min_dist=12)
        plant_tree(r, c, max_height=8.0, sigma=1.5, tree_type="suppressed")

    # 4. Plant 3 Co-Dominant Twin Clusters (Interlocking Crowns)
    for _ in range(3):
        r1, c1 = get_valid_location(min_dist=20)
        plant_tree(r1, c1, max_height=22.0, sigma=2.0, tree_type="co_dominant_A")

        # Plant the twin exactly 6 pixels to the right, slightly shorter
        r2, c2 = r1, c1 + 6
        plant_tree(
            r2, c2, max_height=21.0, sigma=2.0, tree_type="co_dominant_B"
        )  # <-- Changed to 21.0

    # Merge canopy over the noise floor
    final_chm = np.maximum(chm_array, canopy_surface)

    da = xr.DataArray(final_chm, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.attrs["ground_truth"] = ground_truth_trees

    return da


@pytest.fixture
def complex_synthetic_chm() -> xr.DataArray:
    return generate_complex_synthetic_chm()


@pytest.fixture
def empty_chm() -> xr.DataArray:
    """Generates a CHM with zero trees (e.g., a lake or bare ground)."""
    pixel_size = 0.5
    width, height = 50, 50
    transform = from_origin(500000.0, 4000000.0, pixel_size, pixel_size)

    # Flat ground at 0.5m elevation (well below our 2.0m min_height)
    chm_array = np.full((height, width), 0.5)

    da = xr.DataArray(chm_array, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


@pytest.fixture
def simple_chm() -> xr.DataArray:
    """Generates a perfectly clean CHM with exactly one tree and zero noise."""
    pixel_size = 0.5
    width, height = 50, 50
    origin_x, origin_y = 500000.0, 4000000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    chm_array = np.zeros((height, width))

    # Plant exactly one 20m tree at row=25, col=25
    y_grid, x_grid = np.ogrid[:height, :width]
    dist_sq = (x_grid - 25) ** 2 + (y_grid - 25) ** 2
    chm_array += 20.0 * np.exp(-dist_sq / (2 * 3.0**2))

    da = xr.DataArray(chm_array, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)

    # X = origin_x + (col * pixel_size) + (pixel_size / 2)
    da.attrs["exact_x"] = 500000.0 + (25 * 0.5) + 0.25
    da.attrs["exact_y"] = 4000000.0 - (25 * 0.5) - 0.25
    return da


def test_vwf_empty_forest(empty_chm: xr.DataArray):
    """Proves the algorithm doesn't crash on bare ground and returns the correct schema."""
    dask_df = variable_window_filter(
        chm_da=empty_chm, min_height=2.0, spatial_resolution=0.5
    )
    df = dask_df.compute()

    assert len(df) == 0, "Should find zero trees."
    assert list(df.columns) == [
        "x",
        "y",
        "height",
    ], "Empty DataFrame must still have correct schema."


def test_vwf_pure_coordinate_math(simple_chm: xr.DataArray):
    """Proves pure coordinate transformation without noise interference."""
    dask_df = variable_window_filter(
        chm_da=simple_chm, min_height=2.0, spatial_resolution=0.5
    )
    df = dask_df.compute()

    assert len(df) == 1
    assert np.isclose(df.iloc[0]["x"], simple_chm.attrs["exact_x"])
    assert np.isclose(df.iloc[0]["y"], simple_chm.attrs["exact_y"])
    assert np.isclose(df.iloc[0]["height"], 20.0)


def test_vwf_handles_interlocking_crowns(complex_synthetic_chm: xr.DataArray):
    """
    Validates that the Variable Window Filter correctly identifies Twin Trees
    without merging them into a single stem or over-segmenting the saddle.
    """
    ground_truth = complex_synthetic_chm.attrs["ground_truth"]

    # Act
    dask_df = variable_window_filter(
        chm_da=complex_synthetic_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    )
    df = dask_df.compute()
    valid_trees = df[df["height"] > 5.0]

    # Assert 1: Total Population Count
    # 5 dominant + 10 suppressed + (3 clusters * 2 twins) = 21 trees
    assert len(valid_trees) == len(
        ground_truth
    ), f"Algorithm failed on complex canopy. Expected {len(ground_truth)}, found {len(valid_trees)}"

    # Assert 2: Specific Co-Dominant Cluster Verification
    # Let's grab the first set of twins we planted
    twins = [
        t for t in ground_truth if t["type"] in ["co_dominant_A", "co_dominant_B"]
    ][:2]

    # We search the Dask outputs for trees within a tight bounding box around this cluster
    # The twins are 3m apart. We draw a 4m box around their center.
    center_x = (twins[0]["x"] + twins[1]["x"]) / 2
    center_y = (twins[0]["y"] + twins[1]["y"]) / 2

    detected_twins = valid_trees[
        (valid_trees["x"].between(center_x - 4.0, center_x + 4.0))
        & (valid_trees["y"].between(center_y - 4.0, center_y + 4.0))
    ]

    # If the window is too large, it merges them. If too small, it over-segments the saddle.
    # A perfect algorithm finds exactly 2 peaks here.
    assert (
        len(detected_twins) == 2
    ), "VWF failed to separate the interlocking crowns of a co-dominant cluster!"

    # Ensure the heights match our new planting targets (22m and 21m)
    # We use a set to avoid strict ordering issues in the DataFrame
    assert set(np.round(detected_twins["height"].values, 1)) == {21.0, 22.0}


def test_find_treetops_fixed_window_limitations(complex_synthetic_chm: xr.DataArray):
    """
    Tests the baseline fixed-window algorithm to explicitly demonstrate
    its limitations (under-segmentation) on a complex canopy.
    """
    ground_truth = complex_synthetic_chm.attrs["ground_truth"]

    # Act: Use a 5m fixed window (a common size used for mature stands)
    dask_df = fixed_window_filter(
        chm_da=complex_synthetic_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        window_size_meters=7.0,
    )
    df = dask_df.compute()
    valid_trees = df[df["height"] > 5.0]

    # --- 1. The Under-segmentation Proof (Twin Trees) ---
    # Our co-dominant twin trees were planted exactly 3m apart.
    # Because our fixed search window is 5m, it is mathematically impossible
    # for it to resolve both peaks. It will merge them into a single stem.

    twins = [
        t for t in ground_truth if t["type"] in ["co_dominant_A", "co_dominant_B"]
    ][:2]
    center_x = (twins[0]["x"] + twins[1]["x"]) / 2
    center_y = (twins[0]["y"] + twins[1]["y"]) / 2

    detected_twins = valid_trees[
        (valid_trees["x"].between(center_x - 4.0, center_x + 4.0))
        & (valid_trees["y"].between(center_y - 4.0, center_y + 4.0))
    ]

    # ASSERTION: The fixed window FAILS to find both trees. It only finds 1.
    assert (
        len(detected_twins) == 1
    ), "A 7m fixed window merged trees that are 3m apart, as expected."

    # --- 2. The Over-segmentation Proof (Dominant Trees) ---
    # If we instead used a tiny 1.5m fixed window to successfully separate the 3m twins,
    # that tiny window would slide across the massive 8m-wide crown of our 25m dominant
    # trees. Any tiny micro-bump of noise on that crown would register as a fake tree.

    # --- 3. Total Count Proof ---
    # Because the 5m window merged all 3 sets of our twin clusters, our total
    # detected tree count will be short of the actual ground truth.
    assert len(valid_trees) < len(
        ground_truth
    ), f"Fixed window under-segmented! Found {len(valid_trees)}, expected {len(ground_truth)}."


def test_find_treetops_vwf_accuracy(complex_synthetic_chm: xr.DataArray):
    """
    Tests that the Variable Window Filter dynamically detects all
    size classes without dropping suppressed trees.
    """
    ground_truth = complex_synthetic_chm.attrs["ground_truth"]
    expected_tree_count = len(ground_truth)

    # Act
    dask_df = variable_window_filter(
        chm_da=complex_synthetic_chm,
        min_height=2.0,  # Ignore the 1m background noise
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    )
    df = dask_df.compute()

    # Filter out any tiny noise artifacts that might have survived
    valid_trees = df[df["height"] > 5.0]

    # Assert 1: We found exactly the number of trees we planted (25)
    assert (
        len(valid_trees) == expected_tree_count
    ), f"Expected {expected_tree_count} trees, but found {len(valid_trees)}"

    # Assert 2: Coordinate Accuracy Validation
    # Let's ensure our algorithm isn't shifting the X/Y coordinates
    # by checking our first planted dominant tree.
    first_dominant = ground_truth[0]

    # FIX: Find the detected tree closest to our known spatial location instead of height.
    # Since we planted 5 dominant trees at exactly 25.0m, height matching is non-deterministic.
    dx = valid_trees["x"] - first_dominant["x"]
    dy = valid_trees["y"] - first_dominant["y"]
    distances = np.sqrt(dx ** 2 + dy ** 2)

    detected_match = valid_trees.iloc[distances.argsort()[:1]]

    assert np.isclose(
        detected_match["x"].values[0], first_dominant["x"]
    ), "X coordinate drift detected!"
    assert np.isclose(
        detected_match["y"].values[0], first_dominant["y"]
    ), "Y coordinate drift detected!"
