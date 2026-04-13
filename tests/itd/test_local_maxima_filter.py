from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from dask.callbacks import Callback
from numpy.random import Generator
import rasterio as rio
from rasterio.transform import from_origin

import fastfuels_core.itd.local_maxima_filter as local_maxima_filter
from fastfuels_core.itd.local_maxima_filter import (
    _extract_block_candidates,
    _union_find_merge,
    fixed_window_filter,
    variable_window_filter,
)
from tests.itd.reference_local_maxima_filter import (
    fixed_window_filter_reference,
    variable_window_filter_reference,
)

FIGURES_PATH = Path(__file__).parent / "figures"

# Toggle to True to save/show diagnostic plots during test runs
SAVE_FIG = True
SHOW_FIG = False


def _get_extent(chm_da: xr.DataArray) -> list[float]:
    """Calculate real-world extent [min_x, max_x, min_y, max_y] for imshow."""
    transform = chm_da.rio.transform()
    width, height = chm_da.shape[1], chm_da.shape[0]
    min_x = transform.c
    max_y = transform.f
    max_x = min_x + (width * transform.a)
    min_y = max_y + (height * transform.e)
    return [min_x, max_x, min_y, max_y]


def _plot_chm_with_treetops(
    ax,
    chm_da: xr.DataArray,
    treetops: pd.DataFrame,
    *,
    title: str = "",
    ground_truth: list[dict] | None = None,
) -> None:
    """Plot a CHM with detected treetops overlaid."""
    extent = _get_extent(chm_da)
    ax.imshow(chm_da.values, extent=extent, cmap="viridis", origin="upper")
    if ground_truth:
        gt_x = [t["x"] for t in ground_truth]
        gt_y = [t["y"] for t in ground_truth]
        ax.scatter(
            gt_x,
            gt_y,
            facecolors="none",
            edgecolors="lime",
            s=150,
            linewidths=2,
            label="Ground Truth",
        )
    ax.scatter(
        treetops["x"],
        treetops["y"],
        c="red",
        marker="x",
        s=60,
        linewidths=1.5,
        label=f"Detected ({len(treetops)})",
    )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Easting (X)")
    ax.set_ylabel("Northing (Y)")


def _plot_chunked_comparison(
    chm_da: xr.DataArray,
    results: dict[str, pd.DataFrame],
    *,
    chunk_boundaries: list[tuple[str, list[float]]] | None = None,
    suptitle: str = "",
    filename: str = "",
) -> None:
    """Plot side-by-side comparison of results from different chunk layouts."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]
    extent = _get_extent(chm_da)

    for ax, (label, df) in zip(axes, results.items()):
        ax.imshow(chm_da.values, extent=extent, cmap="viridis", origin="upper")
        ax.scatter(
            df["x"],
            df["y"],
            c="red",
            marker="x",
            s=80,
            linewidths=2,
            label=f"Detected ({len(df)})",
        )
        ax.set_title(f"{label}\n{len(df)} treetops", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Easting (X)")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if SAVE_FIG and filename:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_PATH / filename, dpi=150)
    if SHOW_FIG:
        plt.show()
    plt.close(fig)


# --- SHARED GENERATOR HELPERS ---


def _get_valid_location(
    min_dist: float, rng: Generator, existing_centers: List, height: int, width: int
) -> tuple[int, int]:
    for _ in range(200):
        r = rng.integers(20, height - 20)
        c = rng.integers(20, width - 20)
        if not existing_centers or all(
            np.sqrt((r - er) ** 2 + (c - ec) ** 2) >= min_dist
            for er, ec in existing_centers
        ):
            return r, c
    raise ValueError("Could not find space to plant tree!")


def _plant_conical_tree(
    canopy_surface: np.ndarray,
    ground_truth: List[Dict[str, Any]],
    row: int,
    col: int,
    max_height: float,
    sigma: float | np.ndarray,
    tree_type: str,
    pixel_size: float,
    origin_x: float,
    origin_y: float,
) -> None:
    """Stamp a 2D Gaussian tree into the canopy surface and log its ground truth.

    ``sigma`` controls the crown shape:
    - scalar: isotropic circular crown (σ_row = σ_col = sigma, no rotation).
    - 2x2 array: full covariance matrix, enabling elliptical and rotated crowns.
    """
    height, width = canopy_surface.shape
    y_grid, x_grid = np.ogrid[:height, :width]

    dy = y_grid - row
    dx = x_grid - col
    cov = np.atleast_2d(sigma)
    if cov.shape == (1, 1):
        cov = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
    inv_cov = np.linalg.inv(cov)
    mahal = (
        inv_cov[0, 0] * dy**2
        + (inv_cov[0, 1] + inv_cov[1, 0]) * dy * dx
        + inv_cov[1, 1] * dx**2
    )
    tree_footprint = max_height * np.exp(-0.5 * mahal)

    np.maximum(canopy_surface, tree_footprint, out=canopy_surface)

    x_coord = origin_x + (col * pixel_size) + (pixel_size / 2)
    y_coord = origin_y - (row * pixel_size) - (pixel_size / 2)

    ground_truth.append(
        {
            "type": tree_type,
            "row": row,
            "col": col,
            "x": x_coord,
            "y": y_coord,
            "height": max_height,
        }
    )


# --- DATA GENERATORS ---


def generate_complex_synthetic_chm() -> xr.DataArray:
    """Generates a realistic synthetic CHM with three distinct stand structures."""
    pixel_size = 0.5
    width, height = 200, 200
    origin_x, origin_y = 500000.0, 4000000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    rng = np.random.default_rng(seed=42)
    chm_array = rng.normal(loc=1.0, scale=0.1, size=(height, width))
    canopy_surface = np.zeros((height, width))

    ground_truth_trees = []
    existing_centers = []

    # 1. Plant 5 Isolated Dominant Trees
    for _ in range(5):
        r, c = _get_valid_location(20, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r,
            c,
            25.0,
            4.0,
            "dominant",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    # 2. Plant 10 Suppressed Trees
    for _ in range(10):
        r, c = _get_valid_location(12, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r,
            c,
            8.0,
            1.5,
            "suppressed",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    # 3. Plant 3 Co-Dominant Twin Clusters
    for _ in range(3):
        r1, c1 = _get_valid_location(20, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r1,
            c1,
            22.0,
            2.0,
            "co_dominant_A",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r1, c1))

        r2, c2 = r1, c1 + 6
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r2,
            c2,
            21.0,
            2.0,
            "co_dominant_B",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r2, c2))

    final_chm = np.maximum(chm_array, canopy_surface)

    da = xr.DataArray(final_chm, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.attrs["ground_truth"] = ground_truth_trees

    return da


def generate_mixed_morphology_chm() -> xr.DataArray:
    """Generates a CHM containing a mixture of conical canopies and L-shaped plateaus."""
    pixel_size = 0.5
    width, height = 200, 200
    origin_x, origin_y = 500000.0, 4000000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    rng = np.random.default_rng(seed=101)
    chm_array = rng.normal(loc=1.0, scale=0.1, size=(height, width))
    canopy_surface = np.zeros((height, width))

    ground_truth = []
    existing_centers = []

    # 1. Plant 12 Normal Conical Trees
    for _ in range(12):
        r, c = _get_valid_location(15, rng, existing_centers, height, width)
        max_height = rng.uniform(15.0, 22.0)
        _plant_conical_tree(
            canopy_surface,
            ground_truth,
            r,
            c,
            max_height,
            3.0,
            "conical",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    # 2. Plant 6 Irregular L-Shaped Plateaus
    for _ in range(6):
        r, c = _get_valid_location(20, rng, existing_centers, height, width)
        max_height = rng.uniform(22.0, 28.0)

        # Draw L-shape
        canopy_surface[r : r + 12, c : c + 3] = np.maximum(
            canopy_surface[r : r + 12, c : c + 3], max_height
        )
        canopy_surface[r + 9 : r + 12, c + 3 : c + 10] = np.maximum(
            canopy_surface[r + 9 : r + 12, c + 3 : c + 10], max_height
        )

        # Calculate coordinates for ground truth
        x_coord = origin_x + (c * pixel_size) + (pixel_size / 2)
        y_coord = origin_y - (r * pixel_size) - (pixel_size / 2)

        ground_truth.append(
            {
                "type": "l_shape",
                "row": r,
                "col": c,
                "x": x_coord,
                "y": y_coord,
                "height": max_height,
            }
        )
        existing_centers.append((r, c))

    final_chm = np.maximum(chm_array, canopy_surface)

    da = xr.DataArray(final_chm, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.attrs["ground_truth"] = ground_truth

    return da


def generate_asymmetric_crown_chm() -> xr.DataArray:
    """Generates a CHM with elliptical and rotated tree crowns.

    Contains three groups of trees whose crowns are non-circular:
    - 5 axis-aligned elliptical crowns (wider in the column direction)
    - 5 axis-aligned elliptical crowns (wider in the row direction)
    - 5 rotated elliptical crowns (45-degree tilt)
    All trees have a single well-defined peak at the Gaussian center.
    """
    pixel_size = 0.5
    width, height = 200, 200
    origin_x, origin_y = 500000.0, 4000000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    rng = np.random.default_rng(seed=555)
    chm_array = rng.normal(loc=1.0, scale=0.1, size=(height, width))
    canopy_surface = np.zeros((height, width))

    ground_truth_trees: List[Dict[str, Any]] = []
    existing_centers: List = []

    # 1. Column-elongated elliptical crowns (σ_row=2, σ_col=6)
    cov_col_wide = np.array([[4.0, 0.0], [0.0, 36.0]])
    for _ in range(5):
        r, c = _get_valid_location(20, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r,
            c,
            rng.uniform(18.0, 25.0),
            cov_col_wide,
            "elliptical_col",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    # 2. Row-elongated elliptical crowns (σ_row=6, σ_col=2)
    cov_row_wide = np.array([[36.0, 0.0], [0.0, 4.0]])
    for _ in range(5):
        r, c = _get_valid_location(20, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r,
            c,
            rng.uniform(18.0, 25.0),
            cov_row_wide,
            "elliptical_row",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    # 3. Rotated elliptical crowns (45-degree tilt)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov_rotated = R @ np.diag([4.0, 36.0]) @ R.T
    for _ in range(5):
        r, c = _get_valid_location(20, rng, existing_centers, height, width)
        _plant_conical_tree(
            canopy_surface,
            ground_truth_trees,
            r,
            c,
            rng.uniform(18.0, 25.0),
            cov_rotated,
            "elliptical_rotated",
            pixel_size,
            origin_x,
            origin_y,
        )
        existing_centers.append((r, c))

    final_chm = np.maximum(chm_array, canopy_surface)

    da = xr.DataArray(final_chm, dims=["y", "x"])
    da.rio.write_crs("EPSG:32611", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.attrs["ground_truth"] = ground_truth_trees

    return da


@pytest.fixture
def asymmetric_crown_chm() -> xr.DataArray:
    return generate_asymmetric_crown_chm()


@pytest.fixture
def complex_synthetic_chm() -> xr.DataArray:
    return generate_complex_synthetic_chm()


@pytest.fixture
def mixed_morphology_chm() -> xr.DataArray:
    return generate_mixed_morphology_chm()


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
    distances = np.sqrt(dx**2 + dy**2)

    detected_match = valid_trees.iloc[distances.argsort()[:1]]

    assert np.isclose(
        detected_match["x"].values[0], first_dominant["x"]
    ), "X coordinate drift detected!"
    assert np.isclose(
        detected_match["y"].values[0], first_dominant["y"]
    ), "Y coordinate drift detected!"


def test_mixed_morphology_extraction(mixed_morphology_chm: xr.DataArray):
    """Proves the algorithm handles regular and irregular canopies simultaneously."""
    dask_df = variable_window_filter(
        chm_da=mixed_morphology_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    )
    df = dask_df.compute()

    # We planted exactly 12 conical + 6 L-shapes = 18 trees
    assert len(df) == 18, "Algorithm failed to accurately count mixed morphologies."


# ---------------------------------------------------------------------------
# Asymmetric crown tests
# ---------------------------------------------------------------------------


def test_vwf_detects_all_asymmetric_crowns(asymmetric_crown_chm: xr.DataArray):
    """VWF must detect every planted tree regardless of crown ellipticity or rotation."""
    ground_truth = asymmetric_crown_chm.attrs["ground_truth"]
    dask_df = variable_window_filter(
        chm_da=asymmetric_crown_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    )
    df = dask_df.compute()
    valid_trees = df[df["height"] > 5.0]

    assert len(valid_trees) == len(
        ground_truth
    ), f"Expected {len(ground_truth)} trees, found {len(valid_trees)}"

    if SAVE_FIG or SHOW_FIG:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        _plot_chm_with_treetops(
            ax,
            asymmetric_crown_chm,
            valid_trees,
            title=f"VWF on Asymmetric Crowns — {len(valid_trees)} treetops",
            ground_truth=ground_truth,
        )
        fig.tight_layout()
        if SAVE_FIG:
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_PATH / "itd_asymmetric_vwf.png", dpi=150)
        if SHOW_FIG:
            plt.show()
        plt.close(fig)


def test_fwf_detects_asymmetric_crowns(asymmetric_crown_chm: xr.DataArray):
    """Fixed window filter must detect every asymmetric crown."""
    ground_truth = asymmetric_crown_chm.attrs["ground_truth"]
    dask_df = fixed_window_filter(
        chm_da=asymmetric_crown_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        window_size_meters=3.0,
    )
    df = dask_df.compute()
    valid_trees = df[df["height"] > 5.0]

    assert len(valid_trees) == len(
        ground_truth
    ), f"Expected {len(ground_truth)} trees, found {len(valid_trees)}"

    if SAVE_FIG or SHOW_FIG:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        _plot_chm_with_treetops(
            ax,
            asymmetric_crown_chm,
            valid_trees,
            title=f"FWF on Asymmetric Crowns — {len(valid_trees)} treetops",
            ground_truth=ground_truth,
        )
        fig.tight_layout()
        if SAVE_FIG:
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES_PATH / "itd_asymmetric_fwf.png", dpi=150)
        if SHOW_FIG:
            plt.show()
        plt.close(fig)


def test_asymmetric_crown_coordinate_accuracy(asymmetric_crown_chm: xr.DataArray):
    """Detected treetop positions must land at the planted Gaussian center."""
    ground_truth = asymmetric_crown_chm.attrs["ground_truth"]
    dask_df = variable_window_filter(
        chm_da=asymmetric_crown_chm,
        min_height=2.0,
        spatial_resolution=0.5,
        crown_ratio=0.10,
        crown_offset=1.0,
    )
    df = dask_df.compute()

    for tree in ground_truth:
        dx = df["x"] - tree["x"]
        dy = df["y"] - tree["y"]
        distances = np.sqrt(dx**2 + dy**2)
        closest = df.iloc[distances.argmin()]

        assert np.isclose(closest["x"], tree["x"]), (
            f"{tree['type']} tree at ({tree['x']:.2f}, {tree['y']:.2f}): "
            f"x drift = {abs(closest['x'] - tree['x']):.4f}"
        )
        assert np.isclose(closest["y"], tree["y"]), (
            f"{tree['type']} tree at ({tree['x']:.2f}, {tree['y']:.2f}): "
            f"y drift = {abs(closest['y'] - tree['y']):.4f}"
        )


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_asymmetric_crowns_chunk_invariant(
    filter_name: str,
    asymmetric_crown_chm: xr.DataArray,
):
    """Asymmetric crowns must produce identical results across chunk layouts."""
    reference = _run_reference(filter_name, asymmetric_crown_chm)
    square = _run_filter(
        filter_name, _chunk_chm(asymmetric_crown_chm, {"y": 100, "x": 100})
    )
    asymmetric = _run_filter(
        filter_name, _chunk_chm(asymmetric_crown_chm, {"y": 50, "x": 200})
    )

    _assert_same_output(reference, square)
    _assert_same_output(reference, asymmetric)

    if SAVE_FIG or SHOW_FIG:
        _plot_chunked_comparison(
            asymmetric_crown_chm,
            {
                "Reference (eager)": reference,
                "4 chunks (100x100)": square,
                "Asymmetric (50x200)": asymmetric,
            },
            suptitle=f"Asymmetric Crowns Chunk Invariance: {filter_name}",
            filename=f"itd_asymmetric_chunk_invariance_{filter_name}.png",
        )


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_asymmetric_crowns_match_reference(
    filter_name: str,
    asymmetric_crown_chm: xr.DataArray,
):
    """Dask implementation must match eager scipy reference for asymmetric crowns."""
    reference = _run_reference(filter_name, asymmetric_crown_chm)
    result = _run_filter(filter_name, asymmetric_crown_chm)
    _assert_same_output(result, reference)

    if SAVE_FIG or SHOW_FIG:
        ground_truth = asymmetric_crown_chm.attrs.get("ground_truth")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        _plot_chm_with_treetops(
            ax1,
            asymmetric_crown_chm,
            reference,
            title=f"Reference (eager scipy) — {len(reference)} treetops",
            ground_truth=ground_truth,
        )
        _plot_chm_with_treetops(
            ax2,
            asymmetric_crown_chm,
            result,
            title=f"New (dask-image) — {len(result)} treetops",
            ground_truth=ground_truth,
        )
        fig.suptitle(
            f"Asymmetric Crowns: Reference vs New ({filter_name})",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        if SAVE_FIG:
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                FIGURES_PATH / f"itd_asymmetric_ref_vs_new_{filter_name}.png", dpi=150
            )
        if SHOW_FIG:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers for new chunked-correctness tests
# ---------------------------------------------------------------------------


def _sort_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["x", "y", "height"])
    return df.sort_values(["x", "y", "height"]).reset_index(drop=True)


def _assert_same_output(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        _sort_output(left),
        _sort_output(right),
        check_exact=False,
        atol=1e-9,
        rtol=1e-9,
    )


def _chunk_chm(chm_da: xr.DataArray, chunking: dict[str, int]) -> xr.DataArray:
    return chm_da.chunk(chunking)


def _run_filter(filter_name: str, chm_da: xr.DataArray) -> pd.DataFrame:
    if filter_name == "fixed":
        return fixed_window_filter(
            chm_da=chm_da,
            min_height=2.0,
            spatial_resolution=0.5,
            window_size_meters=3.0,
        ).compute()
    if filter_name == "variable":
        return variable_window_filter(
            chm_da=chm_da,
            min_height=2.0,
            spatial_resolution=0.5,
            crown_ratio=0.10,
            crown_offset=1.0,
        ).compute()
    raise ValueError(f"Unsupported filter: {filter_name}")


def _run_reference(filter_name: str, chm_da: xr.DataArray) -> pd.DataFrame:
    if filter_name == "fixed":
        return fixed_window_filter_reference(
            chm_da=chm_da,
            min_height=2.0,
            spatial_resolution=0.5,
            window_size_meters=3.0,
        )
    if filter_name == "variable":
        return variable_window_filter_reference(
            chm_da=chm_da,
            min_height=2.0,
            spatial_resolution=0.5,
            crown_ratio=0.10,
            crown_offset=1.0,
        )
    raise ValueError(f"Unsupported filter: {filter_name}")


# ---------------------------------------------------------------------------
# Fixtures for boundary and benchmark tests
# ---------------------------------------------------------------------------


def rio_coords(transform, row: int, col: int) -> tuple[float, float]:
    import rasterio

    x, y = rasterio.transform.xy(transform, [row], [col])
    return float(x[0]), float(y[0])


def generate_boundary_plateau_chm(
    row_slice: slice,
    col_slice: slice,
    *,
    width: int = 64,
    height: int = 64,
    pixel_size: float = 1.0,
) -> xr.DataArray:
    origin_x, origin_y = 1000.0, 2000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    chm_array = np.zeros((height, width), dtype=np.float64)
    chm_array[row_slice, col_slice] = 20.0

    chm_da = xr.DataArray(chm_array, dims=["y", "x"])
    chm_da.rio.write_crs("EPSG:32611", inplace=True)
    chm_da.rio.write_transform(transform, inplace=True)
    expected_x, expected_y = rio_coords(transform, row_slice.start, col_slice.start)
    chm_da.attrs["expected_x"] = expected_x
    chm_da.attrs["expected_y"] = expected_y
    return chm_da


def generate_benchmark_chm(size: int = 1024, pixel_size: float = 1.0) -> xr.DataArray:
    origin_x, origin_y = 250000.0, 4100000.0
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    chm_array = np.full((size, size), 0.25, dtype=np.float64)
    y_grid, x_grid = np.ogrid[:size, :size]
    for row, col, ht, sigma in [
        (size // 4, size // 4, 24.0, 9.0),
        (size // 4, (3 * size) // 4, 22.0, 8.0),
        ((3 * size) // 4, size // 4, 21.0, 10.0),
        ((3 * size) // 4, (3 * size) // 4, 23.0, 7.0),
        (size // 2, size // 2, 28.0, 12.0),
    ]:
        dist_sq = (x_grid - col) ** 2 + (y_grid - row) ** 2
        canopy = ht * np.exp(-dist_sq / (2 * sigma**2))
        np.maximum(chm_array, canopy, out=chm_array)

    chm_da = xr.DataArray(chm_array, dims=["y", "x"])
    chm_da.rio.write_crs("EPSG:32611", inplace=True)
    chm_da.rio.write_transform(transform, inplace=True)
    return chm_da


@pytest.fixture
def boundary_split_half_chm() -> xr.DataArray:
    return generate_boundary_plateau_chm(slice(28, 32), slice(30, 34))


@pytest.fixture
def boundary_split_three_quarters_chm() -> xr.DataArray:
    return generate_boundary_plateau_chm(slice(28, 32), slice(29, 33))


@pytest.fixture
def corner_split_chm() -> xr.DataArray:
    return generate_boundary_plateau_chm(slice(30, 34), slice(30, 34))


# ---------------------------------------------------------------------------
# Chunked correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_new_implementation_matches_eager_reference(
    filter_name: str,
    complex_synthetic_chm: xr.DataArray,
):
    """New dask-image implementation matches the old eager scipy reference."""
    reference = _run_reference(filter_name, complex_synthetic_chm)
    result = _run_filter(filter_name, complex_synthetic_chm)
    _assert_same_output(result, reference)

    if SAVE_FIG or SHOW_FIG:
        ground_truth = complex_synthetic_chm.attrs.get("ground_truth")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        _plot_chm_with_treetops(
            ax1,
            complex_synthetic_chm,
            reference,
            title=f"Reference (eager scipy) — {len(reference)} treetops",
            ground_truth=ground_truth,
        )
        _plot_chm_with_treetops(
            ax2,
            complex_synthetic_chm,
            result,
            title=f"New (dask-image) — {len(result)} treetops",
            ground_truth=ground_truth,
        )
        fig.suptitle(
            f"Reference vs New: {filter_name} filter", fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        if SAVE_FIG:
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                FIGURES_PATH / f"itd_reference_vs_new_{filter_name}.png", dpi=150
            )
        if SHOW_FIG:
            plt.show()
        plt.close(fig)


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_output_is_identical_across_chunk_layouts(
    filter_name: str,
    mixed_morphology_chm: xr.DataArray,
):
    """Results must be identical whether CHM is 1, 2, or 4 chunks."""
    one_chunk = _run_filter(
        filter_name, _chunk_chm(mixed_morphology_chm, {"y": 200, "x": 200})
    )
    two_chunks = _run_filter(
        filter_name, _chunk_chm(mixed_morphology_chm, {"y": 200, "x": 100})
    )
    four_chunks = _run_filter(
        filter_name, _chunk_chm(mixed_morphology_chm, {"y": 100, "x": 100})
    )

    _assert_same_output(one_chunk, two_chunks)
    _assert_same_output(one_chunk, four_chunks)

    if SAVE_FIG or SHOW_FIG:
        _plot_chunked_comparison(
            mixed_morphology_chm,
            {
                "1 chunk (200×200)": one_chunk,
                "2 chunks (200×100)": two_chunks,
                "4 chunks (100×100)": four_chunks,
            },
            suptitle=f"Chunk Layout Invariance: {filter_name} filter",
            filename=f"itd_chunk_invariance_{filter_name}.png",
        )


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "boundary_split_half_chm",
        "boundary_split_three_quarters_chm",
        "corner_split_chm",
    ],
)
def test_boundary_straddle_plateaus_emit_one_treetop(
    filter_name: str,
    fixture_name: str,
    request: pytest.FixtureRequest,
):
    """A flat plateau straddling a chunk boundary must produce exactly 1 treetop."""
    boundary_chm = request.getfixturevalue(fixture_name)
    one_chunk = _chunk_chm(boundary_chm, {"y": 64, "x": 64})
    chunked = _chunk_chm(boundary_chm, {"y": 32, "x": 32})

    reference = _run_filter(filter_name, one_chunk)
    result = _run_filter(filter_name, chunked)

    assert len(result) == 1
    _assert_same_output(result, reference)

    if SAVE_FIG or SHOW_FIG:
        extent = _get_extent(boundary_chm)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, df, label in [
            (ax1, reference, "1 chunk (64×64)"),
            (ax2, result, "4 chunks (32×32)"),
        ]:
            ax.imshow(
                boundary_chm.values, extent=extent, cmap="viridis", origin="upper"
            )
            ax.scatter(
                df["x"],
                df["y"],
                c="red",
                marker="x",
                s=120,
                linewidths=2,
                label=f"Detected ({len(df)})",
            )
            # Draw chunk boundaries
            mid_x = extent[0] + (extent[1] - extent[0]) / 2
            mid_y = extent[2] + (extent[3] - extent[2]) / 2
            ax.axhline(mid_y, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axvline(mid_x, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_title(f"{label}", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"Boundary Plateau: {fixture_name} / {filter_name}",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        if SAVE_FIG:
            FIGURES_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                FIGURES_PATH / f"itd_boundary_{fixture_name}_{filter_name}.png", dpi=150
            )
        if SHOW_FIG:
            plt.show()
        plt.close(fig)


def test_returned_dataframe_is_lazy(
    mixed_morphology_chm: xr.DataArray,
):
    """Graph construction must not trigger computation."""
    chunked = _chunk_chm(mixed_morphology_chm, {"y": 100, "x": 100})

    class TaskCounter(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.task_count = 0

        def _pretask(self, key, dsk, state) -> None:  # noqa: ANN001
            self.task_count += 1

    callback = TaskCounter()
    with callback:
        ddf = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=0.5,
            window_size_meters=3.0,
        )

    assert callback.task_count == 0
    assert isinstance(ddf, dd.DataFrame)


def test_graph_contains_more_tasks_for_multi_chunk_inputs(
    boundary_split_half_chm: xr.DataArray,
):
    """More chunks should produce a larger task graph."""
    one_chunk = fixed_window_filter(
        chm_da=_chunk_chm(boundary_split_half_chm, {"y": 64, "x": 64}),
        min_height=2.0,
        spatial_resolution=1.0,
        window_size_meters=3.0,
    )
    four_chunks = fixed_window_filter(
        chm_da=_chunk_chm(boundary_split_half_chm, {"y": 32, "x": 32}),
        min_height=2.0,
        spatial_resolution=1.0,
        window_size_meters=3.0,
    )

    graph_one = len(one_chunk.__dask_graph__())
    graph_four = len(four_chunks.__dask_graph__())

    assert graph_four > graph_one


# ---------------------------------------------------------------------------
# Tests: interior/boundary split correctness
# ---------------------------------------------------------------------------


class TestBoundaryFlagClassification:
    """Unit tests for is_boundary flag in _extract_block_candidates."""

    @staticmethod
    def _call(chm_block, mask_block, **kwargs):
        """Helper: call _extract_block_candidates and return the candidates df."""
        result_tuple = _extract_block_candidates(
            chm_block,
            mask_block,
            row_offset=kwargs.get("row_offset", 0),
            col_offset=kwargs.get("col_offset", 0),
            label_offset=kwargs.get("label_offset", 0),
        )
        return result_tuple[0]  # candidates DataFrame

    def test_interior_label_flagged_false(self):
        """A label entirely inside the block (no edge pixels) is is_boundary=False."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True

        chm_block = np.zeros((10, 10), dtype=np.float64)
        chm_block[4:6, 4:6] = 20.0

        result = self._call(chm_block, mask)
        assert len(result) == 1
        assert result.iloc[0]["is_boundary"] == False  # noqa: E712

    def test_edge_touching_label_flagged_true(self):
        """A label touching the top row of the block is is_boundary=True."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 4:6] = True

        chm_block = np.zeros((10, 10), dtype=np.float64)
        chm_block[0, 4:6] = 15.0

        result = self._call(chm_block, mask)
        assert len(result) == 1
        assert result.iloc[0]["is_boundary"] == True  # noqa: E712

    def test_corner_label_flagged_true(self):
        """A label touching a corner pixel is is_boundary=True."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True

        chm_block = np.zeros((10, 10), dtype=np.float64)
        chm_block[0, 0] = 12.0

        result = self._call(chm_block, mask)
        assert len(result) == 1
        assert result.iloc[0]["is_boundary"] == True  # noqa: E712

    def test_mixed_interior_and_boundary_labels(self):
        """Multiple disconnected labels: some interior, some boundary."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True  # interior
        mask[0, 5] = True  # boundary (top edge)
        mask[9, 5] = True  # boundary (bottom edge)
        mask[5, 0] = True  # boundary (left edge)
        mask[5, 9] = True  # boundary (right edge)
        mask[3, 3] = True  # interior

        chm_block = np.ones((10, 10), dtype=np.float64) * 10.0

        result = self._call(chm_block, mask)
        n_interior = (result["is_boundary"] == False).sum()  # noqa: E712
        n_boundary = (result["is_boundary"] == True).sum()  # noqa: E712

        assert len(result) == 6
        assert n_interior == 2
        assert n_boundary == 4

    def test_label_spanning_edge_and_interior_flagged_true(self):
        """A connected component spanning edge and interior is is_boundary=True."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 5] = True
        mask[1, 5] = True
        mask[2, 5] = True  # connected, extends from edge

        chm_block = np.zeros((10, 10), dtype=np.float64)
        chm_block[0:3, 5] = 15.0

        result = self._call(chm_block, mask)
        assert len(result) == 1
        assert result.iloc[0]["is_boundary"] == True  # noqa: E712
        centroid_row = (
            result.iloc[0]["centroid_row_sum"] / result.iloc[0]["centroid_count"]
        )
        centroid_col = (
            result.iloc[0]["centroid_col_sum"] / result.iloc[0]["centroid_count"]
        )
        assert centroid_row == pytest.approx(1.0)
        assert centroid_col == pytest.approx(5.0)

    def test_empty_block_returns_empty_with_is_boundary_column(self):
        """An empty mask returns an empty DataFrame with is_boundary column."""
        mask = np.zeros((10, 10), dtype=bool)
        chm_block = np.zeros((10, 10), dtype=np.float64)

        result = self._call(chm_block, mask)
        assert len(result) == 0
        assert "is_boundary" in result.columns

    def test_returns_edge_label_arrays(self):
        """_extract_block_candidates returns 4 edge label arrays."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 5] = True  # top edge
        mask[9, 3] = True  # bottom edge

        chm_block = np.ones((10, 10), dtype=np.float64) * 20.0

        _, bottom, right, top, left = _extract_block_candidates(
            chm_block,
            mask,
            row_offset=0,
            col_offset=0,
            label_offset=0,
        )
        assert top[5] > 0  # label at top edge col 5
        assert bottom[3] > 0  # label at bottom edge col 3
        assert len(top) == 10
        assert len(bottom) == 10
        assert len(left) == 10
        assert len(right) == 10


class TestUnionFindMerge:
    """Unit tests for _union_find_merge."""

    def test_single_candidate_no_merges(self):
        """A single boundary candidate with no merge pairs passes through."""
        candidates = pd.DataFrame(
            {
                "label": [1],
                "height": [25.0],
                "centroid_row_sum": [10.0],
                "centroid_col_sum": [20.0],
                "centroid_count": [1],
                "is_boundary": [True],
            }
        )
        transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
        result = _union_find_merge(candidates, [], transform)

        assert len(result) == 1
        assert list(result.columns) == ["x", "y", "height"]
        assert result.iloc[0]["height"] == 25.0

    def test_merge_pair_combines_centroids(self):
        """Two candidates linked by a merge pair: centroids are combined."""
        candidates = pd.DataFrame(
            {
                "label": [1, 2],
                "height": [20.0, 20.0],
                "centroid_row_sum": [10.0, 14.0],
                "centroid_col_sum": [40.0, 60.0],
                "centroid_count": [4, 4],
                "is_boundary": [True, True],
            }
        )
        transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
        result = _union_find_merge(candidates, [(1, 2)], transform)

        assert len(result) == 1
        expected_x, expected_y = rio.transform.xy(transform, [3.0], [12.5])
        assert result.iloc[0]["x"] == pytest.approx(expected_x[0])
        assert result.iloc[0]["y"] == pytest.approx(expected_y[0])

    def test_transitive_merge_three_labels(self):
        """Three labels linked transitively: A-B and B-C merges all three."""
        candidates = pd.DataFrame(
            {
                "label": [1, 2, 3],
                "height": [20.0, 20.0, 20.0],
                "centroid_row_sum": [6.0, 10.0, 14.0],
                "centroid_col_sum": [20.0, 30.0, 40.0],
                "centroid_count": [2, 2, 2],
                "is_boundary": [True, True, True],
            }
        )
        transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
        result = _union_find_merge(candidates, [(1, 2), (2, 3)], transform)

        assert len(result) == 1
        # centroid: row=(6+10+14)/6=5.0, col=(20+30+40)/6=15.0
        expected_x, expected_y = rio.transform.xy(transform, [5.0], [15.0])
        assert result.iloc[0]["x"] == pytest.approx(expected_x[0])
        assert result.iloc[0]["y"] == pytest.approx(expected_y[0])

    def test_independent_labels_stay_separate(self):
        """Two unlinked boundary labels are not merged."""
        candidates = pd.DataFrame(
            {
                "label": [1, 2],
                "height": [20.0, 15.0],
                "centroid_row_sum": [5.0, 30.0],
                "centroid_col_sum": [10.0, 50.0],
                "centroid_count": [2, 5],
                "is_boundary": [True, True],
            }
        )
        transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
        result = _union_find_merge(candidates, [], transform)

        assert len(result) == 2

    def test_empty_input_returns_empty_output(self):
        """Empty input returns empty DataFrame with correct columns."""
        candidates = pd.DataFrame(
            columns=[
                "label",
                "height",
                "centroid_row_sum",
                "centroid_col_sum",
                "centroid_count",
                "is_boundary",
            ]
        )
        transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
        result = _union_find_merge(candidates, [], transform)

        assert len(result) == 0
        assert list(result.columns) == ["x", "y", "height"]


class TestInteriorBoundarySplit:
    """Integration tests verifying the interior/boundary split in the full pipeline."""

    def test_output_has_more_partitions_than_one(self):
        """Multi-chunk input produces multi-partition output (not single-partition)."""
        chm_da = generate_boundary_plateau_chm(
            slice(28, 32), slice(30, 34), width=64, height=64
        )
        chunked = _chunk_chm(chm_da, {"y": 32, "x": 32})

        ddf = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        )
        assert ddf.npartitions > 1

    def test_interior_only_chm_has_empty_boundary_partition(self):
        """When all treetops are interior to their chunks, the boundary partition
        should be empty and the result should still be correct."""
        # Place a single tree well inside a chunk (center of a 64x64 block,
        # chunked as a single 64x64 chunk — all labels touch the "boundary"
        # of the one chunk). Instead, use a large chunk so trees are interior.
        chm_array = np.zeros((128, 128), dtype=np.float64)
        # Place Gaussian trees far from any chunk boundary (chunk size 128)
        for r, c in [(32, 32), (32, 96), (96, 32), (96, 96)]:
            y_grid, x_grid = np.ogrid[:128, :128]
            dist_sq = (x_grid - c) ** 2 + (y_grid - r) ** 2
            canopy = 20.0 * np.exp(-dist_sq / (2 * 3.0**2))
            np.maximum(chm_array, canopy, out=chm_array)

        chm_da = xr.DataArray(chm_array, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
        # Chunk so each tree is well within its chunk
        chunked = _chunk_chm(chm_da, {"y": 64, "x": 64})

        ddf = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        )
        result = ddf.compute()
        assert len(result) == 4

        # The last partition is the boundary dedup partition — should be empty
        boundary_partition = ddf.get_partition(ddf.npartitions - 1).compute()
        assert len(boundary_partition) == 0

    def test_boundary_partition_contains_only_boundary_labels(
        self,
        boundary_split_half_chm: xr.DataArray,
    ):
        """The boundary dedup partition should contain only labels that
        straddled chunk edges, not interior labels."""
        # This CHM has a single plateau at rows 28-31, cols 30-33
        # in a 64x64 grid. With 32x32 chunks, the plateau straddles
        # the row boundary (row 31/32) and col boundary (col 31/32).
        chunked = _chunk_chm(boundary_split_half_chm, {"y": 32, "x": 32})

        ddf = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        )
        result = ddf.compute()
        assert len(result) == 1

        # The single treetop should come from the boundary partition (last one)
        boundary_partition = ddf.get_partition(ddf.npartitions - 1).compute()
        assert len(boundary_partition) == 1

        # Interior partitions should all be empty (no interior trees in this CHM)
        for i in range(ddf.npartitions - 1):
            interior_part = ddf.get_partition(i).compute()
            assert len(interior_part) == 0

    @pytest.mark.parametrize("filter_name", ["fixed", "variable"])
    def test_boundary_materialization_is_small_fraction(
        self,
        filter_name: str,
    ):
        """The boundary partition should be a small fraction of total treetops.

        This is the core memory-safety property: only O(perimeter) candidates
        are materialized for dedup, not O(area).

        Uses a dense random CHM so that treetops are uniformly distributed
        across each chunk's interior, not concentrated at chunk boundaries.
        """
        rng = np.random.default_rng(777)
        arr = rng.uniform(0, 30, (256, 256)).astype(np.float64)
        chm_da = xr.DataArray(arr, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
        chunked = _chunk_chm(chm_da, {"y": 64, "x": 64})  # 16 chunks

        ddf = (
            fixed_window_filter(
                chm_da=chunked,
                min_height=2.0,
                spatial_resolution=1.0,
                window_size_meters=3.0,
            )
            if filter_name == "fixed"
            else variable_window_filter(
                chm_da=chunked,
                min_height=2.0,
                spatial_resolution=1.0,
                crown_ratio=0.10,
                crown_offset=1.0,
            )
        )

        total = len(ddf)
        boundary_count = len(ddf.get_partition(ddf.npartitions - 1))

        assert total > 0, "Test expects a non-empty result"
        boundary_fraction = boundary_count / total
        # Boundary labels should be a small fraction of total.
        # For 64x64 chunks, perimeter/area = 4*64 / 64^2 ≈ 6%.
        # In practice, boundary labels are typically 2-8% of total.
        assert boundary_fraction < 0.15, (
            f"Boundary fraction {boundary_fraction:.1%} is too high — "
            f"dedup materialization may not be bounded"
        )

    @pytest.mark.parametrize("filter_name", ["fixed", "variable"])
    def test_split_matches_reference_on_dense_random_chm(
        self,
        filter_name: str,
    ):
        """Interior + boundary paths combined must match the eager reference
        on a dense random CHM where both paths are exercised."""
        rng = np.random.default_rng(12345)
        arr = rng.uniform(0, 30, (128, 128)).astype(np.float64)
        chm_da = xr.DataArray(arr, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 0.5, 0.5), inplace=True)
        chunked = _chunk_chm(chm_da, {"y": 32, "x": 32})

        reference = _run_reference(filter_name, chm_da)
        result = _run_filter(filter_name, chunked)

        _assert_same_output(result, reference)

    def test_all_zeros_chm_produces_empty_output(self):
        """A CHM with all zero values produces empty output from both paths."""
        chm_array = np.zeros((64, 64), dtype=np.float64)
        chm_da = xr.DataArray(chm_array, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
        chunked = _chunk_chm(chm_da, {"y": 32, "x": 32})

        result = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        ).compute()
        assert len(result) == 0
        assert list(result.columns) == ["x", "y", "height"]

    def test_all_below_min_height_produces_empty_output(self):
        """A CHM with all values below min_height produces empty output."""
        rng = np.random.default_rng(99)
        chm_array = rng.uniform(0.1, 1.5, (64, 64)).astype(np.float64)
        chm_da = xr.DataArray(chm_array, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
        chunked = _chunk_chm(chm_da, {"y": 32, "x": 32})

        result = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        ).compute()
        assert len(result) == 0

    def test_small_chunks_mostly_boundary_still_correct(self):
        """With small chunks where most labels touch an edge, the pipeline
        must still produce correct results via the boundary dedup path."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0, 30, (64, 64)).astype(np.float64)
        chm_da = xr.DataArray(arr, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(from_origin(0, 0, 0.5, 0.5), inplace=True)
        chunked = _chunk_chm(chm_da, {"y": 16, "x": 16})  # 16 chunks

        reference = _run_reference("fixed", chm_da)
        result = _run_filter("fixed", chunked)

        _assert_same_output(result, reference)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation in public API functions."""

    def test_fixed_window_rejects_negative_min_height(self, simple_chm: xr.DataArray):
        with pytest.raises(ValueError, match="min_height cannot be negative"):
            fixed_window_filter(
                chm_da=simple_chm, min_height=-1.0, spatial_resolution=0.5
            )

    def test_fixed_window_rejects_zero_spatial_resolution(
        self, simple_chm: xr.DataArray
    ):
        with pytest.raises(ValueError, match="spatial_resolution must be positive"):
            fixed_window_filter(
                chm_da=simple_chm, min_height=2.0, spatial_resolution=0.0
            )

    def test_fixed_window_rejects_negative_spatial_resolution(
        self, simple_chm: xr.DataArray
    ):
        with pytest.raises(ValueError, match="spatial_resolution must be positive"):
            fixed_window_filter(
                chm_da=simple_chm, min_height=2.0, spatial_resolution=-1.0
            )

    def test_variable_window_rejects_negative_min_height(
        self, simple_chm: xr.DataArray
    ):
        with pytest.raises(ValueError, match="min_height cannot be negative"):
            variable_window_filter(
                chm_da=simple_chm, min_height=-1.0, spatial_resolution=0.5
            )

    def test_variable_window_rejects_zero_spatial_resolution(
        self, simple_chm: xr.DataArray
    ):
        with pytest.raises(ValueError, match="spatial_resolution must be positive"):
            variable_window_filter(
                chm_da=simple_chm, min_height=2.0, spatial_resolution=0.0
            )

    def test_prepare_chm_rejects_non_2d_input(self):
        arr = np.zeros((10, 10, 3))
        da = xr.DataArray(arr, dims=["y", "x", "band"])
        da.rio.write_crs("EPSG:32611", inplace=True)
        da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
        with pytest.raises(ValueError, match="CHM must be a 2D DataArray"):
            fixed_window_filter(chm_da=da, min_height=2.0, spatial_resolution=1.0)

    def test_fixed_window_small_window_floors_to_three(self):
        """A window_size_meters that resolves to < 3 pixels is floored to 3."""
        pixel_size = 1.0
        chm_array = np.zeros((50, 50), dtype=np.float64)
        chm_array[25, 25] = 20.0
        chm_da = xr.DataArray(chm_array, dims=["y", "x"])
        chm_da.rio.write_crs("EPSG:32611", inplace=True)
        chm_da.rio.write_transform(
            from_origin(0, 0, pixel_size, pixel_size), inplace=True
        )

        result = fixed_window_filter(
            chm_da=chm_da,
            min_height=2.0,
            spatial_resolution=pixel_size,
            window_size_meters=1.0,  # 1.0 / 1.0 = 1 pixel, floored to 3
        ).compute()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Asymmetric chunk layout test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_asymmetric_chunk_layout(
    filter_name: str,
    mixed_morphology_chm: xr.DataArray,
):
    """Results must be identical with non-square (asymmetric) chunk layouts."""
    reference = _run_reference(filter_name, mixed_morphology_chm)
    tall_chunks = _run_filter(
        filter_name, _chunk_chm(mixed_morphology_chm, {"y": 200, "x": 50})
    )
    wide_chunks = _run_filter(
        filter_name, _chunk_chm(mixed_morphology_chm, {"y": 50, "x": 200})
    )

    _assert_same_output(reference, tall_chunks)
    _assert_same_output(reference, wide_chunks)


# ---------------------------------------------------------------------------
# Benchmark and parallel execution tests (gated by env vars)
# ---------------------------------------------------------------------------


# @pytest.mark.skipif(
#     os.getenv("RUN_HEAVY_ITD_BENCHMARK") != "1",
#     reason="Set RUN_HEAVY_ITD_BENCHMARK=1 to run the ITD benchmark suite.",
# )
def test_memory_bounded_execution():
    """Peak memory during chunked execution must be bounded by chunk size,
    not by the full CHM size.  This is acceptance criterion 2 from issue #70."""
    import tracemalloc

    size = 4096
    chunk_size = 512
    rng = np.random.default_rng(42)
    arr = rng.normal(1.0, 0.1, (size, size)).astype(np.float64)
    for r, c in [(500, 500), (500, 3500), (3500, 500), (3500, 3500), (2048, 2048)]:
        y_grid, x_grid = np.ogrid[:size, :size]
        canopy = 25.0 * np.exp(-((x_grid - c) ** 2 + (y_grid - r) ** 2) / (2 * 8**2))
        np.maximum(arr, canopy, out=arr)

    chm_da = xr.DataArray(arr, dims=["y", "x"])
    chm_da.rio.write_crs("EPSG:32611", inplace=True)
    chm_da.rio.write_transform(from_origin(0, 0, 1.0, 1.0), inplace=True)
    chunked = _chunk_chm(chm_da, {"y": chunk_size, "x": chunk_size})

    full_array_bytes = arr.nbytes  # 128 MB
    chunk_bytes = chunk_size * chunk_size * arr.itemsize  # 2 MB

    import dask

    tracemalloc.start()
    with dask.config.set(scheduler="synchronous"):
        result = fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=5.0,
        ).compute()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(result) == 5
    # Peak memory must be well below the full array size.
    # Allow a generous bound: 10 chunks worth of memory for concurrent
    # intermediate arrays (overlap buffers, mask, labels, etc.)
    memory_bound = 10 * chunk_bytes
    print(
        f"Peak memory: {peak / 1024**2:.1f} MB, "
        f"bound: {memory_bound / 1024**2:.1f} MB, "
        f"full array: {full_array_bytes / 1024**2:.1f} MB"
    )
    assert peak < memory_bound, (
        f"Peak memory {peak / 1024**2:.1f} MB exceeds bound "
        f"{memory_bound / 1024**2:.1f} MB (full array is "
        f"{full_array_bytes / 1024**2:.1f} MB)"
    )


# @pytest.mark.skipif(
#     os.getenv("RUN_HEAVY_ITD_BENCHMARK") != "1",
#     reason="Set RUN_HEAVY_ITD_BENCHMARK=1 to run the ITD benchmark suite.",
# )
@pytest.mark.parametrize("filter_name", ["fixed", "variable"])
def test_benchmark_old_vs_new_chunked(filter_name: str):
    benchmark_chm = generate_benchmark_chm(size=1024)

    def measure(label: str, runner: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        start = time.perf_counter()
        result = runner()
        elapsed = time.perf_counter() - start
        print(f"{filter_name} {label}: elapsed={elapsed:.4f}s")
        return result

    reference = measure(
        label="reference",
        runner=lambda: _run_reference(filter_name, benchmark_chm),
    )
    one_chunk = measure(
        label="new_one_chunk",
        runner=lambda: _run_filter(
            filter_name, _chunk_chm(benchmark_chm, {"y": 1024, "x": 1024})
        ),
    )
    four_chunks = measure(
        label="new_four_chunks",
        runner=lambda: _run_filter(
            filter_name, _chunk_chm(benchmark_chm, {"y": 512, "x": 512})
        ),
    )

    _assert_same_output(reference, one_chunk)
    _assert_same_output(one_chunk, four_chunks)


# @pytest.mark.skipif(
#     os.getenv("RUN_HEAVY_ITD_PARALLEL") != "1",
#     reason="Set RUN_HEAVY_ITD_PARALLEL=1 to run the ITD parallel-execution proof.",
# )
def test_maximum_filter_blocks_execute_in_parallel():
    benchmark_chm = generate_benchmark_chm(size=128)
    chunked = _chunk_chm(benchmark_chm, {"y": 32, "x": 32})

    records: list[tuple[float, float, int]] = []
    lock = threading.Lock()

    import scipy.ndimage

    original_scipy_mf = scipy.ndimage.maximum_filter

    def wrapped_scipy_mf(*args, **kwargs):
        start = time.perf_counter()
        thread_id = threading.get_ident()
        time.sleep(0.05)
        result = original_scipy_mf(*args, **kwargs)
        end = time.perf_counter()
        with lock:
            records.append((start, end, thread_id))
        return result

    import unittest.mock

    with (
        unittest.mock.patch("scipy.ndimage.maximum_filter", wrapped_scipy_mf),
        unittest.mock.patch("scipy.ndimage._filters.maximum_filter", wrapped_scipy_mf),
        dask.config.set(scheduler="threads", num_workers=4),
    ):
        fixed_window_filter(
            chm_da=chunked,
            min_height=2.0,
            spatial_resolution=1.0,
            window_size_meters=3.0,
        ).compute()

    assert len(records) >= 4
    assert len({thread_id for _, _, thread_id in records}) > 1
    assert any(
        start_a < end_b and start_b < end_a
        for index, (start_a, end_a, _) in enumerate(records)
        for start_b, end_b, _ in records[index + 1 :]
    )
