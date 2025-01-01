import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from rasterio.enums import Resampling

from fastfuels_core.onramps.hag_pim import (
    check_same_crs,
    check_projected_crs,
    check_resolution,
    resample_raster,
    compute_cover_from_hag,
    interpolate_hag,
    interpolate_pim,
    combine_hag_pim,
    create_plots_gdf_from_resampled_pim,
)


def create_toy_raster(
    bounds=(-12, 0, 0, 8), crs="EPSG:5070", nodata_value=0, res=(1, -1), data=None
):
    x_min, x_max, y_min, y_max = bounds

    width = int((x_max - x_min) / abs(res[0]))
    height = int((y_max - y_min) / abs(res[1]))
    transform = from_origin(x_min, y_max, abs(res[0]), abs(res[1]))
    if data is None:
        data = np.ones((height, width))
    toy_raster = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "x": np.linspace(x_min + res[0] / 2, x_max - res[0] / 2, width),
            "y": np.linspace(y_min - res[1] / 2, y_max + res[1] / 2, height)[::-1],
        },
        name="toy_data",
    )
    toy_raster = toy_raster.rio.write_crs(crs)
    toy_raster = toy_raster.rio.write_transform(transform)
    toy_raster.rio.write_nodata(nodata_value, inplace=True)
    return toy_raster


@pytest.fixture
def hag_4326():
    data = np.array(
        [
            [0, 1, 2, 11, 12, 3],
            [4, 5, 6, 7, 8, 13],
            [9, 14, 15, 16, 17, 18],
            [0, 19, 10, 20, 21, 22],
        ]
    )
    return create_toy_raster(
        bounds=(-6, 0, 0, 4), crs="EPSG:4326", res=(1, -1), data=data
    )


@pytest.fixture
def pim_5070():
    bounds = (-12, 0, 0, 8)
    data = np.array(
        [[0, 1, 0, 1, 0, 1], [0, 2, 0, 2, 0, 2], [0, 3, 0, 3, 0, 3], [0, 4, 0, 4, 0, 4]]
    )
    return create_toy_raster(bounds=bounds, data=data, res=(2, -2))


@pytest.fixture
def hag_5070():
    data = np.arange(24).reshape((4, 6))
    return create_toy_raster(bounds=(-12.5, -0.5, 0.5, 8.5), res=(2, -2), data=data)


def test_check_same_crs_matching(hag_5070, pim_5070):
    try:
        check_same_crs(hag_5070, pim_5070)
    except ValueError:
        pytest.fail("check_same_crs raised ValueError for matching CRS.")


def test_check_same_crs_mismatching(hag_4326, hag_5070):
    with pytest.raises(ValueError, match="do not have the same CRS"):
        check_same_crs(hag_4326, hag_5070)


def test_check_projected_crs_true(pim_5070):
    try:
        check_projected_crs(pim_5070)
    except ValueError:
        pytest.fail(
            "check_projected_crs raised ValueError for raster with a projected CRS."
        )


def test_check_projected_crs_false(hag_4326):
    with pytest.raises(ValueError, match="do not have a projected CRS"):
        check_projected_crs(hag_4326)


def test_check_resolution_true(pim_5070, hag_4326):
    try:
        check_resolution(pim_5070, hag_4326, 2)
    except ValueError:
        pytest.fail(
            "check_resolution raised ValueError for rasters with correct resolutions."
        )


def test_check_resolution_false1(pim_5070, hag_4326):
    with pytest.raises(ValueError, match="is finer than the resolution of the height"):
        check_resolution(hag_4326, pim_5070, 1)


def test_check_resolution_false2(pim_5070, hag_5070):
    with pytest.raises(ValueError, match="is finer than the desired resolution"):
        check_resolution(pim_5070, hag_5070, 100)


def test_compute_cover(hag_4326):
    min_value = 10
    desired_res = 2
    cover_raster = compute_cover_from_hag(hag_4326, min_value, desired_res)
    correct_values = np.array([[0, 0.25, 0.5], [0.5, 0.75, 1]])
    assert cover_raster.rio.resolution() == (desired_res, -desired_res)
    assert cover_raster.rio.crs == hag_4326.rio.crs
    assert cover_raster.rio.nodata == hag_4326.rio.nodata
    assert np.all(cover_raster.values == correct_values)


def test_resample_raster_pim(pim_5070):
    desired_res = 1
    resampled = resample_raster(pim_5070, desired_res, Resampling.nearest)
    correct_values = np.array(
        [
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2],
            [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2],
            [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3],
            [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3],
            [0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4],
            [0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4],
        ]
    )
    assert resampled.rio.resolution() == (desired_res, -desired_res)
    assert resampled.rio.crs == pim_5070.rio.crs
    assert resampled.rio.nodata == pim_5070.rio.nodata
    assert np.all(resampled.values == correct_values)


def test_resample_raster_hag(hag_4326):
    desired_res = 2
    correct_values = np.array([[10, 26, 36], [42, 61, 78]])
    resampled = resample_raster(hag_4326, desired_res, Resampling.sum)
    assert resampled.rio.resolution() == (desired_res, -desired_res)
    assert resampled.rio.crs == hag_4326.rio.crs
    assert resampled.rio.nodata == hag_4326.rio.nodata
    assert np.all(resampled.values == correct_values)


def test_interpolate_pim(pim_5070):
    correct_values = np.array(
        [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4]]
    )
    interpolated = interpolate_pim(pim_5070)
    assert interpolated.rio.resolution() == pim_5070.rio.resolution()
    assert interpolated.rio.crs == pim_5070.rio.crs
    assert interpolated.rio.nodata == pim_5070.rio.nodata
    assert np.all(interpolated.values == correct_values)


def test_interpolate_hag(hag_5070, pim_5070):
    interpolated = interpolate_hag(hag_5070, pim_5070)
    assert interpolated.rio.resolution() == pim_5070.rio.resolution()
    assert interpolated.rio.crs == pim_5070.rio.crs
    assert interpolated.rio.nodata == pim_5070.rio.nodata
    assert interpolated.coords.equals(pim_5070.coords)
    assert np.all(interpolated.values == hag_5070.values)


def test_combine_hag_pim():
    bounds = (-3, 0, 0, 2)
    pim_data = np.arange(6).reshape((2, 3))
    hag_data = np.array([[0, 0.25, 0.5], [0.5, 0.75, 1]])
    correct_25 = np.array([[0, 0, 2], [3, 4, 5]])
    correct_50 = np.array([[0, 0, 0], [0, 4, 5]])
    correct_75 = np.array([[0, 0, 0], [0, 0, 5]])
    pim = create_toy_raster(bounds=bounds, data=pim_data)
    hag = create_toy_raster(bounds=bounds, data=hag_data)
    combined_25 = combine_hag_pim(hag, pim, 0.25)
    combined_50 = combine_hag_pim(hag, pim, 0.5)
    combined_75 = combine_hag_pim(hag, pim, 0.75)
    assert combined_25.rio.resolution() == pim.rio.resolution()
    assert combined_25.rio.crs == pim.rio.crs
    assert combined_25.rio.nodata == pim.rio.nodata
    assert combined_25.coords.equals(pim.coords)
    assert np.all(combined_25.values == correct_25)
    assert np.all(combined_50.values == correct_50)
    assert np.all(combined_75.values == correct_75)


def test_create_gdf(pim_5070):
    gdf1 = create_plots_gdf_from_resampled_pim(pim_5070)
    gdf2 = create_plots_gdf_from_resampled_pim(pim_5070, "test_name")
    assert np.all(gdf1.columns == ["Y", "X", "PLOT_ID", "geometry"])
    assert np.all(gdf2.columns == ["Y", "X", "test_name", "geometry"])
    assert np.all(gdf1.X.unique() == pim_5070["x"].values)
    assert np.all(gdf1.Y.unique() == pim_5070["y"].values)
    assert np.all(gdf1.PLOT_ID.unique() == [0, 1, 2, 3, 4])
