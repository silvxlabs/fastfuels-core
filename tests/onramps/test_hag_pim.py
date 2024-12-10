import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin

from fastfuels_core.onramps.hag_pim import (
    check_same_crs,
    check_projected_crs,
    check_resolution,
    resample_raster,
    convert_to_cover,
    interpolate_hag,
    interpolate_pim,
    combine_hag_pim,
    create_gdf
)

def create_toy_data(bounds,pim_res, hag_res, 
                    pim_crs, hag_crs, nodata_value):

    x_min, x_max, y_min, y_max = bounds
   
    # Create toy tree imputation raster
    width = int((x_max - x_min) / abs(pim_res[0]))
    height = int((y_max - y_min) / abs(pim_res[1]))
    transform = from_origin(x_min, y_max, pim_res[0], -pim_res[1])
    data = np.arange(0,height).reshape(height, 1)*np.ones((1, width))
    data[:,::2] = 0
    toy_pim = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "x": np.linspace(x_min + pim_res[0]/2, 
                             x_max - pim_res[0]/2, width),
            "y": np.linspace(y_min - pim_res[1]/2, 
                             y_max + pim_res[1]/2, height),
        },
        name="toy_pim_data")
    toy_pim = toy_pim.rio.write_crs(pim_crs)
    toy_pim = toy_pim.rio.write_transform(transform)
    toy_pim.rio.write_nodata(nodata_value, inplace=True)
    
    # Create toy height above ground raster
    width = int((x_max - x_min) / hag_res[0])
    height = int((y_max - y_min) / abs(hag_res[1]))
    transform = from_origin(x_min, y_max, hag_res[0], abs(hag_res[1]))
    data = np.zeros((height,width), dtype=int)
    block_height = height // 3  # Divide into 3 rows of blocks
    block_width = width // 6   # Divide into 6 columns of blocks
    block_values = np.arange(1,10)
    block_idx = 0
    for i in range(3):  # Three rows of blocks
        for j in range(6):  # Six columns of blocks    
            start_row = i * block_height
            end_row = (i + 1) * block_height
            start_col = j * block_width
            end_col = (j + 1) * block_width
            if (i%2 == 0 and j%2 == 0) or (i%2 != 0 and j%2 != 0):
                data[start_row:end_row, start_col:end_col] = 0
            else:
                data[start_row:end_row, 
                start_col:end_col] = block_values[block_idx]
                block_idx += 1
    toy_hag = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "x": np.linspace(x_min, 
                             x_max, width),
            "y": np.linspace(y_min, 
                             y_max, height),
        },
        name="toy_hag_data")
    toy_hag = toy_hag.rio.write_crs(hag_crs)
    toy_hag = toy_hag.rio.write_transform(transform)
    toy_hag.rio.write_nodata(nodata_value, inplace=True)

    return toy_pim, toy_hag


def create_toy_raster(bounds = (-10,0,0,10),
                      crs="EPSG:5070",
                      nodata_value = 0,
                      res = (1,-1)):
    x_min, x_max, y_min, y_max = bounds
   
    width = int((x_max - x_min) / abs(res[0]))
    height = int((y_max - y_min) / abs(res[1]))
    transform = from_origin(x_min, y_max, abs(res[0]), abs(res[1]))
    data = np.ones((height, width))
    toy_raster = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "x": np.linspace(x_min + res[0]/2, 
                             x_max - res[0]/2, width),
            "y": np.linspace(y_min - res[1]/2, 
                             y_max + res[1]/2, height),
        },
        name="toy_data")
    toy_raster = toy_raster.rio.write_crs(crs)
    toy_raster = toy_raster.rio.write_transform(transform)
    toy_raster.rio.write_nodata(nodata_value, inplace=True)
    return toy_raster

@pytest.fixture
def toy_4326():
    return create_toy_raster(crs='EPSG:4326')

@pytest.fixture
def toy_5070():
    return create_toy_raster(crs='EPSG:5070')

def test_check_same_crs_matching(toy_5070):
    try:
        check_same_crs(toy_5070, toy_5070)
    except ValueError:
        pytest.fail("check_same_crs raised ValueError for matching CRS.")

def test_check_same_crs_mismatching(toy_4326, toy_5070):
    with pytest.raises(ValueError, match="do not have the same CRS"):
        check_same_crs(toy_4326, toy_5070)


def test_check_projected_crs_true(toy_5070):
    try:
        check_projected_crs(toy_5070)
    except ValueError:
        pytest.fail("check_prpjected_crs raised ValueError for raster with a projected CRS.")

def test_check_projected_crs_false(toy_4326):
    with pytest.raises(ValueError, match="do not have a projected CRS"):
        check_projected_crs(toy_4326)   

    
'''
def test_check_resolution(pim_raster, hag_raster):
    assert

def test_convert_to_cover(raster, min_hag, desired_res):
    assert

def test_resample_raster(raster, desired_res):
    assert

def test_interpolate_hag(hag_raster, pim_raster):
    assert

def test_interpolate_pim(pim_raster):
    assert

def test_combine_hag_pim(hag_raster, pim_raster, threshold):
    assert

def test_create_gdf(combined_raster):
    assert

'''
