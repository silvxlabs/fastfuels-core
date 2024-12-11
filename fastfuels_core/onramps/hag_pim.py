"""
This module contains functions for sampling tree imputation maps based on height above ground maps into a TreePopulation
"""

#Internal imports

#Extrernal inports
import numpy as np
import xarray as xr
import rioxarray as rio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.transform import from_origin

def onramp(pim_raster, 
           hag_raster, 
           desired_res=7.5, 
           min_hag=1,
           hag_threshold = 0.25):
    """
    Parameters
    -----------
    pim_raster : DataArray
        plot imputation map raster likely TreeMap
    hag_raster : DataArray
        height above ground raster likely canopy height map
    desired_res : float
        desired resolution (in meters) of assimilated raster
    min_hag : float
        cutoff value for lowest height above ground value to be considered
    hag_threshold : float
        threshold for height above ground full percentage to sample plot imputation map

    Returns
    --------
    GeoDataFrame : df of tree plot points with locations informed by the height above ground raster
    """

    check_same_crs(pim_raster, hag_raster)
    check_projected_crs(pim_raster)
    check_resolution(pim_raster, hag_raster, desired_res)

    pim_resampled = resample_raster(pim_raster, desired_res)

    hag_cover = convert_to_cover(hag_raster, min_hag, desired_res)
    
    hag_resampled = resample_raster(hag_cover, desired_res)

    hag_interpolated = interpolate_hag(hag_resampled, pim_resampled)
    
    pim_interpolated = interpolate_pim(pim_resampled)

    combined_raster = combine_hag_pim(hag_interpolated, pim_interpolated, hag_threshold)

    combined_gdf = create_gdf(combined_raster) 

    return combined_gdf

def check_same_crs(pim_raster, hag_raster):
    if pim_raster.rio.crs != hag_raster.rio.crs:
        raise ValueError(
                "The plot imputation map and height above ground raster do not have the same CRS."
            )

def check_projected_crs(pim_raster):
    if not pim_raster.rio.crs.is_projected:
        raise ValueError(
                "The plot imputation map and height above ground raster do not have a projected CRS."
            )

def check_resolution(pim_raster, hag_raster, desired_res):
    pim_res = np.array(pim_raster.rio.resolution())
    hag_res = np.array(hag_raster.rio.resolution())
    
    if not np.all(abs(pim_res) >= abs(hag_res)):
        raise ValueError(
                f"The resolution of the plot imputation map, {pim_res} is finer than the resolution of the height above ground, {hag_res}."
            )
    if not abs(pim_res[0]) >= desired_res:
        raise ValueError(
                f"The resolution of the plot imputation map, {pim_res} is finer than the desired resolution, {desired_res}."
            )

def convert_to_cover(raster, min_value, desired_res):
    '''
    Returns: 
        raster with values where the hag value is more than the min_value
        - values = 1/(# of cells in one cell desired resolution raster)
    '''
    cover_raster = xr.where(raster > min_value, 1.0, 0.0) 
    cover_raster.rio.write_nodata(0, inplace=True)
    cover_raster.rio.write_crs(raster.rio.crs,
                               inplace=True)
    res = np.array(raster.rio.resolution())
    res_scale = desired_res/abs(res)
    cover_raster /= (res_scale[0]*res_scale[1])
    return cover_raster

def resample_raster(raster, desired_res):
    '''
    Returns: 
        raster with desired resolution 
        - if desired resolution is finer, the values 
        are determined by the nearest previous cell
        - if the desired resolution is more coarse, the 
        values are determined by summing previous cells 
    '''
    res = np.array(raster.rio.resolution())
    x_min = raster["x"].min().item() - abs(res[0]/2)
    y_max = raster["y"].max().item() + abs(res[1]/2)
    new_transform = from_origin(x_min, y_max, 
                                desired_res, desired_res)
    res_scale = abs(res)/desired_res
    old_shape = np.array((raster.rio.height,
                          raster.rio.width))
    new_shape = old_shape*res_scale
    new_shape = (int(new_shape[0]),
                 int(new_shape[1]))
    if res[0] < desired_res:
        resample = Resampling.sum
    else:
        resample = Resampling.nearest
    raster_resampled = raster.rio.reproject(
        raster.rio.crs,
        transform=new_transform,
        shape=new_shape,
        resampling=resample
    )
    raster_resampled = raster_resampled.fillna(0)
    raster_resampled.rio.write_nodata(0, inplace=True)
    return raster_resampled


def interpolate_hag(hag_raster, pim_raster):
    '''
    Returns: 
        hag raster with same coordinates as pim raster
        - values were determined using nearest neighbor from hag input
    '''
    hag_interp = hag_raster.rio.reproject_match(pim_raster)
    hag_interp.rio.write_crs(hag_raster.rio.crs, inplace=True)
    hag_interp.rio.write_nodata(hag_raster.rio.nodata, inplace=True)
    return hag_interp

def interpolate_pim(pim_raster):
    '''
    Returns: 
        plot imputation raster with all cells filled with data
        - values were determined using nearest neighbor
    '''
    pim_interp = pim_raster.rio.interpolate_na(
        method='nearest')
    pim_interp.rio.write_crs(pim_raster.rio.crs, inplace=True)
    pim_interp.rio.write_nodata(pim_raster.rio.nodata, inplace=True)
    return pim_interp

def combine_hag_pim(hag_raster, pim_raster, threshold):
    '''
    Returns: 
        plot imputation raster 
        - values are only included where the value of the 
        hag_raster is greater than the threshold value
    '''
    combined_raster = xr.where(
        hag_raster > threshold,   
        pim_raster,                     
        pim_raster.rio.nodata              
    )
    combined_raster.rio.write_crs(
        pim_raster.rio.crs, inplace=True)
    combined_raster.rio.write_nodata(
        pim_raster.rio.nodata, inplace=True)
    return combined_raster

def create_gdf(raster):
    '''
    Returns: 
        Geopandas dataframe from the values in the raster
    '''
    raster.name = "PLOT_ID"
    df = raster.to_dataframe().reset_index()
    df.drop(columns=['spatial_ref'], inplace=True)
    df.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)
    combined_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['X'], df['Y']),
        crs=raster.rio.crs
    )
    return combined_gdf



    