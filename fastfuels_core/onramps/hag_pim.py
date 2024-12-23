"""
This module contains functions for sampling tree imputation maps based on height above ground maps into a TreePopulation
"""

# Extrernal inports
import numpy as np
import xarray as xr
import rioxarray as rio  # noqa: F401
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.transform import from_origin


def sample_plots_from_hag(
    pim_raster: xr.DataArray,
    hag_raster: xr.DataArray,
    desired_res: float = 7.5,
    min_hag: float = 1,
    hag_threshold: float = 0.25,
):
    """
    Parameters
    -----------
    pim_raster : DataArray
        Plot imputation map raster likely TreeMap
    hag_raster : DataArray
        Height above ground raster likely canopy height map
    desired_res : float
        Desired resolution (in meters) of assimilated raster
    min_hag : float
        Cutoff value for lowest height above ground value to be considered
    hag_threshold : float
        Threshold for height above ground full percentage to
        sample from the plot imputation map

    Returns
    --------
    GeoDataFrame :
        DataFrame of tree plot points with locations informed by the height above ground raster
    """

    check_same_crs(pim_raster, hag_raster)
    check_projected_crs(pim_raster)
    check_resolution(pim_raster, hag_raster, desired_res)

    pim_resampled = resample_raster(pim_raster, desired_res, Resampling.nearest)

    hag_cover = compute_cover_from_hag(hag_raster, min_hag, desired_res)

    hag_interpolated = interpolate_hag(hag_cover, pim_resampled)

    pim_interpolated = interpolate_pim(pim_resampled)

    combined_raster = combine_hag_pim(hag_interpolated, pim_interpolated, hag_threshold)

    combined_gdf = create_plots_gdf_from_resampled_pim(combined_raster)

    return combined_gdf


def check_same_crs(pim_raster: xr.DataArray, hag_raster: xr.DataArray):
    """
    Checks that the plot imputation map and height above ground rasters have the same CRS

    Parameters
    ----------
    pim_raster : xr.DataArray
        The plot imputation map raster
    hag_raster : xr.DataArray
        The height above ground raster

    Raises
    ------
    ValueError
        If the two rasters do no have the same CRS
    """

    if pim_raster.rio.crs != hag_raster.rio.crs:
        raise ValueError(
            "The plot imputation map and height above ground raster do not have the same CRS."
        )


def check_projected_crs(pim_raster: xr.DataArray):
    """
    Checks that the rasters have a projected CRS

    Parameters
    ----------
    pim_raster : xr.DataArray
        The plot imputation map raster that has been checked to have the same CRS as the HAG raster

    Raises
    ------
    ValueError
        If the plot imputation raster does not have a projected CRS
    """
    if not pim_raster.rio.crs.is_projected:
        raise ValueError(
            "The plot imputation map and height above ground raster do not have a projected CRS."
        )


def check_resolution(
    pim_raster: xr.DataArray, hag_raster: xr.DataArray, desired_res: float
):
    """
    Checks that the height above ground raster has a
    finer resolution than the plot imputation map raster
    and that the desired resolution is finer than
    the resolution of the plot imputation map raster

    Parameters
    ----------
    pim_raster : xr.DataArray
        The plot imputation map raster
    hag_raster : xr.DataArray
        The height above ground raster
    desired_res : float
        The desired resolution

    Raises
    ------
    ValueError
        If the height above ground resolution is more
        coarse than the plot imputation map resolution
    ValueError
        If the desired resolution is more coarse than
        the plot imputation map resolution
    """
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


def compute_cover_from_hag(raster: xr.DataArray, min_hag: float, desired_res: float):
    """
    Creates a new raster from the height above ground raster that has
    the desired resolution where the values represent the
    percentage of previous cells with a value above min_hag

    Parameters
    ----------
    raster : xr.DataArray
        The height above ground raster
    min_hag : float
        Minimum height for a cell to be considered occupied on the height above ground raster
    desired_res : float
        The desired resolution

    Returns
    -------
    raster: xr.DataArray
        Height above ground cover raster
    """
    # Create cover raster from HAG
    hag_cover = xr.where((raster > min_hag) & (raster != raster.rio.nodata), 1.0, 0.0)
    hag_cover.rio.write_nodata(raster.rio.nodata, inplace=True)
    hag_cover.rio.write_crs(raster.rio.crs, inplace=True)

    # Resample with summing
    hag_resampled = resample_raster(hag_cover, desired_res, Resampling.sum)

    # Scale the values
    res = np.array(hag_cover.rio.resolution())
    res_scale = desired_res / abs(res)
    hag_resampled /= res_scale[0] * res_scale[1]

    return hag_resampled


def resample_raster(
    raster: xr.DataArray, desired_res: float, resampling_method: Resampling
):
    """
    Creates a new raster with the desired resolution
    using the resampling_method provided

    Parameters
    ----------
    raster : xr.DataArray
        A raster to be resampled
    desired_res : float
        The desired resolution of the new raster
    resampling_method : rasterio.edum.Resampling
        The method used for resampling

    Returns
    -------
    raster : xr.DataArray
        A resampled raster
    """
    # ensure correct edges
    res = np.array(raster.rio.resolution())
    x_min = raster["x"].min().item() - abs(res[0] / 2)
    y_max = raster["y"].max().item() + abs(res[1] / 2)
    new_transform = from_origin(x_min, y_max, desired_res, desired_res)
    # determine new shape
    res_scale = abs(res) / desired_res
    old_shape = np.array((raster.rio.height, raster.rio.width))
    new_shape = old_shape * res_scale
    new_shape = (int(new_shape[0]), int(new_shape[1]))
    # resample raster
    raster_resampled = raster.rio.reproject(
        raster.rio.crs,
        transform=new_transform,
        shape=new_shape,
        resampling=resampling_method,
    )
    raster_resampled = raster_resampled.fillna(raster.rio.nodata)
    raster_resampled.rio.write_nodata(raster.rio.nodata, inplace=True)
    return raster_resampled


def interpolate_hag(hag_raster: xr.DataArray, pim_raster: xr.DataArray):
    """
    Interpolates the height above ground raster to have the same
    coordinates as the plot imputation map raster. Values are determined
    using nearest neighbor from height above ground raster.

    Parameters
    ----------
    hag_raster : xr.DataArray
        The height above ground raster
    pim_raster : xr.DataArray
        The plot imputation map raster

    Returns
    -------
    raster: xr.DataArray
        The interpolated height above ground raster
    """
    hag_interp = hag_raster.rio.reproject_match(pim_raster)
    hag_interp.rio.write_crs(hag_raster.rio.crs, inplace=True)
    hag_interp.rio.write_nodata(hag_raster.rio.nodata, inplace=True)
    return hag_interp


def interpolate_pim(pim_raster: xr.DataArray):
    """
    Fills all nodata values with the nearest value

    Parameters
    ----------
    pim_raster : xr.DataArray
        The plot imputation map raster

    Returns
    -------
    raster: xr.DataArray
        The interpolated plot imputation map raster
    """
    pim_interp = pim_raster.rio.interpolate_na(method="nearest")
    pim_interp.rio.write_crs(pim_raster.rio.crs, inplace=True)
    pim_interp.rio.write_nodata(pim_raster.rio.nodata, inplace=True)
    return pim_interp


def combine_hag_pim(
    hag_raster: xr.DataArray, pim_raster: xr.DataArray, threshold: float
):
    """
    Creates a raster with the values from the interpolated
    plot imputation map raster only where the values from the
    height above ground cover raster is greater than the
    threshold value.

    Parameters
    ----------
    hag_raster : xr.DataArray
        The height above ground raster
    pim_raster : xr.DataArray
        The plot imputation map raster
    threshold : float
        A value between 0 and 1 that indicates the minimum
        percentage full required to sample from the plot
        imputation map.

    Returns
    -------
    _raster: xr.DataArray
        The plot imputation map raster informed by the height above ground raster
    """
    combined_raster = xr.where(
        hag_raster > threshold, pim_raster, pim_raster.rio.nodata
    )
    combined_raster.rio.write_crs(pim_raster.rio.crs, inplace=True)
    combined_raster.rio.write_nodata(pim_raster.rio.nodata, inplace=True)
    return combined_raster


def create_plots_gdf_from_resampled_pim(
    raster: xr.DataArray, plot_column_name: str = "PLOT_ID"
):
    """
    Creates a GeoDataFrame from the height above
    ground informed plot imputation map raster

    Parameters
    ----------
    raster : xr.DataArray
        The height above ground informed plot imputation map raster
    plot_column_name : str, optional
        The name fo the column given to the values from
        the raster, by default "PLOT_ID"

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame created by the height above
        ground informed plot imputation map raster
    """
    raster.name = plot_column_name
    df = raster.to_dataframe().reset_index()
    df.drop(columns=["spatial_ref"], inplace=True)
    df.rename(columns={"x": "X", "y": "Y"}, inplace=True)
    combined_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs=raster.rio.crs
    )
    return combined_gdf
