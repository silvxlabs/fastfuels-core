import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
from scipy.ndimage import maximum_filter, label, maximum_position
import rasterio as rio


def variable_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    crown_ratio: float = 0.10,
    crown_offset: float = 1.0,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Variable Window Filter (VWF).

    Calculates the search window size dynamically using a linear allometric
    relationship: Crown_Width_m = (Height_m * crown_ratio) + crown_offset.

    Algorithm Validation & Scientific Context:
    - Popescu & Wynne (2004): Validated dynamic window sizing based on allometry.
      https://doi.org/10.14358/PERS.70.5.589
    - Chen et al. (2006): Validated VWF applied specifically to continuous CHMs.
      https://doi.org/10.14358/PERS.72.8.923

    Args:
        chm_da (xr.DataArray): The Canopy Height Model data array.
        min_height (float): Minimum height threshold in CHM units (meters).
        spatial_resolution (float): Pixel size of the CHM in meters (e.g., 0.5, 1.0).
        crown_ratio (float): The multiplier for tree height to estimate crown width.
            Defaults to 0.10 (10%).
        crown_offset (float): The base crown width in meters. Defaults to 1.0m.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    chm = chm_da.values
    transform = chm_da.rio.transform()

    crown_width_meters = (chm * crown_ratio) + crown_offset
    required_windows = (crown_width_meters / spatial_resolution).astype(int)
    required_windows = np.where(
        required_windows % 2 == 0, required_windows + 1, required_windows
    )

    vw_max = np.zeros_like(chm)
    unique_windows = np.unique(required_windows)

    for w in unique_windows:
        if w <= 1:
            mask = required_windows == w
            vw_max[mask] = chm[mask]
            continue

        y, x = np.ogrid[-w // 2 : w // 2 + 1, -w // 2 : w // 2 + 1]
        footprint = x * x + y * y <= (w // 2) ** 2
        chm_max_filtered = maximum_filter(chm, footprint=footprint)
        mask = required_windows == w
        vw_max[mask] = chm_max_filtered[mask]

    local_maxima_mask = (chm == vw_max) & (chm > min_height)
    labeled_maxima, num_labels = label(local_maxima_mask)

    if num_labels == 0:
        return dd.from_pandas(pd.DataFrame(columns=["x", "y", "height"]), npartitions=1)

    # Use maximum_position
    # This guarantees the selected coordinate physically exists ON the canopy plateau,
    # avoiding the negative-space trap of C-shaped or L-shaped bounding boxes/centers of mass.
    indices = np.arange(1, num_labels + 1)
    positions = maximum_position(chm, labels=labeled_maxima, index=indices)

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]

    heights = chm[rows, cols]
    xs, ys = rio.transform.xy(transform, rows, cols)

    return dd.from_pandas(
        pd.DataFrame({"x": xs, "y": ys, "height": heights}), npartitions=1
    )


def fixed_window_filter(
    chm_da: xr.DataArray,
    min_height: float,
    spatial_resolution: float,
    window_size_meters: float = 3.0,
) -> dd.DataFrame:
    """Finds treetops from a CHM using a Fixed Window Local Maxima (FW-LM) filter.

    Applies a static, circular search window across the entire Canopy Height Model
    to identify local maxima.

    Algorithm Validation & Scientific Context:
    - Wulder et al. (2000): The foundational paper validating the use of fixed-size
      optical/spatial windows for extracting tree locations from high-resolution data.
      https://doi.org/10.1016/S0034-4257(00)00103-6
    - Chen et al. (2006): Used this exact fixed-window methodology as the baseline
      to compare against Variable Window Filters, demonstrating that fixed windows
      are prone to high omission errors in mixed stands.
      https://doi.org/10.14358/PERS.72.8.923

    Args:
        chm_da (xr.DataArray): The Canopy Height Model data array.
        min_height (float): Minimum height threshold in CHM units (meters).
        spatial_resolution (float): Pixel size of the CHM in meters (e.g., 0.5, 1.0).
        window_size_meters (float): The fixed diameter of the search window in meters.
            Defaults to 3.0m.

    Returns:
        dd.DataFrame: Detected treetops with explicit 'x', 'y', and 'height' columns.
    """
    # Extract raw numpy array and transform directly from xarray
    chm = chm_da.values
    transform = chm_da.rio.transform()

    # Convert the real-world window size into a pixel count based on CHM resolution
    window_size_pixels = int(window_size_meters / spatial_resolution)

    # Force window size to be an odd integer (required for center-pixel alignment)
    if window_size_pixels % 2 == 0:
        window_size_pixels += 1

    # Ensure the window is at least 3x3 pixels to perform a valid neighborhood search
    if window_size_pixels < 3:
        window_size_pixels = 3

    # Generate a single circular footprint for the filter
    y, x = np.ogrid[
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
        -window_size_pixels // 2 : window_size_pixels // 2 + 1,
    ]
    footprint = x * x + y * y <= (window_size_pixels // 2) ** 2

    # Apply the maximum filter across the entire array at once
    chm_max_filtered = maximum_filter(chm, footprint=footprint)

    # Identify peaks (pixels that are the highest in their fixed window AND above min_height)
    local_maxima_mask = (chm == chm_max_filtered) & (chm > min_height)

    # Find connected components of local maxima
    labeled_maxima, num_labels = label(local_maxima_mask)

    if num_labels == 0:
        # Return empty Dask DataFrame with the correct schema
        empty_df = pd.DataFrame(columns=["x", "y", "height"])
        return dd.from_pandas(empty_df, npartitions=1)

    # Use maximum_position
    # This guarantees the selected coordinate physically exists ON the canopy plateau,
    # avoiding the negative-space trap of C-shaped or L-shaped bounding boxes/centers of mass.
    indices = np.arange(1, num_labels + 1)
    positions = maximum_position(chm, labels=labeled_maxima, index=indices)

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]

    # Array indexing and coordinate transformation
    heights = chm[rows, cols]
    xs, ys = rio.transform.xy(transform, rows, cols)

    # Create Dask DataFrame directly from native types
    pdf = pd.DataFrame({"x": xs, "y": ys, "height": heights})

    return dd.from_pandas(pdf, npartitions=1)
