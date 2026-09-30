"""Assign each grid cell its LANDFIRE map zone."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from shapely.geometry import box


@lru_cache(maxsize=1)
def _lf_map_zones() -> gpd.GeoDataFrame:
    """LANDFIRE CONUS map zone polygons (EPSG:5070).

    Read from disk on first call and cached.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per map zone, with the zone number in ``ZONE_NUM``.
    """
    return gpd.read_parquet(files("fastfuels_core.data") / "LF_MAP_ZONES.parquet")


def lookup_lf_zones(
    transform: Affine,
    shape: tuple[int, int],
    crs,
    nodata: int = -9999,
) -> np.ndarray:
    """LANDFIRE map zone number for every cell of a grid.

    Parameters
    ----------
    transform : affine.Affine
        The grid's transform, e.g. ``da.rio.transform()``.
    shape : tuple of int
        Grid shape, ``(ny, nx)``.
    crs
        The grid's CRS, e.g. ``da.rio.crs``.
    nodata : int, optional
        Value for cells outside every map zone. Default -9999.

    Returns
    -------
    numpy.ndarray
        int32 zone numbers, shape ``shape``.

    Notes
    -----
    Most grids fall inside a single zone, which is filled directly. Only
    grids that cross a zone boundary or the CONUS edge are rasterized, and
    then only with the zones they touch. A rasterized cell gets the zone
    containing its center.

    Only the grid's outline is reprojected to find the zones it touches.
    Its edges can bend slightly after reprojection, but over typical grid
    sizes the error is well under one 30 m cell.

    Examples
    --------
    A 90 m x 90 m grid near Missoula, MT, which lies in zone 10:

    >>> from pyproj import Transformer
    >>> from rasterio.transform import from_origin
    >>> x, y = Transformer.from_crs(
    ...     "EPSG:4326", "EPSG:32611", always_xy=True
    ... ).transform(-114.099, 46.829)
    >>> zones = lookup_lf_zones(from_origin(x, y, 30, 30), (3, 3), "EPSG:32611")
    >>> zones.shape
    (3, 3)
    >>> np.unique(zones).tolist()
    [10]
    """
    zones = _lf_map_zones()
    outline = (
        gpd.GeoSeries([box(*array_bounds(*shape, transform))], crs=crs)
        .to_crs(zones.crs)
        .iloc[0]
    )
    touching = zones.iloc[zones.sindex.query(outline, predicate="intersects")]

    if touching.empty:
        return np.full(shape, nodata, dtype=np.int32)
    if len(touching) == 1 and touching.geometry.iloc[0].contains(outline):
        return np.full(shape, touching["ZONE_NUM"].iloc[0], dtype=np.int32)

    touching = touching.to_crs(crs)
    return rasterize(
        zip(touching.geometry, touching["ZONE_NUM"]),
        out_shape=shape,
        transform=transform,
        fill=nodata,
        dtype="int32",
    )
