"""Tests for :mod:`fastfuels_core.fuel_models.lf_zone_lookup`.

Most tests replace the packaged map zones with two adjacent squares in
EPSG:5070, so every expected zone is known exactly:

    zone 1: x 0-1000, y 0-1000
    zone 2: x 1000-2000, y 0-1000

That covers the three paths through :func:`lookup_lf_zones` -- one zone
containing the whole grid (filled without rasterizing), a grid crossing a
zone boundary or the edge of every zone (rasterized), and a grid outside
every zone -- plus the cell-center rule and grids in another CRS. The
last class checks the real packaged file.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box

import fastfuels_core.fuel_models.lf_zone_lookup as lf_zone_lookup
from fastfuels_core.fuel_models.lf_zone_lookup import (
    _lf_map_zones,  # noqa
    lookup_lf_zones,
)

ALBERS = "EPSG:5070"
UTM_14N = "EPSG:32614"
POLYGON_PATH = Path(__file__).parent.parent / "data" / "polygon.geojson"

TWO_ZONES = gpd.GeoDataFrame(
    {"ZONE_NUM": [1, 2]},
    geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
    crs=ALBERS,
)


@pytest.fixture
def two_zones(monkeypatch):
    """Use the two square zones instead of the packaged file."""
    monkeypatch.setattr(lf_zone_lookup, "_lf_map_zones", lambda: TWO_ZONES)


def _grid_in_utm(center_x, center_y, cell, shape):
    """A UTM 14N grid centered on an EPSG:5070 point."""
    x, y = Transformer.from_crs(ALBERS, UTM_14N, always_xy=True).transform(
        center_x, center_y
    )
    ny, nx = shape
    return from_origin(x - nx * cell / 2, y + ny * cell / 2, cell, cell)


@pytest.mark.usefixtures("two_zones")
class TestLookupZones:
    def test_grid_inside_one_zone_is_filled_without_rasterizing(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("single-zone grid should not be rasterized")

        monkeypatch.setattr(lf_zone_lookup, "rasterize", fail)
        zones = lookup_lf_zones(from_origin(100, 900, 100, 100), (3, 3), ALBERS)
        np.testing.assert_array_equal(zones, np.full((3, 3), 1))
        assert zones.dtype == np.int32

    def test_grid_crossing_a_boundary(self):
        # Columns are centered at x = 850, 950 | 1050, 1150.
        zones = lookup_lf_zones(from_origin(800, 900, 100, 100), (2, 4), ALBERS)
        np.testing.assert_array_equal(zones, [[1, 1, 2, 2], [1, 1, 2, 2]])
        assert zones.dtype == np.int32

    def test_cell_takes_the_zone_at_its_center(self):
        # The first cell spans x 940-1040: 40% of it is in zone 2, but its
        # center (990) is in zone 1.
        zones = lookup_lf_zones(from_origin(940, 900, 100, 100), (1, 2), ALBERS)
        np.testing.assert_array_equal(zones, [[1, 2]])

    def test_grid_past_the_edge_of_every_zone(self):
        # Only zone 2 touches this grid, but it doesn't contain it, so the
        # cells past x = 2000 must still be nodata.
        zones = lookup_lf_zones(from_origin(1900, 500, 100, 100), (1, 3), ALBERS)
        np.testing.assert_array_equal(zones, [[2, -9999, -9999]])

    def test_grid_outside_every_zone(self):
        zones = lookup_lf_zones(from_origin(5000, 5000, 100, 100), (2, 2), ALBERS)
        np.testing.assert_array_equal(zones, np.full((2, 2), -9999))
        assert zones.dtype == np.int32

    def test_custom_nodata(self):
        zones = lookup_lf_zones(
            from_origin(1900, 500, 100, 100), (1, 3), ALBERS, nodata=0
        )
        np.testing.assert_array_equal(zones, [[2, 0, 0]])

    @pytest.mark.parametrize("center_x, expected", [(500, 1), (1500, 2)])
    def test_grid_in_another_crs_inside_one_zone(self, center_x, expected):
        transform = _grid_in_utm(center_x, 500, cell=30, shape=(4, 4))
        zones = lookup_lf_zones(transform, (4, 4), UTM_14N)
        np.testing.assert_array_equal(zones, np.full((4, 4), expected))

    def test_grid_in_another_crs_crossing_a_boundary(self):
        # Two 100 m cells straddling the boundary at x = 1000; their centers
        # are 50 m either side of it.
        transform = _grid_in_utm(1000, 500, cell=100, shape=(1, 2))
        zones = lookup_lf_zones(transform, (1, 2), UTM_14N)
        np.testing.assert_array_equal(zones, [[1, 2]])


class TestPackagedMapZones:
    def test_loads_in_albers_with_zone_numbers(self):
        zones = _lf_map_zones()
        assert zones.crs.to_epsg() == 5070
        assert "ZONE_NUM" in zones.columns
        assert np.issubdtype(zones["ZONE_NUM"].dtype, np.integer)
        assert len(zones) > 0

    def test_loader_is_cached(self):
        assert _lf_map_zones() is _lf_map_zones()

    def test_known_location_is_zone_10(self):
        # tests/data/polygon.geojson (near Missoula, MT) lies wholly in zone 10.
        polygon = gpd.read_file(POLYGON_PATH)
        crs = polygon.estimate_utm_crs()
        west, south, east, north = polygon.to_crs(crs).total_bounds
        shape = (int(np.ceil((north - south) / 30)), int(np.ceil((east - west) / 30)))
        zones = lookup_lf_zones(from_origin(west, north, 30, 30), shape, crs)
        np.testing.assert_array_equal(zones, np.full(shape, 10))
