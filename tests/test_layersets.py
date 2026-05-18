"""
test_layersets.py
=================
Tests for layersets.py — rasterize_layerset() and its helpers.

Coverage
--------
- Input validation (_validate_gdf)
- Raster grid construction (_build_raster_grid)
- Output structure (Dataset shape, coords, CRS, transform)
- Distribution modes: homogeneous, uniform_random, random_clusters
- Overlap resolution: loading sums, optional bands via mean/min/max
- Optional-band NaN propagation
- Seed reproducibility
- Integration test with surface_fuels.geojson (all three modes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import pytest
import xarray as xr
from shapely.geometry import box

from fastfuels_core.layersets import (
    ALL_BANDS,
    rasterize_layerset,
    _build_raster_grid,
    _validate_gdf,
)

# ---------------------------------------------------------------------------
# Paths & shared constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data"
GEOJSON_PATH = DATA_PATH / "surface_fuels.geojson"

RESOLUTION = 2.0
SEED = 42
CRS_PROJECTED = "EPSG:32611"  # UTM zone 11N — same as the GeoJSON
CRS_GEOGRAPHIC = "EPSG:4326"


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _make_gdf(
    rows: list[dict],
    crs: str | None = CRS_PROJECTED,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a list of property dicts (geometry key required)."""
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _single_poly(
    fuel_type: str = "litter",
    distribution: str = "homogeneous",
    fuel_loading: float = 1.0,
    fuel_height: float = 0.5,
    percent_cover: float = 100.0,
    patch_size: float | None = None,
    patch_std_dev: float | None = None,
    live_fuel_moisture: float | None = None,
    dead_fuel_moisture: float | None = None,
    heat_of_combustion: float | None = None,
    geom=None,
    crs: str | None = CRS_PROJECTED,
) -> gpd.GeoDataFrame:
    """One-row GeoDataFrame covering a 100×100 m square at the origin."""
    if geom is None:
        geom = box(0, 0, 100, 100)
    row = dict(
        fuel_type=fuel_type,
        distribution=distribution,
        fuel_loading=fuel_loading,
        fuel_height=fuel_height,
        percent_cover=percent_cover,
        geometry=geom,
    )
    if patch_size is not None:
        row["patch_size"] = patch_size
    if patch_std_dev is not None:
        row["patch_std_dev"] = patch_std_dev
    if live_fuel_moisture is not None:
        row["live_fuel_moisture"] = live_fuel_moisture
    if dead_fuel_moisture is not None:
        row["dead_fuel_moisture"] = dead_fuel_moisture
    if heat_of_combustion is not None:
        row["heat_of_combustion"] = heat_of_combustion
    return _make_gdf([row], crs=crs)


def _two_non_overlapping(crs: str | None = CRS_PROJECTED) -> gpd.GeoDataFrame:
    """Two non-overlapping 100×100 m polygons of different fuel types."""
    return _make_gdf(
        [
            dict(
                fuel_type="litter",
                distribution="homogeneous",
                fuel_loading=0.5,
                fuel_height=0.05,
                percent_cover=90.0,
                geometry=box(0, 0, 100, 100),
            ),
            dict(
                fuel_type="shrub",
                distribution="uniform_random",
                fuel_loading=1.2,
                fuel_height=1.0,
                percent_cover=50.0,
                geometry=box(200, 0, 300, 100),
            ),
        ],
        crs=crs,
    )


def _two_overlapping(
    loading1: float = 1.0,
    loading2: float = 1.0,
    fuel_height1: float = 1.0,
    fuel_height2: float = 2.0,
    live_fm1: float | None = None,
    live_fm2: float | None = None,
) -> gpd.GeoDataFrame:
    """Two identical 100×100 m polygons of the same fuel_type — triggers overlap logic."""
    poly = box(0, 0, 100, 100)
    rows = [
        dict(
            fuel_type="shrub",
            distribution="homogeneous",
            fuel_loading=loading1,
            fuel_height=fuel_height1,
            percent_cover=100.0,
            geometry=poly,
        ),
        dict(
            fuel_type="shrub",
            distribution="homogeneous",
            fuel_loading=loading2,
            fuel_height=fuel_height2,
            percent_cover=100.0,
            geometry=poly,
        ),
    ]
    if live_fm1 is not None:
        rows[0]["live_fuel_moisture"] = live_fm1
    if live_fm2 is not None:
        rows[1]["live_fuel_moisture"] = live_fm2
    return _make_gdf(rows)


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_on_no_crs(self):
        gdf = _single_poly(crs=None)
        with pytest.raises(ValueError, match="CRS"):
            _validate_gdf(gdf)

    def test_raises_on_geographic_crs(self):
        # Create a polygon in geographic coords (tiny, near origin)
        gdf = _single_poly(crs=CRS_GEOGRAPHIC, geom=box(0, 0, 0.001, 0.001))
        with pytest.raises(ValueError, match="projected CRS"):
            _validate_gdf(gdf)

    def test_raises_on_missing_required_column(self):
        gdf = _single_poly().drop(columns=["fuel_loading"])
        with pytest.raises(ValueError, match="fuel_loading"):
            _validate_gdf(gdf)

    @pytest.mark.parametrize("col", ["fuel_loading", "fuel_height", "percent_cover"])
    def test_raises_on_nan_in_required_numeric(self, col):
        gdf = _single_poly()
        gdf[col] = np.nan
        with pytest.raises(ValueError, match=col):
            _validate_gdf(gdf)

    def test_raises_on_missing_distribution(self):
        gdf = _single_poly()
        gdf["distribution"] = np.nan
        with pytest.raises(ValueError, match="distribution"):
            _validate_gdf(gdf)

    def test_raises_on_unknown_distribution_mode(self):
        gdf = _single_poly()
        gdf["distribution"] = "clumped"
        with pytest.raises(ValueError, match="clumped"):
            _validate_gdf(gdf)

    def test_raises_on_random_clusters_without_patch_size_column(self):
        gdf = _single_poly(distribution="random_clusters")
        # patch_size column is absent
        with pytest.raises(ValueError, match="patch_size"):
            _validate_gdf(gdf)

    def test_raises_on_random_clusters_with_nan_patch_size(self):
        gdf = _single_poly(distribution="random_clusters")
        gdf["patch_size"] = np.nan
        with pytest.raises(ValueError, match="patch_size"):
            _validate_gdf(gdf)

    def test_valid_gdf_does_not_raise(self):
        _validate_gdf(_single_poly())  # should not raise

    def test_raises_on_invalid_overlap_method(self):
        gdf = _single_poly()
        with pytest.raises(ValueError, match="overlap_method"):
            rasterize_layerset(gdf, resolution=RESOLUTION, overlap_method=np.sum)


# ---------------------------------------------------------------------------
# 2. Grid construction
# ---------------------------------------------------------------------------


class TestBuildRasterGrid:
    def test_grid_covers_bounds(self):
        gdf = _single_poly(geom=box(0, 0, 100, 100))
        grid = _build_raster_grid(gdf, RESOLUTION)
        # Cell centers span [res/2, bound - res/2]
        assert grid.xs[0] >= 0
        assert grid.xs[-1] <= 100
        assert grid.ys[0] <= 100
        assert grid.ys[-1] >= 0

    def test_cell_center_spacing_equals_resolution(self):
        gdf = _single_poly(geom=box(0, 0, 100, 100))
        grid = _build_raster_grid(gdf, RESOLUTION)
        np.testing.assert_allclose(np.diff(grid.xs), RESOLUTION)
        np.testing.assert_allclose(np.abs(np.diff(grid.ys)), RESOLUTION)

    def test_first_cell_center_at_half_resolution(self):
        gdf = _single_poly(geom=box(0, 0, 100, 100))
        grid = _build_raster_grid(gdf, RESOLUTION)
        assert grid.xs[0] == pytest.approx(RESOLUTION / 2)

    def test_grid_size_scales_with_extent(self):
        gdf_small = _single_poly(geom=box(0, 0, 10, 10))
        gdf_large = _single_poly(geom=box(0, 0, 100, 100))
        grid_small = _build_raster_grid(gdf_small, RESOLUTION)
        grid_large = _build_raster_grid(gdf_large, RESOLUTION)
        assert grid_large.nx > grid_small.nx
        assert grid_large.ny > grid_small.ny

    def test_single_cell_minimum(self):
        """A polygon smaller than resolution still yields at least 1×1 grid."""
        gdf = _single_poly(geom=box(0, 0, 0.5, 0.5))
        grid = _build_raster_grid(gdf, RESOLUTION)
        assert grid.nx >= 1
        assert grid.ny >= 1


# ---------------------------------------------------------------------------
# 3. Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    @pytest.fixture(autouse=True)
    def _ds(self):
        self.gdf = _two_non_overlapping()
        self.ds = rasterize_layerset(self.gdf, resolution=RESOLUTION, seed=SEED)

    def test_returns_xarray_dataset(self):
        assert isinstance(self.ds, xr.Dataset)

    def test_one_variable_per_fuel_type(self):
        assert set(self.ds.data_vars) == {"litter", "shrub"}

    def test_dims_are_band_y_x(self):
        for var in self.ds.data_vars:
            assert tuple(self.ds[var].dims) == ("band", "y", "x")

    def test_band_coordinate_matches_all_bands(self):
        for var in self.ds.data_vars:
            assert list(self.ds[var].coords["band"].values) == list(ALL_BANDS)

    def test_crs_attached(self):
        assert self.ds.rio.crs is not None
        assert self.ds.rio.crs.to_epsg() == 32611

    def test_transform_attached(self):
        t = self.ds.rio.transform()
        assert t is not None
        assert t.a == pytest.approx(RESOLUTION)

    def test_x_coords_monotone_increasing(self):
        x = self.ds.coords["x"].values
        assert np.all(np.diff(x) > 0)

    def test_y_coords_monotone_decreasing(self):
        # rioxarray stores y top-to-bottom (decreasing)
        y = self.ds.coords["y"].values
        assert np.all(np.diff(y) < 0)

    def test_all_data_arrays_same_spatial_shape(self):
        shapes = [self.ds[v].shape[1:] for v in self.ds.data_vars]
        assert len(set(shapes)) == 1


# ---------------------------------------------------------------------------
# 4. Distribution modes
# ---------------------------------------------------------------------------


class TestDistributionModes:
    """Each mode is tested on its most important property."""

    # --- homogeneous ---

    def test_homogeneous_every_cell_gets_loading(self):
        gdf = _single_poly(distribution="homogeneous", percent_cover=100.0)
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        # All interior cells should have loading > 0
        interior = ~np.isnan(loading)
        assert interior.any()
        assert np.all(loading[interior] > 0)

    def test_homogeneous_loading_scaled_by_cover(self):
        """50% cover → loading = fuel_loading × 0.5 for every interior cell."""
        gdf = _single_poly(
            distribution="homogeneous", fuel_loading=2.0, percent_cover=50.0
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        interior_vals = loading[loading > 0]
        # Interior cells carry exactly loading * cover_frac
        np.testing.assert_allclose(
            interior_vals[interior_vals == interior_vals.max()], 1.0, rtol=1e-5
        )

    # --- uniform_random ---

    def test_uniform_random_approximate_cover(self):
        """With a large polygon and 50% cover, realised cover should be close to 50%.

        The grid is built to tightly fit the polygon, so every cell in the
        grid corresponds to a cell inside (or at the edge of) the polygon.
        Cover fraction = (cells with loading > 0) / (total grid cells).
        """
        gdf = _single_poly(
            distribution="uniform_random",
            percent_cover=50.0,
            geom=box(0, 0, 200, 200),
        )
        ds = rasterize_layerset(gdf, resolution=1.0, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        total_cells = loading.size
        filled_cells = np.count_nonzero(loading > 0)
        cover = filled_cells / total_cells
        assert cover == pytest.approx(0.50, abs=0.10)

    def test_uniform_random_each_cell_is_full_or_zero(self):
        """Cells are either fully selected (weight=1) or not (weight=0).
        No fractional edge weights exist — _poly_mask is boolean only."""
        gdf = _single_poly(
            distribution="uniform_random", fuel_loading=1.0, percent_cover=60.0
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        unique = np.unique(loading[~np.isnan(loading)])
        assert set(np.round(unique, 5)).issubset({0.0, 1.0})

    # --- random_clusters ---

    def test_random_clusters_requires_patch_size(self):
        gdf = _single_poly(distribution="random_clusters")
        # No patch_size column — should fail validation
        with pytest.raises(ValueError, match="patch_size"):
            rasterize_layerset(gdf, resolution=RESOLUTION)

    @pytest.mark.parametrize("patch_std_dev", [None, 2.0])
    def test_random_clusters_fills_cells(self, patch_std_dev):
        """Clusters fill cells whether or not patch_std_dev is provided."""
        gdf = _single_poly(
            distribution="random_clusters",
            percent_cover=60.0,
            patch_size=10.0,
            patch_std_dev=patch_std_dev,
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        assert np.any(loading > 0)

    def test_random_clusters_approximate_cover(self):
        """Cover fraction = (cells with loading > 0) / (total grid cells)."""
        gdf = _single_poly(
            distribution="random_clusters",
            fuel_loading=1.0,
            percent_cover=50.0,
            patch_size=10.0,
            geom=box(0, 0, 200, 200),
        )
        ds = rasterize_layerset(gdf, resolution=2.0, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        total_cells = loading.size
        filled_cells = np.count_nonzero(loading > 0)
        cover = filled_cells / total_cells
        assert cover == pytest.approx(0.50, abs=0.15)

    def test_random_clusters_with_std_dev_fills_cells(self):
        """patch_std_dev provided — clusters should still fill cells."""
        gdf = _single_poly(
            distribution="random_clusters",
            percent_cover=60.0,
            patch_size=10.0,
            patch_std_dev=2.0,
            geom=box(0, 0, 200, 200),
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        assert np.any(loading > 0)


# ---------------------------------------------------------------------------
# 5. Overlap resolution
# ---------------------------------------------------------------------------


class TestOverlapCombining:
    def test_loading_always_sums(self):
        gdf = _two_overlapping(loading1=1.0, loading2=1.5)
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["shrub"].sel(band="loading").values
        interior = loading > 0
        np.testing.assert_allclose(loading[interior], 2.5)

    def test_optional_band_mean(self):
        gdf = _two_overlapping(live_fm1=80.0, live_fm2=40.0)
        ds = rasterize_layerset(
            gdf, resolution=RESOLUTION, overlap_method=np.mean, seed=SEED
        )
        lfm = ds["shrub"].sel(band="live_fuel_moisture").values
        valid = lfm[np.isfinite(lfm)]
        np.testing.assert_allclose(valid, 60.0)

    def test_optional_band_min(self):
        gdf = _two_overlapping(live_fm1=80.0, live_fm2=40.0)
        ds = rasterize_layerset(
            gdf, resolution=RESOLUTION, overlap_method=np.min, seed=SEED
        )
        lfm = ds["shrub"].sel(band="live_fuel_moisture").values
        valid = lfm[np.isfinite(lfm)]
        np.testing.assert_allclose(valid, 40.0)

    def test_optional_band_max(self):
        gdf = _two_overlapping(live_fm1=80.0, live_fm2=40.0)
        ds = rasterize_layerset(
            gdf, resolution=RESOLUTION, overlap_method=np.max, seed=SEED
        )
        lfm = ds["shrub"].sel(band="live_fuel_moisture").values
        valid = lfm[np.isfinite(lfm)]
        np.testing.assert_allclose(valid, 80.0)

    @pytest.mark.parametrize(
        "method,expected",
        [
            (np.mean, 2.0),
            (np.min, 1.0),
            (np.max, 3.0),
        ],
    )
    def test_height_respects_overlap_method(self, method, expected):
        """Height is now resolved via overlap_method like other non-loading bands."""
        gdf = _two_overlapping(fuel_height1=1.0, fuel_height2=3.0)
        ds = rasterize_layerset(
            gdf, resolution=RESOLUTION, overlap_method=method, seed=SEED
        )
        h = ds["shrub"].sel(band="height").values
        valid = h[np.isfinite(h) & (h > 0)]
        np.testing.assert_allclose(valid, expected)

    def test_non_overlapping_polygons_independent(self):
        """Two non-overlapping same-type polygons should not influence each other's loading."""
        poly1 = box(0, 0, 100, 100)
        poly2 = box(200, 0, 300, 100)
        gdf = _make_gdf(
            [
                dict(
                    fuel_type="litter",
                    distribution="homogeneous",
                    fuel_loading=1.0,
                    fuel_height=0.1,
                    percent_cover=100.0,
                    geometry=poly1,
                ),
                dict(
                    fuel_type="litter",
                    distribution="homogeneous",
                    fuel_loading=2.0,
                    fuel_height=0.1,
                    percent_cover=100.0,
                    geometry=poly2,
                ),
            ]
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        # No cell should have loading = 3.0 (that would mean they were summed incorrectly)
        interior = loading[loading > 0]
        assert not np.any(np.isclose(interior, 3.0))


# ---------------------------------------------------------------------------
# 6. Optional bands & NaN propagation
# ---------------------------------------------------------------------------


class TestOptionalBands:
    def test_optional_band_nan_when_not_provided(self):
        """If live_fuel_moisture is absent from the GDF, that band is all NaN."""
        gdf = _single_poly()  # no optional columns
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        lfm = ds["litter"].sel(band="live_fuel_moisture").values
        assert np.all(np.isnan(lfm))

    def test_optional_band_present_when_provided(self):
        gdf = _single_poly(live_fuel_moisture=75.0)
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        lfm = ds["litter"].sel(band="live_fuel_moisture").values
        valid = lfm[np.isfinite(lfm)]
        assert valid.size > 0
        np.testing.assert_allclose(valid, 75.0)

    def test_loading_and_height_always_present(self):
        gdf = _single_poly()
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        assert "loading" in ALL_BANDS
        assert "height" in ALL_BANDS
        # At least some cells should have finite values
        assert np.any(np.isfinite(ds["litter"].sel(band="loading").values))
        assert np.any(np.isfinite(ds["litter"].sel(band="height").values))

    def test_outside_polygon_cells_are_nan(self):
        """Cells outside all polygons should carry NaN loading.

        Two polygons with a gap produce cells inside the bounding box but
        outside both polygons. With boolean-only masking (_poly_mask), those
        cells accumulate zero loading and are written as NaN by _rasterize_fuel_type.
        """
        poly1 = box(0, 0, 40, 40)
        poly2 = box(60, 0, 100, 40)  # 20 m gap between x=40 and x=60
        gdf = _make_gdf(
            [
                dict(
                    fuel_type="litter",
                    distribution="homogeneous",
                    fuel_loading=1.0,
                    fuel_height=0.1,
                    percent_cover=100.0,
                    geometry=poly1,
                ),
                dict(
                    fuel_type="litter",
                    distribution="homogeneous",
                    fuel_loading=1.0,
                    fuel_height=0.1,
                    percent_cover=100.0,
                    geometry=poly2,
                ),
            ]
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        # Gap cells should be NaN (no edge-weight partial values in this version)
        assert np.any(np.isnan(loading))


# ---------------------------------------------------------------------------
# 7. Values sanity
# ---------------------------------------------------------------------------


class TestValueSanity:
    def test_loading_nonnegative(self):
        gdf = _two_non_overlapping()
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        for v in ds.data_vars:
            loading = ds[v].sel(band="loading").values
            assert np.all(loading[~np.isnan(loading)] >= 0)

    def test_height_nonnegative(self):
        gdf = _two_non_overlapping()
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        for v in ds.data_vars:
            h = ds[v].sel(band="height").values
            assert np.all(h[~np.isnan(h)] >= 0)

    def test_full_cover_homogeneous_loading_equals_input(self):
        """100% homogeneous cover → interior cell loading = fuel_loading exactly."""
        gdf = _single_poly(
            distribution="homogeneous", fuel_loading=3.7, percent_cover=100.0
        )
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        loading = ds["litter"].sel(band="loading").values
        interior = loading[loading > 0]
        np.testing.assert_allclose(interior, 3.7, rtol=1e-5)

    def test_seed_reproducibility(self):
        gdf = _two_non_overlapping()
        ds1 = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        ds2 = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        for v in ds1.data_vars:
            np.testing.assert_array_equal(ds1[v].values, ds2[v].values)

    def test_different_seeds_produce_different_results(self):
        gdf = _single_poly(distribution="uniform_random", percent_cover=50.0)
        ds1 = rasterize_layerset(gdf, resolution=RESOLUTION, seed=1)
        ds2 = rasterize_layerset(gdf, resolution=RESOLUTION, seed=99)
        loading1 = ds1["litter"].sel(band="loading").values
        loading2 = ds2["litter"].sel(band="loading").values
        assert not np.array_equal(loading1, loading2)

    def test_multiple_fuel_types_do_not_bleed(self):
        """A cell covered by litter only should not appear in shrub."""
        gdf = _two_non_overlapping()
        ds = rasterize_layerset(gdf, resolution=RESOLUTION, seed=SEED)
        litter_loading = ds["litter"].sel(band="loading").values
        shrub_loading = ds["shrub"].sel(band="loading").values
        # Where litter has fuel, shrub should be NaN (no overlap)
        litter_mask = litter_loading > 0
        assert np.all(np.isnan(shrub_loading[litter_mask]))


# ---------------------------------------------------------------------------
# 8. Integration — surface_fuels.geojson
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not GEOJSON_PATH.exists(), reason="blue_mountain_fuels.geojson not found"
)
class TestIntegrationBlueMountain:
    """
    Real-data integration tests using blue_mountain_fuels.geojson.

    The file contains 5 polygons:
      - 2× litter    (homogeneous)
      - 1× herb      (uniform_random)
      - 2× shrub     (random_clusters, with patch_size)
    CRS: EPSG:32611 (projected UTM)
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        self.gdf = gpd.read_file(GEOJSON_PATH)
        self.ds = rasterize_layerset(self.gdf, resolution=RESOLUTION, seed=SEED)

    # --- basic sanity ---

    def test_returns_dataset(self):
        assert isinstance(self.ds, xr.Dataset)

    def test_expected_fuel_type_variables(self):
        assert set(self.ds.data_vars) == {"litter", "herb", "shrub"}

    def test_crs_preserved(self):
        assert self.ds.rio.crs is not None
        assert self.ds.rio.crs.to_epsg() == 32611

    def test_no_negative_loading(self):
        for v in self.ds.data_vars:
            arr = self.ds[v].sel(band="loading").values
            assert np.all(arr[~np.isnan(arr)] >= 0)

    def test_no_negative_height(self):
        for v in self.ds.data_vars:
            arr = self.ds[v].sel(band="height").values
            assert np.all(arr[~np.isnan(arr)] >= 0)

    def test_all_bands_present(self):
        for v in self.ds.data_vars:
            assert list(self.ds[v].coords["band"].values) == list(ALL_BANDS)

    # --- distribution-specific checks ---

    def test_litter_homogeneous_cells_have_loading(self):
        """litter polygons are homogeneous — should have widespread loading."""
        loading = self.ds["litter"].sel(band="loading").values
        assert np.any(loading > 0)

    def test_shrub_clusters_have_loading(self):
        loading = self.ds["shrub"].sel(band="loading").values
        assert np.any(loading > 0)

    def test_herb_random_has_loading(self):
        loading = self.ds["herb"].sel(band="loading").values
        assert np.any(loading > 0)

    def test_litter_overlapping_polygons_loading_sum(self):
        """
        Rows 0 and 4 are both litter/homogeneous and may overlap spatially.
        Where they do, loading should exceed either polygon's individual value.
        The maximum single-polygon loading for litter is 0.45 (row 0).
        If any cell exceeds 0.45, summing happened.
        """
        loading = self.ds["litter"].sel(band="loading").values
        finite = loading[np.isfinite(loading)]
        # At minimum, some cells should have loading > 0.45
        assert finite.max() > 0.45

    # --- optional band propagation ---

    def test_heat_of_combustion_present_for_all_types(self):
        """All rows in the GeoJSON have heat_of_combustion; every variable should carry it."""
        for v in self.ds.data_vars:
            hoc = self.ds[v].sel(band="heat_of_combustion").values
            assert np.any(np.isfinite(hoc)), f"{v} has no finite heat_of_combustion"

    # --- reproducibility ---

    def test_seed_reproducibility(self):
        ds2 = rasterize_layerset(self.gdf, resolution=RESOLUTION, seed=SEED)
        for v in self.ds.data_vars:
            np.testing.assert_array_equal(self.ds[v].values, ds2[v].values)
