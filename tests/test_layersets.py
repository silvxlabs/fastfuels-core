# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.layersets import create_layerset, _resolve_strata_names

# External imports
import pytest
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from rasterio.transform import from_origin

TEST_PATH = Path(__file__).parent
LAYERSETS_PARQUET = Path(__file__).parent / "data" / "surface_fuels_ex.parquet"

RESOLUTION = 2.0
SEED = 42
CRS = "EPSG:5070"


def _load_gdf(path=LAYERSETS_PARQUET):
    return gpd.read_parquet(path)


def _make_simple_gdf(crs: str | None = CRS):
    """Two non-overlapping polygons, same strata."""
    poly1 = box(0, 0, 100, 100)
    poly2 = box(200, 0, 300, 100)
    return gpd.GeoDataFrame(
        [
            {
                "strata": "Shrub",
                "loading": 2.0,
                "height": 3.0,
                "spatial_pattern": "Uniform",
                "percent_cover": 50,
                "geometry": poly1,
            },
            {
                "strata": "Shrub",
                "loading": 1.0,
                "height": 1.0,
                "spatial_pattern": "Uniform",
                "percent_cover": 100,
                "geometry": poly2,
            },
        ],
        geometry="geometry",
        crs=crs,
    )


def _make_overlapping_gdf(height1=2.0, height2=4.0, crs=CRS):
    """Two identical polygons with different heights and 100% cover."""
    poly = box(0, 0, 100, 100)
    return gpd.GeoDataFrame(
        [
            {"strata": "Shrub", "loading": 1.0, "height": height1,
             "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly},
            {"strata": "Shrub", "loading": 1.0, "height": height2,
             "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly},
        ],
        geometry="geometry",
        crs=crs,
    )


# ---------------------------------------------------------------------------
# _resolve_strata_names
# ---------------------------------------------------------------------------

class TestResolveStrataNames:
    def test_no_trailing_digits(self):
        result = _resolve_strata_names(["Shrub", "Herb"], combine=True)
        assert result == {"Shrub": "Shrub", "Herb": "Herb"}

    def test_combine_true_merges_variants(self):
        result = _resolve_strata_names(["Shrub1", "Shrub2"], combine=True)
        assert result["Shrub1"] == "Shrub"
        assert result["Shrub2"] == "Shrub"

    def test_combine_false_primary_secondary(self):
        result = _resolve_strata_names(["Shrub1", "Shrub2"], combine=False)
        assert result["Shrub1"] == "Shrub_primary"
        assert result["Shrub2"] == "Shrub_secondary"

    def test_combine_false_number_gt_2_warns(self):
        with pytest.warns(UserWarning, match="trailing number > 2"):
            result = _resolve_strata_names(["Shrub3"], combine=False)
        assert result["Shrub3"] == "Shrub_3"

    def test_mixed_named_and_numbered(self):
        result = _resolve_strata_names(["Shrub1", "GroundFuels"], combine=True)
        assert result["Shrub1"] == "Shrub"
        assert result["GroundFuels"] == "GroundFuels"


# ---------------------------------------------------------------------------
# create_layerset — validation
# ---------------------------------------------------------------------------

class TestCreateLayersetValidation:
    def test_raises_on_geographic_crs(self):
        gdf = _make_simple_gdf(crs="EPSG:4326")
        with pytest.raises(ValueError, match="projected CRS"):
            create_layerset(gdf, RESOLUTION)

    def test_raises_on_missing_crs(self):
        gdf = _make_simple_gdf(crs=None)
        with pytest.raises(ValueError, match="CRS"):
            create_layerset(gdf, RESOLUTION)

    def test_raises_on_missing_column(self):
        gdf = _make_simple_gdf().drop(columns=["loading"])
        with pytest.raises(ValueError, match="loading"):
            create_layerset(gdf, RESOLUTION)

    def test_raises_when_all_filtered_out(self):
        gdf = _make_simple_gdf()
        gdf["loading"] = 0
        with pytest.raises(ValueError, match="No rows remain"):
            create_layerset(gdf, RESOLUTION)

    def test_raises_when_spatial_pattern_not_uniform(self):
        gdf = _make_simple_gdf()
        gdf["spatial_pattern"] = "Clumped"
        with pytest.raises(ValueError, match="No rows remain"):
            create_layerset(gdf, RESOLUTION)

    def test_raises_when_no_resolution_and_no_transform(self):
        gdf = _make_simple_gdf()
        with pytest.raises(ValueError, match="horizontal_resolution"):
            create_layerset(gdf)

    def test_raises_when_only_transform_given(self):
        gdf = _make_simple_gdf()
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        with pytest.raises(ValueError, match="together"):
            create_layerset(gdf, transform=tf)

    def test_raises_when_only_shape_given(self):
        gdf = _make_simple_gdf()
        with pytest.raises(ValueError, match="together"):
            create_layerset(gdf, shape=(50, 150))

    def test_warns_on_tiny_polygon_no_intersection(self):
        """
        Test that a polygon smaller than the resolution triggers a warning
        and results in an empty grid rather than a crash.
        """
        # Create a tiny box (0.1 x 0.1) that won't hit a 2.0 resolution cell center
        tiny_poly = box(0, 0, 0.1, 0.1)
        gdf = gpd.GeoDataFrame(
            [
                {
                    "strata": "Shrub",
                    "loading": 1.0,
                    "height": 1.0,
                    "spatial_pattern": "Uniform",
                    "percent_cover": 100,
                    "geometry": tiny_poly,
                }
            ],
            geometry="geometry",
            crs=CRS,
        )

        with pytest.warns(UserWarning, match="does not intersect any grid cells"):
            ds = create_layerset(gdf, RESOLUTION)

        # Verify the output is still a valid Dataset but contains only zeros
        assert (ds["loading"] == 0).all()
        assert (ds["height"] == 0).all()

# ---------------------------------------------------------------------------
# create_layerset — output structure
# ---------------------------------------------------------------------------

class TestCreateLayersetOutput:
    def setup_method(self):
        self.gdf = _make_simple_gdf()
        self.ds = create_layerset(self.gdf, RESOLUTION, seed=SEED)

    def test_returns_dataset(self):
        assert isinstance(self.ds, xr.Dataset)

    def test_has_loading_and_height_variables(self):
        assert "loading" in self.ds
        assert "height" in self.ds

    def test_dimensions(self):
        assert set(self.ds["loading"].dims) == {"strata", "y", "x"}
        assert set(self.ds["height"].dims) == {"strata", "y", "x"}

    def test_strata_coordinate(self):
        assert "strata" in self.ds.coords
        assert "Shrub" in self.ds.coords["strata"].values

    def test_crs_attached(self):
        assert self.ds.rio.crs is not None
        assert self.ds.rio.crs.to_epsg() == 5070

    def test_transform_attached(self):
        transform = self.ds.rio.transform()
        assert transform is not None
        assert transform.a == pytest.approx(RESOLUTION)

    def test_cell_size_is_exact(self):
        x = self.ds.coords["x"].values
        assert np.diff(x) == pytest.approx(RESOLUTION)

    def test_cell_centers_alignment(self):
        # If minx is 0 and res is 2.0, the first center should be 1.0
        gdf = _make_simple_gdf()  # bounds are 0, 0, 300, 100
        ds = create_layerset(gdf, horizontal_resolution=2.0)

        assert ds.coords["x"].values[0] == pytest.approx(1.0)
        assert ds.coords["y"].values[0] == pytest.approx(99.0)  # y is usually top-down


# ---------------------------------------------------------------------------
# create_layerset — external transform + shape
# ---------------------------------------------------------------------------

class TestCreateLayersetTransformShape:
    def test_grid_dimensions_match_shape(self):
        gdf = _make_simple_gdf()
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        n_rows, n_cols = 50, 150
        ds = create_layerset(gdf, transform=tf, shape=(n_rows, n_cols))
        assert ds["loading"].shape == (1, n_rows, n_cols)

    def test_coords_derived_from_transform(self):
        gdf = _make_simple_gdf()
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        n_rows, n_cols = 50, 150
        ds = create_layerset(gdf, transform=tf, shape=(n_rows, n_cols))
        expected_x = tf.c + (np.arange(n_cols) + 0.5) * tf.a
        expected_y = tf.f + (np.arange(n_rows) + 0.5) * tf.e
        np.testing.assert_allclose(ds.coords["x"].values, expected_x)
        np.testing.assert_allclose(ds.coords["y"].values, expected_y)

    def test_transform_written_to_dataset(self):
        gdf = _make_simple_gdf()
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        ds = create_layerset(gdf, transform=tf, shape=(50, 150))
        assert ds.rio.transform() is not None

    def test_horizontal_resolution_not_required_when_transform_given(self):
        gdf = _make_simple_gdf()
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        ds = create_layerset(gdf, transform=tf, shape=(50, 150))
        assert isinstance(ds, xr.Dataset)


# ---------------------------------------------------------------------------
# create_layerset — strata combining
# ---------------------------------------------------------------------------

class TestCombinePrimarySecondaryStrata:
    @staticmethod
    def _make_numbered_gdf():
        poly = box(0, 0, 100, 100)
        return gpd.GeoDataFrame(
            [
                {"strata": "Shrub1", "loading": 1.0, "height": 2.0,
                 "spatial_pattern": "Uniform", "percent_cover": 50, "geometry": poly},
                {"strata": "Shrub2", "loading": 1.0, "height": 2.0,
                 "spatial_pattern": "Uniform", "percent_cover": 50, "geometry": poly},
            ],
            geometry="geometry",
            crs=CRS,
        )

    def test_combine_true_yields_one_strata(self):
        gdf = self._make_numbered_gdf()
        ds = create_layerset(gdf, RESOLUTION, seed=SEED)
        assert ds.coords["strata"].to_index().tolist() == ["Shrub"]

    def test_combine_false_yields_two_strata(self):
        gdf = self._make_numbered_gdf()
        ds = create_layerset(gdf, RESOLUTION, seed=SEED, combine_primary_secondary_strata=False)
        strata = ds.coords["strata"].to_index().tolist()
        assert "Shrub_primary" in strata
        assert "Shrub_secondary" in strata


# ---------------------------------------------------------------------------
# create_layerset — height_func
# ---------------------------------------------------------------------------

class TestHeightFunc:
    def test_default_is_mean(self):
        """Default height_func produces the same result as np.mean."""
        gdf = _make_overlapping_gdf(height1=2.0, height2=4.0)
        ds_default = create_layerset(gdf, RESOLUTION)
        ds_mean = create_layerset(gdf, RESOLUTION, height_func=np.mean)
        np.testing.assert_array_equal(
            ds_default["height"].values, ds_mean["height"].values
        )

    @pytest.mark.parametrize("func, expected_val", [
        (np.mean, 3.0),
        (np.min, 2.0),
        (np.max, 4.0),
    ])
    def test_height_functions(self, func, expected_val):
        gdf = _make_overlapping_gdf(height1=2.0, height2=4.0)
        ds = create_layerset(gdf, RESOLUTION, height_func=func)
        h = ds["height"].sel(strata="Shrub").values
        # Only check cells that were actually hit
        np.testing.assert_allclose(h[h > 0], expected_val)

    def test_mean_three_ploygons(self):
        poly = box(0, 0, 100, 100)
        gdf = gpd.GeoDataFrame(
            [
                {"strata": "Shrub", "loading": 1.0, "height": 1.0,
                 "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly},
                {"strata": "Shrub", "loading": 1.0, "height": 2.0,
                 "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly},
                {"strata": "Shrub", "loading": 1.0, "height": 3.0,
                 "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly},
            ],
            geometry="geometry",
            crs=CRS,
        )
        ds = create_layerset(gdf, RESOLUTION, height_func=np.mean)
        h = ds["height"].sel(strata="Shrub").values
        np.testing.assert_allclose(h[h > 0], 2.0)

    def test_single_hit_unaffected_by_func(self):
        """A cell covered by only one polygon keeps that polygon's height for all funcs."""
        poly = box(0, 0, 100, 100)
        gdf = gpd.GeoDataFrame(
            [{"strata": "Shrub", "loading": 1.0, "height": 5.0,
              "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly}],
            geometry="geometry",
            crs=CRS,
        )
        for func in (np.mean, np.min, np.max):
            ds = create_layerset(gdf, RESOLUTION, height_func=func)
            h = ds["height"].sel(strata="Shrub").values
            np.testing.assert_allclose(h[h > 0], 5.0)


# ---------------------------------------------------------------------------
# create_layerset — values
# ---------------------------------------------------------------------------

class TestCreateLayersetValues:
    def test_loading_nonnegative(self):
        ds = create_layerset(_make_simple_gdf(), RESOLUTION, seed=SEED)
        assert (ds["loading"].values >= 0).all()

    def test_height_nonnegative(self):
        ds = create_layerset(_make_simple_gdf(), RESOLUTION, seed=SEED)
        assert (ds["height"].values >= 0).all()

    def test_full_cover_loading_equals_polygon_value(self):
        poly = box(0, 0, 100, 100)
        gdf = gpd.GeoDataFrame(
            [{"strata": "Shrub", "loading": 2.5, "height": 1.0,
              "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly}],
            geometry="geometry", crs=CRS,
        )
        ds = create_layerset(gdf, RESOLUTION)
        loading = ds["loading"].sel(strata="Shrub").values
        np.testing.assert_allclose(loading[loading > 0], 2.5)

    def test_loading_accumulates_across_overlapping_polygons(self):
        gdf = _make_overlapping_gdf(height1=1, height2=1)  # two rows, each loading=1.0, 100% cover
        ds = create_layerset(gdf, RESOLUTION)
        loading = ds["loading"].sel(strata="Shrub").values
        np.testing.assert_allclose(loading[loading > 0], 2.0)

    def test_loading_zero_outside_polygon(self):
        poly = box(0, 0, 50, 50)
        gdf = gpd.GeoDataFrame(
            [{"strata": "Shrub", "loading": 3.0, "height": 1.0,
              "spatial_pattern": "Uniform", "percent_cover": 100, "geometry": poly}],
            geometry="geometry", crs=CRS,
        )
        # Grid larger than the polygon so cells outside it exist
        tf = from_origin(0, 100, RESOLUTION, RESOLUTION)
        ds = create_layerset(gdf, transform=tf, shape=(100, 100))
        loading = ds["loading"].sel(strata="Shrub").values
        assert loading.max() > 0
        assert loading.min() == pytest.approx(0.0)

    def test_seed_reproducibility(self):
        gdf = _make_simple_gdf()
        ds1 = create_layerset(gdf, RESOLUTION, seed=SEED)
        ds2 = create_layerset(gdf, RESOLUTION, seed=SEED)
        np.testing.assert_array_equal(ds1["loading"].values, ds2["loading"].values)

    def test_different_seeds_differ(self):
        gdf = _make_simple_gdf()
        ds1 = create_layerset(gdf, RESOLUTION, seed=1)
        ds2 = create_layerset(gdf, RESOLUTION, seed=2)
        assert not np.array_equal(ds1["loading"].values, ds2["loading"].values)

    def test_percent_cover(self):
        poly = box(0, 0, 100, 100)
        gdf = gpd.GeoDataFrame([{
            "strata": "Shrub", "loading": 1.0, "height": 1.0,
            "spatial_pattern": "Uniform", "percent_cover": 50, "geometry": poly
        }], crs=CRS)

        ds = create_layerset(gdf, horizontal_resolution=1.0, seed=42)
        loading = ds["loading"].sel(strata="Shrub").values

        total_cells = loading.size
        filled_cells = np.count_nonzero(loading)
        actual_cover = (filled_cells / total_cells) * 100

        assert actual_cover == pytest.approx(50.0, abs=5.0)


# ---------------------------------------------------------------------------
# create_layerset — real data (integration)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAYERSETS_PARQUET.exists(), reason="sample parquet not present")
class TestCreateLayersetIntegration:
    def setup_method(self):
        self.gdf = _load_gdf()
        self.ds = create_layerset(self.gdf, RESOLUTION, seed=SEED)

    def test_returns_dataset(self):
        assert isinstance(self.ds, xr.Dataset)

    def test_strata_are_strings(self):
        for name in self.ds.coords["strata"].values:
            assert isinstance(name, str)

    def test_no_negative_loading(self):
        assert (self.ds["loading"].values >= 0).all()

    def test_no_negative_height(self):
        assert (self.ds["height"].values >= 0).all()

    def test_crs_matches_input(self):
        assert self.ds.rio.crs.to_epsg() == 5070
