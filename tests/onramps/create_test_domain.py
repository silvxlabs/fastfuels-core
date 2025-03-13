from os import environ

environ["FASTFUELS_API_KEY"] = "c51bd50bfde04de7abbfacc15c56c844"
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon
from fastfuels_sdk.domains import Domain
from fastfuels_sdk.features import Features
from fastfuels_sdk.inventories import Inventories
from fastfuels_sdk.grids import (
    Grids,
    TopographyGridBuilder,
    TreeGridBuilder,
)

DUET_DIR = Path(__file__).parent / "duet-data"

# Define the polygon coordinates for Blue Mountain area
coordinates = [
    [-114.09957018646286, 46.82933208815811],
    [-114.10141707482919, 46.828370407248826],
    [-114.10010954324228, 46.82690548814563],
    [-114.09560673134018, 46.8271123684554],
    [-114.09592544216444, 46.829058122675065],
    [-114.09957018646286, 46.82933208815811],
]

# Create a GeoDataFrame
polygon = Polygon(coordinates)
roi = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")  # WGS 84 coordinate system

domain = Domain.from_geodataframe(
    geodataframe=roi,
    name="Blue Mountain ROI",
    description="Test area in Blue Mountain Recreation Area",
    horizontal_resolution=2.0,  # 2-meter horizontal resolution
    vertical_resolution=1.0,  # 1-meter vertical resolution
)

print(f"Created domain with ID: {domain.id}")

# Initialize Features for our domain
features = Features.from_domain_id(domain.id)

# Create features from OpenStreetMap
road_feature = features.create_road_feature_from_osm()
water_feature = features.create_water_feature_from_osm()

# Wait for features to be ready
road_feature.wait_until_completed()
water_feature.wait_until_completed()

# Create feature grid
feature_grid = Grids.from_domain_id(domain.id).create_feature_grid(
    attributes=["road", "water"],
)

feature_grid.wait_until_completed()

topography_grid = (
    TopographyGridBuilder(domain_id=domain.id)
    .with_elevation_from_3dep(interpolation_method="linear")
    .build()
)

topography_grid.wait_until_completed()

# Create tree inventory
tree_inventory = Inventories.from_domain_id(
    domain.id
).create_tree_inventory_from_treemap(feature_masks=["road", "water"])
tree_inventory.wait_until_completed()

# Create tree grid
tree_grid = (
    TreeGridBuilder(domain_id=domain.id).with_bulk_density_from_tree_inventory().build()
)
tree_grid.wait_until_completed()

export = tree_grid.create_export("zarr")
export.wait_until_completed()
export.to_file(DUET_DIR / "test_tree_grid.zarr")
