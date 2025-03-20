from __future__ import annotations

from fastfuels_core.onramps import duet
from duet_tools import DuetRun
import numpy as np
from pathlib import Path
import pytest
import xarray
import zarr

DUET_DIR = Path(__file__).parent / "duet-data"


@pytest.fixture(autouse=True)
def delete_dat_files() -> None:
    dat_list = [
        "treesrhof.dat",
        "treesmoist.dat",
        "treesfueldepth.dat",
        "surface_rhof_layered.dat",
        "surface_moist_layered.dat",
        "surface_depth_layered.dat",
        "surface_ss_layered.dat",
        "surface_rhof.dat",
        "surface_species.dat",
        "flattrees.dat",
        "canopy.dat",
    ]
    for dat in dat_list:
        dat_path = DUET_DIR / dat
        dat_path.unlink(missing_ok=True)


def get_zarr_grid_and_attrs(zarr_path: Path, group: str) -> tuple:
    tree_grid = zarr.open(zarr_path, mode="r")
    array = tree_grid[group][...]
    return array


def create_dataarray(array: np.ndarray) -> xarray.DataArray:
    x_len = array.shape[2] * 2
    y_len = array.shape[1] * 2
    z_len = array.shape[0] * 1
    x_coords = np.linspace(0, x_len, array.shape[2])
    y_coords = np.linspace(0, y_len, array.shape[1])
    z_coords = np.linspace(0, z_len, array.shape[0])
    xarr = xarray.DataArray(
        data=array,
        dims=["z", "y", "x"],
        coords={"z": z_coords, "y": y_coords, "x": x_coords},
    )
    return xarr


@pytest.fixture
def density_array():
    array = get_zarr_grid_and_attrs(DUET_DIR / "test_tree_grid.zarr", "bulkDensity")
    xarr = create_dataarray(array)
    return xarr


@pytest.fixture
def spcd_array():
    array = get_zarr_grid_and_attrs(DUET_DIR / "test_tree_grid.zarr", "SPCD")
    xarr = create_dataarray(array)
    return xarr


@pytest.fixture
def moisture_array():
    array = get_zarr_grid_and_attrs(DUET_DIR / "test_tree_grid.zarr", "fuelMoisture")
    xarr = create_dataarray(array)
    return xarr


def test_run_duet(density_array, moisture_array, spcd_array):
    duet_run = duet.run_duet(
        DUET_DIR,
        "duet_v2.1_FF_mac.exe",
        density_array,
        moisture_array,
        spcd_array,
        270,
        359,
        5,
    )
    assert Path(DUET_DIR / "treesrhof.dat").exists()
    assert Path(DUET_DIR / "treesmoist.dat").exists()
    assert Path(DUET_DIR / "treesspcd.dat").exists()
    assert Path(DUET_DIR / "surface_rhof_layered.dat").exists()
    assert isinstance(duet_run, DuetRun)
