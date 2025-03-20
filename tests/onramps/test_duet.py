from __future__ import annotations

from fastfuels_core.onramps import duet
from duet_tools import DuetRun
from duet_tools.utils import read_dat_to_array
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


def test_export_to_duet(
    density_array: xarray.DataArray,
    moisture_array: xarray.DataArray,
    spcd_array: xarray.DataArray,
):
    duet.export_to_duet(
        DUET_DIR, density_array, moisture_array, spcd_array, 270, 359, 5, 47
    )
    assert Path(DUET_DIR / "treesrhof.dat").exists()
    assert Path(DUET_DIR / "treesmoist.dat").exists()
    assert Path(DUET_DIR / "treesspcd.dat").exists()

    nz, ny, nx = density_array.shape

    treesrhof = read_dat_to_array(DUET_DIR, "treesrhof.dat", nx, ny, nz=nz)
    treesmoist = read_dat_to_array(DUET_DIR, "treesmoist.dat", nx, ny, nz=nz)
    treesspcd = read_dat_to_array(
        DUET_DIR, "treesspcd.dat", nx, ny, nz=nz, dtype=np.int32
    )

    density_numpy = density_array.to_numpy()
    moisture_numpy = moisture_array.to_numpy()
    spcd_numpy = spcd_array.to_numpy()

    assert np.allclose(treesrhof, density_numpy)
    assert np.allclose(treesmoist, moisture_numpy)
    assert np.allclose(treesspcd, spcd_numpy)

    dz, dy, dx = [
        int(np.diff(density_array.coords[dim]).mean()) for dim in density_array.dims
    ]

    with open(DUET_DIR / "duet.in") as f:
        lines = f.readlines()
        assert int(lines[0].strip().split("!")[0]) == nx
        assert int(lines[1].strip().split("!")[0]) == ny
        assert int(lines[2].strip().split("!")[0]) == nz
        assert float(lines[3].strip().split("!")[0]) == dx
        assert float(lines[4].strip().split("!")[0]) == dy
        assert float(lines[5].strip().split("!")[0]) == dz
        assert float(lines[6].strip().split("!")[0]) == 47
        assert float(lines[7].strip().split("!")[0]) == 270
        assert float(lines[8].strip().split("!")[0]) == 359
        assert float(lines[9].strip().split("!")[0]) == 5


def test_run_duet(
    density_array: xarray.DataArray,
    moisture_array: xarray.DataArray,
    spcd_array: xarray.DataArray,
):
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
    assert Path(DUET_DIR / "surface_rhof_layered.dat").exists()
    assert Path(DUET_DIR / "surface_moist_layered.dat").exists()
    assert Path(DUET_DIR / "surface_depth_layered.dat").exists()
    assert isinstance(duet_run, DuetRun)

    nz, ny, nx = density_array.shape

    assert duet_run.density.shape == (2, ny, nx)
    assert duet_run.moisture.shape == (2, ny, nx)
    assert duet_run.height.shape == (2, ny, nx)


def test_run_duet_no_exe(
    density_array: xarray.DataArray,
    moisture_array: xarray.DataArray,
    spcd_array: xarray.DataArray,
):
    with pytest.raises(FileNotFoundError):
        duet.run_duet(
            DUET_DIR,
            "duet_wrong_name.exe",
            density_array,
            moisture_array,
            spcd_array,
            270,
            359,
            5,
        )

    with pytest.raises(FileNotFoundError):
        duet.run_duet(
            DUET_DIR.parent,
            "duet_v2.1_FF_mac.exe",
            density_array,
            moisture_array,
            spcd_array,
            270,
            359,
            5,
        )
