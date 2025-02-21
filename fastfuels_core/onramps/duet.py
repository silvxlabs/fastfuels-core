from __future__ import annotations

# Core imports
from importlib.resources import files
from pathlib import Path
from shutil import copy
from subprocess import run, PIPE
from time import sleep

# External imports
import duet_tools as duet
from duet_tools.inputs import InputFile
from duet_tools.utils import write_array_to_dat
import numpy as np
from numpy import ndarray
from xarray import DataArray, Dataset

DATA_PATH = files("fastfuels_core.data")


def run_duet_no_calibration(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    canopy_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
    duet_version: str = "v2",
) -> Dataset:
    """
    Runs the specified DUET executable using the supplied treeGrid.
    No calibration of values is performed.

    Parameters
    ----------
    duet_exe_directory: Path | str
        Path to directory containing the DUET executable.
    duet_exe_name: str
        Name of DUET exe file, with extension.
    bulk_density_grid: DataArray
        Grid of bulk density values with dimensions ["z", "y", "x"].
    species_code_grid: DataArray
        Grid of FIA species codes with dimensions ["z", "y", "x"].
    fuel_moisture_grid: DataArray
        Grid of fuel moisture content values with dimensions ["z", "y", "x"].
    wind_direction: float
        Direction of prevailing wind in DUET simulation (degrees)
    wind_variability: float
        Variability in wind direction for the DUET simulation (degrees)
    random_seed: int = None
        Seed for reproducibility in DUET. If None, will be generated from computer time.
    duet_version: str = "v2"
        Version of DUET executable. Accepts "v1" and "v2".

    Returns
    -------
    An Xarray Dataset with dimensions ["parameter","fuel_type","x","y"] shape (2,2,ny,nx).
    """
    duet_run = _run_duet(
        duet_exe_directory,
        duet_exe_name,
        canopy_grid,
        spcd_grid,
        wind_direction,
        wind_variability,
        duration,
        random_seed,
        duet_version,
    )
    return _assemble_dataset(duet_run, canopy_grid)


def run_and_calibrate_duet(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    canopy_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
    duet_version: str = "v2",
    bulk_density_targets: duet.FuelParameter = None,
    fuel_height_targets: duet.FuelParameter = None,
) -> Dataset:
    """
    Runs the specified DUET executable using the supplied treeGrid, then
    calibrates the resulting values based on targets and methods the supplied
    FuelParameter object(s).

    Parameters
    ----------
    duet_exe_directory: Path | str
        Path to directory containing the DUET executable.
    duet_exe_name: str
        Name of DUET exe file, with extension.
    bulk_density_grid: DataArray
        Grid of bulk density values with dimensions ["z", "y", "x"].
    species_code_grid: DataArray
        Grid of FIA species codes with dimensions ["z", "y", "x"].
    fuel_moisture_grid: DataArray
        Grid of fuel moisture content values with dimensions ["z", "y", "x"].
    wind_direction: float
        Direction of prevailing wind in DUET simulation (degrees)
    wind_variability: float
        Variability in wind direction for the DUET simulation (degrees)
    random_seed: int = None
        Seed for reproducibility in DUET. If None, will be generated from computer time.
    duet_version: str = "v2"
        Version of DUET executable. Accepts "v1" and "v2".

    Returns
    -------
    An Xarray Dataset with dimensions ["parameter","fuel_type","x","y"] shape (2,2,ny,nx).
    """
    duet_run = _run_duet(
        duet_exe_directory,
        duet_exe_name,
        canopy_grid,
        spcd_grid,
        wind_direction,
        wind_variability,
        duration,
        random_seed,
        duet_version,
    )
    targets = [bulk_density_targets, fuel_height_targets]
    fuel_parameter_targets = [target for target in targets if target is not None]
    calibrated_duet = duet.calibrate(duet_run, fuel_parameter_targets)
    return _assemble_dataset(calibrated_duet, canopy_grid)


def run_and_calibrate_duet_from_surface_grid(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    canopy_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    surface_grid: DataArray,  # TODO
    random_seed: int = None,
    duet_version: str = "v2",
) -> Dataset:
    """
    Runs the specified DUET executable using the supplied treeGrid, then
    calibrates the resulting values based on targets and methods the supplied
    surfaceGrid.

    Parameters
    ----------
    duet_exe_directory: Path | str
        Path to directory containing the DUET executable.
    duet_exe_name: str
        Name of DUET exe file, with extension.
    bulk_density_grid: DataArray
        Grid of bulk density values with dimensions ["z", "y", "x"].
    species_code_grid: DataArray
        Grid of FIA species codes with dimensions ["z", "y", "x"].
    fuel_moisture_grid: DataArray
        Grid of fuel moisture content values with dimensions ["z", "y", "x"].
    wind_direction: float
        Direction of prevailing wind in DUET simulation (degrees)
    wind_variability: float
        Variability in wind direction for the DUET simulation (degrees)
    surface_grid: DataArray? Dataset?
        TODO
    random_seed: int = None
        Seed for reproducibility in DUET. If None, will be generated from computer time.
    duet_version: str = "v2"
        Version of DUET executable. Accepts "v1" and "v2".

    Returns
    -------
    An Xarray Dataset with dimensions ["parameter","fuel_type","x","y"] shape (2,2,ny,nx).
    """
    duet_run = _run_duet(
        duet_exe_directory,
        duet_exe_name,
        canopy_grid,
        spcd_grid,
        wind_direction,
        wind_variability,
        duration,
        random_seed,
        duet_version,
    )
    # TODO


# def run_and_calibrate_duet_with_mask(grass_mask, litter_mask) -> ndarray:
#     output = run_duet(...)
#     calibrated_output = dt.calibrate(output)
#     return calibrated_output


def _run_duet(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    canopy_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
    duet_version: str = "v2",
) -> duet.DuetRun:
    """
    Runs the specified DUET executable using the supplied treeGrid.

    Parameters
    ----------
    duet_exe_directory: Path | str
        Path to directory containing the DUET executable.
    duet_exe_name: str
        Name of DUET exe file, with extension.
    bulk_density_grid: DataArray
        Grid of bulk density values with dimensions ["z", "y", "x"].
    species_code_grid: DataArray
        Grid of FIA species codes with dimensions ["z", "y", "x"].
    fuel_moisture_grid: DataArray
        Grid of fuel moisture content values with dimensions ["z", "y", "x"].
    wind_direction: float
        Direction of prevailing wind in DUET simulation (degrees)
    wind_variability: float
        Variability in wind direction for the DUET simulation (degrees)
    random_seed: int = None
        Seed for reproducibility in DUET. If None, will be generated from computer time.
    duet_version: str = "v2"
        Version of DUET executable. Accepts "v1" and "v2".
    """
    if isinstance(duet_exe_directory, str):
        duet_exe_directory = Path(duet_exe_directory)

    # write input file
    nz, ny, nx = canopy_grid.shape
    dz, dy, dx = [
        int(
            np.diff(canopy_grid.coords[dim]).mean()
        )  # this only works if grid is constant
        for dim in canopy_grid.dims
    ]
    duet_in = InputFile.create(
        nx, ny, nz, duration, wind_direction, dx, dy, dz, random_seed, wind_variability
    )
    duet_in.to_file(duet_exe_directory)

    # write fuel grids to .dat files
    write_array_to_dat(canopy_grid, "treesrhof.dat", duet_exe_directory, reshape=False)
    write_array_to_dat(spcd_grid, "treesspcd.dat", duet_exe_directory, reshape=False)

    exit_code = _execute_duet(duet_exe_directory, duet_exe_name)
    if exit_code != 0:
        raise (Exception)  # TODO
    else:
        nsp = len(
            np.unique(canopy_grid.values)
        )  # this should be the number of species + 1
        duet_run = duet.import_duet(duet_exe_directory, nx, ny, nsp, duet_version)
    return duet_run


def _assemble_dataset(duet_run: duet.DuetRun, canopy_grid: DataArray) -> Dataset:
    """ """
    litter_density = duet_run.to_numpy("litter", "density")
    grass_density = duet_run.to_numpy("grass", "density")
    litter_height = duet_run.to_numpy("litter", "height")
    grass_height = duet_run.to_numpy("grass", "height")

    # Define fuel parameter and type labels
    parameter = ["density", "height"]
    fuel_type = ["grass", "litter"]

    data = np.array(
        [[grass_density, litter_density], [grass_height, litter_height]]
    )  # shape: (parameter, fuel_type, x, y)

    # Create an xarray Dataset
    dataset = Dataset(
        {"data": (["group", "type", "x", "y"], data)},
        coords={
            "parameter": parameter,
            "fuel_type": fuel_type,
            "x": canopy_grid["nx"].coords,
            "y": canopy_grid["ny"].coords,
        },
    )
    return dataset


def _execute_duet(
    duet_exe_directory: Path,
    duet_exe_name: str,
) -> int:
    """
    Parameters
    ----------
    duet_exe_directory: Path
        Path to directory containing the DUET executable.
    duet_exe_name: str
        Name of DUET exe file, with extension.
    """

    # copy necessary txt file from data directory
    copy(
        DATA_PATH / "FIA_FastFuels_fin_fulllist_populated.txt",
        duet_exe_directory / "FIA_FastFuels_fin_fulllist_populated.txt",
    )

    # run duet executable
    duet_exe_path = duet_exe_directory / duet_exe_name
    subprocess_call = [f"./{duet_exe_path}"]
    exit_code = _run_and_report(subprocess_call)
    return exit_code


def _run_and_report(subprocess_call: list) -> int:
    with run(subprocess_call, stdout=PIPE, sterr=PIPE) as process:

        def poll_and_read():
            print(f"{process.stdout.read1().decode('utf-8')}")

        while process.poll() is None:
            poll_and_read()
            sleep(1)
        if process.poll() == 0:
            print("DUET run successfully")
        else:
            print(f"DUET ERROR with exit code {process.stdout.read1().decode('utf-8')}")
            print(process.sterr.read1().decode("utf-8"))
        return process.stdout.read1().decode("utf-8")
