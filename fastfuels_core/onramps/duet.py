from __future__ import annotations

# Core imports
from importlib.resources import files
from pathlib import Path
from shutil import copy
from subprocess import Popen, PIPE
from time import sleep

# External imports
from duet_tools import DuetRun, InputFile, import_duet
from duet_tools.utils import write_array_to_dat
import numpy as np
from xarray import DataArray

DATA_PATH = files("fastfuels_core.data")


def run_duet(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    bulk_density_grid: DataArray,
    fuel_moisture_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
    duet_version: str = "v2",
) -> DuetRun:
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
    fuel_moisture_grid: DataArray
        Grid of fuel moisture values with dimensions ["z","y","x"].
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
    A DuetRun object
    """
    export_to_duet(
        duet_exe_directory,
        bulk_density_grid,
        fuel_moisture_grid,
        spcd_grid,
        wind_direction,
        wind_variability,
        duration,
        random_seed,
    )

    nz, ny, nx = bulk_density_grid.shape

    exit_code = _execute_duet(duet_exe_directory, duet_exe_name)
    if exit_code != 0:
        raise (Exception)  # TODO
    else:
        nsp = len(
            np.unique(spcd_grid.values)
        )  # this should be the number of species + 1
        duet_run = import_duet(duet_exe_directory, nx, ny, nsp, duet_version)
    return duet_run


def export_to_duet(
    directory: Path | str,
    bulk_density_grid: DataArray,
    fuel_moisture_grid: DataArray,
    spcd_grid: DataArray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
):
    """
    Runs the specified DUET executable using the supplied treeGrid.

    Parameters
    ----------
    directory: Path | str
        Path to directory where DUET files are written.
    bulk_density_grid: DataArray
        Grid of bulk density values with dimensions ["z", "y", "x"].
    fuel_moisture_grid: DataArray
        Grid of fuel moisture values with dimensions ["z","y","x"].
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

    Returns
    -------
    None
        DUET input files (duet.in and trees*.dat) are saved to the directory
    """
    if isinstance(directory, str):
        directory = Path(directory)

    # write input file
    nz, ny, nx = bulk_density_grid.shape
    dz, dy, dx = [
        int(
            np.diff(bulk_density_grid.coords[dim]).mean()
        )  # this only works if grid is constant
        for dim in bulk_density_grid.dims
    ]
    duet_in = InputFile.create(
        nx, ny, nz, duration, wind_direction, dx, dy, dz, random_seed, wind_variability
    )
    duet_in.to_file(directory)

    # write fuel grids to .dat files
    write_array_to_dat(
        bulk_density_grid.data, "treesrhof.dat", directory, reshape=False
    )
    write_array_to_dat(
        fuel_moisture_grid.data, "treesmoist.dat", directory, reshape=False
    )
    write_array_to_dat(
        spcd_grid.data,
        "treesspcd.dat",
        directory,
        reshape=False,
        dtype=np.int32,
    )
    print(f"DUET input files written to {directory}")


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
    subprocess_call = ["bash", "-c", f'cd "{duet_exe_directory}" && ./{duet_exe_name}']
    exit_code = _run_and_report(subprocess_call)
    return exit_code


def _run_and_report(subprocess_call: list) -> int:
    with Popen(
        subprocess_call,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    ) as process:
        while process.poll() is None:
            output = process.stdout.read()
            if output:
                print(output, end="")
            sleep(1)

        # Final output after process ends
        output = process.stdout.read()
        if output:
            print(output, end="")

        # Check the exit code
        if process.poll() == 0:
            print("DUET run successfully")
            return process.poll()
        else:
            print(f"DUET ERROR with exit code: {process.poll()}")
            print(process.stderr.read())
            return process.poll()
