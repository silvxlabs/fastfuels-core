"""
This commit contains a first draft with lots more flexibility. As of 3/11/25 Anthony and Niko decided that
the calibration would take place in the FastFuels API repo, and that using SB40 to calibrate would not be
supported for now.
"""

from __future__ import annotations

# Core imports
from importlib.resources import files
from pathlib import Path
from shutil import copy
from subprocess import run, PIPE
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
    canopy_grid: DataArray,
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
    A calibrated DuetRun object
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
        duet_run = import_duet(duet_exe_directory, nx, ny, nsp, duet_version)
    return duet_run


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
