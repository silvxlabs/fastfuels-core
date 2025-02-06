from __future__ import annotations

# Core imports
from importlib.resources import files
from pathlib import Path
from shutil import copy
from subprocess import Popen, PIPE
from time import sleep

# External imports
import numpy as np
from numpy import ndarray
import duet_tools as dt
from duet_tools.inputs import InputFile
from duet_tools.utils import write_array_to_dat

DATA_PATH = files("fastfuels_core.data")


def run_duet(
    duet_exe_directory: Path | str,
    duet_exe_name: str,
    bulk_density_grid: ndarray,
    species_code_grid: ndarray,
    fuel_moisture_grid: ndarray,
    wind_direction: float,
    wind_variability: float,
    duration: int,
    random_seed: int = None,
) -> ndarray:
    """"""
    if isinstance(duet_exe_directory, str):
        duet_exe_directory = Path(duet_exe_directory)

    # write input file
    nz, ny, nx = bulk_density_grid.shape
    dz, dy, dx = (1, 2, 2)  # how to get this from grid(s)?
    duet_in = InputFile.create(
        nx, ny, nz, duration, wind_direction, dx, dy, dz, random_seed, wind_variability
    )
    duet_in.to_file(duet_exe_directory)

    # copy necessary txt file from data directory
    copy(
        DATA_PATH / "FIA_FastFuels_fin_fulllist_populated.txt",
        duet_exe_directory / "FIA_FastFuels_fin_fulllist_populated.txt",
    )

    # write fuel grids to .dat files
    write_array_to_dat(
        bulk_density_grid, "treesrhof.dat", duet_exe_directory, reshape=False
    )
    write_array_to_dat(
        species_code_grid, "treesspcd.dat", duet_exe_directory, reshape=False
    )
    write_array_to_dat(
        fuel_moisture_grid, "treesmoist.dat", duet_exe_directory, reshape=False
    )

    # run duet executable
    duet_exe_path = duet_exe_directory / duet_exe_name
    subprocess_call = [f"./{duet_exe_path}"]
    _run_and_report(subprocess_call)


def run_and_calibrate_duet() -> ndarray:
    output = run_duet(...)
    calibrated_output = dt.calibrate(output)
    return calibrated_output


def run_and_calibrate_duet_from_surface_grid(surface_grid) -> ndarray:
    output = run_duet(...)
    calibrated_output = dt.calibrate(output)
    return calibrated_output


def run_and_calibrate_duet_with_mask(grass_mask, litter_mask) -> ndarray:
    output = run_duet(...)
    calibrated_output = dt.calibrate(output)
    return calibrated_output


def _run_and_report(subprocess_call: list):
    with Popen(subprocess_call, stdout=PIPE) as process:

        def poll_and_read():
            print(f"{process.stdout.read1().decode('utf-8')}")

        while process.poll() is None:
            poll_and_read()
            sleep(1)
        if process.poll() == 0:
            print("DUET run successfully")
        else:
            poll_and_read()
