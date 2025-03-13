from __future__ import annotations

from pathlib import Path
import pytest
from fastfuels_core.onramps import duet
import xarray
import zarr

DUET_DIR = Path(__file__).parent / "duet-data"

"""
We will need .dat files to run duet.
Get them from fastfuels-sdk?
Write a script to pull down a fuelgrid zarr
Put that zarr in duet-data
Create a pytest fixture?? that imports and processes the zarr into an xarray DataArray
Test run_duet with those arrays
"""


class TestRunDuet:
    def test_run_duet(self):
        """"""
