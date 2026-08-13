"""FuelCalc reference tables for canopy fuel computation, loaded lazily.

Transcribed from the FuelCalc 1.7 User's Guide, Appendix D (pp. 68-81).
The vertical-distribution cubics originate in Reinhardt, Scott, Gray &
Keane 2006 (Can. J. For. Res. 36:2803-2814, Table 4); FuelCalc's PP, PS,
and IC rows match the Ninemile, Flagstaff, and Blodgett "Available" fits
exactly, while DF and LP are pooled fits by Kathy Gray. Cumulative fuel
fraction is ``pw(ph) = B1*ph + B2*ph**2 + B3*ph**3`` with pw(0)=0 and
pw(1)=1 (B1+B2+B3=1); ph is fractional height within the crown.

The species table is FuelCalc's Default Equation Table keyed by FIA
species code. ``INCL_CBD`` is FuelCalc's default species inclusion
(hardwoods excluded). The "Ponderosa Pine SW" row has no distinct FIA
code (122 covers both varieties) and is omitted; SPCD 122 maps to PP.

Tables are read from disk on first use and cached, so importing
fastfuels_core (or this module) does no I/O.
"""

from functools import lru_cache
from importlib.resources import files

import pandas as pd


@lru_cache(maxsize=1)
def fuelcalc_vdist() -> pd.DataFrame:
    """Vertical-distribution cubic coefficients, indexed by VDIST_CODE."""
    return pd.read_csv(
        files("fastfuels_core.data") / "FUELCALC_VERTICAL_DISTRIBUTION.csv",
        index_col="VDIST_CODE",
    )


@lru_cache(maxsize=1)
def fuelcalc_crown_class_factors() -> pd.DataFrame:
    """Crown-class biomass adjustment factors, indexed by CROWN_REDUC_CODE."""
    return pd.read_csv(
        files("fastfuels_core.data") / "FUELCALC_CROWN_CLASS_FACTORS.csv",
        index_col="CROWN_REDUC_CODE",
    )


@lru_cache(maxsize=1)
def fuelcalc_species() -> pd.DataFrame:
    """FuelCalc Default Equation Table, indexed by FIA species code."""
    table = pd.read_csv(
        files("fastfuels_core.data") / "FUELCALC_SPECIES_TABLE.csv",
    ).dropna(subset=["SPCD"])
    table["SPCD"] = table["SPCD"].astype(int)
    return table.set_index("SPCD")
