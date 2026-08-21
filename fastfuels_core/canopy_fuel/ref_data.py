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
(hardwoods excluded). The guide prints no species code for the
"Ponderosa Pine SW" row; ``sr_ESD[]`` in ``FC_DLL/NC_ESD.C`` keys it
to NRCS symbol PIAR5, which is *Pinus arizonica*, SPCD 135. That row
is the one place the PS vertical distribution and the PS crown-class
factors are reachable.

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
    )
    table["SPCD"] = table["SPCD"].astype(int)
    return table.set_index("SPCD")


@lru_cache(maxsize=1)
def fuelcalc_crown_width() -> pd.DataFrame:
    """FVS crown width coefficients, indexed by FVS species index.

    ``COVER_EQ`` in the species table selects a row. The C table numbers
    the catch-all "Other" row 39 and the User Guide's printed table
    numbers it 38; ``sr_ESD[]`` points species at 39, so that is the
    number carried here. Every other row agrees between the two.
    """
    return pd.read_csv(
        files("fastfuels_core.data") / "FUELCALC_CROWN_WIDTH.csv",
        index_col="COVER_EQ",
    )


@lru_cache(maxsize=1)
def fuelcalc_small_tree_biomass() -> pd.DataFrame:
    """Small-tree component weights (lb), indexed by (CODE, HT_CLASS_FT).

    Measured crown component weights for trees of one inch dbh and
    under, tabulated by the same equation code the crown-weight
    equations use and by height in ten one-foot classes -- class 1 is
    ``h <= 1`` and class 10 everything over 9 ft. Diameter selects the
    table, height selects the row.

    The rows are Brown (1978) Tables 6 and 19 evaluated per foot of
    height, as the FuelCalc 1.7 User Guide (Appendix D) prints them.
    """
    table = pd.read_csv(
        files("fastfuels_core.data") / "FUELCALC_SMALL_TREE_BIOMASS.csv",
    )
    return table.set_index(["CODE", "HT_CLASS_FT"])
