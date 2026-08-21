"""Crown width from the FVS/FOFEM species coefficients.

Crookston, N.L. & Stage, A.R. 1999. Percent Canopy Cover and Stand
Structure Statistics from the Forest Vegetation Simulator. USDA For.
Serv. Gen. Tech. Rep. RMRS-GTR-24. Reached through the Fire and Fuels
Extension to FVS (Reinhardt & Crookston, eds.); the coefficients are
fitted to the R6 Permanent Plot Grid Inventory and are printed in the
FuelCalc guide (p. 78) and carried in ``FC_DLL/nc_ca2.h``.

Two forms, split at breast height because below it diameter is measured
at the root collar and the power fit does not hold::

    CW = A * D**B      height > 4.5 ft
    CW = R * D         height <= 4.5 ft

with ``CW`` the crown width in feet and ``D`` the diameter in inches.
Species resolve to a coefficient row through the FuelCalc species
table's ``COVER_EQ`` column, an FVS species index.

This is the crown width behind FuelCalc's canopy cover. It is a
different quantity from the Purves radius used by default elsewhere in
fastfuels-core: Purves is fitted to FIA crown-width data continent-wide
and varies with height and stand position, these coefficients are
regional and depend on diameter alone above breast height.
"""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.ref_data import fuelcalc_crown_width, fuelcalc_species

# Above this height the power form applies; at or below it the ratio
# form does (``NC_CA.C CA_CrnArea``).
BREAST_HEIGHT_FT = 4.5


def crown_width(
    cover_eq: np.ndarray, dia_in: np.ndarray, height_ft: np.ndarray
) -> np.ndarray:
    """Crown width (feet) from FVS coefficients.

    Parameters
    ----------
    cover_eq : numpy.ndarray
        FVS species indices, from the species table's ``COVER_EQ``.
    dia_in : numpy.ndarray
        Diameter at breast height, inches.
    height_ft : numpy.ndarray
        Tree height, feet.

    Returns
    -------
    numpy.ndarray
        Crown width in feet, floored at zero.

    Raises
    ------
    ValueError
        For an FVS index absent from the coefficient table.
    """
    eq = np.asarray(cover_eq)
    dia = np.asarray(dia_in, dtype=np.float64)
    height = np.asarray(height_ft, dtype=np.float64)

    table = fuelcalc_crown_width()
    unknown = np.setdiff1d(eq, table.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"FVS crown width index/indices {unknown.tolist()} are not in "
            f"the coefficient table."
        )
    rows = table.index.get_indexer(eq)
    a = table["A"].to_numpy()[rows]
    b = table["B"].to_numpy()[rows]
    ratio = table["RATIO"].to_numpy()[rows]

    small = height <= BREAST_HEIGHT_FT
    # dia**b is nan for dia == 0; the ratio form covers seedlings and
    # zero diameter is zero width either way.
    with np.errstate(invalid="ignore", divide="ignore"):
        large = a * np.power(np.maximum(dia, 0.0), b)
    return np.maximum(np.where(small, ratio * dia, large), 0.0)


def crown_width_for_species(
    species_code: np.ndarray, dia_in: np.ndarray, height_ft: np.ndarray
) -> np.ndarray:
    """:func:`crown_width`, resolving FIA species codes through COVER_EQ.

    Raises
    ------
    ValueError
        For species outside the FuelCalc species table.
    """
    spcd = np.asarray(species_code)
    species = fuelcalc_species()
    unknown = np.setdiff1d(spcd, species.index.to_numpy())
    if unknown.size:
        raise ValueError(
            f"Species code(s) {unknown.tolist()} are not in the FuelCalc "
            f"species table, so they have no crown width equation."
        )
    return crown_width(species["COVER_EQ"].loc[spcd].to_numpy(), dia_in, height_ft)
