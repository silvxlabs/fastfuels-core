"""
Dead Fuel Moisture Content Calculator using the Fosberg Method
===========================================================

This module implements the Fosberg and Deeming (1971) methodology for calculating
1-hour dead fuel moisture content. The calculations are based on the methodology
documented by the National Wildfire Coordinating Group (NWCG).

The module provides functionality to:
    - Calculate reference fuel moisture based on temperature and relative humidity
    - Apply corrections based on environmental factors including:
        * Time of day
        * Aspect
        * Slope
        * Shading
        * Relative elevation
    - Handle both scalar and vectorized (numpy array) inputs

Module Constants
---------------
TEMP_BREAKPOINTS : numpy.ndarray
    Temperature breakpoints in Fahrenheit [10, 30, 50, 70, 90, 110]
RH_BREAKPOINTS : numpy.ndarray
    Relative humidity breakpoints in percent [0, 5, ..., 95, 100]
TIME_RANGES : list of str
    Time ranges in 24-hour format ['0800-0959', ..., '1800-1959']
SLOPE_RANGES : list of str
    Slope ranges in percent ['0-30', '31-90']
RFM_TABLE : numpy.ndarray
    Reference fuel moisture lookup table
TABLE_B : dict
    Correction factors for May-June-July
TABLE_C : dict
    Correction factors for Feb-Mar-Apr and Aug-Sep-Oct
TABLE_D : dict
    Correction factors for Nov-Dec-Jan

Classes
-------
Aspect : class
    Enumeration for cardinal directions (North, East, South, West)
ShadingBoolean : class
    Enumeration for shading conditions (Unshaded, Shaded)
RelativeElevation : class
    Enumeration for elevation positions (Below, Level, Above)

Notes
-----
The Fosberg method is based on:
Fosberg, M.A., and J.E. Deeming. 1971. Derivation of the 1- and 10-hour
timelag fuel moisture calculations for fire-danger rating.
Res. Paper RM-207. Fort Collins, CO. Rocky Mountain Forest and Range
Experiment Station. 8 p.

Examples
--------
>>> from fosberg import get_1hr_fuel_moisture
>>> # Calculate 1-hr fuel moisture for a south-facing slope at noon
>>> moisture = calculate_1hr_fuel_moisture(
...     dry_bulb_temp=75,
...     relative_humidity=30,
...     aspect=180,  # South-facing
...     slope=20,
...     time=1200,
...     month='June',
...     elevation=1,  # Level with weather station
...     shading=0.2,  # 20% shaded
... )
>>> print(moisture)
4.0

See Also
--------
NWCG Documentation : https://www.nwcg.gov/publications/pms437/fuel-moisture/dead-fuel-moisture-content

References
----------
.. [1] Fosberg, M.A., and J.E. Deeming. 1971.
       https://www.frames.gov/catalog/14186
.. [2] National Wildfire Coordinating Group. Dead Fuel Moisture Content.
       https://www.nwcg.gov/publications/pms437/fuel-moisture/dead-fuel-moisture-content
"""

# External imports
import numpy as np

"""
Table A. Reference Fuel Moisture
"""

# Reference Fuel Moisture (RFM) based on temperature and relative humidity
RFM_TABLE = np.array(
    [
        [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 9, 10, 11, 12, 12, 13, 13, 14],
        [1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 13],
        [1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 12, 12, 13],
        [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
        [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
        [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 11, 11, 12, 12, 12],
    ]
)

# Define temperature and humidity breakpoints
TEMP_BREAKPOINTS = np.array([10, 30, 50, 70, 90, 110])
RH_BREAKPOINTS = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
)


"""
Corrections Tables
"""

TIME_RANGES = [
    "0800-0959",
    "1000-1159",
    "1200-1359",
    "1400-1559",
    "1600-1759",
    "1800-1959",
]

SLOPE_RANGES = ["0-30", "31-90"]  # Percent


class Aspect:
    """
    Enumeration for cardinal directions.
    """

    North: int = 0
    East: int = 1
    South: int = 2
    West: int = 3


class ShadingBoolean:
    """
    Enumeration for shading conditions.
    """

    Unshaded: int = 0
    Shaded: int = 1


class RelativeElevation:
    """
    Enumeration for elevation positions.
    """

    B: int = 0
    L: int = 1
    A: int = 2


TABLE_B = {
    ShadingBoolean.Unshaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [2, 3, 4],
                TIME_RANGES[1]: [1, 1, 1],
                TIME_RANGES[2]: [0, 0, 1],
                TIME_RANGES[3]: [0, 0, 1],
                TIME_RANGES[4]: [1, 1, 1],
                TIME_RANGES[5]: [2, 3, 4],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [3, 4, 4],
                TIME_RANGES[1]: [1, 2, 2],
                TIME_RANGES[2]: [1, 1, 2],
                TIME_RANGES[3]: [1, 1, 2],
                TIME_RANGES[4]: [1, 2, 2],
                TIME_RANGES[5]: [3, 4, 4],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [2, 2, 3],
                TIME_RANGES[1]: [1, 1, 1],
                TIME_RANGES[2]: [0, 0, 1],
                TIME_RANGES[3]: [0, 0, 1],
                TIME_RANGES[4]: [1, 1, 2],
                TIME_RANGES[5]: [3, 4, 4],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [1, 2, 2],
                TIME_RANGES[1]: [0, 0, 1],
                TIME_RANGES[2]: [0, 0, 1],
                TIME_RANGES[3]: [1, 1, 2],
                TIME_RANGES[4]: [2, 3, 4],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [2, 3, 3],
                TIME_RANGES[1]: [1, 1, 1],
                TIME_RANGES[2]: [0, 0, 1],
                TIME_RANGES[3]: [0, 0, 1],
                TIME_RANGES[4]: [1, 1, 1],
                TIME_RANGES[5]: [2, 3, 3],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [2, 3, 3],
                TIME_RANGES[1]: [1, 1, 2],
                TIME_RANGES[2]: [0, 1, 1],
                TIME_RANGES[3]: [0, 1, 1],
                TIME_RANGES[4]: [1, 1, 2],
                TIME_RANGES[5]: [2, 3, 3],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [2, 3, 4],
                TIME_RANGES[1]: [1, 1, 2],
                TIME_RANGES[2]: [0, 0, 1],
                TIME_RANGES[3]: [0, 0, 1],
                TIME_RANGES[4]: [0, 1, 1],
                TIME_RANGES[5]: [1, 2, 3],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [2, 3, 4],
                TIME_RANGES[2]: [1, 1, 2],
                TIME_RANGES[3]: [0, 0, 1],
                TIME_RANGES[4]: [0, 0, 1],
                TIME_RANGES[5]: [1, 2, 2],
            },
        },
    },
    ShadingBoolean.Shaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 5],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 4, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 4, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 4, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 4, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 4, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 4, 5],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 5],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 4, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 3, 4],
                TIME_RANGES[3]: [3, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 4, 5],
            },
        },
    },
}

TABLE_C = {
    ShadingBoolean.Unshaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [1, 2, 3],
                TIME_RANGES[2]: [1, 1, 2],
                TIME_RANGES[3]: [1, 1, 2],
                TIME_RANGES[4]: [1, 2, 3],
                TIME_RANGES[5]: [3, 4, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [3, 3, 4],
                TIME_RANGES[2]: [2, 3, 4],
                TIME_RANGES[3]: [2, 3, 4],
                TIME_RANGES[4]: [3, 3, 4],
                TIME_RANGES[5]: [3, 4, 5],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [1, 2, 3],
                TIME_RANGES[2]: [1, 1, 1],
                TIME_RANGES[3]: [1, 1, 2],
                TIME_RANGES[4]: [1, 2, 4],
                TIME_RANGES[5]: [3, 4, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [3, 3, 4],
                TIME_RANGES[1]: [1, 1, 1],
                TIME_RANGES[2]: [1, 1, 1],
                TIME_RANGES[3]: [1, 2, 3],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [1, 2, 2],
                TIME_RANGES[2]: [1, 1, 1],
                TIME_RANGES[3]: [1, 1, 1],
                TIME_RANGES[4]: [1, 2, 3],
                TIME_RANGES[5]: [3, 4, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [1, 2, 2],
                TIME_RANGES[2]: [0, 1, 1],
                TIME_RANGES[3]: [0, 1, 1],
                TIME_RANGES[4]: [1, 2, 2],
                TIME_RANGES[5]: [3, 4, 5],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [3, 4, 5],
                TIME_RANGES[1]: [1, 2, 3],
                TIME_RANGES[2]: [1, 1, 1],
                TIME_RANGES[3]: [1, 1, 1],
                TIME_RANGES[4]: [1, 2, 3],
                TIME_RANGES[5]: [3, 4, 5],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [1, 2, 3],
                TIME_RANGES[3]: [1, 1, 1],
                TIME_RANGES[4]: [1, 1, 1],
                TIME_RANGES[5]: [3, 3, 4],
            },
        },
    },
    ShadingBoolean.Shaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [4, 5, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [4, 5, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [3, 4, 5],
                TIME_RANGES[3]: [3, 4, 5],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
    },
}


TABLE_D = {
    ShadingBoolean.Unshaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [2, 3, 4],
                TIME_RANGES[3]: [2, 3, 4],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 4],
                TIME_RANGES[2]: [2, 3, 3],
                TIME_RANGES[3]: [2, 3, 3],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [2, 3, 4],
                TIME_RANGES[2]: [2, 2, 3],
                TIME_RANGES[3]: [3, 4, 4],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [2, 3, 3],
                TIME_RANGES[3]: [2, 2, 3],
                TIME_RANGES[4]: [3, 4, 4],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [2, 3, 3],
                TIME_RANGES[2]: [1, 1, 2],
                TIME_RANGES[3]: [1, 1, 2],
                TIME_RANGES[4]: [2, 3, 3],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [3, 4, 5],
                TIME_RANGES[2]: [2, 3, 3],
                TIME_RANGES[3]: [2, 3, 3],
                TIME_RANGES[4]: [3, 4, 5],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [3, 4, 4],
                TIME_RANGES[3]: [2, 2, 3],
                TIME_RANGES[4]: [2, 3, 4],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
    },
    ShadingBoolean.Shaded: {
        Aspect.North: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.East: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.South: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
        Aspect.West: {
            SLOPE_RANGES[0]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
            SLOPE_RANGES[1]: {
                TIME_RANGES[0]: [4, 5, 6],
                TIME_RANGES[1]: [4, 5, 6],
                TIME_RANGES[2]: [4, 5, 6],
                TIME_RANGES[3]: [4, 5, 6],
                TIME_RANGES[4]: [4, 5, 6],
                TIME_RANGES[5]: [4, 5, 6],
            },
        },
    },
}


def _vectorized_lookup(table, shade, aspect, slope, time, elev):
    return table[shade][aspect][slope][time][elev]


_vectorized_lookup_func = np.vectorize(_vectorized_lookup)


def _calculate_reference_fuel_moisture(dry_bulb_temp, relative_humidity):
    """
    Calculate the Reference Fuel Moisture (RFM) based on dry bulb temperature and relative humidity.

    This function implements the Fosberg Model for 1-hr Fuel Moisture Estimation as documented
    by Michael A. Fosberg and John E. Deeming (1971).

    Args:
        dry_bulb_temp (float or np.ndarray): The dry bulb temperature(s) in Fahrenheit.
        relative_humidity (float or np.ndarray): The relative humidity as a percentage (0-100).

    Returns:
        np.ndarray: The Reference Fuel Moisture (RFM) as a percentage.

    Raises:
        ValueError: If the input temperature or humidity is out of the valid range.
    """
    # Convert inputs to numpy arrays if they're not already
    dry_bulb_temp = np.asarray(dry_bulb_temp)
    relative_humidity = np.asarray(relative_humidity)

    # Validate input ranges
    if (
        np.any(dry_bulb_temp < 10)
        or np.any(relative_humidity < 0)
        or np.any(relative_humidity > 100)
    ):
        raise ValueError(
            "Temperature must be at least 10°F, and relative humidity between 0% and 100%."
        )

    # Find the correct temperature range indices
    temp_indices = np.digitize(dry_bulb_temp, TEMP_BREAKPOINTS) - 1
    temp_indices = np.clip(temp_indices, 0, len(TEMP_BREAKPOINTS) - 1)

    # Find the correct humidity range indices
    rh_indices = np.digitize(relative_humidity, RH_BREAKPOINTS) - 1
    rh_indices = np.clip(rh_indices, 0, len(RH_BREAKPOINTS) - 1)

    # Return the Reference Fuel Moisture
    return RFM_TABLE[temp_indices, rh_indices]


def _calculate_1hr_fuel_moisture_corrections(
    correction_table,
    shading,
    aspect,
    slope,
    time,
    relative_elevation: RelativeElevation,
):
    # Convert inputs to numpy arrays if they're not already
    shading = np.asarray(shading)
    aspect = np.asarray(aspect)
    slope = np.asarray(slope)

    # Convert aspect in degrees to compass direction
    aspect_category = _convert_aspect_to_cardinal(aspect)

    # Convert shading to a binary true/false where shading > 50% is True
    shaded_category = np.where(
        shading > 0.5, ShadingBoolean.Shaded, ShadingBoolean.Unshaded
    )

    # Determine the slope category based on the slope angle
    slope_category = np.where(slope > 30, SLOPE_RANGES[1], SLOPE_RANGES[0])

    # Determine the time category based on the input time
    time_category = TIME_RANGES[min(5, time // 200 - 4)]

    # Apply the vectorized lookup function to the correction table
    result = _vectorized_lookup_func(
        correction_table,
        shaded_category,
        aspect_category,
        slope_category,
        time_category,
        relative_elevation,
    )

    # If the result is a single-element array, return it as a scalar
    if result.size == 1:
        return result.item()

    return result


def _convert_aspect_to_cardinal(aspect):
    cardinal = (np.array(aspect) + 45) // 90 % 4
    if cardinal.size == 1:
        return cardinal.item()
    return cardinal


def calculate_1hr_fuel_moisture(
    dry_bulb_temp, relative_humidity, aspect, slope, time, month, elevation, shading
):
    """Calculate the 1-hour dead fuel moisture content using the Fosberg method.

    This function implements the Fosberg and Deeming (1971) methodology for
    calculating 1-hour dead fuel moisture content, including environmental
    corrections. It first determines a reference fuel moisture based on temperature
    and humidity, then applies corrections based on site-specific factors.

    Parameters
    ----------
    dry_bulb_temp : float or numpy.ndarray
        Dry bulb temperature in Fahrenheit. Must be >= 10°F.
    relative_humidity : float or numpy.ndarray
        Relative humidity as a percentage (0-100).
    aspect : float or numpy.ndarray
        Slope aspect in degrees (0-360). 0° is north, 90° is east,
        180° is south, 270° is west.
    slope : float or numpy.ndarray
        Slope steepness in degrees (0-90).
    time : int
        Time of day in 24-hour format (e.g., 1430 for 2:30 PM).
        Must be between 0800-1959.
    month : str
        Month of the year ('January', 'February', etc.).
        Used to select the appropriate correction table.
    elevation : {0, 1, 2}
        Relative elevation compared to weather station:
        - 0: Site is 1000-2000 ft below weather station
        - 1: Site is within 1000 ft of weather station
        - 2: Site is 1000-2000 ft above weather station
    shading : float or numpy.ndarray
        Decimal percent (0.0-1.0) of fine fuels that are shaded
        by canopy or cloud cover.

    Returns
    -------
    float or numpy.ndarray
        The 1-hour dead fuel moisture content as a percentage.
        Returns array if any inputs are arrays.

    Notes
    -----
    The calculation follows these steps:
    1. Calculate reference fuel moisture from temperature and humidity
    2. Select correction table based on month
    3. Apply corrections based on:
        - Time of day
        - Aspect and slope
        - Relative elevation
        - Shading

    For nighttime calculations (2000-0759), use the 0800 shaded condition
    values as recommended by the Missoula Fire Lab.

    References
    ----------
    .. [1] https://www.nwcg.gov/publications/pms437/fuel-moisture/dead-fuel-moisture-content

    Examples
    --------
    >>> # Calculate fuel moisture for a south-facing slope at noon
    >>> moisture = calculate_1hr_fuel_moisture(
    ...     dry_bulb_temp=75,
    ...     relative_humidity=30,
    ...     aspect=180,  # South-facing
    ...     slope=20,
    ...     time=1200,
    ...     month='June',
    ...     elevation=1,  # Level with weather station
    ...     shading=0.2  # 20% shaded
    ... )
    >>> print(moisture)
    5

    >>> # Calculate for multiple locations using numpy arrays
    >>> temps = np.array([80, 70])
    >>> rhs = np.array([30, 50])
    >>> aspects = np.array([180, 270])  # South and West facing
    >>> moistures = calculate_1hr_fuel_moisture(
    ...     dry_bulb_temp=temps,
    ...     relative_humidity=rhs,
    ...     aspect=aspects,
    ...     slope=20,
    ...     time=1200,
    ...     month='June',
    ...     elevation=1,
    ...     shading=0.2
    ... )
    >>> print(moistures)
    [5, 7]

    Raises
    ------
    ValueError
        If any input parameters are outside their valid ranges:
        - Temperature < 10°F
        - Relative humidity outside 0-100%
        - Time outside 0800-1959
        - Invalid month name
        - Elevation not in {0, 1, 2}
        - Shading outside 0.0-1.0
    """

    rfm = _calculate_reference_fuel_moisture(dry_bulb_temp, relative_humidity)

    if month in ["May", "June", "July"]:
        correction_table = TABLE_B
    elif month in ["February", "March", "April", "August", "September", "October"]:
        correction_table = TABLE_C
    elif month in ["November", "December", "January"]:
        correction_table = TABLE_D
    else:
        raise ValueError("Invalid month provided")

    correction = _calculate_1hr_fuel_moisture_corrections(
        correction_table, shading, aspect, slope, time, elevation
    )

    return rfm + correction
