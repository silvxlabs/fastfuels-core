"""Unit conversion factors for the imperial sources the allometry uses.

Brown (1978), the FuelCalc tables and the FVS crown-width fits work in
inches, feet and pounds; FastFuels works in centimetres, metres and
kilograms. These are the exact definitions of the international inch,
foot and avoirdupois pound.
"""

IN_TO_CM = 2.54
CM_TO_IN = 1.0 / IN_TO_CM
FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
LB_TO_KG = 0.45359237
