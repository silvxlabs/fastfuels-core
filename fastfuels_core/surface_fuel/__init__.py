"""Gridded surface fuel properties, such as fuel loads, depth, moisture and curing.

Each method builds a 2D grid per band. Methods can share building blocks:
one sets where values land across the grid, another sets what the values
are, so a new method can reuse whichever parts it needs.

========================= ============================================
``distribution_grids``    grids filled from a distribution per band
``distributions``         distribution types: percentiles -> values
``patterns``              spatial patterns: cells -> percentiles
``correlation``           correlation between bands
========================= ============================================
"""

from fastfuels_core.surface_fuel.distribution_grids import build_distribution_grids

__all__ = [
    "build_distribution_grids",
]
