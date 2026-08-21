"""Unit conversion via pint, loaded lazily.

Use :func:`conversion_factor` for scalar factors applied to numpy
arrays — it avoids wrapping large arrays in Quantities while keeping
every conversion constant derived from pint rather than hand-typed.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def unit_registry():
    """The shared pint UnitRegistry, constructed on first use."""
    import pint

    return pint.UnitRegistry()


@lru_cache(maxsize=None)
def conversion_factor(source: str, target: str) -> float:
    """Multiplicative factor converting ``source`` units to ``target``."""
    return unit_registry().Quantity(1.0, source).to(target).magnitude
