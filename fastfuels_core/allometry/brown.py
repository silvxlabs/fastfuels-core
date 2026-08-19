"""Crown component proportions from Brown (1978) and Snell & Little (1983).

Primary sources, transcribed and verified against the original tables:

- Brown, J.K. 1978. Weight and Density of Crowns of Rocky Mountain
  Conifers. USDA For. Serv. Res. Pap. INT-197. **Table 1** (p. 10):
  live crown weight of dominant and codominant trees > 1 in dbh, and
  **Table 16** (p. 53): accumulative proportions of foliage and
  branchwood by size class for the same trees, for 11 conifer species.
- Snell, J.A.K. & Little, S.N. 1983. Predicting Crown Weight and Bole
  Volume of Five Western Hardwoods. PNW-GTR-151. **Table 3** (p. 6):
  cumulative fractions of live crown component weights for five western
  hardwoods.

Both sources define the same accumulative structure over total live
crown weight: P1 is the foliage fraction, P2 the foliage *plus* 0-1/4 in
branchwood fraction, so the fine (0-1/4 in) branchwood fraction is
``P2 - P1`` — the individual fractions "must be subtracted from one
another" (SL-83 p. 4).

Species are keyed by FuelCalc's two-letter equation Ids (the FuelCalc
Default Equation Table maps FIA species codes to Ids; see
``fastfuels_core.canopy_fuel.ref_data``). Brown's own species letters
map onto them as L->WL, S->ES, AF->SF, WBP->WB, C->WC. Diameters are in
inches throughout, matching the sources.

The AL Id is a FuelCalc cross-reference that borrows whitebark pine's P1
and western larch's P2. Neither primary source sanctions that pairing,
so it is kept only because nothing resolves to it -- subalpine larch
uses the WL Ids for every component. The QA Id was the same borrow and
*was* reachable, via SPCD 746; it has been dropped rather than left to
return an unsourced number, so quaking aspen now raises like any other
species without equations. Its real source is Loomis & Roussopoulos
1978 (NC-156), not yet implemented.
"""

from __future__ import annotations

import numpy as np

# Proportions are clipped to [0, 0.95]: the linear and sqrt forms go
# negative at large diameters, and FuelCalc's own code caps proportions
# at 0.95. The upper clip also bounds the fine-share denominator away
# from zero.
PROPORTION_MAX = 0.95


def _exponential(dia, a, b):
    """a * exp(b * dia)"""
    return a * np.exp(b * dia)


def _linear(dia, a, b):
    """a + b * dia"""
    return a + b * dia


def _reciprocal(dia, a, b, c=1.0):
    """1 / (a + b * dia**c)"""
    return 1.0 / (a + b * dia**c)


def _sqrt_linear(dia, a, b):
    """a + b * sqrt(dia)"""
    return a + b * np.sqrt(dia)


def _log_log(dia, a, b):
    """exp(a + b * ln(dia))"""
    return np.exp(a + b * np.log(dia))


def _quadratic(dia, a, b):
    """a + b * dia**2"""
    return a + b * dia * dia


def _douglas_fir_crown_weight(dia, a, b, c, d, break_in):
    """Brown Table 1's two-part Douglas-fir fit.

    ``exp(a + b*ln(dia))`` below the break, ``c*dia**2 + d`` at or above
    it. The two branches agree to 0.15% where they meet, which is what
    identifies them as halves of one predictor rather than alternatives.
    """
    return np.where(dia < break_in, _log_log(dia, a, b), c * dia * dia + d)


# Live crown weight (foliage plus all branchwood) in POUNDS, for
# dominant and codominant trees over 1 in dbh: Brown 1978 Table 1,
# p. 10. Brown fits several predictors per species; these are the
# diameter-only forms, the ones FuelCalc uses and the ones Gray &
# Reinhardt (2003) evaluated. Species without a diameter-only entry in
# Brown -- the Snell & Little hardwoods -- are absent by design; see
# crown_weight().
CROWN_WEIGHT_EQUATIONS = {
    "GF": (_log_log, {"a": 1.3094, "b": 1.6076}),
    "WL": (_log_log, {"a": 0.4373, "b": 1.6786}),
    "ES": (_log_log, {"a": 1.0404, "b": 1.7096}),
    "SF": (_quadratic, {"a": 7.345, "b": 1.255}),
    "LP": (_log_log, {"a": 0.1224, "b": 1.8820}),
    "WP": (_log_log, {"a": 0.7276, "b": 1.5497}),
    "WB": (_quadratic, {"a": -1.00, "b": 0.8371}),
    "WC": (_log_log, {"a": 0.8815, "b": 1.6389}),
    "PP": (_log_log, {"a": 0.2680, "b": 2.0740}),
    "DF": (
        _douglas_fir_crown_weight,
        {"a": 1.1368, "b": 1.5819, "c": 1.0237, "d": -20.74, "break_in": 17.0},
    ),
    "WH": (_log_log, {"a": 0.7218, "b": 1.7502}),
}


# P1: foliage fraction of total live crown weight.
# Conifers: Brown 1978 Table 16. Hardwoods: SL-83 Table 3 f(1).
P1_EQUATIONS = {
    "GF": (_reciprocal, {"a": 1.592, "b": 0.0529}),
    "WL": (_exponential, {"a": 0.347, "b": -0.0434}),
    "ES": (_exponential, {"a": 0.578, "b": -0.0325}),
    "SF": (_exponential, {"a": 0.597, "b": -0.0425}),
    "LP": (_linear, {"a": 0.493, "b": -0.0117}),
    "WP": (_exponential, {"a": 0.550, "b": -0.0345}),
    "WB": (_exponential, {"a": 0.512, "b": -0.0374}),
    "WC": (_exponential, {"a": 0.617, "b": -0.0233}),
    "PP": (_exponential, {"a": 0.558, "b": -0.0475}),
    "DF": (_exponential, {"a": 0.484, "b": -0.0210}),
    "WH": (_exponential, {"a": 0.547, "b": -0.0370}),
    "RA": (_reciprocal, {"a": 2.7638, "b": 0.2155, "c": 1.3364}),
    "GC": (_reciprocal, {"a": 1.6048, "b": 0.5630, "c": 0.6828}),
    "BM": (_reciprocal, {"a": 4.6762, "b": 0.1091, "c": 2.0390}),
    "MA": (_reciprocal, {"a": 1.6013, "b": 0.3591, "c": 1.3090}),
    "TO": (_reciprocal, {"a": 1.7936, "b": 0.5952, "c": 0.7239}),
    # FuelCalc cross-reference Id: whitebark pine's foliage fraction.
    "AL": (_exponential, {"a": 0.512, "b": -0.0374}),
}

# P2: foliage plus 0-1/4 in branchwood fraction of total live crown
# weight (accumulative). Same sources, class 2 rows.
P2_EQUATIONS = {
    "GF": (_reciprocal, {"a": 1.150, "b": 0.0416}),
    "WL": (_exponential, {"a": 0.745, "b": -0.0362}),
    "ES": (_exponential, {"a": 0.852, "b": -0.0281}),
    "SF": (_exponential, {"a": 0.864, "b": -0.0373}),
    "LP": (_linear, {"a": 0.777, "b": -0.0146}),
    "WP": (_sqrt_linear, {"a": 0.914, "b": -0.0978}),
    "WB": (_exponential, {"a": 0.864, "b": -0.0585}),
    "WC": (_exponential, {"a": 0.756, "b": -0.0241}),
    "PP": (_exponential, {"a": 0.625, "b": -0.0511}),
    "DF": (_exponential, {"a": 0.729, "b": -0.0233}),
    "WH": (_exponential, {"a": 0.835, "b": -0.0380}),
    "RA": (_reciprocal, {"a": 1.2860, "b": 0.1016, "c": 1.3525}),
    "GC": (_reciprocal, {"a": 1.0700, "b": 0.2525, "c": 0.7637}),
    "BM": (_reciprocal, {"a": 3.3212, "b": 0.0777, "c": 2.0496}),
    "MA": (_reciprocal, {"a": 1.0357, "b": 0.2263, "c": 1.3567}),
    "TO": (_reciprocal, {"a": 0.9940, "b": 0.4229, "c": 0.6520}),
    # FuelCalc cross-reference Id: western larch's P2.
    "AL": (_exponential, {"a": 0.745, "b": -0.0362}),
}

# Large-diameter constant overrides from Brown Table 16's Conditions
# column: {Id: (break_in, P1, P2)}. The FuelCalc guide reprints only the
# GF and WH ones; DF, ES, and WB are in the primary source.
LARGE_DIAMETER_CONSTANTS = {
    "GF": (36.0, 0.286, 0.378),
    "DF": (36.0, 0.227, 0.315),
    "ES": (40.0, 0.158, 0.277),
    "WH": (40.0, 0.125, 0.183),
    "WB": (20.0, 0.242, 0.268),
}

# Brown's PP condition, Table 16 (p. 53): P2 = P1 + 0.01. The printed
# inequality reads "If d <=31 in", but that has to be a typo for ">":
#   - the two fitted curves cross at d = ln(0.625/0.558)/0.0036 = 31.5 in,
#     and only above the crossing does the fine fraction go negative, so
#     only above it does P2 need an override at all;
#   - applied below 31 in the override would supersede a P2 equation
#     Brown fits with R2 = 0.89 over almost the whole diameter range,
#     and would contradict his own Table 15, which gives ponderosa a
#     fine fraction of 0.14 -- not 0.01 -- at d <= 1 in;
#   - every other Conditions entry that patches P1 or P2 is a high-side
#     "d >" rule; the "d <=" rules all patch P3 or P4 to 1.0.
# So the override applies above the crossing, where it repairs the sign.
PP_CROSSOVER_IN = 31.0


def _evaluate(equations: dict, equation_id: np.ndarray, dia: np.ndarray) -> np.ndarray:
    """Evaluate per-Id proportion equations over aligned tree arrays.

    Ids without an equation raise — mirroring FuelCalc's treelist error
    for species without assigned equations (FuelCalc guide p. 16). That
    currently covers the pinyon/juniper Ids (Grier 1992, structurally
    different) and the eastern "newer species" Ids (FuelCalc guide p. 74
    cites their sources without printing the equations).
    """
    missing = sorted(set(np.unique(equation_id)) - set(equations))
    if missing:
        raise ValueError(
            f"No crown proportion equations for FuelCalc equation Id(s) "
            f"{missing}. FuelCalc errors on species without assigned "
            f"equations; supply per-tree fuel via fuel_column to bypass "
            f"allometry for these species."
        )
    result = np.empty(dia.shape, dtype=np.float64)
    for eq_id in np.unique(equation_id):
        form, params = equations[eq_id]
        mask = equation_id == eq_id
        result[mask] = form(dia[mask], **params)
    return np.clip(result, 0.0, PROPORTION_MAX)


def foliage_fraction(equation_id: np.ndarray, dia_in: np.ndarray) -> np.ndarray:
    """P1: foliage fraction of total live crown weight, by equation Id."""
    p1 = _evaluate(P1_EQUATIONS, equation_id, dia_in)
    for eq_id, (break_in, p1_const, _) in LARGE_DIAMETER_CONSTANTS.items():
        p1 = np.where((equation_id == eq_id) & (dia_in > break_in), p1_const, p1)
    return p1


def foliage_plus_fine_fraction(
    equation_id: np.ndarray, dia_in: np.ndarray
) -> np.ndarray:
    """P2: foliage plus fine (0-1/4 in) branchwood fraction of total
    live crown weight (accumulative), by equation Id."""
    p2 = _evaluate(P2_EQUATIONS, equation_id, dia_in)
    for eq_id, (break_in, _, p2_const) in LARGE_DIAMETER_CONSTANTS.items():
        p2 = np.where((equation_id == eq_id) & (dia_in > break_in), p2_const, p2)
    # PP guard: past the curve crossing, hold the fine fraction at 0.01.
    pp_past_crossover = (equation_id == "PP") & (dia_in > PP_CROSSOVER_IN)
    if pp_past_crossover.any():
        pp_p1 = 0.558 * np.exp(-0.0475 * dia_in) + 0.01
        p2 = np.where(pp_past_crossover, pp_p1, p2)
    return p2


def fine_branchwood_share(
    fol_id: np.ndarray, twig_id: np.ndarray, dia_in: np.ndarray
) -> np.ndarray:
    """Fine (0-1/4 in) share of total branchwood weight.

    ``(P2 - P1) / (1 - P1)``: the fine fraction of crown weight over the
    branchwood fraction of crown weight. ``fol_id`` resolves P1 and
    ``twig_id`` resolves P2 (they differ only for FuelCalc's
    cross-referenced Ids); the difference is floored at zero since
    cross-Id subtraction can dip negative.

    Parameters
    ----------
    fol_id, twig_id : numpy.ndarray
        FuelCalc equation Ids from the species table's FOL and TWIG
        columns, aligned with ``dia_in``.
    dia_in : numpy.ndarray
        Diameter at breast height, inches.

    Returns
    -------
    numpy.ndarray
        Fine share in [0, 1].
    """
    p1 = foliage_fraction(fol_id, dia_in)
    p2 = foliage_plus_fine_fraction(twig_id, dia_in)
    fine = np.maximum(p2 - p1, 0.0)
    return np.minimum(fine / (1.0 - p1), 1.0)


def crown_weight(equation_id: np.ndarray, dia_in: np.ndarray) -> np.ndarray:
    """Total live crown weight (pounds), by equation Id.

    Brown 1978 Table 1 (p. 10), diameter-only forms, for dominant and
    codominant trees over 1 in dbh. Foliage plus every branchwood size
    class; multiply by :func:`foliage_fraction` for the foliage alone.

    Two of Brown's fits are quadratics that go negative at the small end
    of their range, so the result is floored at zero. Below 1 in dbh the
    equations are extrapolations -- Brown fits trees under an inch
    separately, in his Table 19, which is not implemented here.

    Parameters
    ----------
    equation_id : numpy.ndarray
        FuelCalc equation Ids from the species table's TOTAL column.
    dia_in : numpy.ndarray
        Diameter at breast height, inches.

    Returns
    -------
    numpy.ndarray
        Crown weight in pounds, aligned with ``dia_in``.

    Raises
    ------
    ValueError
        For Ids Brown has no diameter-only equation for. That is every
        Snell & Little hardwood: their crown weights come from PNW-GTR-
        151, a different source, and only their *proportions* are
        implemented today.
    """
    equation_id = np.asarray(equation_id)
    dia = np.asarray(dia_in, dtype=np.float64)
    missing = sorted(
        str(i) for i in np.unique(equation_id) if i not in CROWN_WEIGHT_EQUATIONS
    )
    if missing:
        raise ValueError(
            f"Brown 1978 Table 1 has no diameter-only crown weight "
            f"equation for Id(s) {missing}. Brown covers Rocky Mountain "
            f"conifers; the hardwood Ids would need Snell & Little "
            f"(1983). Use the nsvb equations, or supply per-tree fuel "
            f"via fuel_column, for these species."
        )
    result = np.empty(dia.shape, dtype=np.float64)
    for eq_id in np.unique(equation_id):
        form, params = CROWN_WEIGHT_EQUATIONS[eq_id]
        mask = equation_id == eq_id
        result[mask] = form(dia[mask], **params)
    return np.maximum(result, 0.0)
