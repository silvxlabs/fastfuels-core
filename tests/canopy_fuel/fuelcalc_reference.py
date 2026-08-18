"""Reference implementation of FuelCalc's canopy fuel arithmetic.

Transcribed line-for-line from the FuelCalc 1.8 C source (FC_DLL, DLL
version 1.6) so the module can serve as an oracle in parity tests. It
reproduces what the binary *does*, including behaviour we deliberately
diverge from and two outright bugs; nothing here should be imported by
library code.

Provenance, by section:

- ``BT`` / :func:`bt_eq` / :func:`bt_get_wc`
  ``FC_DLL/NC_BM.C`` — Brown (1978) crown weight and accumulative
  proportions, table ``sr_BT[]`` and functions ``BT_Eq``/``BT_GetWC``.
  Field order follows ``d_BT`` in ``NC_BM.H``.
- ``SL`` / :func:`sl_pwc` / :func:`sl_pc`
  ``FC_DLL/NC_BMSL.C`` — Snell & Little (1983) hardwoods, table
  ``sr_SL[]`` and functions ``SL_pWC``/``SL_PC``.
- :func:`available_canopy_fuel_lb`
  ``FC_DLL/NC_PTL.C:1408`` ``_AvailFuel()`` — foliage plus half the
  0-1/4 in branchwood.
- :func:`crown_fraction` / :func:`vd_calc`
  ``FC_DLL/NC_VD.C`` — the ``VD_*`` cubics and ``VD_Calc`` with its
  ``_Bot``/``_Top``/``_Mid`` case analysis, in that test order.
- :func:`bulk_density`
  ``FC_DLL/NC_PTL.C:613-698`` ``_BulkDensity()`` — the running mean, the
  threshold rule, and the two raw-profile clamps on CBH and SH.

Two deliberate departures from a literal transcription, both noted where
they occur: this module works in SI (a bulk-density profile in kg m^-3
and depths in metres) rather than lb ac^-1 and feet, because FuelCalc's
conversion helper ``lbAc_To_kgm3`` lives in ``nc_util.c``, which is not
part of the distributed source; and every quantity the running mean and
threshold touch is linear in those units, so the conversion cancels.
Diameters stay in inches, matching Brown and Snell & Little.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------
# Brown (1978): NC_BM.C sr_BT[]
# fields: (A, B, LoLim, LoVal, HiLim, HiVal, i_EF)
# --------------------------------------------------------------------
BT: dict[str, dict[str, tuple[float, ...]]] = {
    "PP": {
        "Tot": (0.2680, 2.0740, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.558, -0.0475, 0.0, 0.0, 0.0, 0.0, 3),
        "Twg": (0.625, -0.0511, 0.0, 0.0, 0.0, 0.0, 3),
        "1in": (0.985, -0.0310, 1.0, 1.0, 0.0, 0.0, 3),
        "3in": (1.083, 0.0131, 6.5, 1.0, 0.0, 0.0, 4),
        "3inP": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "GF": {
        "Tot": (1.3094, 1.6076, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (1.5920, 0.0529, 0.0, 0.0, 36.0, 0.286, 5),
        # NOTE: HiVal here is 0.286 — identical to the Fol HiVal — so the
        # fine fraction collapses to zero above 36 in. Brown's Table 16
        # gives 0.378. Reproduced as-is; this is a FuelCalc bug.
        "Twg": (1.1500, 0.0416, 0.0, 0.0, 36.0, 0.286, 5),
        "1in": (1.0270, 0.0150, 2.9, 1.0, 36.0, 0.488, 4),
        "3in": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "DF": {
        "Tot": (1.1368, 1.5819, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.484, -0.0210, 0.0, 0.0, 36.0, 0.227, 3),
        "Twg": (0.729, -0.0233, 0.0, 0.0, 36.0, 0.315, 3),
        "1in": (1.034, 0.0158, 2.9, 1.0, 36.0, 0.465, 4),
        "3in": (1.022, 0.00182, 14.0, 1.0, 0.0, 0.0, 4),
        "3inP": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "LP": {
        "Tot": (0.1224, 1.8820, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.493, 0.0117, 0.0, 0.0, 0.0, 0.0, 4),
        "Twg": (0.777, 0.0146, 0.0, 0.0, 0.0, 0.0, 4),
        "1in": (1.049, 0.0140, 3.9, 1.0, 0.0, 0.0, 4),
        "3in": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "WL": {
        "Tot": (0.4373, 1.6786, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.3470, -0.0434, 0.0, 0.0, 0.0, 0.0, 3),
        # NOTE: -0.0632 here; the User Guide and Brown's Table 16 as we
        # read it give -0.0362. See EXPECTED_DIVERGENCES in the parity
        # tests.
        "Twg": (0.7450, -0.0632, 0.0, 0.0, 0.0, 0.0, 3),
        "1in": (1.0540, -0.0213, 2.9, 1.0, 0.0, 0.0, 3),
        "3in": (0.9220, -0.7200, 11.0, 1.0, 0.0, 0.0, 6),
        "3inP": (0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "SF": {
        "Tot": (7.345, 1.2550, 0.0, 0.0, 0.0, 0.0, 2),
        "Fol": (0.597, -0.0425, 0.0, 0.0, 0.0, 0.0, 3),
        "Twg": (0.864, -0.0373, 0.0, 0.0, 0.0, 0.0, 3),
        "1in": (1.022, 0.0108, 2.9, 1.0, 0.0, 0.0, 4),
        "3in": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "ES": {
        "Tot": (1.0404, 1.7096, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.5780, -0.0325, 0.0, 0.0, 40.0, 0.158, 3),
        "Twg": (0.8520, -0.0281, 0.0, 0.0, 40.0, 0.277, 3),
        "1in": (1.0380, 0.0154, 2.9, 1.0, 40.0, 0.423, 4),
        "3in": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "WB": {
        "Tot": (-1.0000, 0.8371, 0.0, 0.0, 0.0, 0.0, 2),
        "Fol": (0.5120, -0.0374, 0.0, 0.0, 20.0, 0.242, 3),
        "Twg": (0.8640, -0.0585, 0.0, 0.0, 20.0, 0.268, 3),
        "1in": (1.0770, -0.0238, 3.9, 1.0, 20.0, 0.669, 3),
        "3in": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "WP": {
        "Tot": (0.7276, 1.5497, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.5500, -0.0345, 0.0, 0.0, 0.0, 0.0, 3),
        "Twg": (0.9140, 0.0978, 0.0, 0.0, 0.0, 0.0, 7),
        "1in": (1.0560, -0.0181, 3.9, 1.0, 0.0, 0.0, 3),
        "3in": (0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "WC": {
        "Tot": (0.8815, 1.6389, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.6170, -0.0233, 0.0, 0.0, 0.0, 0.0, 3),
        "Twg": (0.7560, -0.0241, 0.0, 0.0, 0.0, 0.0, 3),
        "1in": (1.0600, -0.0223, 2.9, 1.0, 0.0, 0.0, 3),
        "3in": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
    "WH": {
        "Tot": (0.7218, 1.7502, 0.0, 0.0, 0.0, 0.0, 1),
        "Fol": (0.5470, -0.0370, 0.0, 0.0, 40.0, 0.125, 3),
        "Twg": (0.8350, -0.0380, 0.0, 0.0, 40.0, 0.183, 3),
        "1in": (1.0781, -0.0274, 2.9, 1.0, 40.0, 0.361, 3),
        "3in": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
        "3inP": (1.0000, 0.0000, 0.0, 0.0, 0.0, 0.0, 0),
    },
}

COMPONENT_ORDER = ("Tot", "Fol", "Twg", "1in", "3in", "3inP")


def bt_eq(code: str, component: str, dia_in: float) -> float:
    """``NC_BM.C BT_Eq()``.

    ``"Tot"`` returns crown weight in pounds; every other component
    returns an *accumulative* proportion of that weight.
    """
    a, b, lolim, loval, hilim, hival, form = BT[code][component]
    if dia_in == 0:
        return 0.0
    if form == 0:
        return a
    if form == 1:
        p = math.exp(a + b * math.log(dia_in))
    elif form == 2:
        p = a + b * (dia_in * dia_in)
    elif form == 3:
        p = a * math.exp(b * dia_in)
    elif form == 4:
        p = a - (b * dia_in)
    elif form == 5:
        p = 1.0 / (a + b * dia_in)
    elif form == 6:
        p = a + (b / dia_in)
    elif form == 7:
        p = a - (b * math.sqrt(dia_in))
    else:  # pragma: no cover - "ERROR BT_Eq()" in the C
        raise ValueError(f"unknown equation form {form}")

    if lolim != 0 and dia_in <= lolim:
        return loval
    if hilim != 0 and dia_in > hilim:
        return hival
    return max(p, 0.0)


def bt_get_wc(code: str, component: str, dia_in: float) -> float:
    """``NC_BM.C BT_GetWC()``: component weight in pounds.

    Proportions are accumulative, so every component above foliage is the
    difference against the next smaller one, floored at zero.
    """
    weight = bt_eq(code, "Tot", dia_in)
    p = bt_eq(code, component, dia_in)
    if component == "Fol":
        return p * weight
    previous = COMPONENT_ORDER[COMPONENT_ORDER.index(component) - 1]
    return weight * max(p - bt_eq(code, previous, dia_in), 0.0)


# --------------------------------------------------------------------
# Snell & Little (1983): NC_BMSL.C sr_SL[]
# fields: (A, B, C)
# --------------------------------------------------------------------
SL: dict[str, dict[str, tuple[float, float, float]]] = {
    "RA": {
        "Tot": (-1.3290, 2.6232, 0.0),
        "Fol": (2.7638, 0.2155, 1.3364),
        "Twg": (1.2860, 0.1016, 1.3525),
        "1in": (0.8847, 0.0441, 1.3021),
        "3in": (0.9550, 0.0013, 1.9736),
        "3inP": (0.0, 0.0, 0.0),
    },
    "GC": {
        "Tot": (-0.8032, 2.2699, 0.0),
        "Fol": (1.6048, 0.5630, 0.6828),
        "Twg": (1.0700, 0.2525, 0.7637),
        "1in": (0.7312, 0.1691, 0.6118),
        "3in": (0.9669, 0.0036, 1.1786),
        "3inP": (0.0, 0.0, 0.0),
    },
    "BM": {
        "Tot": (-0.0582, 2.1505, 0.0),
        "Fol": (4.6762, 0.1091, 2.0390),
        "Twg": (3.3212, 0.0777, 2.0496),
        "1in": (0.9341, 0.0158, 2.1627),
        "3in": (0.8625, 0.0093, 1.7070),
        "3inP": (0.0, 0.0, 0.0),
    },
    "MA": {
        "Tot": (-0.7881, 2.4839, 0.0),
        "Fol": (1.6013, 0.3591, 1.3090),
        "Twg": (1.0357, 0.2263, 1.3567),
        "1in": (1.0281, 0.0084, 2.1850),
        "3in": (0.8778, 0.0115, 1.6394),
        "3inP": (0.0, 0.0, 0.0),
    },
    # Tan oak has no 3+ in factors in the book; -999 is FuelCalc's sentinel.
    "TO": {
        "Tot": (-0.3169, 2.2774, 0.0),
        "Fol": (1.7936, 0.5952, 0.7239),
        "Twg": (0.9940, 0.4229, 0.6520),
        "1in": (0.8759, 0.0927, 0.7843),
        "3in": (0.0, 0.0, 0.0),
        "3inP": (-999.0, 0.0, 0.0),
    },
}


def sl_pwc(code: str, component: str, dia_in: float) -> float:
    """``NC_BMSL.C SL_pWC()``: accumulative proportion ``1/(A + B*d**C)``."""
    a, b, c = SL[code][component]
    return 1.0 / (a + b * dia_in**c)


def sl_pc(code: str, component: str, dia_in: float) -> float:
    """``NC_BMSL.C SL_PC()``: proportion for one component alone."""
    a, _, _ = SL[code][component]
    if a == -999.0:
        return 0.0
    if a == 0.0:  # the 3+ in class is whatever is left above 3 in
        previous = COMPONENT_ORDER[COMPONENT_ORDER.index(component) - 1]
        return max(1.0 - sl_pwc(code, previous, dia_in), 0.0)
    p = sl_pwc(code, component, dia_in)
    if component == "Fol":
        return p
    previous = COMPONENT_ORDER[COMPONENT_ORDER.index(component) - 1]
    return max(p - sl_pwc(code, previous, dia_in), 0.0)


# --------------------------------------------------------------------
# unified accessors over both paths
# --------------------------------------------------------------------
BROWN_IDS = tuple(BT)
SNELL_LITTLE_IDS = tuple(SL)
ALL_IDS = BROWN_IDS + SNELL_LITTLE_IDS


def p1(equation_id: str, dia_in: float) -> float:
    """Accumulative foliage proportion of total crown weight."""
    if equation_id in BT:
        return bt_eq(equation_id, "Fol", dia_in)
    return sl_pwc(equation_id, "Fol", dia_in)


def p2(equation_id: str, dia_in: float) -> float:
    """Accumulative foliage plus 0-1/4 in proportion of crown weight."""
    if equation_id in BT:
        return bt_eq(equation_id, "Twg", dia_in)
    return sl_pwc(equation_id, "Twg", dia_in)


def fine_fraction_of_crown(equation_id: str, dia_in: float) -> float:
    """Fine (0-1/4 in) branchwood as a fraction of total crown weight."""
    return max(p2(equation_id, dia_in) - p1(equation_id, dia_in), 0.0)


def available_canopy_fuel_lb(equation_id: str, dia_in: float) -> float:
    """``NC_PTL.C _AvailFuel()``: ``Fol + 0.5 * Twg`` in pounds.

    Before the crown-class factor, which FuelCalc applies to the
    component weights in ``PTL_SetBioMass`` (``NC_PTL.C:867-875``).
    """
    if equation_id in BT:
        return bt_get_wc(equation_id, "Fol", dia_in) + 0.5 * bt_get_wc(
            equation_id, "Twg", dia_in
        )
    a, b, _ = SL[equation_id]["Tot"]
    weight = math.exp(a + b * math.log(dia_in))
    return weight * (
        sl_pc(equation_id, "Fol", dia_in) + 0.5 * sl_pc(equation_id, "Twg", dia_in)
    )


# --------------------------------------------------------------------
# vertical distribution: NC_VD.C
# --------------------------------------------------------------------
VDIST_CUBICS: dict[str, tuple[float, float, float]] = {
    "PP": (0.0, 2.3637, -1.3637),
    "DF": (0.0, 2.3284, -1.3284),
    "LP": (0.0, 1.6045, -0.6045),
    "IC": (0.0, 2.5395, -1.5395),
    "PS": (0.1251, 2.8072, -1.9322),
    "WF": (1.0, 0.0, 0.0),
    "UN": (1.0, 0.0, 0.0),
}


def crown_fraction(vdist_code: str, ph: float) -> float:
    """``NC_VD.C Crown_Fraction()``: cumulative fuel fraction at ``ph``.

    Zero at or below the crown base, as the C does before dispatching.
    """
    if ph <= 0.0:
        return 0.0
    b1, b2, b3 = VDIST_CUBICS[vdist_code]
    return b1 * ph + b2 * ph**2 + b3 * ph**3


def vd_calc(
    vdist_code: str,
    crown_base: float,
    crown_top: float,
    layer_depth: float,
    fuel: float,
    n_layers: int,
) -> list[float]:
    """``NC_VD.C VD_Calc()``: distribute ``fuel`` into fixed-depth layers.

    The three cases are tested in the C's order — ``_Bot``, ``_Top``,
    ``_Mid`` — which matters when a whole crown falls inside one layer:
    ``_Bot`` catches it first and clamps its ratio to 1.0.
    """
    out = [0.0] * n_layers
    layer_bottom, layer_top = 0.0, layer_depth
    index = 0
    while True:
        if layer_bottom >= crown_top:
            break
        span = crown_top - crown_base
        if layer_top > crown_base and layer_bottom < crown_base:  # _Bot
            ph = float("inf") if span == 0 else (layer_top - crown_base) / span
            weight = crown_fraction(vdist_code, min(ph, 1.0))
        elif layer_top > crown_top and layer_bottom < crown_top:  # _Top
            ph = float("inf") if span == 0 else (layer_bottom - crown_base) / span
            weight = 1.0 - crown_fraction(vdist_code, ph)
        elif layer_bottom >= crown_base and layer_top <= crown_top:  # _Mid
            lower = crown_fraction(vdist_code, (layer_bottom - crown_base) / span)
            upper = crown_fraction(vdist_code, (layer_top - crown_base) / span)
            weight = upper - lower
        else:
            weight = 0.0
        out[index] += fuel * weight
        index += 1
        if index == n_layers:
            return out
        layer_bottom += layer_depth
        layer_top += layer_depth
    return out


# --------------------------------------------------------------------
# profile reduction: NC_PTL.C _BulkDensity()
# --------------------------------------------------------------------
RUNNING_MEAN_SPREAD = 5  # e_RASprD, FC_DLL.h:634 — counted in layers
CBD_CRITICAL_THRESHOLD = 0.012  # e_LayOvr, FC_DLL.h:639 — kg m^-3


class BulkDensityResult(dict):
    """CBD (kg m^-3), CBH and SH (m), the threshold used, and the
    smoothed profile — mirroring the fields ``_BulkDensity`` fills in."""

    __getattr__ = dict.__getitem__


def bulk_density(
    profile: list[float],
    layer_depth: float,
    *,
    spread: int = RUNNING_MEAN_SPREAD,
    threshold: float = CBD_CRITICAL_THRESHOLD,
) -> BulkDensityResult:
    """``NC_PTL.C _BulkDensity()`` over a bulk-density profile.

    ``profile[i]`` is the mean density of the layer spanning
    ``[i*layer_depth, (i+1)*layer_depth)``. FuelCalc carries lb ac^-1 per
    layer and converts on read; every step below is linear in that
    conversion, so working in kg m^-3 throughout is equivalent.

    Heights are reported the way the C reports them, as ``(i+1) *
    layer_depth`` — the *top* of the layer — for both CBH and SH.
    """
    n = len(profile)
    half = spread // 2
    n_ra = n + half

    # Running mean. The window is truncated at the ground (blay clamps to
    # 0 and the denominator shrinks) but not above the canopy, where it
    # reads past the end of a zeroed array with the denominator intact.
    smoothed = []
    for i in range(n_ra):
        top = i + half
        bottom = max(i - half, 0)
        total = sum(profile[j] for j in range(bottom, top + 1) if j < n)
        smoothed.append(total / (top - bottom + 1))

    effective = min(max(smoothed) / 10.0, threshold)

    cbd = -1.0  # e_MissCBD
    cbh = sh = None
    for i, value in enumerate(smoothed):
        if value > effective:
            if cbh is None:
                cbh = (i + 1) * layer_depth
            sh = (i + 1) * layer_depth
        cbd = max(cbd, value)

    # CBH may not sit below the lowest layer that actually holds fuel,
    # and SH may not sit above the highest — the raw profile, not the
    # smoothed one, is what bounds them.
    occupied = [i for i, v in enumerate(profile) if v > 0]
    if occupied:
        lowest = (occupied[0] + 1) * layer_depth
        highest = (occupied[-1] + 1) * layer_depth
        if cbh is None or lowest > cbh:
            cbh = lowest
        if sh is None or highest < sh:
            sh = highest

    return BulkDensityResult(
        cbd=cbd, cbh=cbh, sh=sh, threshold=effective, smoothed=smoothed
    )
