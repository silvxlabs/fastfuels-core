"""
Solver behind :class:`fastfuels_core.treatments.ThinToClumpPattern`.

Simulated annealing over keep/cut on 200 m tiles, for the ICO pattern
(individuals, clumps and openings) of Churchill et al. (2016), *The ICO
approach to quantifying and restoring forest spatial pattern:
Implementation guide*, version 3.0. See the class for the targets; this
module holds the numba kernel and the tiled driver that feeds it.

Importing this module compiles the kernel: every function called from
Python has an explicit signature, and all are ``cache=True``, so a
Docker build can compile once by importing it. ``treatments`` imports it
lazily, inside ``apply``.

Energy of one tile, in tree units::

    E = W * (w_D * density miss + sum of bin misses
             + open-space miss / grid cells per tree)
        - D * (mean score * rho + w_L * pair ratio)

where each miss is 0 inside its tolerance band, ``D`` is the tile's
density target in trees, ``rho = min(1, N / D)`` and the pair ratio is
the number of kept pairs within the clump distance relative to random
placement. Moves: flip one tree, swap a kept tree with a cut tree within
the swap radius, relocate a clump (dissolve it and build one elsewhere),
dissolve a clump, build a clump. Temperature falls geometrically from
``T_START`` to ``T_END`` over ``SWEEPS`` sweeps and the best state seen
at the end of a sweep is kept.

With a cap on clump size, a clump over the cap falls in one more class,
with share 0 and no tolerance, so each of its trees counts as a miss (the
prototype's forbidden 26+ class). That class is never built, and the
build and relocate moves draw no clump larger than the cap. Without a cap
(``NO_CAP``) the class stays empty and the result is the same as before
the cap existed.

Tiles are solved in four passes of a 2 x 2 checkerboard, tiles of one
pass in parallel. A tree within the clump distance of a tree kept in an
earlier pass, or of a keep tree in another tile, is blocked, so no clump
crosses a tile edge. Each tile's unmet density and per-bin deficits are
carried to its unsolved neighbours. Each tile's seed is derived from the
seed and the tile index, and tiles are assigned to threads in fixed
bins, so the result does not depend on the thread count.

Everything below in capitals is a solver setting, not a public
parameter. The kernel reads none of them as globals (numba would freeze
them at compile time): the driver passes them in, so tests and
benchmarks can patch them.
"""

from __future__ import annotations

import heapq

import numba
import numpy as np
from numba import njit, prange, types

# ---------------------------------------------------------------------------
# Solver settings. Not public parameters; tests and benchmarks may patch them.
# ---------------------------------------------------------------------------

#: Cooling schedule: start and end temperature, and sweeps (moves per tree).
T_START = 1.0
T_END = 0.01
SWEEPS = 100

#: ``W``: weight of the target misses against the leave-tree score.
TARGET_WEIGHT = 10.0

#: ``w_D``: extra weight on the density miss, so an infeasible pattern
#: gives way before density does.
DENSITY_WEIGHT = 10.0

#: Share of each move. Must sum to 1.
MOVE_MIX = {
    "relocate": 0.15,
    "dissolve": 0.085,
    "build": 0.085,
    "swap": 0.34,
    "flip": 0.34,
}

#: A kept tree swaps with a cut tree at most this far away, in meters
#: (or the clump distance plus 4 m, if that is larger).
SWAP_RADIUS = 10.0

#: Side of a tile, in meters.
TILE_SIZE = 200.0

#: Cell of the open-space grid, in meters.
OPEN_SPACE_CELL = 3.0

#: Cell of the treed-area mask, in meters (the TreeMap pixel). The stand
#: is the cells of this size that hold at least one stand tree.
FOREST_CELL = 30.0

# ---------------------------------------------------------------------------
# Fixed by the targets' definitions; compile-time constants of the kernel.
# ---------------------------------------------------------------------------

#: Clump-size bins: 1, 2-4, 5-9, 10-15 and 16+ trees (guide Table 2).
BIN_LABELS = ("1", "2-4", "5-9", "10-15", "16+")
NK = 5
#: Classes the kernel tracks: the five bins, then clumps over the cap
#: (``max_clump_size``). That class has share 0 and no tolerance, so
#: every tree in it is a miss, like the prototype's forbidden 26+ class.
NC = NK + 1
#: The cap that means no cap.
NO_CAP = 1 << 40
BIN_LOW = np.array([1, 2, 5, 10, 16], np.int64)
#: Largest clump the build and relocate moves draw for each bin. Clumps
#: of the open top bin can still grow past it by single-tree moves.
BUILD_HIGH = np.array([1, 4, 9, 15, 20], np.int64)
#: Average clump size of each bin (guide Table 3).
MEAN_CLUMP_SIZE = np.array([1.0, 3.0, 7.0, 12.0, 17.0])

#: Open space is the share of the stand more than this far from a kept
#: tree (F > 9 m).
OPEN_SPACE_DISTANCE = 9.0
_OPEN_R2 = OPEN_SPACE_DISTANCE**2

#: Reference model of Churchill (2013), Fig. 3:
#: F>9 = 1.841 + 0.144 ln(mean clump size) - 0.396 ln(TPH).
F9_INTERCEPT = 1.841
F9_CLUMP_SIZE = 0.144
F9_TPH = -0.396

# Layout of a tile's target vector.
I_D = 0  # density target, in tree units
I_TOL_D = 1  # density tolerance, in tree units
I_TOL_S = 2  # share tolerance, absolute
I_SHARE = 3  # NC shares (the last, over the cap, is 0)
I_OFFSET = I_SHARE + NC  # NC per-class tree deficits carried from neighbours
I_F = I_OFFSET + NC  # open-space target, fraction; < 0 is off
I_TOL_F = I_F + 1  # open-space tolerance, absolute
I_CELLS = I_F + 2  # open-space cells in the tile
I_CELLS_PER_TREE = I_F + 3  # grid cells one tree covers
I_HA = I_F + 4  # tile's treed area, ha
I_WD = I_F + 5  # w_D
I_WL = I_F + 6  # w_L
I_PAIR_AREA = I_F + 7  # pi * clump distance^2, m2
I_CAP = I_F + 8  # largest clump allowed, trees (NO_CAP = none)
TG = I_F + 9

# A cover count that marks a grid cell outside the tile's open-space region.
_BIG = 1 << 30

# Signature shorthands.
_f8 = types.float64
_i8 = types.int64
_f8v = types.float64[::1]
_i8v = types.int64[::1]
_i4v = types.int32[::1]
_b1v = types.boolean[::1]
_u1v = types.uint8[::1]
_f8m = types.float64[:, ::1]


# The kernel keeps the prototype's compact layout: its moves pass 20-odd
# state arrays positionally, which black would put one per line.
# fmt: off
# ===========================================================================
# Annealer
# ===========================================================================
# State shared by the moves:
#   lab[i]   clump id of tree i, -1 if cut    siz[c]   size of clump c
#   fre      stack of free clump ids          kep/pos  kept list, positions
#   cov[c]   kept trees within 9 m of cell c  cip/cix  tree -> covered cells
#   cnt/tre  clumps / trees per bin           ip/ix    clump graph (CSR)
#   si = [open cells, free ids, kept trees, epoch, kept pairs ("links"), clump cap]
#   sv = [total score, total density units]


@njit(cache=True, inline="always")
def cls_of(s, cap):
    if s > cap:
        return NK                                             # over the cap
    if s <= 1:
        return 0
    if s <= 4:
        return 1
    if s <= 9:
        return 2
    if s <= 15:
        return 3
    return 4                                                  # 16+: the open top bin


@njit(cache=True, inline="always")
def u64(rs):
    x = rs[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    rs[0] = x
    return x


@njit(cache=True, inline="always")
def rnd(rs):
    return (u64(rs) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def rint(rs, n):
    return np.int64(u64(rs) % np.uint64(n))


@njit(cache=True, inline="always")
def stat(s, sign, cnt, tre, cap):
    k = cls_of(s, cap)
    cnt[k] += sign
    tre[k] += sign * s


@njit(cache=True, inline="always")
def kept_add(i, kep, pos, cip, cix, cov, lip, lix, si):
    pos[i] = si[2]
    kep[si[2]] = i
    si[2] += 1
    for e in range(lip[i], lip[i + 1]):                       # si[4] = kept pairs within the clump distance
        if pos[lix[e]] >= 0:
            si[4] += 1
    for e in range(cip[i], cip[i + 1]):                       # si[0] = open cells (> 9 m from a kept tree)
        c = cix[e]
        if cov[c] == 0:
            si[0] -= 1
        cov[c] += 1


@njit(cache=True, inline="always")
def kept_del(i, kep, pos, cip, cix, cov, lip, lix, si):
    p = pos[i]
    si[2] -= 1
    last = kep[si[2]]
    kep[p] = last
    pos[last] = p
    pos[i] = -1
    for e in range(lip[i], lip[i + 1]):
        if pos[lix[e]] >= 0:
            si[4] -= 1
    for e in range(cip[i], cip[i + 1]):
        c = cix[e]
        cov[c] -= 1
        if cov[c] == 0:
            si[0] += 1


@njit(cache=True, inline="always")
def kept_add_nl(i, kep, pos, cip, cix, cov, si):
    pos[i] = si[2]
    kep[si[2]] = i
    si[2] += 1
    for e in range(cip[i], cip[i + 1]):
        c = cix[e]
        if cov[c] == 0:
            si[0] -= 1
        cov[c] += 1


@njit(cache=True, inline="always")
def kept_del_nl(i, kep, pos, cip, cix, cov, si):
    p = pos[i]
    si[2] -= 1
    last = kep[si[2]]
    kep[p] = last
    pos[last] = p
    pos[i] = -1
    for e in range(cip[i], cip[i + 1]):
        c = cix[e]
        cov[c] -= 1
        if cov[c] == 0:
            si[0] += 1


@njit(cache=True, inline="always")
def new_cid(fre, si):
    si[1] -= 1
    return fre[si[1]]


@njit(cache=True, inline="always")
def free_cid(c, fre, si):
    fre[si[1]] = c
    si[1] += 1


@njit(cache=True)
def relabel(start, frm, to, ip, ix, lab, stk):
    lab[start] = to
    stk[0] = start
    top = 1
    while top:
        top -= 1
        u = stk[top]
        for e in range(ip[u], ip[u + 1]):
            v = ix[e]
            if lab[v] == frm:
                lab[v] = to
                stk[top] = v
                top += 1


@njit(cache=True)
def add(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc):
    """keep tree i, merging the clumps it touches into the largest"""
    si[3] += 1
    ep = si[3]
    m = 0
    base = -1
    bsz = 0
    nk = 0
    for e in range(ip[i], ip[i + 1]):
        v = ix[e]
        c = lab[v]
        if c >= 0:
            nk += 1                                           # every kept tree within the clump distance is a link
            if cst[c] != ep:
                cst[c] = ep
                nbc[m] = v
                m += 1
                if siz[c] > bsz:
                    bsz = siz[c]
                    base = c
    if m == 0:
        c = new_cid(fre, si)
        lab[i] = c
        siz[c] = 1
        stat(1, 1, cnt, tre, si[5])
    else:
        tot = 1
        for q in range(m):
            c = lab[nbc[q]]
            stat(siz[c], -1, cnt, tre, si[5])
            tot += siz[c]
        for q in range(m):
            c = lab[nbc[q]]
            if c != base:
                relabel(nbc[q], c, base, ip, ix, lab, stk)
                free_cid(c, fre, si)
        lab[i] = base
        siz[base] = tot
        stat(tot, 1, cnt, tre, si[5])
    si[4] += nk
    sv[0] += cr[i, 0]
    sv[1] += cr[i, 1]
    kept_add_nl(i, kep, pos, cip, cix, cov, si)


@njit(cache=True)
def remove(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, stm, stk, nbc, pst, psz, sdm):
    """cut tree i, splitting its clump if it was a bridge"""
    c = lab[i]
    s = siz[c]
    stat(s, -1, cnt, tre, si[5])
    lab[i] = -1
    sv[0] -= cr[i, 0]
    sv[1] -= cr[i, 1]
    kept_del_nl(i, kep, pos, cip, cix, cov, si)
    if s == 1:
        free_cid(c, fre, si)
        return
    m = 0
    for e in range(ip[i], ip[i + 1]):
        v = ix[e]
        if lab[v] == c:
            nbc[m] = v
            m += 1
    si[4] -= m                                                # kept trees within the clump distance = clump neighbours
    if m == 1:
        siz[c] = s - 1
        stat(s - 1, 1, cnt, tre, si[5])
        return
    si[3] += 1
    ep = si[3]
    # BFS from the first clump neighbour; the clump stays connected iff it reaches all the others, so stop there
    for q in range(1, m):
        sdm[nbc[q]] = ep
    s0 = nbc[0]
    stm[s0] = ep
    stk[0] = s0
    top = 1
    k = 0
    left = m - 1
    while top > 0 and left > 0:
        top -= 1
        u = stk[top]
        k += 1
        for e in range(ip[u], ip[u + 1]):
            v = ix[e]
            if lab[v] == c and stm[v] != ep:
                stm[v] = ep
                stk[top] = v
                top += 1
                if sdm[v] == ep:
                    left -= 1
    if left == 0:
        siz[c] = s - 1
        stat(s - 1, 1, cnt, tre, si[5])
        return
    pst[0] = s0
    psz[0] = k
    npc = 1
    covered = k              # split: that BFS ran to exhaustion = the first part
    for q in range(1, m):
        s0 = nbc[q]
        if stm[s0] == ep:
            continue
        stm[s0] = ep
        stk[0] = s0
        top = 1
        k = 0
        while top:
            top -= 1
            u = stk[top]
            k += 1
            for e in range(ip[u], ip[u + 1]):
                v = ix[e]
                if lab[v] == c and stm[v] != ep:
                    stm[v] = ep
                    stk[top] = v
                    top += 1
        pst[npc] = s0
        psz[npc] = k
        npc += 1
        covered += k
        if covered == s - 1:
            break
    if npc == 1:
        siz[c] = s - 1
        stat(s - 1, 1, cnt, tre, si[5])
        return
    big = 0
    for q in range(1, npc):
        if psz[q] > psz[big]:
            big = q
    for q in range(npc):
        if q != big:
            c2 = new_cid(fre, si)
            relabel(pst[q], c, c2, ip, ix, lab, stk)
            siz[c2] = psz[q]
            stat(psz[q], 1, cnt, tre, si[5])
    siz[c] = psz[big]
    stat(psz[big], 1, cnt, tre, si[5])


@njit(cache=True)
def add_energy(i, ip, ix, cr, lab, siz, cip, cix, cov, tre, si, sv, cst, tg, W, trh, sih, svh):
    """energy after add(i), state untouched (same integer/float ops as add, so bit-identical)"""
    for k in range(NC):
        trh[k] = tre[k]
    si[3] += 1
    ep = si[3]
    tot = 1
    nk = 0
    for e in range(ip[i], ip[i + 1]):
        c = lab[ix[e]]
        if c >= 0:
            nk += 1
            if cst[c] != ep:
                cst[c] = ep
                sc = siz[c]
                trh[cls_of(sc, si[5])] -= sc
                tot += sc
    trh[cls_of(tot, si[5])] += tot
    o = si[0]
    for e in range(cip[i], cip[i + 1]):
        if cov[cix[e]] == 0:
            o -= 1
    sih[0] = o
    sih[1] = si[1]
    sih[2] = si[2] + 1
    sih[3] = si[3]
    sih[4] = si[4] + nk
    svh[0] = sv[0] + cr[i, 0]
    svh[1] = sv[1] + cr[i, 1]
    return energy(trh, tg, sih, svh, W)


@njit(cache=True)
def members(c, start, ip, ix, lab, si, stm, buf):
    """trees of clump c in the order dissolve(c, start) visits them (state untouched)"""
    si[3] += 1
    ep = si[3]
    stm[start] = ep
    buf[0] = start
    m = 1
    head = 0
    while head < m:
        u = buf[head]
        head += 1
        for e in range(ip[u], ip[u + 1]):
            v = ix[e]
            if lab[v] == c and stm[v] != ep:
                stm[v] = ep
                buf[m] = v
                m += 1
    return m


@njit(cache=True)
def noop_relocate(buf, m, cr, kep, pos, si, sv):
    """the net effect of dissolve(buf) + restore(buf) on the state that does not come back exactly: the order of
    the kept list and the rounding of the running sums (labels, sizes, cover counts and links return unchanged)"""
    for q in range(m):
        u = buf[q]
        sv[0] -= cr[u, 0]
        sv[1] -= cr[u, 1]
        p = pos[u]
        si[2] -= 1
        last = kep[si[2]]
        kep[p] = last
        pos[last] = p
        pos[u] = -1
    for q in range(m):
        u = buf[q]
        sv[0] += cr[u, 0]
        sv[1] += cr[u, 1]
        pos[u] = si[2]
        kep[si[2]] = u
        si[2] += 1


@njit(cache=True, inline="always")
def seed_fails(i, c, ip, ix, lab):
    """build(i) would return 0 at once after clump c is dissolved"""
    if lab[i] >= 0 and lab[i] != c:
        return True
    for e in range(ip[i], ip[i + 1]):
        lv = lab[ix[e]]
        if lv >= 0 and lv != c:
            return True
    return False


@njit(cache=True)
def dissolve(c, start, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, buf):
    stat(siz[c], -1, cnt, tre, si[5])
    lab[start] = -1
    buf[0] = start
    m = 1
    head = 0
    while head < m:
        u = buf[head]
        head += 1
        sv[0] -= cr[u, 0]
        sv[1] -= cr[u, 1]
        kept_del(u, kep, pos, cip, cix, cov, lip, lix, si)
        for e in range(ip[u], ip[u + 1]):
            v = ix[e]
            if lab[v] == c:
                lab[v] = -1
                buf[m] = v
                m += 1
    free_cid(c, fre, si)
    return m


@njit(cache=True)
def restore(buf, m, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv):
    c = new_cid(fre, si)
    for q in range(m):
        u = buf[q]
        lab[u] = c
        sv[0] += cr[u, 0]
        sv[1] += cr[u, 1]
        kept_add(u, kep, pos, cip, cix, cov, lip, lix, si)
    siz[c] = m
    stat(m, 1, cnt, tre, si[5])


@njit(cache=True)
def build(i, s, rs, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, stm, cst, stk, nbc,
          buf, scratch, front):
    """grow a clump of s trees from cut tree i by random frontier growth, touching no other clump; undo it and
    return 0 if it cannot reach s"""
    if lab[i] >= 0:
        return 0
    for e in range(ip[i], ip[i + 1]):
        if lab[ix[e]] >= 0:
            return 0
    si[3] += 1
    ep = si[3]
    stm[i] = ep
    front[0] = i
    nf = 1
    k = 0
    mine = -1
    while nf > 0 and k < s:
        r = rint(rs, nf)
        u = front[r]
        nf -= 1
        front[r] = front[nf]
        ok = True
        for e in range(ip[u], ip[u + 1]):
            lv = lab[ix[e]]
            if lv >= 0 and lv != mine:
                ok = False
                break
        if not ok:
            continue
        add(u, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
        buf[k] = u
        k += 1
        mine = lab[u]
        for e in range(ip[u], ip[u + 1]):
            v = ix[e]
            if stm[v] != ep and lab[v] < 0:
                stm[v] = ep
                front[nf] = v
                nf += 1
    if k < s:
        if k > 0:
            dissolve(mine, buf[0], ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, scratch)
        return 0
    return k


@njit(cache=True, inline="always")
def any_pin(buf, m, pin):
    for q in range(m):
        if pin[buf[q]]:
            return True
    return False


@njit(cache=True, inline="always")
def energy(tre, tgt, si, sv, W):
    N = si[2]
    d = abs(sv[1] - tgt[I_D]) - tgt[I_TOL_D]
    if d < 0.0:
        d = 0.0
    d *= tgt[I_WD]                                            # density first (guide step 3), pattern second
    for k in range(NC):
        tol = tgt[I_TOL_S] * N if tgt[I_SHARE + k] > 0.0 else 0.0
        x = abs(tre[k] - tgt[I_SHARE + k] * N - tgt[I_OFFSET + k]) - tol
        if x > 0.0:
            d += x
    if tgt[I_F] >= 0.0 and tgt[I_CELLS] > 0.0:               # open space F>9 m against one stand-level target
        x = abs(si[0] - tgt[I_F] * tgt[I_CELLS]) - tgt[I_TOL_F] * tgt[I_CELLS]
        if x > 0.0:
            d += x / tgt[I_CELLS_PER_TREE]
    # leave-tree score: below the density target reward the TOTAL score (more trees help), at/above it the
    # MEAN score scaled to the target (better trees, not more). Continuous at the target; a pure mean would make
    # every added tree from an empty start cost ~D/N x its score gap and trap the annealer near N = 1.
    msc = sv[0] / N if N > 0 else 0.0
    rho = sv[1] / tgt[I_D] if tgt[I_D] > 0.0 else 1.0
    if rho > 1.0:
        rho = 1.0
    # compact clumps (guide step 4: "patches of contiguous canopy"): kept pairs within the clump distance relative to
    # random placement of the same trees on the tile's stand area (pair correlation at r). Density-neutral, so a BA
    # target is not pushed toward many small trees; below the density target M is the count at target.
    lterm = 0.0
    if tgt[I_WL] > 0.0 and N > 1 and tgt[I_HA] > 0.0:
        M = N / rho if 0.0 < rho < 1.0 else N * 1.0
        lterm = 2.0 * si[4] * tgt[I_HA] * 1e4 / (tgt[I_PAIR_AREA] * M * M)
    return W * d - tgt[I_D] * (msc * rho + tgt[I_WL] * lterm)


@njit(cache=True)
def sa_tile(ip, ix, nip, nix, cip, cix, cov, cr, pin, tgt, W, sweeps, t0, t1, seed, mix, out):
    """anneal one tile: ip/ix clump graph, nip/nix swap list, cip/cix open-space cells each tree covers, cov cover
    counts (_BIG = not in the region), cr [score, density units], pin keep trees; the best state goes to out"""
    n = cr.shape[0]
    lab = np.full(n, -1, np.int32)
    siz = np.zeros(n, np.int32)
    fre = np.arange(n).astype(np.int32)
    kep = np.zeros(n, np.int32)
    pos = np.full(n, -1, np.int32)
    cnt = np.zeros(NC, np.int64)
    tre = np.zeros(NC, np.int64)
    si = np.zeros(6, np.int64)
    si[1] = n
    si[5] = np.int64(tgt[I_CAP])
    lip = ip                                                  # link graph = the clump graph
    lix = ix
    sv = np.zeros(2)
    stm = np.zeros(n, np.int64)
    cst = np.zeros(n, np.int64)
    sdm = np.zeros(n, np.int64)
    stk = np.empty(n, np.int32)
    nbc = np.empty(n, np.int32)
    pst = np.empty(n, np.int32)
    psz = np.empty(n, np.int32)
    bufA = np.empty(n, np.int32)
    bufB = np.empty(n, np.int32)
    bufC = np.empty(n, np.int32)
    front = np.empty(n, np.int32)
    trh = np.zeros(NC, np.int64)
    sih = np.zeros(5, np.int64)
    svh = np.zeros(2)
    alw = np.zeros(NC, np.bool_)                              # the over-cap class is never built
    bcl = np.empty(NC, np.int64)
    nbcl = 0
    for k in range(NK):
        alw[k] = (tgt[I_SHARE + k] > 0.0 or tgt[I_OFFSET + k] > 0.5) and BIN_LOW[k] <= si[5]
        if alw[k]:
            bcl[nbcl] = k
            nbcl += 1
    rs = np.empty(1, np.uint64)
    rs[0] = (np.uint64(seed) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15) | np.uint64(1)
    p_relocate = mix[0]
    rest = mix[1]
    c_dissolve = mix[2]
    c_build = mix[3]
    c_swap = mix[4]

    for q in range(cov.shape[0]):                             # open cells given earlier passes' trees
        if cov[q] == 0:
            si[0] += 1
    for q in range(n):                                        # keep trees start kept, never leave
        if pin[q]:
            add(q, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
    E = energy(tre, tgt, si, sv, W)
    bestE = E
    for q in range(n):
        out[q] = pin[q]
    moves = sweeps * n
    T = t0
    ratio = (t1 / t0) ** (1.0 / max(moves, 1))
    cdn = n
    for step in range(moves):
        T *= ratio
        r = rnd(rs)
        if r < p_relocate:                                    # relocate
            if si[2] > 0:
                u = kep[rint(rs, si[2])]
                c = lab[u]
                k = cls_of(siz[c], si[5])
                continue_ = True
                i0 = 0
                s = 0
                if alw[k]:
                    # moves that would dissolve the clump, fail to build, and restore it (pinned member, or a seed
                    # next to another clump) only permute the kept list: do just that, same RNG draws
                    m = members(c, u, ip, ix, lab, si, stm, bufA)
                    hi = min(BUILD_HIGH[k], n, si[5])
                    s = BIN_LOW[k] + rint(rs, hi - BIN_LOW[k] + 1)
                    if any_pin(bufA, m, pin):
                        noop_relocate(bufA, m, cr, kep, pos, si, sv)
                        continue_ = True
                    else:
                        i0 = rint(rs, n)
                        continue_ = seed_fails(i0, c, ip, ix, lab)
                        if continue_:
                            noop_relocate(bufA, m, cr, kep, pos, si, sv)
                if alw[k] and not continue_:
                    m = dissolve(c, u, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, bufA)
                    nb = build(i0, s, rs, ip, ix, cr, lab, siz, fre,
                               kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, stm, cst, stk, nbc, bufB, bufC, front)
                    if nb == 0:
                        restore(bufA, m, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv)
                    else:
                        newE = energy(tre, tgt, si, sv, W)
                        dE = newE - E
                        if dE <= 0 or rnd(rs) < np.exp(-dE / T):
                            E = newE
                        else:
                            dissolve(lab[bufB[0]], bufB[0], ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, bufC)
                            restore(bufA, m, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv)
        else:
            r = (r - p_relocate) / rest
            if r < c_dissolve:                                # dissolve
                if si[2] > 0:
                    u = kep[rint(rs, si[2])]
                    m = dissolve(lab[u], u, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, bufA)
                    newE = energy(tre, tgt, si, sv, W)
                    dE = newE - E
                    if not any_pin(bufA, m, pin) and (dE <= 0 or rnd(rs) < np.exp(-dE / T)):
                        E = newE
                    else:
                        restore(bufA, m, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv)
            elif r < c_build:                                 # build
                if nbcl > 0:
                    k = bcl[rint(rs, nbcl)]
                    hi = min(BUILD_HIGH[k], n, si[5])
                    nb = 0 if hi < BIN_LOW[k] else build(rint(rs, n), BIN_LOW[k] + rint(rs, hi - BIN_LOW[k] + 1), rs, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv,
                                                         stm, cst, stk, nbc, bufB, bufC, front)
                    if nb > 0:
                        newE = energy(tre, tgt, si, sv, W)
                        dE = newE - E
                        if dE <= 0 or rnd(rs) < np.exp(-dE / T):
                            E = newE
                        else:
                            dissolve(lab[bufB[0]], bufB[0], ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, bufC)
            elif r < c_swap and si[2] > 0:                    # swap kept i -> cut j within the swap radius
                i = kep[rint(rs, si[2])]
                if nip[i + 1] > nip[i] and not pin[i]:
                    j = nix[nip[i] + rint(rs, nip[i + 1] - nip[i])]
                    if lab[j] < 0:
                        remove(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, stm, stk, nbc, pst, psz, sdm)
                        newE = add_energy(j, ip, ix, cr, lab, siz, cip, cix, cov, tre, si, sv, cst, tgt, W, trh, sih, svh)
                        dE = newE - E
                        if dE <= 0 or rnd(rs) < np.exp(-dE / T):
                            E = newE
                            add(j, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
                        else:                                 # sv as after add(j) + remove(j), then put i back
                            sv[0] = svh[0] - cr[j, 0]
                            sv[1] = svh[1] - cr[j, 1]
                            add(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
            else:                                             # flip (keep trees never flip)
                i = rint(rs, n)
                if not pin[i]:
                    if lab[i] >= 0:
                        remove(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, stm, stk, nbc, pst, psz, sdm)
                        newE = energy(tre, tgt, si, sv, W)
                        dE = newE - E
                        if dE <= 0 or rnd(rs) < np.exp(-dE / T):
                            E = newE
                        else:
                            add(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
                    else:                                     # score the add before applying it
                        newE = add_energy(i, ip, ix, cr, lab, siz, cip, cix, cov, tre, si, sv, cst, tgt, W, trh, sih, svh)
                        dE = newE - E
                        if dE <= 0 or rnd(rs) < np.exp(-dE / T):
                            E = newE
                            add(i, ip, ix, cr, lab, siz, fre, kep, pos, cip, cix, cov, lip, lix, cnt, tre, si, sv, cst, stk, nbc)
                        else:                                 # sv as after add(i) + remove(i)
                            sv[0] = svh[0] - cr[i, 0]
                            sv[1] = svh[1] - cr[i, 1]
        cdn -= 1
        if cdn == 0:
            cdn = n
            if E < bestE:
                bestE = E
                for q in range(n):
                    out[q] = lab[q] >= 0
    if E < bestE:
        for q in range(n):
            out[q] = lab[q] >= 0


# ===========================================================================
# Per-tile structures, built inside the parallel region
# ===========================================================================
# Nothing O(pairs) or O(tree x cell) is held for the whole stand: one grid hash of the candidates (cell = the swap
# radius) answers every neighbour query, and each tile builds its own clump graph, swap list and open-space cover
# lists, solves, and frees them. Cross-tile state is two uint8 rasters on the open-space grid (stand region;
# covered by an earlier pass) and per-tree flags.


@njit(types.Tuple((_i8v, _i4v))(_f8v, _f8v, _f8, _i8, _i8), cache=True)
def grid_hash(x, y, hc, hnx, hny):
    """counting sort of the trees into hc-sized cells; trees ascending within a cell"""
    n = x.shape[0]
    st = np.zeros(hnx * hny + 1, np.int64)
    cid = np.empty(n, np.int32)
    for i in range(n):
        gx = min(int(x[i] // hc), hnx - 1)
        gy = min(int(y[i] // hc), hny - 1)
        c = gx * hny + gy
        cid[i] = c
        st[c + 1] += 1
    for c in range(hnx * hny):
        st[c + 1] += st[c]
    fill = st[:-1].copy()
    idx = np.empty(n, np.int32)
    for i in range(n):
        c = cid[i]
        idx[fill[c]] = i
        fill[c] += 1
    return st, idx


@njit(cache=True, inline="always")
def isort(a, lo, hi):
    for k in range(lo + 1, hi):
        v = a[k]
        m = k - 1
        while m >= lo and a[m] > v:
            a[m + 1] = a[m]
            m -= 1
        a[m + 1] = v


@njit(types.void(_i8v, _i8v, _i4v, _f8v, _f8v, _i8v, _b1v, _b1v, _i8v, _i4v, _f8, _i8, _i8, _f8, _b1v, _i8v),
      parallel=True, cache=True)
def mark_blocked(tiles, toff, tord, x, y, tile_of, pin, keep, hst, hidx, hc, hnx, hny, cd2, blocked, tcount):
    """a non-keep tree within the clump distance of a keep tree in another tile, or of a tree kept in an earlier
    pass, is never kept (no clump crosses a tile edge)"""
    for q in prange(tiles.shape[0]):
        t = tiles[q]
        c = 0
        for k in range(toff[t], toff[t + 1]):
            i = tord[k]
            b = False
            if not pin[i]:
                gx = min(int(x[i] // hc), hnx - 1)
                gy = min(int(y[i] // hc), hny - 1)
                for ax in range(max(gx - 1, 0), min(gx + 2, hnx)):
                    for ay in range(max(gy - 1, 0), min(gy + 2, hny)):
                        h = ax * hny + ay
                        for e in range(hst[h], hst[h + 1]):
                            j = hidx[e]
                            if tile_of[j] != t and (pin[j] or keep[j]):
                                dx = x[i] - x[j]
                                dy = y[i] - y[j]
                                if dx * dx + dy * dy <= cd2:
                                    b = True
                                    break
                        if b:
                            break
                    if b:
                        break
            blocked[i] = b
            if not b:
                c += 1
        tcount[t] = c


@njit(cache=True)
def tile_solve(t, x, y, tv, pin, tile_of, blocked, tord, toff, hst, hidx, hc, hnx, hny, cd2, r2,
               reg, covd, CS, gnx, gny, R, L, nx, ny, open_on, tg, W, sweeps, t0, t1, seed, mix, keep, loc, gtr, ngd):
    k0 = toff[t]
    k1 = toff[t + 1]
    m = 0
    for k in range(k0, k1):
        if not blocked[tord[k]]:
            m += 1
    if m == 0:
        return
    act = np.empty(m, np.int64)
    li = 0
    for k in range(k0, k1):
        i = tord[k]
        if not blocked[i]:
            act[li] = i
            loc[i] = li
            li += 1
    # ---- clump graph (ip/ix, <= clump distance) and swap list (nip/nix, <= swap radius), local ids, rows ascending
    ip = np.zeros(m + 1, np.int64)
    nip = np.zeros(m + 1, np.int64)
    for li in range(m):
        i = act[li]
        xi = x[i]
        yi = y[i]
        gx = min(int(xi // hc), hnx - 1)
        gy = min(int(yi // hc), hny - 1)
        for ax in range(max(gx - 1, 0), min(gx + 2, hnx)):
            for ay in range(max(gy - 1, 0), min(gy + 2, hny)):
                h = ax * hny + ay
                for e in range(hst[h], hst[h + 1]):
                    j = hidx[e]
                    if j == i or tile_of[j] != t or blocked[j]:
                        continue
                    dx = xi - x[j]
                    dy = yi - y[j]
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        nip[li + 1] += 1
                        if d2 <= cd2:
                            ip[li + 1] += 1
    for li in range(m):
        ip[li + 1] += ip[li]
        nip[li + 1] += nip[li]
    ix = np.empty(ip[m], np.int32)
    nix = np.empty(nip[m], np.int32)
    for li in range(m):
        i = act[li]
        xi = x[i]
        yi = y[i]
        p6 = ip[li]
        p10 = nip[li]
        gx = min(int(xi // hc), hnx - 1)
        gy = min(int(yi // hc), hny - 1)
        for ax in range(max(gx - 1, 0), min(gx + 2, hnx)):
            for ay in range(max(gy - 1, 0), min(gy + 2, hny)):
                h = ax * hny + ay
                for e in range(hst[h], hst[h + 1]):
                    j = hidx[e]
                    if j == i or tile_of[j] != t or blocked[j]:
                        continue
                    dx = xi - x[j]
                    dy = yi - y[j]
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        nix[p10] = loc[j]
                        p10 += 1
                        if d2 <= cd2:
                            ix[p6] = loc[j]
                            p6 += 1
        isort(ix, ip[li], ip[li + 1])
        isort(nix, nip[li], nip[li + 1])
    # ---- open-space cover lists: tree -> region cells of this tile within 9 m (local cell ids over the tile bbox)
    cip = np.zeros(m + 1, np.int64)
    cix = np.zeros(0, np.int32)
    if open_on:
        tx = t // ny
        ty = t % ny
        gi0 = max(0, int(tx * L / CS) - 2)
        gi1 = gnx if tx == nx - 1 else min(gnx, int((tx + 1) * L / CS) + 3)
        gj0 = max(0, int(ty * L / CS) - 2)
        gj1 = gny if ty == ny - 1 else min(gny, int((ty + 1) * L / CS) + 3)
        wj = max(gj1 - gj0, 0)
        nc = max(gi1 - gi0, 0) * wj
        cov = np.full(nc, _BIG, np.int32)
        for gi in range(gi0, gi1):
            if min(int(((gi + 0.5) * CS) // L), nx - 1) != tx:
                continue
            for gj in range(gj0, gj1):
                if min(int(((gj + 0.5) * CS) // L), ny - 1) != ty:
                    continue
                g = gi * gny + gj
                if reg[g]:
                    cov[(gi - gi0) * wj + gj - gj0] = covd[g]
        for pas in range(2):
            if pas == 1:
                for li in range(m):
                    cip[li + 1] += cip[li]
                cix = np.empty(cip[m], np.int32)
            for li in range(m):
                i = act[li]
                ci = int(x[i] // CS)
                cj = int(y[i] // CS)
                p = cip[li]
                for gi in range(max(ci - R, gi0), min(ci + R + 1, gi1)):
                    dx = (gi + 0.5) * CS - x[i]
                    for gj in range(max(cj - R, gj0), min(cj + R + 1, gj1)):
                        dy = (gj + 0.5) * CS - y[i]
                        if dx * dx + dy * dy <= _OPEN_R2:
                            lc = (gi - gi0) * wj + gj - gj0
                            if cov[lc] != _BIG:
                                if pas == 0:
                                    cip[li + 1] += 1
                                else:
                                    cix[p] = lc
                                    p += 1
    else:
        cov = np.zeros(0, np.int32)
    cr = np.empty((m, 2))
    pn = np.empty(m, np.bool_)
    for li in range(m):
        cr[li, 0] = tv[act[li], 0]
        cr[li, 1] = tv[act[li], 1]
        pn[li] = pin[act[li]]
    out = np.zeros(m, np.bool_)
    sa_tile(ip, ix, nip, nix, cip, cix, cov, cr, pn, tg, W, sweeps, t0, t1, seed, mix, out)
    # ---- write back; this tile's clumps -> trees per bin, kept trees and density (carried to neighbours)
    ng = 0
    dg = 0.0
    for li in range(m):
        if out[li]:
            keep[act[li]] = True
            ng += 1
            dg += cr[li, 1]
    ngd[t, 0] = ng
    ngd[t, 1] = dg
    seen = np.zeros(m, np.bool_)
    stk = np.empty(m, np.int64)
    for s0 in range(m):
        if out[s0] and not seen[s0]:
            seen[s0] = True
            stk[0] = s0
            top = 1
            k = 0
            while top:
                top -= 1
                u = stk[top]
                k += 1
                for e in range(ip[u], ip[u + 1]):
                    v = ix[e]
                    if out[v] and not seen[v]:
                        seen[v] = True
                        stk[top] = v
                        top += 1
            gtr[t, cls_of(k, np.int64(tg[I_CAP]))] += k


@njit(types.void(_i8v, _i8v, _f8v, _f8v, _f8m, _b1v, _i8v, _b1v, _i4v, _i8v, _i8v, _i4v, _f8, _i8, _i8, _f8, _f8,
                 _u1v, _u1v, _f8, _i8, _i8, _i8, _f8, _i8, _i8, types.boolean, _f8m, _f8, _i8, _f8, _f8, _i8v, _f8v,
                 _b1v, _i8v, _f8m, _f8m),
      parallel=True, cache=True)
def run_pass(order, boff, x, y, tv, pin, tile_of, blocked, tord, toff, hst, hidx, hc, hnx, hny, cd2, r2,
             reg, covd, CS, gnx, gny, R, L, nx, ny, open_on, tgt, W, sweeps, t0, t1, seeds, mix, keep, loc, gtr, ngd):
    # one prange iteration per thread bin (LPT-balanced by the driver): numba's workqueue layer splits a prange
    # statically into contiguous blocks, so a per-tile prange over size-sorted tiles is badly unbalanced
    for b in prange(boff.shape[0] - 1):
        for q in range(boff[b], boff[b + 1]):
            t = order[q]
            tile_solve(t, x, y, tv, pin, tile_of, blocked, tord, toff, hst, hidx, hc, hnx, hny, cd2, r2,
                       reg, covd, CS, gnx, gny, R, L, nx, ny, open_on, tgt[t], W, sweeps, t0, t1, seeds[t], mix,
                       keep, loc, gtr, ngd)


@njit(types.void(_i8v, _f8v, _f8v, _f8, _i8, _i8, _i8, _u1v), cache=True)
def stamp(idx, x, y, CS, gnx, gny, R, covd):
    """covd = 1 on every cell within 9 m of the trees idx"""
    for k in range(idx.shape[0]):
        i = idx[k]
        ci = int(x[i] // CS)
        cj = int(y[i] // CS)
        for gi in range(max(ci - R, 0), min(ci + R + 1, gnx)):
            dx = (gi + 0.5) * CS - x[i]
            for gj in range(max(cj - R, 0), min(cj + R + 1, gny)):
                dy = (gj + 0.5) * CS - y[i]
                if dx * dx + dy * dy <= _OPEN_R2:
                    covd[gi * gny + gj] = 1


@njit(_i8v(_b1v, _i8, _i8, _f8, _f8, _i8, _i8), cache=True)
def cells_per_tile(mask, gnx, gny, cs, L, nx, ny):
    out = np.zeros(nx * ny, np.int64)
    for gi in range(gnx):
        tx = min(int(((gi + 0.5) * cs) // L), nx - 1)
        for gj in range(gny):
            if mask[gi * gny + gj]:
                out[tx * ny + min(int(((gj + 0.5) * cs) // L), ny - 1)] += 1
    return out


@njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(types.Tuple((_i8v, _i8v, _i8, _i8, _i8))(_b1v, _f8v, _f8v, _i8v, _i8v, _i4v, _f8, _i8, _i8, _f8), cache=True)
def global_clumps(keep, x, y, tile_of, hst, hidx, hc, hnx, hny, cd2):
    """clumps of all kept trees (union-find over the grid hash): trees and clumps per bin, the largest clump,
    clumps spanning tiles, and kept pairs within the clump distance"""
    n = x.shape[0]
    par = np.arange(n)
    links = 0
    for i in range(n):
        if not keep[i]:
            continue
        gx = min(int(x[i] // hc), hnx - 1)
        gy = min(int(y[i] // hc), hny - 1)
        for ax in range(max(gx - 1, 0), min(gx + 2, hnx)):
            for ay in range(max(gy - 1, 0), min(gy + 2, hny)):
                h = ax * hny + ay
                for e in range(hst[h], hst[h + 1]):
                    j = hidx[e]
                    if j > i and keep[j]:
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        if dx * dx + dy * dy <= cd2:
                            links += 1
                            ri = _find(par, i)
                            rj = _find(par, j)
                            if ri != rj:
                                par[max(ri, rj)] = min(ri, rj)
    siz = np.zeros(n, np.int64)
    ftile = np.full(n, -1, np.int64)
    cross = np.zeros(n, np.bool_)
    for i in range(n):
        if keep[i]:
            r = _find(par, i)
            siz[r] += 1
            if ftile[r] < 0:
                ftile[r] = tile_of[i]
            elif ftile[r] != tile_of[i]:
                cross[r] = True
    trees_bin = np.zeros(NK, np.int64)
    clumps_bin = np.zeros(NK, np.int64)
    largest = 0
    ncross = 0
    for r in range(n):
        if siz[r] > 0:
            k = cls_of(siz[r], NO_CAP)
            trees_bin[k] += siz[r]
            clumps_bin[k] += 1
            largest = max(largest, siz[r])
            if cross[r]:
                ncross += 1
    return trees_bin, clumps_bin, largest, ncross, links
# fmt: on


# ===========================================================================
# Driver
# ===========================================================================


def reference_open_space(shares, tph: float) -> float:
    """
    Open space (F > 9 m) of the reference model of Churchill (2013),
    Fig. 3, at a density in trees per hectare: ``1.841 + 0.144 ln(mean
    clump size) - 0.396 ln(TPH)``, clipped to [0, 1]. The mean clump size
    is the one the shares imply with the guide's average bin sizes (1, 3,
    7, 12, 17; Table 3).
    """
    if tph <= 0:
        return 1.0
    mean_size = 1.0 / float(np.sum(np.asarray(shares, float) / MEAN_CLUMP_SIZE))
    f = F9_INTERCEPT + F9_CLUMP_SIZE * np.log(mean_size) + F9_TPH * np.log(tph)
    return float(np.clip(f, 0.0, 1.0))


def _move_thresholds() -> np.ndarray:
    """MOVE_MIX as the kernel's thresholds: the relocate share, the
    remainder, then dissolve, build and swap as cumulative shares of the
    remainder."""
    mix = MOVE_MIX
    if set(mix) != {"relocate", "dissolve", "build", "swap", "flip"}:
        raise ValueError(f"MOVE_MIX has the wrong moves: {sorted(mix)}")
    if abs(sum(mix.values()) - 1.0) > 1e-9 or min(mix.values()) < 0:
        raise ValueError("MOVE_MIX shares must be non-negative and sum to 1.")
    p = mix["relocate"]
    rest = 1.0 - p
    d, b, s = mix["dissolve"], mix["build"], mix["swap"]
    return np.array([p, rest, d / rest, (d + b) / rest, (d + b + s) / rest])


def _lpt_bins(tiles, cost, nb):
    """Longest processing time first: tiles (cost descending) -> nb bins
    of near-equal total cost. Returns the tiles grouped by bin and the bin
    offsets."""
    heap = [(0, b) for b in range(nb)]
    members_ = [[] for _ in range(nb)]
    for t, c in zip(tiles, cost):
        load, b = heapq.heappop(heap)
        members_[b].append(t)
        heapq.heappush(heap, (load + int(c), b))
    boff = np.zeros(nb + 1, np.int64)
    boff[1:] = np.cumsum([len(m) for m in members_])
    order = np.array([t for m in members_ for t in m], np.int64)
    return order, boff


def _occupancy(x, y, C, W, H):
    """Cells of side C, anchored at the origin, that hold a tree. A tree on
    the far edge of the extent belongs to the last cell inside it."""
    onx, ony = max(1, int(np.ceil(W / C))), max(1, int(np.ceil(H / C)))
    gx = np.minimum((x // C).astype(np.int64), onx - 1)
    gy = np.minimum((y // C).astype(np.int64), ony - 1)
    occ = np.zeros(onx * ony, bool)
    occ[gx * ony + gy] = True
    return occ, onx, ony


def _tile_areas(occ, onx, ony, C, poly, L, nx, ny):
    """
    Treed area of each tile, m2, measured from the geometry: the occupied
    C-cells, intersected with the polygon if there is one, cut at the tile
    edges. Cells and tiles are split into rectangles that each lie in one
    cell and one tile.
    """
    import shapely

    def edges(n_cells, n_tiles):
        e = np.arange(n_cells + 1) * C
        e = np.unique(np.concatenate([e, np.arange(1, n_tiles) * L]))
        return e[e <= n_cells * C]

    xs, ys = edges(onx, nx), edges(ony, ny)
    xm, ym = 0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:])
    ci = np.minimum((xm // C).astype(np.int64), onx - 1)
    cj = np.minimum((ym // C).astype(np.int64), ony - 1)
    tx = np.minimum((xm // L).astype(np.int64), nx - 1)
    ty = np.minimum((ym // L).astype(np.int64), ny - 1)
    I, J = np.nonzero(occ[ci[:, None] * ony + cj[None, :]])
    x0, x1, y0, y1 = xs[I], xs[I + 1], ys[J], ys[J + 1]
    area = (x1 - x0) * (y1 - y0)
    if poly is not None:
        boxes = shapely.box(x0, y0, x1, y1)
        part = ~shapely.contains(poly, boxes)
        area[part] = shapely.area(shapely.intersection(boxes[part], poly))
    return np.bincount(tx[I] * ny + ty[J], weights=area, minlength=nx * ny)


def _stand_raster(cs, W, H, poly, occ, onx, ony, C):
    """Open-space region at the centres of a cs grid (ij order): inside an
    occupied C-cell and, if there is one, the polygon. Row-chunked so no
    full-grid coordinate arrays are built."""
    import shapely

    gnx, gny = max(1, int(np.ceil(W / cs))), max(1, int(np.ceil(H / cs)))
    out = np.zeros(gnx * gny, bool)
    cy = (np.arange(gny) + 0.5) * cs
    rows = max(1, 1_000_000 // gny)
    for g0 in range(0, gnx, rows):
        g1 = min(gnx, g0 + rows)
        cx = np.repeat((np.arange(g0, g1) + 0.5) * cs, gny)
        cyy = np.tile(cy, g1 - g0)
        gx = np.minimum((cx // C).astype(np.int64), onx - 1)
        gy = np.minimum((cyy // C).astype(np.int64), ony - 1)
        m = occ[gx * ony + gy]
        if poly is not None:
            w = np.flatnonzero(m)
            m[w] = shapely.intersects_xy(poly, cx[w], cyy[w])
        out[g0 * gny : g1 * gny] = m
    return out, gnx, gny


def _regroup_keep(x, y, pin, tile_of, tx, ty, ny, cd):
    """Keep trees within the clump distance of each other must be solved
    together: move each multi-tile group of keep trees into the tile of
    its first member, provided every member's tile is adjacent to it
    (tiles of one pass are never adjacent, so tiles stay independent).
    Groups that spread further are left as they are and show up as clumps
    crossing tiles."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    kp = np.flatnonzero(pin)
    if len(kp) < 2:
        return
    xy = np.column_stack([x[kp], y[kp]])
    pk = cKDTree(xy).query_pairs(cd, output_type="ndarray")
    if len(pk) == 0:
        return
    graph = coo_matrix(
        (np.ones(len(pk)), (pk[:, 0], pk[:, 1])), shape=(len(kp), len(kp))
    )
    _, klab = connected_components(graph, directed=False)
    for g in np.flatnonzero(np.bincount(klab) > 1):
        mem = kp[klab == g]
        home = tile_of[mem[0]]
        if (tile_of[mem] != home).any():
            hx, hy = divmod(int(home), ny)
            near = (np.abs(tx[mem] - hx) <= 1) & (np.abs(ty[mem] - hy) <= 1)
            if near.all():
                tile_of[mem] = home


def solve(
    x,
    y,
    dbh,
    crown_ratio,
    candidate,
    pin,
    polygon,
    stand_ha,
    *,
    density,
    density_metric,
    shares,
    open_space,
    clump_distance,
    clump_compactness,
    max_clump_size,
    density_tolerance,
    share_tolerance,
    open_space_tolerance,
    seed,
):
    """
    Thin the stand trees to the pattern.

    ``x``, ``y``, ``dbh``, ``crown_ratio``: the stand trees (eligible, and
    inside the polygon if there is one). ``candidate``: the trees that may
    be kept, a superset of ``pin``, the keep trees. ``polygon``: the stand
    polygon, or None with ``stand_ha`` the stand area in hectares. The
    other arguments are the treatment's parameters, already validated.

    Returns a boolean mask over the stand trees of the trees to keep, and
    a report dict of each target against what was achieved.
    """
    import shapely

    L, CS, C = float(TILE_SIZE), float(OPEN_SPACE_CELL), float(FOREST_CELL)
    tph_mode = density_metric == "tph"
    shares = np.asarray(shares, float)

    # ---- stand frame, treed area and its split across tiles ----------------
    if polygon is not None:
        bx0, by0, bx1, by1 = polygon.bounds
        poly = shapely.affinity.translate(polygon, -bx0, -by0)
        shapely.prepare(poly)
        ox, oy, W, H = bx0, by0, bx1 - bx0, by1 - by0
    else:
        poly = None
        ox, oy = float(np.min(x)), float(np.min(y))
        W, H = float(np.max(x)) - ox, float(np.max(y)) - oy
    xs = np.clip(np.asarray(x, float) - ox, 0.0, W)
    ys = np.clip(np.asarray(y, float) - oy, 0.0, H)
    nx, ny = max(1, int(np.ceil(W / L))), max(1, int(np.ceil(H / L)))
    ntiles = nx * ny
    occ, onx, ony = _occupancy(xs, ys, C, W, H)
    treed = _tile_areas(occ, onx, ony, C, poly, L, nx, ny)
    if not treed.sum() > 0:
        raise ValueError("The stand polygon has no area in the cells of its trees.")
    if polygon is not None:
        stand_ha = float(treed.sum()) / 1e4
    tile_ha = stand_ha * treed / treed.sum()

    # ---- leave-tree candidates ----------------------------------------------
    cand = np.flatnonzero(candidate)
    n = len(cand)
    x = np.ascontiguousarray(xs[cand])
    y = np.ascontiguousarray(ys[cand])
    dbh = np.asarray(dbh, float)[cand]
    cr = np.asarray(crown_ratio, float)[cand]
    pin = np.ascontiguousarray(np.asarray(pin, bool)[cand])
    ba = np.pi * (dbh / 200.0) ** 2  # m2
    rank = np.argsort(np.argsort(dbh, kind="stable"), kind="stable")
    rank = (rank + 0.5) / max(n, 1)
    score = 0.5 * rank + 0.5 * cr  # guide step 8: DBH percentile, crown ratio
    ba_unit = float(ba.mean()) if n and ba.sum() > 0 else 1.0  # BA -> tree units
    dw = np.ones(n) if tph_mode else ba / ba_unit
    tv = np.ascontiguousarray(np.stack([score, dw], axis=1))
    D_total = density * stand_ha / (1.0 if tph_mode else ba_unit)

    # ---- open space -----------------------------------------------------------
    if open_space is None:
        F_t = -1.0
    elif isinstance(open_space, str):  # "reference"
        tph = density if tph_mode else density / ba_unit
        F_t = reference_open_space(shares, tph)
    else:
        F_t = float(open_space)
    refresh = isinstance(open_space, str) and not tph_mode
    stand_mask, gnx, gny = _stand_raster(CS, W, H, poly, occ, onx, ony, C)
    reg = stand_mask.view(np.uint8)
    n_reg = int(stand_mask.sum())
    cells_in_tile = cells_per_tile(stand_mask, gnx, gny, CS, L, nx, ny)
    covd = np.zeros(gnx * gny, np.uint8)  # 1 = within 9 m of an earlier pass
    R = int(np.ceil(OPEN_SPACE_DISTANCE / CS)) + 1
    open_on = bool(F_t >= 0 and n_reg > 0)

    tgt = np.zeros((ntiles, TG))
    tgt[:, I_D] = D_total * treed / treed.sum()
    tgt[:, I_TOL_D] = density_tolerance * tgt[:, I_D]
    tgt[:, I_TOL_S] = share_tolerance
    tgt[:, I_SHARE : I_SHARE + NK] = shares  # over the cap: share 0
    tgt[:, I_F] = F_t
    tgt[:, I_TOL_F] = open_space_tolerance
    tgt[:, I_CELLS] = cells_in_tile
    tgt[:, I_CELLS_PER_TREE] = np.pi * _OPEN_R2 / CS**2
    tgt[:, I_HA] = tile_ha
    tgt[:, I_WD] = DENSITY_WEIGHT
    tgt[:, I_WL] = clump_compactness
    tgt[:, I_PAIR_AREA] = np.pi * clump_distance**2
    tgt[:, I_CAP] = NO_CAP if max_clump_size is None else max_clump_size

    # ---- tiles and neighbour hash ------------------------------------------
    cd2 = clump_distance**2
    r10 = max(float(SWAP_RADIUS), clump_distance + 4.0)
    tx = np.minimum((x // L).astype(np.int64), nx - 1)
    ty = np.minimum((y // L).astype(np.int64), ny - 1)
    tile_of = np.ascontiguousarray(tx * ny + ty)
    _regroup_keep(x, y, pin, tile_of, tx, ty, ny, clump_distance)
    tord = np.argsort(tile_of, kind="stable").astype(np.int32)
    toff = np.zeros(ntiles + 1, np.int64)
    toff[1:] = np.cumsum(np.bincount(tile_of, minlength=ntiles))
    trees_in_tile = np.diff(toff)
    hnx, hny = int(W // r10) + 1, int(H // r10) + 1
    hst, hidx = grid_hash(x, y, r10, hnx, hny)
    gx, gy = np.divmod(np.arange(ntiles), ny)
    color = (gx % 2) + 2 * (gy % 2)

    keep = np.zeros(n, np.bool_)
    blocked = np.zeros(n, np.bool_)
    solved = np.zeros(ntiles, bool)
    loc = np.zeros(n, np.int64)
    tcount = np.zeros(ntiles, np.int64)
    seeds = np.arange(ntiles, dtype=np.int64) + 7919 * int(seed)
    mix = _move_thresholds()
    nthr = numba.get_num_threads()
    sched = (float(TARGET_WEIGHT), int(SWEEPS), float(T_START), float(T_END))

    for c in range(4):
        tiles_c = np.flatnonzero(color == c).astype(np.int64)
        if len(tiles_c) == 0 or n == 0:
            continue
        mark_blocked(
            tiles_c, toff, tord, x, y, tile_of, pin, keep,
            hst, hidx, r10, hnx, hny, cd2, blocked, tcount,
        )  # fmt: skip
        big_first = tiles_c[np.argsort(-tcount[tiles_c], kind="stable")]
        order, boff = _lpt_bins(big_first, tcount[big_first], nthr)
        gtr = np.zeros((ntiles, NC))
        ngd = np.zeros((ntiles, 2))
        run_pass(
            order, boff, x, y, tv, pin, tile_of, blocked, tord, toff, hst, hidx,
            r10, hnx, hny, cd2, r10 * r10, reg, covd, CS, gnx, gny, R, L, nx, ny,
            open_on, tgt, *sched, seeds, mix, keep, loc, gtr, ngd,
        )  # fmt: skip

        # Open-space start state of later passes, then carry unmet targets.
        in_pass = np.zeros(ntiles, bool)
        in_pass[tiles_c] = True
        now = np.flatnonzero(keep & in_pass[tile_of]).astype(np.int64)
        stamp(now, x, y, CS, gnx, gny, R, covd)
        solved[tiles_c] = True
        for tt in tiles_c:
            if trees_in_tile[tt] == 0:
                continue
            rD = tgt[tt, I_D] - ngd[tt, 1]
            rk = (
                tgt[tt, I_SHARE : I_SHARE + NC] * ngd[tt, 0]
                + tgt[tt, I_OFFSET : I_OFFSET + NC]
                - gtr[tt]
            )
            ix_, iy_ = divmod(int(tt), ny)
            nbrs = [
                xx * ny + yy
                for xx in range(ix_ - 1, ix_ + 2)
                for yy in range(iy_ - 1, iy_ + 2)
                if 0 <= xx < nx
                and 0 <= yy < ny
                and not solved[xx * ny + yy]
                and trees_in_tile[xx * ny + yy] > 0
            ]
            for nb in nbrs:
                tgt[nb, I_D] = max(0.0, tgt[nb, I_D] + rD / len(nbrs))
                tgt[nb, I_TOL_D] = density_tolerance * tgt[nb, I_D]
                tgt[nb, I_OFFSET : I_OFFSET + NC] += rk / len(nbrs)
        # BA mode: one stand-level open-space target, from the stand TPH
        # expected given the trees kept so far.
        if refresh and keep.any():
            K = int(keep.sum())
            B = float(dw[keep].sum())
            expected = K + max(0.0, D_total - B) / (B / K)
            F_t = reference_open_space(shares, expected / stand_ha)
            tgt[:, I_F] = F_t

    # ---- report -----------------------------------------------------------------
    if n:
        trees_bin, clumps_bin, largest, ncross, links = global_clumps(
            keep, x, y, tile_of, hst, hidx, r10, hnx, hny, cd2
        )
    else:
        trees_bin = clumps_bin = np.zeros(NK, np.int64)
        largest = ncross = links = 0
    N = int(keep.sum())
    got_density = N / stand_ha if tph_mode else float(ba[keep].sum()) / stand_ha
    got_shares = trees_bin / max(N, 1)
    eps = 1e-9
    density_met = abs(got_density - density) <= density_tolerance * density + eps

    report = {
        "stand_area_ha": float(stand_ha),
        "trees": {"candidates": n, "keep": int(pin.sum()), "kept": N},
        "density": {
            "metric": density_metric,
            "target": float(density),
            "achieved": float(got_density),
            "tolerance": float(density_tolerance),
            "met": bool(density_met),
        },
        "shares": {
            "bins": list(BIN_LABELS),
            "target": [float(v) for v in shares],
            "achieved": [float(v) for v in got_shares],
            "tolerance": float(share_tolerance),
            "met": [
                bool(abs(g - s) <= share_tolerance + eps)
                for g, s in zip(got_shares, shares)
            ],
        },
        "open_space": None,
        "clumps": {
            "per_bin": [int(v) for v in clumps_bin],
            "largest": int(largest),
            "crossing_tiles": int(ncross),
            "links_per_kept_tree": float(2 * links / max(N, 1)),
            "max_size": max_clump_size,
            "max_size_met": (
                None if max_clump_size is None else bool(largest <= max_clump_size)
            ),
        },
        "mean_score": {
            "candidates": float(score.mean()) if n else None,
            "kept": float(score[keep].mean()) if N else None,
        },
    }
    if F_t >= 0:
        open_rep = {
            "mode": "reference" if isinstance(open_space, str) else "fixed",
            "target": float(F_t),
            "tolerance": float(open_space_tolerance),
            "achieved": None,
            "met": None,
            "floor": None,
            "infeasible": False,
        }
        if n_reg:
            got_F = float((covd[stand_mask] == 0).mean())
            every = np.zeros(gnx * gny, np.uint8)
            stamp(np.arange(n, dtype=np.int64), x, y, CS, gnx, gny, R, every)
            floor = float((every[stand_mask] == 0).mean())
            open_rep.update(
                achieved=got_F,
                met=bool(abs(got_F - F_t) <= open_space_tolerance + eps),
                floor=floor,
                infeasible=bool(floor > F_t + open_space_tolerance + eps),
            )
        report["open_space"] = open_rep

    kept = np.zeros(len(xs), bool)
    kept[cand[keep]] = True
    return kept, report
