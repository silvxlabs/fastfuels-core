# Plan — #126 FuelCalc crown biomass

Spec: `specs/126-fuelcalc-crown-biomass.md`. Worktree:
`~/Work/NMC/dev/fastfuels-core-126`, branch `126-fuelcalc-crown-biomass`.

Commands: `uv sync --extra test`; `uv run pytest` (full);
`uv run pytest tests/canopy_fuel` (fast loop). Formatting: `uv run black`
if pre-commit config asks for it (`.pre-commit-config.yaml`).

FuelCalc source: `~/Work/NMC/dev/FOF-FC/FC_DLL/` (read-only).

---

- [ ] **1. Per-component dispatch for the `brown_1978` arm**
  - Blockers: none.
  - Restructure so a crown component (total, foliage, 0–¼″ twig) is
    evaluated per Id the way `sr_EFD[]` dispatches it, keyed by the
    species table's `TOTAL`/`FOL`/`TWIG` columns, instead of "Brown crown
    weight × P1". Brown and Snell & Little Ids go through their existing
    proportion code (`BT_GetWC` / `SL_PC` semantics); the structure must
    let a later item add an Id whose C function returns components
    directly (Grier `d_PJ`, the `NC_BM2/3` functions) with one registry
    entry. Read `NC_PTL*.C` for how FuelCalc assembles a tree (where it
    calls `_Tot`/`_Fol`/`_Twg`, how the ≤1″ small-tree table and the
    crown-class factor enter) and match it.
  - The `brown_proportions` fine share under `nsvb`/`jenkins` should come
    from the same component functions (`Twg / (Tot − Fol)` of the `FOL`
    Id, or whatever FuelCalc's structure implies), equal to today's
    `(P2 − P1)/(1 − P1)` for Brown/SL Ids.
  - Keep `brown.py`'s public functions working (they are tested against
    Brown's tables directly); add rather than break. Follow existing
    module conventions — no new base classes.
  - Acceptance: whole suite green with **no expected-value edits**;
    oracle (`tests/canopy_fuel/fuelcalc_reference.py`) gains a
    per-component dispatcher transcribed from `sr_EFD[]`, and a parity
    test compares library vs oracle per component over a diameter sweep
    for every currently-implemented Id.

- [ ] **2. Grier et al. 1992 pinyon and juniper (`PY`, `JP`)**
  - Blockers: 1.
  - Port `NC_BMPJ.C` `Pinyon()` / `Juniper()` (including the DBH → root-
    collar diameter conversion) as library code; transcribe independently
    into the oracle. Sapling (≤1″) and crown-class (`PJ` row) behaviour as
    FuelCalc does it.
  - Acceptance: per-component parity vs oracle for both Ids over
    0.1–60″; the issue's quoted coefficients appear (Tot −1.157/2.086,
    stem+bark −2.297/2.437, foliage mean of −1.358/1.841 and
    −0.862/1.399, twig −1.737/1.382) or a reported reason why not; every
    juniper/pinyon SPCD in the species table prices without raising; JP
    trees count in CBD under `exclude_hardwoods=True` (`INCL_CBD=Yes`).

- [ ] **3. Snell & Little 1983 crown weights (`RA`, `GC`, `MA`, `TO`, `BM`)**
  - Blockers: 1.
  - Port the `SL` total-weight function from `NC_BMSL.C` so the five
    hardwood Ids price under `brown_1978` (proportions already exist).
  - Acceptance: per-component parity vs oracle; `crown_weight` (or its
    successor) no longer raises for these Ids; the hardwood arm of the
    docstrings is updated.

- [ ] **4. Quaking aspen (`QA`) and northern red oak (`RO`)**
  - Blockers: 1.
  - Loomis & Roussopoulos 1978 and Loomis & Blank 1981, from whichever of
    `NC_BM2.C` / `NC_BM3.C` defines `QA_*` / `RO_*`. Remove the
    `brown.py` module-docstring note that QA "raises like any other
    species without equations".
  - Acceptance: per-component parity vs oracle; aspen (SPCD 746) and the
    red-oak SPCDs price under every `equations` × partition combination
    that consults the species table.

- [ ] **5. Eastern Ids, Ker 1980 group (`PB`, `YB`, `AC`, `BF`, `EH`, `EP`, `BS`)**
  - Blockers: 1.
  - `e_Ker108` / `e_Ker114` `Tot`/`Fol`; their `Twg` per `sr_EFD[]`
    (some route twig through `e_SL83` or `e_Brn78` — follow exactly what
    the C function computes).
  - Acceptance: per-component parity vs oracle for all seven.

- [ ] **6. Eastern Ids, remaining (`RM`, `SM`, `RS`, `LO`, `PI`)**
  - Blockers: 1.
  - Freedman et al. 1982 (`e_Fre134`), Rogerson (`LO`), Duveneck 2005
    (`PI`).
  - Acceptance: per-component parity vs oracle for all five; a test
    asserts every Id in the species table's `TOTAL`/`FOL`/`TWIG` columns
    has an implementation (i.e. `sr_EFD[]` coverage is complete), and
    that no SPCD in the table raises under `brown_1978` over 0.1–60″.

- [ ] **7. Error text and docstrings**
  - Blockers: 2–6.
  - Spec requirement 5, in full. Also sweep `available_canopy_fuel`'s
    `Raises` section, the `compute_canopy_metrics` docstring, and any
    test asserting old message text. Messages should name the core
    argument *and* the concept (e.g. "species inclusion that keeps
    hardwoods"), so an API caller can map it.
  - Acceptance: grep shows no remaining claim that pinyon/juniper/eastern
    Ids are unimplemented; tests on message fragments updated.

- [ ] **8. Stand-level parity and the Blackfoot acceptance run**
  - Blockers: 2–7.
  - Extend the stand CBD/CBH/CFL parity test with a tree list mixing old
    and new Ids (juniper, pinyon, aspen, a Snell & Little hardwood, an
    eastern Id), compared against the oracle.
  - Outside the suite: run `compute_canopy_metrics` on
    `~/Work/NMC/dev/blackfoot_gdam_repro.csv` with the argument sets
    griddle passes for each preset
    (`~/Work/NMC/dev/FastFuels-API-v2/services/griddle/griddle/handlers/canopy_inventory.py`),
    `exclude_hardwoods` both ways. Report results; do not commit the CSV.
  - Acceptance: parity test green; Blackfoot run succeeds for every
    preset with plausible CBD/CBH/CFL.
