# Spec — #126: FuelCalc crown biomass for every Id in its equation registry

Source: silvxlabs/fastfuels-core#126. This spec is the FuelCalc half of
that issue. The FFE-FVS port (issue steps 2–3 for FFE, and the FVS side of
step 4) is a follow-up.

## Goal

When `available_canopy_fuel(..., equations="brown_1978")`, and through it
`compute_canopy_metrics`, prices a tree, the result matches FuelCalc for
every species row in `FUELCALC_SPECIES_TABLE.csv`. A FuelCalc preset fed
the same tree list as FuelCalc gives the same per-tree crown components and
the same stand CBD/CBH/CFL.

Today every Id outside Brown (1978) Table 1 raises: `JP`, `PY`, `QA`, the
Snell & Little crown weights (`BM`, `GC`, `MA`, `RA`, `TO`), `RO`, `PI` and
the eastern Ids. That makes every preset fail on western Montana GDAM
inventories, which draw Rocky Mountain juniper and quaking aspen.

## Reference implementation

FuelCalc C source, local clone at `~/Work/NMC/dev/FOF-FC`, `FC_DLL/`:

- `NC_EFD.C` `sr_EFD[]` — the registry: for each Id and component (`Tot`,
  `Fol`, `Twg`, `1in`, `3in`, `3P`), the function and the paper.
- `NC_ESD.C` `sr_ESD[]` — species → per-component Ids (frozen in
  `tests/canopy_fuel/fuelcalc_sr_esd.csv`; mirrored by
  `fastfuels_core/data/FUELCALC_SPECIES_TABLE.csv`).
- `NC_BM.C` (Brown 1978), `NC_BMSL.C` (Snell & Little 1983), `NC_BMPJ.C`
  (Grier et al. 1992), `NC_BM2.C` / `NC_BM3.C` (the eastern / "newer
  species" Ids: Ker 1980, Freedman et al. 1982, Rogerson, Duveneck 2005,
  Loomis & Roussopoulos 1978, Loomis & Blank 1981).
- `NC_PTL*.C` — how a tree's components are assembled (small-tree branch,
  crown-class factor, `_AvailFuel`).

The C code is the spec for structure and arithmetic. For coefficients,
keep the convention `fuelcalc_reference.py` already states: where a shipped
C value is a transcription slip against the paper it cites, follow the
paper, say so at the value, and register it in the parity test. Every such
departure must be reported back to the maintainer, not decided silently.
If the paper is not in hand, follow the C.

## Requirements

1. **Per-component dispatch.** The `brown_1978` arm evaluates each crown
   component through the Id FuelCalc assigns that component (the species
   table's `TOTAL`/`FOL`/`TWIG` columns), as `sr_EFD[]` dispatches it. For
   the Ids implemented today this changes no number (every row has
   `TOTAL == FOL == TWIG`).
2. **Every biomass Id in `sr_EFD[]` has an implementation** for the
   components available fuel needs (foliage, 0–¼″ twig) and total crown,
   following its C function: `JP`, `PY`, `QA`, `RO`, `PI`, the Snell &
   Little `Tot` for `BM`/`GC`/`MA`/`RA`/`TO`, and `AC`, `BF`, `BS`, `EH`,
   `EP`, `LO`, `RS`, `PB`, `YB`, `RM`, `SM`.
3. **Sapling, crown-class and partition behaviour follow FuelCalc** for the
   new Ids: whatever FuelCalc does for a ≤1″ tree of that Id, the crown-
   class factor row it reads (`PJ` etc.), and the fine share the
   `brown_proportions` partition applies under `nsvb`/`jenkins`.
4. **Species outside the table** keep raising, as FuelCalc rejects them.
5. **Error text and docstrings are accurate:**
   - `brown._evaluate`'s docstring no longer claims FuelCalc errors on
     `JP`/`PY`.
   - `brown.crown_weight`'s error no longer sends `QA` to Snell & Little.
   - The species-table errors (`available_fuel._fuelcalc_species_for_trees`,
     `metrics._conifers_only`) make sense to a caller who reached core
     through the API and cannot set `exclude_hardwoods`.
   - `available_canopy_fuel` / `compute_canopy_metrics` docstrings stop
     saying the `brown_1978` arm is scoped to Brown's eleven conifers.
6. **Parity.** Each new Id is pinned against an independent transcription
   of its C function in `tests/canopy_fuel/fuelcalc_reference.py` (the
   oracle; never imported by library code), over a diameter sweep, per
   component. Stand CBD/CBH/CFL parity is extended to a tree list that
   includes the new Ids.

## Non-goals

- FFE-FVS crown biomass, variant selection, and FFE surrogates (follow-up;
  decided: core will take an explicit `variant` parameter).
- Renaming the `brown_1978` equations value. It is a public API enum in
  FastFuels-API-v2; the docstring is corrected instead.
- Components no canopy metric reads (`1in`, `3in`, `3P`) — implement only
  if they fall out of the same C function for free.

## Acceptance

- `uv run pytest` green.
- `compute_canopy_metrics` on `~/Work/NMC/dev/blackfoot_gdam_repro.csv`
  (5,704 trees, juniper and aspen included) succeeds for the argument sets
  the four API presets pass, with and without `exclude_hardwoods`.
- No `ValueError` for any SPCD in `FUELCALC_SPECIES_TABLE.csv` under the
  `brown_1978` arm across a 0.1–60″ diameter sweep.

## Go-ahead

Scope, variant approach, and parity method confirmed by the maintainer on
2026-11-04 (FuelCalc complete + error text in this PR; explicit FFE
variant parameter; transcription oracle for FuelCalc, frozen FVS runs for
the FFE follow-up).
