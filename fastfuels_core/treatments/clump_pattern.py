# Core imports
from __future__ import annotations

import warnings

# External imports
import numpy as np
from pandas import DataFrame


class ThinToClumpPattern:
    """
    Thinning treatment that leaves the ICO pattern (individuals, clumps
    and openings) of the ICO implementation guide (Churchill et al.
    2016): a residual density, a share of the kept trees in each
    clump-size bin and a share of open space, with the best leave trees.

    Targets, each met when it falls inside its tolerance band:

    - **Density** (guide step 3): ``density`` trees per hectare, or m2 of
      basal area per hectare, of the stand.
    - **Clump-size shares** (guide steps 4-5, Table 2): the share of the
      kept trees in clumps of 1, 2-4, 5-9, 10-15 and 16+ trees. A clump
      is the trees within ``clump_distance`` of at least one other tree
      in the clump. The top bin is open, as in the guide (Table 2):
      clumps of any size over 15 count toward it, unless
      ``max_clump_size`` caps them.
    - **Open space**: the share of the stand more than 9 m from a kept
      tree (F > 9 m), measured on a 3 m grid. ``"reference"`` takes it
      from the reference-stand model of Churchill (2013), Fig. 3,
      ``F>9 = 1.841 + 0.144 ln(mean clump size) - 0.396 ln(TPH)``, with
      the mean clump size the shares imply and the guide's average bin
      sizes (1, 3, 7, 12, 17; Table 3). The model is a stand-level
      relationship and is evaluated once for the stand: at ``density``
      in TPH mode, and in BA mode at the stand TPH expected from the
      trees kept so far, refreshed after each pass of the solver. It was
      fitted with 6 m clumps.
    - **Leave-tree score** (guide step 8): the mean of a tree's DBH
      percentile and crown ratio, maximized once the other targets are
      met.

    Density is weighted above the pattern, so when the targets can't all
    be met, the pattern gives way first.

    The solver is simulated annealing over keep/cut on 200 m tiles; see
    :mod:`fastfuels_core.treatments.clump_pattern_solver`. It uses every
    numba thread (``numba.set_num_threads``), and the result does not
    depend on how many there are. The same ``seed`` gives the same
    result.

    Which trees take part:

    - **Ineligible trees** (``eligible`` False) are returned unchanged
      and ignored by the solver: they form no clumps, count toward no
      target and cover no open space. Pass the whole neighbourhood with
      the trees outside the treatment marked ineligible. A kept clump
      may then touch ineligible trees across the treatment's edge, and
      those trees are not counted in its size. Liveness is the caller's
      job too: mark dead trees ineligible to leave them standing as
      snags, or drop them first to cut them.
    - **Keep trees** (``keep`` True) are eligible trees that are never
      cut, such as old trees (guide step 8). They count toward every
      target. ``keep`` has no effect on an ineligible tree.
    - **Trees below** ``min_dbh`` are cut, unless they are keep trees.
      The guide ignores trees under 5 in. (12.7 cm) when it counts
      density; cutting them, rather than leaving them, is this
      treatment's choice.
    - With a polygon for ``stand_area``, eligible trees outside it are
      treated as ineligible.

    After ``apply``, :attr:`report` holds a machine-readable report of
    each target against what was achieved. A ``UserWarning`` is issued
    when a target is missed.

    Parameters
    ----------
    density : float
        Target residual density, per stand hectare: trees per hectare,
        or m2 of basal area per hectare (see ``density_metric``).
    shares : sequence of float
        Five shares of the kept trees, for clumps of 1, 2-4, 5-9, 10-15
        and 16+ trees. Each in [0, 1], summing to 1.
    density_metric : {"tph", "ba"}, optional
        Whether ``density`` is in trees per hectare or basal area. By
        default "tph".
    open_space : "reference", float or None, optional
        The open-space target: "reference" for the reference model, a
        fraction in [0, 1], or None to turn it off. By default
        "reference". Without it, the solver controls clump sizes but not
        where clumps sit, and openings fuse about as much as they would
        in a random thin.
    clump_distance : float, optional
        Distance that links two trees into a clump, in meters. By
        default 6 m (20 ft), the guide's.
    clump_compactness : float, optional
        How strongly the solver prefers compact clumps (``w_L``): the
        weight on the number of kept pairs within ``clump_distance``
        relative to random placement. Single linkage alone lets an
        optimizer meet the bins with sprawling chains, while the guide
        describes clumps as "patches of contiguous canopy". 0 turns the
        preference off, which matched the clump shape of real ICO crews
        in the prototype's tests; the default of 0.1 gives most of the
        compaction for a 3-10% leave-tree score cost. **The default is
        provisional** and still to be chosen.
    max_clump_size : int or None, optional
        Largest clump the treatment may leave, in trees. By default None:
        no cap, which is the guide's behaviour, since its top bin
        (16-20+, Table 2) is open. Without a cap, nothing in the bins
        tells one clump of 16+ trees from several, so on dense stands
        the solver may merge 16+ clumps into a few very large ones. A cap
        is the forester's choice for stands where clump size should be
        limited. A bin that straddles the cap is cut at it (with 12, the
        10-15 bin holds clumps of 10-12), and a bin that starts above it
        must have a share of 0.
    density_tolerance : float, optional
        Tolerance on density, as a fraction of ``density``. By default
        0.02.
    share_tolerance : float, optional
        Tolerance on each share, absolute. By default 0.02 (2 points).
    open_space_tolerance : float, optional
        Tolerance on open space, absolute. By default 0.02 (2 points).
    seed : int, optional
        Non-negative seed of the solver. By default 0.
    min_dbh : float, optional
        Smallest DBH of a leave tree, in cm. By default 12.7 (5 in.,
        guide step 3).

    Attributes
    ----------
    report : dict or None
        Set by ``apply``; None before, or when there was no eligible tree
        to treat. Keys: ``stand_area_ha``; ``trees`` (``candidates``,
        the eligible trees that may be kept; ``keep``; ``kept``;
        ``ineligible``; ``outside_stand``); ``density``, ``shares`` and
        ``open_space``, each with ``target``, ``achieved``,
        ``tolerance`` and ``met`` (per bin for shares; ``open_space`` is
        None when off, and adds ``mode``, ``floor``, the open space left
        if every candidate were kept, and ``infeasible``, True when that
        floor is already above the band); ``clumps`` (``per_bin``,
        ``largest``, ``crossing_tiles``, ``links_per_kept_tree``,
        ``max_size``, the cap or None, and ``max_size_met``, None without
        a cap);
        ``mean_score`` (``candidates``, ``kept``); and ``all_met``.

    References
    ----------
    Churchill, D.J., A.J. Larson, S.M.A. Jeronimo, P.W. Fischer, M.C.
    Dahlgreen and J.F. Franklin. 2016. *The ICO approach to quantifying
    and restoring forest spatial pattern: Implementation guide.* Version
    3.0. Stewardship Forestry and Science, Vashon, WA.

    Churchill, D.J. 2013. *Quantifying and restoring stand-level spatial
    pattern in dry forests of the eastern Washington Cascades.* PhD
    dissertation, University of Washington.
    """

    def __init__(
        self,
        density: float,
        shares,
        density_metric: str = "tph",
        open_space: float | str | None = "reference",
        clump_distance: float = 6.0,
        clump_compactness: float = 0.1,
        max_clump_size: int | None = None,
        density_tolerance: float = 0.02,
        share_tolerance: float = 0.02,
        open_space_tolerance: float = 0.02,
        seed: int = 0,
        min_dbh: float = 12.7,
    ) -> None:
        self.density = _number("density", density, positive=True)

        try:
            shares = np.asarray(shares, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"shares must be five numbers; got {shares!r}.")
        if shares.shape != (5,):
            raise ValueError(
                "shares must have five entries, for clumps of 1, 2-4, 5-9, "
                f"10-15 and 16+ trees; got {shares.size}."
            )
        if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 1).any():
            raise ValueError(f"Each share must be in [0, 1]; got {shares.tolist()}.")
        if abs(shares.sum() - 1.0) > 1e-6:
            raise ValueError(f"shares must sum to 1; they sum to {shares.sum():g}.")
        self.shares = tuple(shares.tolist())

        if density_metric not in ("tph", "ba"):
            raise ValueError(
                f'density_metric must be "tph" or "ba"; got {density_metric!r}.'
            )
        self.density_metric = density_metric

        if isinstance(open_space, str):
            if open_space != "reference":
                raise ValueError(
                    'open_space must be "reference", a fraction in [0, 1] or '
                    f"None; got {open_space!r}."
                )
        elif open_space is not None:
            open_space = _number("open_space", open_space)
            if open_space > 1:
                raise ValueError(
                    f"open_space must be a fraction in [0, 1]; got {open_space}."
                )
        self.open_space = open_space

        self.clump_distance = _number("clump_distance", clump_distance, positive=True)
        self.clump_compactness = _number("clump_compactness", clump_compactness)
        if max_clump_size is not None:
            if (
                isinstance(max_clump_size, bool)
                or not isinstance(max_clump_size, (int, np.integer))
                or max_clump_size < 1
            ):
                raise ValueError(
                    "max_clump_size must be None or an integer >= 1; got "
                    f"{max_clump_size!r}."
                )
            max_clump_size = int(max_clump_size)
            for low, label, share in zip(_BIN_LOW, _BIN_LABELS, self.shares):
                if low > max_clump_size and share > 0:
                    raise ValueError(
                        f"With max_clump_size={max_clump_size}, no clump can "
                        f"reach the {label} bin, so its share must be 0; got "
                        f"{share}."
                    )
        self.max_clump_size = max_clump_size
        self.density_tolerance = _number("density_tolerance", density_tolerance)
        self.share_tolerance = _number("share_tolerance", share_tolerance)
        self.open_space_tolerance = _number(
            "open_space_tolerance", open_space_tolerance
        )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
        self.seed = int(seed)
        self.min_dbh = _number("min_dbh", min_dbh)
        self.report = None

    def apply(
        self,
        trees: DataFrame,
        stand_area,
        eligible: np.ndarray | None = None,
        keep: np.ndarray | None = None,
        x_column: str = "x",
        y_column: str = "y",
        dia_column: str = "dbh",
        crown_ratio_column: str = "crown_ratio",
    ) -> DataFrame:
        """
        Apply the treatment to the DataFrame of trees.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data, with stem coordinates in
            meters in a projected CRS, DBH in cm and crown ratio as a
            fraction in [0, 1].
        stand_area : float or shapely geometry
            The stand: its area in hectares, or its polygon in the same
            CRS as the stem coordinates. Targets are per stand hectare,
            and only the treed part of the stand counts: the 30 m cells
            that hold an eligible tree. With a polygon, the stand area is
            measured from the geometry, as the polygon intersected with
            those cells; with an area, that is the stand area, and the
            treed cells only split the targets across the solver's
            tiles.
        eligible : np.ndarray, optional
            Boolean mask, one entry per row of ``trees``, of the trees
            the treatment applies to. Ineligible trees are returned
            unchanged and ignored. By default every tree is eligible.
        keep : np.ndarray, optional
            Boolean mask, one entry per row of ``trees``, of eligible
            trees that must be kept. They count toward the targets. By
            default none.
        x_column, y_column : str, optional
            Names of the stem coordinate columns, by default "x" and
            "y".
        dia_column : str, optional
            Name of the DBH column, in cm, by default "dbh".
        crown_ratio_column : str, optional
            Name of the crown ratio column, by default "crown_ratio".

        Returns
        -------
        DataFrame
            The retained trees, ineligible trees included, in their
            original order.

        Raises
        ------
        ValueError
            If ``eligible`` or ``keep`` does not match ``trees``; if
            ``stand_area`` is not a positive area or a polygon; or if an
            eligible tree is missing a position, DBH or crown ratio, or
            has a negative DBH or a crown ratio outside [0, 1].
        """
        import shapely

        # Not copied up front: the input is only read, and the result is a
        # new frame (a copy where it is returned unchanged).
        df = trees
        n = len(df)
        self.report = None

        eligible = _tree_mask("eligible", eligible, n, default=True)
        keep = _tree_mask("keep", keep, n, default=False)

        polygon, stand_ha = None, None
        if isinstance(stand_area, shapely.Geometry):
            if stand_area.is_empty or not stand_area.area > 0:
                raise ValueError("stand_area must be a polygon with a positive area.")
            polygon = stand_area
        elif isinstance(stand_area, bool) or not isinstance(
            stand_area, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                "stand_area must be an area in hectares or a shapely polygon; "
                f"got {type(stand_area).__name__}."
            )
        else:
            stand_ha = _number("stand_area", stand_area, positive=True)

        if not eligible.any():
            return df.copy()

        def column(name, what):
            if name not in df.columns:
                raise ValueError(f"trees has no {what} column {name!r}.")
            return df[name].to_numpy(dtype=float, na_value=np.nan)

        x = column(x_column, "x coordinate")
        y = column(y_column, "y coordinate")
        dbh = column(dia_column, "DBH")
        cr = column(crown_ratio_column, "crown ratio")

        bad = (np.isnan(x) | np.isnan(y)) & eligible
        if bad.any():
            raise ValueError(
                f"Every eligible tree needs a position; {x_column!r} or "
                f"{y_column!r} is missing on {bad.sum()} eligible trees."
            )
        for values, name, what in (
            (dbh, dia_column, "DBH"),
            (cr, crown_ratio_column, "crown ratio"),
        ):
            bad = np.isnan(values) & eligible
            if bad.any():
                raise ValueError(
                    f"Every eligible tree needs a {what}; {name!r} is missing "
                    f"on {bad.sum()} eligible trees."
                )
        bad = (dbh < 0) & eligible
        if bad.any():
            raise ValueError(
                f"DBH must be non-negative; {dia_column!r} is negative on "
                f"{bad.sum()} eligible trees."
            )
        bad = ((cr < 0) | (cr > 1)) & eligible
        if bad.any():
            raise ValueError(
                "Crown ratio must be a fraction in [0, 1]; "
                f"{crown_ratio_column!r} is outside it on {bad.sum()} "
                "eligible trees."
            )

        in_stand = eligible.copy()
        if polygon is not None:
            prepared = shapely.affinity.translate(polygon, 0.0, 0.0)  # a copy
            shapely.prepare(prepared)
            in_stand[eligible] = shapely.intersects_xy(
                prepared, x[eligible], y[eligible]
            )
        stand = np.flatnonzero(in_stand)
        if len(stand) == 0:
            return df.copy()

        # Imported here: importing it compiles the solver.
        from fastfuels_core.treatments import clump_pattern_solver

        kept, report = clump_pattern_solver.solve(
            x[stand],
            y[stand],
            dbh[stand],
            cr[stand],
            (dbh[stand] >= self.min_dbh) | keep[stand],
            keep[stand],
            polygon,
            stand_ha,
            density=self.density,
            density_metric=self.density_metric,
            shares=self.shares,
            open_space=self.open_space,
            clump_distance=self.clump_distance,
            clump_compactness=self.clump_compactness,
            max_clump_size=self.max_clump_size,
            density_tolerance=self.density_tolerance,
            share_tolerance=self.share_tolerance,
            open_space_tolerance=self.open_space_tolerance,
            seed=self.seed,
        )
        report["trees"]["ineligible"] = int((~eligible).sum())
        report["trees"]["outside_stand"] = int((eligible & ~in_stand).sum())
        self.report = report
        _warn_misses(report)

        retained = ~in_stand
        retained[stand[kept]] = True
        df = df[retained]

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df


#: Clump-size bins of ThinToClumpPattern, by their smallest clump (guide
#: Table 2): clump_pattern_solver.BIN_LOW, without importing the solver.
_BIN_LOW = (1, 2, 5, 10, 16)
_BIN_LABELS = ("1", "2-4", "5-9", "10-15", "16+")


def _number(name: str, value, positive: bool = False) -> float:
    """A finite number, non-negative or positive, or a ValueError."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a number; got {value!r}.")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        sign = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite, {sign} number; got {value}.")
    return value


def _tree_mask(name: str, value, n: int, default: bool) -> np.ndarray:
    """A boolean mask with one entry per tree, or a ValueError."""
    if value is None:
        return np.full(n, default, dtype=bool)
    value = np.asarray(value)
    if value.dtype != bool or value.shape != (n,):
        raise ValueError(
            f"{name} must be a boolean array with one entry per tree "
            f"({n}); got dtype {value.dtype} and shape {value.shape}."
        )
    return value


def _warn_misses(report: dict) -> None:
    """Set report["all_met"], and warn about every missed target."""
    missed = []
    d = report["density"]
    if not d["met"]:
        unit = "TPH" if d["metric"] == "tph" else "m2/ha BA"
        missed.append(
            f"density {d['achieved']:.4g} against {d['target']:.4g} {unit} "
            f"+/- {d['tolerance']:.0%}"
        )
    s = report["shares"]
    for label, got, want, met in zip(s["bins"], s["achieved"], s["target"], s["met"]):
        if not met:
            missed.append(f"share of bin {label} {got:.3f} against {want:.3f}")
    c = report["clumps"]
    if c["max_size_met"] is False:
        missed.append(
            f"largest clump {c['largest']} trees against a cap of {c['max_size']}"
        )
    o = report["open_space"]
    if o is not None and o["met"] is False:
        text = f"open space {o['achieved']:.3f} against {o['target']:.3f}"
        if o["infeasible"]:
            text += (
                f" (infeasible: keeping every candidate tree still leaves "
                f"{o['floor']:.3f})"
            )
        missed.append(text)
    report["all_met"] = not missed
    if missed:
        warnings.warn(
            "ThinToClumpPattern missed its targets: " + "; ".join(missed) + ".",
            stacklevel=3,
        )
