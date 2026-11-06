"""Tests for :class:`fastfuels_core.treatments.ThinToCrownSpacing`."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.treatments import ThinToCrownSpacing


def random_stand(n: int, side: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, side, n),
            "y": rng.uniform(0, side, n),
            "crown_radius": rng.uniform(0.5, 3.0, n),
            "dbh": rng.uniform(5, 60, n),
        }
    )


def gaps(df: pd.DataFrame) -> np.ndarray:
    """Edge-to-edge distance of every pair of trees, by brute force."""
    xy = df[["x", "y"]].to_numpy()
    r = df["crown_radius"].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    g = d - r[:, None] - r[None, :]
    np.fill_diagonal(g, np.inf)
    return g


def pair(dbh_a=30.0, dbh_b=20.0, distance=4.0, radius=1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, distance],
            "y": [0.0, 0.0],
            "crown_radius": [radius, radius],
            "dbh": [dbh_a, dbh_b],
        }
    )


class TestSpacing:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("spacing", [0.5, 3.0, 10.0])
    def test_no_kept_pair_is_too_close(self, seed, spacing):
        df = random_stand(300, 60.0, seed)
        out = ThinToCrownSpacing(spacing).apply(df)
        assert (gaps(out) >= spacing).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_every_removed_tree_conflicts_with_a_kept_one(self, seed):
        """The kept set is maximal: no removed tree could be put back."""
        df = random_stand(300, 60.0, seed)
        out = ThinToCrownSpacing(3.0).apply(df)
        removed = ~df.index.isin(out.index)
        g = gaps(df)[np.ix_(removed, ~removed)]
        assert (g.min(axis=1) < 3.0).all()

    def test_a_gap_exactly_at_spacing_is_allowed(self):
        out = ThinToCrownSpacing(2.0).apply(pair(distance=4.0, radius=1.0))
        assert len(out) == 2

    def test_a_gap_just_under_spacing_is_not(self):
        out = ThinToCrownSpacing(2.0).apply(pair(distance=3.99, radius=1.0))
        assert len(out) == 1

    def test_large_crowns_conflict_from_far_apart_stems(self):
        """The search reaches out to the largest crown, not just spacing."""
        df = pair(distance=11.0)
        df["crown_radius"] = [10.0, 0.5]
        assert len(ThinToCrownSpacing(1.0).apply(df)) == 1

    def test_spacing_wider_than_the_stand_keeps_one_tree(self):
        df = random_stand(50, 10.0)
        out = ThinToCrownSpacing(100.0).apply(df)
        assert out.index.tolist() == [df["dbh"].idxmax()]


def brute_force(df, spacing, eligible):
    """The greedy rule written out directly, for small stands."""
    g = gaps(df)
    order = sorted(
        range(len(df)),
        key=lambda k: (eligible[k], -df["dbh"].iloc[k] if eligible[k] else 0, k),
    )
    kept = []
    for k in order:
        if not eligible[k] or all(g[k, m] >= spacing for m in kept):
            kept.append(k)
    return sorted(kept)


class TestRanking:
    @pytest.mark.parametrize("seed", range(10))
    def test_it_matches_the_rule_written_out(self, seed):
        df = random_stand(200, 40.0, seed)
        # Coarse diameters, so ties are common.
        df["dbh"] = df["dbh"].round(-1)
        eligible = np.random.default_rng(seed).random(len(df)) < 0.8
        out = ThinToCrownSpacing(2.0).apply(df, eligible=eligible)
        assert out.index.tolist() == brute_force(df, 2.0, eligible)

    def test_the_larger_tree_survives(self):
        out = ThinToCrownSpacing(5.0).apply(pair(dbh_a=20.0, dbh_b=30.0))
        assert out["dbh"].tolist() == [30.0]

    def test_a_tie_goes_to_the_first_tree(self):
        out = ThinToCrownSpacing(5.0).apply(pair(dbh_a=25.0, dbh_b=25.0))
        assert out.index.tolist() == [0]

    def test_rank_by_another_column(self):
        df = pair(dbh_a=30.0, dbh_b=20.0)
        df["height"] = [10.0, 15.0]
        out = ThinToCrownSpacing(5.0, rank_by="height").apply(df)
        assert out["height"].tolist() == [15.0]

    def test_the_greedy_order_can_keep_two_small_over_one_large(self):
        """A large tree between two small ones that clear each other."""
        df = pd.DataFrame(
            {
                "x": [0.0, 3.0, 6.0],
                "y": [0.0, 0.0, 0.0],
                "crown_radius": [1.0, 1.0, 1.0],
                "dbh": [10.0, 50.0, 10.0],
            }
        )
        out = ThinToCrownSpacing(2.0).apply(df)
        assert out.index.tolist() == [1]


class TestEligibility:
    def test_ineligible_trees_are_never_removed(self):
        df = random_stand(300, 60.0)
        eligible = np.arange(len(df)) % 3 != 0
        out = ThinToCrownSpacing(3.0).apply(df, eligible=eligible)
        assert df.index[~eligible].isin(out.index).all()

    def test_kept_eligible_trees_clear_every_kept_tree(self):
        df = random_stand(300, 60.0)
        eligible = np.arange(len(df)) % 3 != 0
        out = ThinToCrownSpacing(3.0).apply(df, eligible=eligible)
        g = gaps(out)
        kept_eligible = eligible[df.index.isin(out.index)]
        assert (g[kept_eligible] >= 3.0).all()

    def test_a_small_fixed_tree_beats_a_large_eligible_one(self):
        out = ThinToCrownSpacing(5.0).apply(
            pair(dbh_a=50.0, dbh_b=10.0), eligible=np.array([True, False])
        )
        assert out.index.tolist() == [1]

    def test_fixed_trees_may_be_closer_than_spacing(self):
        df = pair(distance=1.0)
        out = ThinToCrownSpacing(5.0).apply(df, eligible=np.array([False, False]))
        pd.testing.assert_frame_equal(out, df)

    def test_a_fixed_tree_needs_no_rank(self):
        df = pair(dbh_a=np.nan, dbh_b=10.0)
        out = ThinToCrownSpacing(5.0).apply(df, eligible=np.array([False, True]))
        assert out.index.tolist() == [0]

    def test_a_region_respects_the_trees_across_its_edge(self):
        """
        A treatment confined to a region, as the caller should apply it:
        the whole neighbourhood is passed in, and only the trees inside
        the region (here, x < 30) are eligible.
        """
        df = random_stand(400, 60.0)
        inside = (df["x"] < 30.0).to_numpy()
        out = ThinToCrownSpacing(3.0).apply(df, eligible=inside)

        kept = df.index.isin(out.index)
        assert kept[~inside].all()
        g = gaps(df)[np.ix_(inside & kept, kept)]
        assert (g >= 3.0).all()

        # Passing only the region instead keeps trees that crowd the
        # ones across the edge -- the reason for the mask.
        alone = ThinToCrownSpacing(3.0).apply(df[inside])
        crowding = gaps(df)[np.ix_(df.index.isin(alone.index), ~inside)]
        assert (crowding < 3.0).any()

    @pytest.mark.parametrize(
        "bad", [np.array([1, 0]), np.array([True]), [True, False, True]]
    )
    def test_a_malformed_mask_raises(self, bad):
        with pytest.raises(ValueError, match="eligible"):
            ThinToCrownSpacing(1.0).apply(pair(), eligible=bad)


class TestInputs:
    def test_an_empty_stand_is_returned_unchanged(self):
        df = pair().iloc[:0]
        pd.testing.assert_frame_equal(ThinToCrownSpacing(1.0).apply(df), df)

    @pytest.mark.parametrize("column", ["crown_radius", "dbh", "x"])
    def test_a_null_value_raises(self, column):
        df = pair()
        df.loc[1, column] = np.nan
        with pytest.raises(ValueError, match=column):
            ThinToCrownSpacing(1.0).apply(df)

    def test_a_nullable_integer_rank_with_a_null_raises(self):
        df = pair()
        df["dbh"] = pd.array([30, None], dtype="Int64")
        with pytest.raises(ValueError, match="dbh"):
            ThinToCrownSpacing(1.0).apply(df)

    def test_a_negative_radius_raises(self):
        df = pair()
        df.loc[0, "crown_radius"] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            ThinToCrownSpacing(1.0).apply(df)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_spacing_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="spacing"):
            ThinToCrownSpacing(bad)

    def test_other_column_names(self):
        df = pair(distance=3.0).rename(
            columns={"x": "X", "y": "Y", "crown_radius": "R", "dbh": "DIA"}
        )
        out = ThinToCrownSpacing(5.0, rank_by="DIA").apply(
            df, x_column="X", y_column="Y", radius_column="R"
        )
        assert out.index.tolist() == [0]


class TestFrameHandling:
    def test_the_input_is_not_modified(self):
        df = random_stand(100, 20.0)
        before = df.copy()
        ThinToCrownSpacing(2.0).apply(df)
        pd.testing.assert_frame_equal(df, before)

    def test_survivors_keep_their_original_order_and_columns(self):
        df = random_stand(100, 20.0)
        out = ThinToCrownSpacing(2.0).apply(df)
        assert out.index.is_monotonic_increasing
        assert list(out.columns) == list(df.columns)

    def test_duplicate_index_labels_do_not_take_extra_trees(self):
        df = pair(distance=3.0)
        df.index = [7, 7]
        out = ThinToCrownSpacing(5.0).apply(df)
        assert len(out) == 1

    def test_it_is_deterministic(self):
        df = random_stand(500, 50.0)
        a = ThinToCrownSpacing(2.0).apply(df)
        b = ThinToCrownSpacing(2.0).apply(df)
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("side", [1_600.0, 500.0])
def test_fifty_thousand_trees_well_under_a_second(side):
    # ~200 and ~2,000 trees/ha.
    df = random_stand(50_000, side)
    treatment = ThinToCrownSpacing(3.0)
    treatment.apply(df.iloc[:100])  # warm imports
    start = time.perf_counter()
    treatment.apply(df)
    assert time.perf_counter() - start < 1.0
