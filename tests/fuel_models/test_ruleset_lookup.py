"""Tests for :mod:`fastfuels_core.fuel_models.ruleset_lookup`.

Master_Rulesets itself isn't packaged, so every test builds a small rules
table by hand. :func:`_rules` gives each rule permissive defaults (any
cover, height and BPS, On, Wildcard "any") so a test states only the
fields it is about, and :func:`_match` matches one default pixel unless a
test says otherwise.

What is pinned:

- A rule qualifies only when the key matches exactly, cover and height
  fall in its inclusive ranges, and BPS matches its BPSRF or BPSRF is
  "any" (which accepts nodata too).
- Among qualifying rules: exact BPSRF beats "any", then On beats Off,
  then Wildcard "any" beats a specific value -- each dominating the next.
- The requested column comes back, unmatched pixels are -9999 (numeric)
  or None (text), ``matched`` marks which pixels matched, and n_unmatched
  counts pixels, not distinct input combinations.
- Bad inputs fail loudly rather than being cast to garbage integers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.fuel_models.ruleset_lookup import (
    MatchResult,
    RulesetIndex,
    build_ruleset_index,
    match_rulesets,
)

_RULE_DEFAULTS = {
    "Zone": 1,
    "EVT": 7000,
    "DIST": 0,
    "Cover_Low": 0,
    "Cover_High": 999,
    "Height_Low": 0,
    "Height_High": 999,
    "BPSRF": "any",
    "OnOff": "On",
    "Wildcard": "any",
}


def _rules(*rows: dict) -> pd.DataFrame:
    """A Master_Rulesets table; fields a row leaves out get permissive defaults."""
    return pd.DataFrame([{**_RULE_DEFAULTS, **row} for row in rows])


def _match(
    rules: pd.DataFrame,
    *,
    zone=1,
    evt=7000,
    dist=0,
    cover=150,
    height=110,
    bpsrf=11,
    column="FBFM13",
) -> MatchResult:
    """Match pixels against ``rules``; scalar inputs broadcast to the others."""
    inputs = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v)) for v in (zone, evt, dist, cover, height, bpsrf))
    )
    return match_rulesets(
        *inputs, index=build_ruleset_index(rules), output_column=column
    )


def _fbfm13(rules: pd.DataFrame, **pixel) -> list:
    """FBFM13 for each pixel, as a plain list."""
    return _match(rules, **pixel).output.tolist()


class TestBuildRulesetIndex:
    def test_groups_by_exact_key(self):
        rules = _rules(
            {"Zone": 1, "FBFM13": 1},
            {"Zone": 1, "FBFM13": 2},
            {"Zone": 2, "FBFM13": 3},
        )
        index = build_ruleset_index(rules)
        assert isinstance(index, RulesetIndex)
        assert set(index.groups) == {(1, 7000, 0), (2, 7000, 0)}
        assert len(index.groups[(1, 7000, 0)].row_index) == 2
        assert all(type(v) is int for key in index.groups for v in key)

    def test_non_default_table_index_is_ignored(self):
        # Rows are gathered by position, so a filtered or re-indexed table
        # must still return the right row.
        rules = _rules(
            {"BPSRF": "11", "FBFM13": 8}, {"BPSRF": "12", "FBFM13": 9}
        ).set_index(pd.Index([40, 7]))
        assert _fbfm13(rules, bpsrf=[11, 12]) == [8, 9]


class TestQualification:
    def test_key_must_match_exactly(self):
        rules = _rules({"FBFM13": 5})
        assert _fbfm13(rules, zone=2) == [-9999]
        assert _fbfm13(rules, evt=7001) == [-9999]
        assert _fbfm13(rules, dist=111) == [-9999]

    def test_cover_range_is_inclusive(self):
        rules = _rules({"Cover_Low": 100, "Cover_High": 200, "FBFM13": 5})
        assert _fbfm13(rules, cover=[99, 100, 200, 201]) == [-9999, 5, 5, -9999]

    def test_height_range_is_inclusive(self):
        rules = _rules({"Height_Low": 100, "Height_High": 200, "FBFM13": 5})
        assert _fbfm13(rules, height=[99, 100, 200, 201]) == [-9999, 5, 5, -9999]

    def test_specific_bpsrf_needs_exact_bps(self):
        rules = _rules({"BPSRF": "11", "FBFM13": 5})
        assert _fbfm13(rules, bpsrf=[11, 12]) == [5, -9999]

    def test_any_bpsrf_accepts_nodata_bps(self):
        rules = _rules({"FBFM13": 5})
        assert _fbfm13(rules, bpsrf=-9999) == [5]


class TestTieBreak:
    def test_exact_bpsrf_beats_any_even_when_off(self):
        rules = _rules(
            {"BPSRF": "any", "OnOff": "On", "FBFM13": 1},
            {"BPSRF": "11", "OnOff": "Off", "FBFM13": 2},
        )
        assert _fbfm13(rules, bpsrf=[11, 12]) == [2, 1]

    def test_on_beats_off_even_with_specific_wildcard(self):
        rules = _rules(
            {"OnOff": "Off", "Wildcard": "any", "FBFM13": 1},
            {"OnOff": "On", "Wildcard": "2", "FBFM13": 2},
        )
        assert _fbfm13(rules) == [2]

    def test_wildcard_any_beats_specific(self):
        rules = _rules(
            {"Wildcard": "2", "FBFM13": 1},
            {"Wildcard": "any", "FBFM13": 2},
        )
        assert _fbfm13(rules) == [2]

    def test_non_qualifying_rule_never_wins(self):
        # The better-ranked rule is out of cover range, so the other wins.
        rules = _rules(
            {"BPSRF": "11", "Cover_High": 100, "FBFM13": 1},
            {"BPSRF": "any", "FBFM13": 2},
        )
        assert _fbfm13(rules, cover=150) == [2]


class TestOutputs:
    @pytest.mark.parametrize(
        "column, expected", [("FBFM13", 5), ("FBFM40", 122), ("FCCS", 49)]
    )
    def test_returns_the_requested_column(self, column, expected):
        rules = _rules({"FBFM13": 5, "FBFM40": 122, "FCCS": 49})
        assert _match(rules, column=column).output.tolist() == [expected]

    def test_matched_marks_matched_pixels(self):
        rules = _rules({"Zone": 1, "FBFM13": 5})
        result = _match(rules, zone=np.array([[1, 2], [2, 1]]))
        np.testing.assert_array_equal(result.matched, [[True, False], [False, True]])

    def test_keeps_input_shape(self):
        rules = _rules({"Zone": 1, "FBFM13": 5})
        result = _match(rules, zone=np.array([[1, 2, 1], [2, 1, 1]]))
        np.testing.assert_array_equal(result.output, [[5, -9999, 5], [-9999, 5, 5]])

    def test_unmatched_text_column_is_none(self):
        rules = _rules({"FBFM13": 5, "Label": "grass"})
        result = _match(rules, zone=[1, 2], column="Label")
        assert result.output.tolist() == ["grass", None]

    def test_n_unmatched_counts_pixels_not_combinations(self):
        # Three unmatched pixels share one input combination.
        rules = _rules({"FBFM13": 5})
        assert _match(rules, zone=[1, 2, 2, 2]).n_unmatched == 3

    def test_unknown_output_column_raises(self):
        with pytest.raises(ValueError, match="FBFM99"):
            _match(_rules({"FBFM13": 5}), column="FBFM99")


class TestInputValidation:
    def _call(self, **overrides):
        inputs = {
            "zone": np.array([1, 1]),
            "evt": np.array([7000, 7000]),
            "dist": np.array([0, 0]),
            "cover": np.array([150, 150]),
            "height": np.array([110, 110]),
            "bpsrf": np.array([11, 11]),
        }
        inputs.update(overrides)
        return match_rulesets(
            **inputs,
            index=build_ruleset_index(_rules({"FBFM13": 5})),
            output_column="FBFM13",
        )

    def test_whole_valued_floats_match_like_ints(self):
        result = self._call(evt=np.array([7000.0, 7000.0]))
        assert result.output.tolist() == [5, 5]

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="cover.*NaN"):
            self._call(cover=np.array([150.0, np.nan]))

    def test_non_integer_float_raises(self):
        with pytest.raises(ValueError, match="height.*non-integer"):
            self._call(height=np.array([110.0, 110.5]))

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="bpsrf"):
            self._call(bpsrf=np.array(["11", "11"]))

    def test_different_length_raises(self):
        with pytest.raises(ValueError, match="evt"):
            self._call(evt=np.array([7000]))

    def test_same_size_different_shape_raises(self):
        # (2, 3) and (3, 2) flatten to the same length; they must not be
        # silently paired pixel by pixel.
        with pytest.raises(ValueError, match="dist"):
            self._call(
                zone=np.ones((2, 3), dtype=int),
                evt=np.full((2, 3), 7000),
                dist=np.zeros((3, 2), dtype=int),
                cover=np.full((2, 3), 150),
                height=np.full((2, 3), 110),
                bpsrf=np.full((2, 3), 11),
            )
