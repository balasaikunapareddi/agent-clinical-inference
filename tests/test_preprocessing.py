"""Tests for Agents 5 + 6 — Preprocessing Recommender + Implementor."""

import pandas as pd

from clinical_inference.features import describe_columns
from clinical_inference.preprocessing import (
    classify_column,
    implement,
    recommend,
)


def _meta(df, col):
    return next(m for m in describe_columns(df) if m.name == col)


def test_binary_is_exactly_two_unique():
    df = pd.DataFrame({"x": [0, 1, 0, 1, 0, 1, 0, 1]})
    assert classify_column(_meta(df, "x"), df["x"]) == "Binary"
    df2 = pd.DataFrame({"x": ["M", "F", "M", "F"]})
    assert classify_column(_meta(df2, "x"), df2["x"]) == "Binary"


def test_categorical_is_low_unique_short_strings():
    df = pd.DataFrame({"site": ["A", "B", "C"] * 20})
    assert classify_column(_meta(df, "site"), df["site"]) == "Categorical"


def test_numerical_needs_unique_over_80_percent_rows():
    # 100 rows, 90 unique floats -> Numerical.
    df = pd.DataFrame({"v": [float(i) + 0.5 for i in range(90)] + [0.5] * 10})
    assert classify_column(_meta(df, "v"), df["v"]) == "Numerical"
    # Same column, 50 unique -> not Numerical (falls to Textual).
    df2 = pd.DataFrame({"v": [float(i % 50) for i in range(100)]})
    assert classify_column(_meta(df2, "v"), df2["v"]) == "Textual"


def test_integer_age_is_textual_by_the_papers_literal_rule():
    # Integer age: 26 unique of 120 rows. Not Binary, not Categorical
    # (numeric), not Numerical (26 <= 0.8*120) -> Textual. The
    # recommender retains it as numeric (see test below); the classifier
    # itself follows the paper exactly.
    df = pd.DataFrame({"age": list(range(65, 91)) * 5})
    assert classify_column(_meta(df, "age"), df["age"]) == "Textual"


def test_recommend_and_implement_round_trip():
    df = pd.DataFrame(
        {
            "age": [70, 80, 75, 65, 72, 68, 77, 71, 69, 74] * 3,
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"] * 3,
            "score": [float(i) + 0.5 for i in range(30)],
            "site": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 3,
            "notes": ["hello world this is a long note"] * 30,
            "target": [0, 1] * 15,
        }
    )
    df.loc[0, "score"] = None  # a null to impute (29 unique left, still > 24)
    plan = recommend(df, describe_columns(df), target="target")
    types = {p.column: p.col_type for p in plan.columns}
    assert types["gender"] == "Binary"
    assert types["site"] == "Binary"  # exactly 2 unique -> Binary, not Categorical
    assert types["score"] == "Numerical"  # 29 unique > 0.8 * 30 rows
    assert types["age"] == "Textual"  # integer, 10 unique <= 24 -> Textual
    assert types["notes"] == "Textual"
    assert "target" not in types  # target never transformed
    assert "notes" in plan.dropped
    assert "age" not in plan.dropped  # numeric Textual is retained
    out, applied = implement(df, plan)
    assert "notes" not in out.columns
    assert "age" in out.columns
    assert out["score"].isna().sum() == 0  # median-imputed
    assert set(out["gender"].unique()) <= {0, 1}  # binary-mapped
    assert abs(out["score"].mean()) < 0.01  # standardized
    assert any("median-imputed" in a for a in applied)
    assert any("binary-mapped" in a for a in applied)


def test_numeric_textual_column_is_retained_not_dropped():
    # Integer age types as Textual (paper's rule) but is numeric, so the
    # recommender retains it as a numeric feature instead of dropping it.
    df = pd.DataFrame(
        {"age": list(range(65, 91)) * 5, "target": [0, 1] * 65}
    )
    plan = recommend(df, describe_columns(df), target="target")
    assert "age" not in plan.dropped
    out, _ = implement(df, plan)
    assert "age" in out.columns
