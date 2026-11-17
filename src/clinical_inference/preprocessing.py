"""Agents 5 + 6 — Preprocessing Recommender + Implementor (paper).

Recommender: the paper's EXACT rule-based column typing, applied to
each column's metadata (null/unique counts, dtype):
  - Binary:      exactly 2 unique values.
  - Categorical: low unique values with short string lengths.
  - Numerical:   numeric dtype AND unique values > 0.8 * total rows.
  - Textual:     everything else.
Per-type recommendations: median/mode imputation, one-hot/ordinal
encoding, standardization. Auto mode when the file exceeds 50 MB
(paper's threshold) — user-guided selection is disabled then.

Implementor: executes the recommended steps on the DataFrame and logs
every applied step to the journal.

Known limitation inherited from the paper's literal rule: an integer
score with few unique values (e.g. pain 0-10 as ints) classifies as
Textual, not Numerical. The mock datasets use continuous scores so
the rule behaves sensibly; see README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from clinical_inference.features import ColumnMeta

AUTO_MODE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB, the paper's threshold

# "Low unique values with short string lengths" — the paper gives no
# numbers; these explicit thresholds are this build's interpretation,
# recorded here so the rule stays deterministic and reviewable.
CATEGORICAL_MAX_UNIQUE = 10
CATEGORICAL_MAX_STR_LEN = 20


def classify_column(meta: ColumnMeta, series: pd.Series) -> str:
    """The paper's exact typing rules, in order."""
    n_unique = meta.unique_count
    n_rows = len(series)
    if n_unique == 2:
        return "Binary"
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        if n_unique <= CATEGORICAL_MAX_UNIQUE:
            lengths = series.dropna().astype(str).str.len()
            if len(lengths) and float(lengths.median()) <= CATEGORICAL_MAX_STR_LEN:
                return "Categorical"
    if pd.api.types.is_numeric_dtype(series) and n_unique > 0.8 * n_rows:
        return "Numerical"
    return "Textual"


@dataclass
class ColumnPlan:
    column: str
    col_type: str
    steps: list[str] = field(default_factory=list)


@dataclass
class PreprocessPlan:
    columns: list[ColumnPlan]
    auto_mode: bool
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _recommend_steps(col_type: str, is_numeric: bool = False) -> list[str]:
    if col_type == "Binary":
        return ["mode_impute", "binary_map_0_1"]
    if col_type == "Categorical":
        return ["mode_impute", "one_hot_encode"]
    if col_type == "Numerical":
        return ["median_impute", "standardize"]
    # Textual. The paper prescribes no step for text. Non-numeric text is
    # dropped from model features; numeric columns that land in Textual
    # under the paper's literal rule (e.g. integer age: too few unique
    # values to be Numerical, too many to be Binary) are retained as
    # numeric features with median imputation — a documented
    # interpretation, not a change to the classifier (see README).
    if is_numeric:
        return ["median_impute", "retain_numeric"]
    return ["drop_from_features"]  # Textual


def recommend(
    df: pd.DataFrame,
    metas: list[ColumnMeta],
    target: str,
    file_size_bytes: int = 0,
) -> PreprocessPlan:
    """Recommend preprocessing per column. The target column is never
    transformed (it is the label, not a feature)."""
    auto_mode = file_size_bytes > AUTO_MODE_SIZE_BYTES
    plans: list[ColumnPlan] = []
    dropped: list[str] = []
    notes: list[str] = []
    if auto_mode:
        notes.append(
            f"Auto mode: file is {file_size_bytes} bytes (> 50 MB); "
            "user-guided preprocessing disabled per the paper."
        )
    for meta in metas:
        col = meta.name
        if col == target:
            notes.append(f"Target column '{col}' left untransformed.")
            continue
        col_type = classify_column(meta, df[col])
        is_numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
        steps = _recommend_steps(col_type, is_numeric)
        plans.append(ColumnPlan(column=col, col_type=col_type, steps=steps))
        if col_type == "Textual" and not is_numeric:
            dropped.append(col)
            notes.append(
                f"Textual column '{col}' dropped from model features "
                "(no numeric encoding in v0.1)."
            )
        elif col_type == "Textual" and is_numeric:
            notes.append(
                f"Column '{col}' is Textual under the paper's literal rule "
                "(numeric, but unique values <= 0.8*rows) — retained as a "
                "numeric feature with median imputation (see README)."
            )
    return PreprocessPlan(
        columns=plans, auto_mode=auto_mode, dropped=dropped, notes=notes
    )


def implement(df: pd.DataFrame, plan: PreprocessPlan) -> tuple[pd.DataFrame, list[str]]:
    """Execute the recommended steps. Returns (processed_df, applied_log)."""
    out = df.copy()
    applied: list[str] = []
    for col_plan in plan.columns:
        col = col_plan.column
        if col not in out.columns:
            continue
        for step in col_plan.steps:
            if step == "median_impute":
                median = out[col].median()
                out[col] = out[col].fillna(median)
                applied.append(f"{col}: median-imputed ({median:.4g})")
            elif step == "mode_impute":
                mode = out[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else ""
                out[col] = out[col].fillna(fill)
                applied.append(f"{col}: mode-imputed ({fill!r})")
            elif step == "binary_map_0_1":
                uniques = sorted(out[col].dropna().unique(), key=str)
                if len(uniques) == 2:
                    mapping = {uniques[0]: 0, uniques[1]: 1}
                    out[col] = out[col].map(mapping)
                    applied.append(f"{col}: binary-mapped {mapping}")
            elif step == "one_hot_encode":
                dummies = pd.get_dummies(out[col], prefix=col, dtype=float)
                out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
                applied.append(
                    f"{col}: one-hot-encoded into {list(dummies.columns)}"
                )
            elif step == "standardize":
                mean, std = out[col].mean(), out[col].std()
                if std and std > 0:
                    out[col] = (out[col] - mean) / std
                    applied.append(f"{col}: standardized (mean={mean:.4g}, std={std:.4g})")
                else:
                    applied.append(f"{col}: standardization skipped (std=0)")
            elif step == "retain_numeric":
                applied.append(f"{col}: retained as numeric (Textual by rule)")
            elif step == "drop_from_features":
                out = out.drop(columns=[col])
                applied.append(f"{col}: dropped (Textual)")
    return out, applied
