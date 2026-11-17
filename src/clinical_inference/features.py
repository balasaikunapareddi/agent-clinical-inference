"""Agent 3 — Feature Extraction (paper Table I: Ingestion_Selector).

Tabular: the paper uses column names as "headers" — candidate variables
for matching and preprocessing. Image: the paper uses a multi-stage
MedGemma pipeline (modality -> disease type). This build keeps the
tabular path exact and ships the image path as a clearly-marked mock
in v0.1, with an optional Gemini-vision --live path (the MedGemma
substitute — same job, different vision model).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnMeta:
    name: str
    dtype: str
    null_count: int
    unique_count: int
    min_value: float | str | None = None
    max_value: float | str | None = None


def extract_headers(df: pd.DataFrame) -> list[str]:
    """Tabular "headers": the column names, per the paper."""
    return [str(c) for c in df.columns]


def describe_columns(df: pd.DataFrame) -> list[ColumnMeta]:
    """Per-column metadata the Preprocessing Recommender consumes."""
    metas: list[ColumnMeta] = []
    for col in df.columns:
        series = df[col]
        meta = ColumnMeta(
            name=str(col),
            dtype=str(series.dtype),
            null_count=int(series.isna().sum()),
            unique_count=int(series.nunique(dropna=True)),
        )
        try:
            if pd.api.types.is_numeric_dtype(series):
                meta.min_value = float(series.min())
                meta.max_value = float(series.max())
        except (TypeError, ValueError):
            pass
        metas.append(meta)
    return metas


# --- Image path (mock in v0.1; Gemini vision optional with --live) ---

MOCK_IMAGE_HEADERS = {
    "modality": "colonoscopy",
    "disease_type": "polyp",
    "source": "mock",
    "note": "MOCK v0.1: no real vision model. --live uses Gemini vision "
    "as the MedGemma substitute.",
}


def extract_image_headers(
    path: str, live: bool = False, llm=None
) -> dict:
    """Multi-stage image headers (modality -> disease type).

    Mock by default: returns the mock colonoscopy/polyp headers, clearly
    labeled. With live=True and an LLM client, asks Gemini vision to
    classify modality and disease type instead.
    """
    if live and llm is not None:
        result = llm.describe_image(path)
        result["source"] = "gemini-vision"
        return result
    return dict(MOCK_IMAGE_HEADERS)
