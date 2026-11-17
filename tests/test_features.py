"""Tests for Agent 3 — Feature Extraction."""

import pandas as pd

from clinical_inference.features import (
    describe_columns,
    extract_headers,
    extract_image_headers,
)


def test_headers_are_column_names():
    df = pd.DataFrame({"age": [70], "gait_speed": [0.9], "fell": [0]})
    assert extract_headers(df) == ["age", "gait_speed", "fell"]


def test_describe_columns_metadata():
    df = pd.DataFrame(
        {"age": [70, 80, None], "gender": ["M", "F", "M"], "score": [1.5, 2.5, 3.5]}
    )
    metas = {m.name: m for m in describe_columns(df)}
    assert metas["age"].null_count == 1
    assert metas["age"].unique_count == 2
    assert metas["gender"].unique_count == 2
    assert metas["score"].min_value == 1.5
    assert metas["score"].max_value == 3.5


def test_image_headers_mock_is_labeled():
    headers = extract_image_headers("fake.png")
    assert headers["source"] == "mock"
    assert headers["modality"] == "colonoscopy"
    assert headers["disease_type"] == "polyp"
    assert "MOCK" in headers["note"]
