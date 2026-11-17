"""Tests for Agent 4 — Model-Data Matcher."""

from clinical_inference.matcher import (
    THRESHOLD,
    match_features,
    match_models,
    pick_best,
    token_overlap,
)
from clinical_inference.registry import load_registry, default_registry_path


def test_exact_header_match_scores_one():
    assert token_overlap("gait_speed", "gait_speed") == 1.0


def test_synonym_match():
    # "sex" is a synonym of "gender" in the matcher.
    assert token_overlap("sex", "gender") == 1.0
    # "walking speed" tokenizes to gait+speed, matching "gait_speed".
    assert token_overlap("walking speed", "gait_speed") == 1.0


def test_unrelated_headers_score_below_threshold():
    assert token_overlap("foot_angle", "antidepressant_use") < THRESHOLD


def test_greedy_one_to_one_assignment():
    # Both required features like column "age_years"; only one may take it.
    # "age_years" matches itself at 1.0; "age" vs "age_years" scores 0.5
    # (Jaccard {age}/{age,years}) — below the 0.6 threshold, so "age"
    # stays unmatched instead of stealing the column.
    mapping, scores, unmatched = match_features(
        ["age", "age_years"], ["age_years", "gender"]
    )
    assert mapping == {"age_years": "age_years"}
    assert scores["age_years"] == 1.0
    assert unmatched == ["age"]
    assert len(set(mapping.values())) == len(mapping)  # each column once


def test_match_below_threshold_leaves_unmatched():
    mapping, _, unmatched = match_features(["gait_speed"], ["email"])
    assert mapping == {}
    assert unmatched == ["gait_speed"]


def test_fall_risk_headers_match_fall_model_only():
    models = load_registry(default_registry_path())
    headers = [
        "age", "gender", "gait_speed", "stride_length", "foot_pressure_mean",
        "foot_angle", "ankle_rom", "step_time_variability", "fall_history",
        "clinic_site",
    ]
    candidates = match_models(headers, models)
    names = [c.model.name for c in candidates]
    assert "fall-risk-gbm" in names
    assert "anxiety-gbm" not in names
    assert "hope-gbm" not in names


def test_anxiety_headers_match_anxiety_and_hope():
    models = load_registry(default_registry_path())
    headers = [
        "age", "gender", "antidepressant_use", "vomiting", "pain_score",
        "fatigue_score", "sleep_hours", "appetite_loss", "hope_score",
    ]
    candidates = match_models(headers, models)
    names = [c.model.name for c in candidates]
    assert "anxiety-gbm" in names
    assert "hope-gbm" in names
    assert "fall-risk-gbm" not in names


def test_pick_best_is_deterministic_mock():
    models = load_registry(default_registry_path())
    headers = ["age", "gender", "antidepressant_use", "vomiting",
               "pain_score", "fatigue_score", "hope_score"]
    candidates = match_models(headers, models)
    best, justification = pick_best(candidates)
    assert best.model.name in ("anxiety-gbm", "hope-gbm")
    assert "Mock pick" in justification
