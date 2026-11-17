"""Tests for Agent 7 — Model Inference."""

import numpy as np
import pandas as pd
import pytest

from clinical_inference.inference import (
    explain,
    infer_image_mock,
    train_and_predict,
)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (x1 + x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(0, 1, n), "y": y})


def test_train_and_predict_metrics_and_shape():
    df = _frame()
    result = train_and_predict(df, "y", ["x1", "x2", "noise"], "test-gbm")
    assert 0.6 < result.metrics["accuracy"] <= 1.0
    assert 0.6 < result.metrics["roc_auc"] <= 1.0
    assert result.metrics["n_train"] + result.metrics["n_test"] == 200
    assert set(result.predictions.columns) >= {"y", "predicted", "predicted_proba"}
    assert len(result.predictions) == result.metrics["n_test"]
    assert result.predictions["predicted"].isin([0, 1]).all()


def test_explain_top10_uses_permutation_when_no_shap():
    df = _frame()
    result = train_and_predict(df, "y", ["x1", "x2", "noise"], "test-gbm")
    assert len(result.top_features) == 3  # only 3 features exist
    assert result.top_features[0]["method"] in (
        "permutation_importance",
        "shap",
    )
    # The two signal features outrank the noise feature.
    ranked = [f["feature"] for f in result.top_features]
    assert ranked[-1] == "noise"


def test_explain_direct():
    df = _frame()
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(df[["x1", "x2", "noise"]], df["y"])
    top = explain(model, df[["x1", "x2", "noise"]], df["y"], ["x1", "x2", "noise"])
    assert len(top) == 3
    assert all("importance" in f for f in top)


def test_single_class_target_raises():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="single class"):
        train_and_predict(df, "y", ["x1"], "test-gbm")


def test_image_inference_is_explicit_mock():
    result = infer_image_mock({"modality": "colonoscopy", "disease_type": "polyp"})
    assert result["result"] == "mock"
    assert result["detections"] == []
    assert "MOCK" in result["note"]
