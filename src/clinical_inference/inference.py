"""Agent 7 — Model Inference (paper Table I: Model_Inferencer).

The paper: small data -> Perpetual Booster (GBM); large data -> custom
deep learning; images -> fine-tuned DETR with bounding boxes; SHAP/LIME
top-10 features + DETR attention maps; structured CSV output.

This build: sklearn HistGradientBoosting as the Perpetual Booster
equivalent (same family, same job), trained on the anonymized
preprocessed data and applied to a held-out split. Explainability is
SHAP when installed, else deterministic permutation importance
(sklearn), top 10 features. Output: structured predictions CSV.
The image path is a clearly-marked mock stub in v0.1 — no fake DETR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class InferenceResult:
    predictions: pd.DataFrame
    metrics: dict
    top_features: list[dict]
    model_name: str


def train_and_predict(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    test_size: float = 0.25,
    seed: int = 42,
) -> InferenceResult:
    """Train the GBM on the processed data, predict the held-out split."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    if len(y.unique()) < 2:
        raise ValueError(f"Target '{target}' has a single class after filtering.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Small-data tuning (the paper's Perpetual Booster use case): the
    # sklearn defaults assume large data — min_samples_leaf=20 and
    # early stopping on a 10% validation split both starve a 90-row
    # training set. These settings let the booster actually split.
    model = HistGradientBoostingClassifier(
        random_state=seed,
        min_samples_leaf=5,
        max_leaf_nodes=15,
        max_iter=200,
        learning_rate=0.08,
        early_stopping=False,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_positive_rate": round(float(y_test.mean()), 4),
    }
    predictions = X_test.copy()
    predictions[target] = y_test.values
    predictions["predicted"] = pred
    predictions["predicted_proba"] = np.round(proba, 4)
    top_features = explain(model, X_train, y_train, list(X.columns))
    return InferenceResult(
        predictions=predictions.reset_index(drop=True),
        metrics=metrics,
        top_features=top_features,
        model_name=model_name,
    )


def explain(model, X_train, y_train, feature_names, top_n: int = 10) -> list[dict]:
    """Top-N influential features. SHAP when installed, else
    deterministic permutation importance (same job, no extra dep)."""
    try:
        import shap  # type: ignore

        explainer = shap.Explainer(model, X_train)
        values = explainer(X_train[:200])
        importance = np.abs(values.values).mean(axis=0)
        method = "shap"
    except ImportError:
        result = permutation_importance(
            model, X_train, y_train, n_repeats=5, random_state=42
        )
        importance = result.importances_mean
        method = "permutation_importance"
    order = np.argsort(-importance)[:top_n]
    return [
        {
            "feature": feature_names[i],
            "importance": round(float(importance[i]), 4),
            "method": method,
        }
        for i in order
    ]


def infer_image_mock(image_headers: dict) -> dict:
    """Image inference stub (v0.1): no real DETR weights exist, so this
    returns an explicit MOCK result instead of faking a detection."""
    return {
        "result": "mock",
        "modality": image_headers.get("modality"),
        "disease_type": image_headers.get("disease_type"),
        "detections": [],
        "note": "MOCK v0.1: DETR inference is a stub. Real fine-tuned DETR "
        "weights are a future step; nothing here is a real detection.",
    }
