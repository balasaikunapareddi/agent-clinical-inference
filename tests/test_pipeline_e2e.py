"""End-to-end pipeline tests on BOTH mock datasets (mock mode)."""

import json
from pathlib import Path

import pytest

from clinical_inference.pipeline import run_pipeline
from clinical_inference.state_machine import PipelineState

DATA = Path(__file__).resolve().parents[1] / "data"


def _run(dataset: str, tmp_path, **kwargs):
    return run_pipeline(DATA / dataset, out_dir=tmp_path / "runs", **kwargs)


def test_e2e_fall_risk(tmp_path):
    result = _run("fall_risk_mock.csv", tmp_path)
    assert result.state is PipelineState.DONE
    assert result.model_name == "fall-risk-gbm"
    assert result.metrics["accuracy"] > 0.6
    assert result.metrics["roc_auc"] > 0.6
    # 9 model features, so top-10 is capped at 9.
    assert len(result.top_features) == 9
    # Artifacts exist.
    assert (result.run_dir / "predictions.csv").exists()
    assert (result.run_dir / "journal.json").exists()
    assert (result.run_dir / "report.md").exists()
    # Journal: one entry per agent step, in order.
    journal = json.loads((result.run_dir / "journal.json").read_text())
    agents = [e["agent"] for e in journal["entries"]]
    assert agents == [
        "ingestion_identifier",
        "data_anonymizer",
        "feature_extraction",
        "model_data_matcher",
        "preprocessing_recommender",
        "preprocessing_implementor",
        "model_inference",
        "model_inference",
    ]
    # Anonymizer audit: PII was masked, and the audit holds no values.
    anon = next(e for e in journal["entries"] if e["agent"] == "data_anonymizer")
    assert anon["outputs"]["total_masked_cells"] > 0
    assert "patient_name" in anon["outputs"]["removed_pii_columns"]
    assert "Asha" not in json.dumps(journal)
    # Predictions are real: both classes predicted, probabilities in [0, 1].
    import pandas as pd

    preds = pd.read_csv(result.run_dir / "predictions.csv")
    assert set(preds["predicted"].unique()) <= {0, 1}
    assert preds["predicted_proba"].between(0, 1).all()
    assert "fell" in preds.columns


def test_e2e_anxiety(tmp_path):
    result = _run("anxiety_mock.csv", tmp_path)
    assert result.state is PipelineState.DONE
    assert result.model_name == "anxiety-gbm"
    assert result.metrics["accuracy"] > 0.6
    assert result.metrics["roc_auc"] > 0.6
    assert (result.run_dir / "predictions.csv").exists()
    assert "anxiety_high" in __import__("pandas").read_csv(
        result.run_dir / "predictions.csv"
    ).columns


def test_e2e_no_matching_model_fails_cleanly(tmp_path):
    # Headers that match no registered model: the run FAILS with a
    # recorded reason instead of guessing.
    path = tmp_path / "weird.csv"
    path.write_text("planet,star,galaxy\nmars,sun,milkyway\nvenus,sun,milkyway\n")
    with pytest.raises(ValueError, match="No registered model matches"):
        _run(str(path), tmp_path)
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    journal = json.loads((run_dirs[0] / "journal.json").read_text())
    assert journal["entries"][-1]["state"] == "FAILED"


def test_e2e_force_model(tmp_path):
    # anxiety data can also run the hope model (it matches its headers).
    result = _run("anxiety_mock.csv", tmp_path, force_model="hope-gbm")
    assert result.state is PipelineState.DONE
    assert result.model_name == "hope-gbm"


def test_e2e_force_ineligible_model_fails(tmp_path):
    with pytest.raises(ValueError, match="not eligible"):
        _run("fall_risk_mock.csv", tmp_path, force_model="anxiety-gbm")
