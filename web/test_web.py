"""Smoke tests for the clinical-inference web UI.

The UI runs the real pipeline in-process (mock mode), so these
assert the same headline numbers the CLI demos print — if a marker
disappears, the wiring broke. Run from the repo root:

    python3 -m pytest web -q
"""

from pathlib import Path

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_loads_with_datasets_and_disclaimer():
    response = client.get("/")
    assert response.status_code == 200
    # Pain framing + honest scope, up top.
    assert "no ML team" in response.text
    assert "Not for clinical use" in response.text
    # Both bundled datasets, the upload form, and the model registry.
    assert "Fall-risk (gait data)" in response.text
    assert "Anxiety (palliative-care data)" in response.text
    assert 'type="file"' in response.text
    assert "fall-risk-gbm" in response.text
    assert "anxiety-gbm" in response.text


def test_system_prompts_shown():
    response = client.get("/")
    assert response.status_code == 200
    assert "System prompts" in response.text
    # The Strands agent's system prompt, verbatim from agent.py.
    assert "You are the assistant for the clinical inference pipeline" in response.text


def test_run_bundled_fall_risk():
    response = client.post("/run", data={"dataset": "fall_risk"})
    assert response.status_code == 200
    # Same numbers as `make demo` (verified in the repo README evals).
    assert "fall-risk-gbm" in response.text
    assert "0.68" in response.text
    assert "ROC-AUC" in response.text
    assert "0.757" in response.text
    # Top explanation + journal timeline + predictions download.
    assert "step_time_variability" in response.text
    assert "model_data_matcher" in response.text
    assert "PII masked locally" in response.text
    assert "data:text/csv;base64," in response.text


def test_run_bundled_anxiety():
    response = client.post("/run", data={"dataset": "anxiety"})
    assert response.status_code == 200
    assert "anxiety-gbm" in response.text
    assert "0.76" in response.text
    assert "0.7389" in response.text  # README rounds this to 0.739


def test_upload_csv():
    csv_bytes = (REPO_ROOT / "data" / "fall_risk_mock.csv").read_bytes()
    response = client.post(
        "/run",
        files={"file": ("my_patients.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    assert "fall-risk-gbm" in response.text
    assert "uploaded CSV (my_patients.csv)" in response.text


def test_upload_unmatchable_csv_fails_cleanly():
    # Headers that match no registered model: the pipeline must stop
    # with its journaled reason, and the page must show it — no 500.
    bad = b"foo,bar,baz\n1,2,3\n4,5,6\n"
    response = client.post(
        "/run", files={"file": ("nope.csv", bad, "text/csv")}
    )
    assert response.status_code == 200
    assert "The pipeline stopped" in response.text
    assert "No registered model matches" in response.text
    # The partial journal shows how far it got (ingest + anonymize ran).
    assert "How far it got" in response.text


def test_upload_non_csv_rejected():
    response = client.post(
        "/run", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 200
    assert "Only CSV uploads are supported" in response.text


def test_run_with_no_input_is_a_friendly_error():
    response = client.post("/run", data={})
    assert response.status_code == 200
    assert "Pick a bundled dataset or choose a CSV file first" in response.text


def test_unknown_dataset_is_a_friendly_error():
    response = client.post("/run", data={"dataset": "no-such-dataset"})
    assert response.status_code == 200
    assert "Pick a bundled dataset or choose a CSV file first" in response.text
