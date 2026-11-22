"""agent-clinical-inference — run the 7-agent pipeline in your browser.

A thin FastAPI front door over the deterministic clinical pipeline:
pick a bundled synthetic dataset (or upload a CSV), press Run, and
watch all seven agents work — ingestion, local anonymization,
feature extraction, model matching, preprocessing, inference, and
explanation — with the typed run journal as the audit trail.

    pip install -r web/requirements.txt
    uvicorn web.app:app --reload        # http://127.0.0.1:8000

Mock mode by default: deterministic, offline, no API key. If
GEMINI_API_KEY is set in the environment, the two LLM steps (model
pick, image headers) use Gemini, with an automatic fall back to
mock mode if the live call fails. Every run executes in a fresh
temp directory; uploaded CSVs are processed there and never stored.

Deployed on Render's free tier (see render.yaml at the repo root).
Demo on synthetic data only — not for clinical use.
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The pipeline package lives in src/ (see pyproject.toml); put it on
# the path so `uvicorn web.app:app` works from the repo root with no
# install step, same trick as the cookbook UI's runners.
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from clinical_inference.prompts import SYSTEM_PROMPT  # noqa: E402
from clinical_inference.pipeline import run_pipeline  # noqa: E402
from clinical_inference.registry import (  # noqa: E402
    default_registry_path,
    load_registry,
)

GITHUB_URL = "https://github.com/balasaikunapareddi/agent-clinical-inference"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB — plenty for a demo CSV

DATASETS = {
    "fall_risk": {
        "label": "Fall-risk (gait data)",
        "path": REPO_ROOT / "data" / "fall_risk_mock.csv",
        "blurb": "200 synthetic rows, GSTRIDE-style: gait speed, stride "
                 "length, foot pressure, step-time variability → fell. "
                 "Includes fake PII columns so you can watch the "
                 "anonymizer mask and drop them.",
    },
    "anxiety": {
        "label": "Anxiety (palliative-care data)",
        "path": REPO_ROOT / "data" / "anxiety_mock.csv",
        "blurb": "200 synthetic rows: age, antidepressant use, pain / "
                 "fatigue / hope scores → anxiety_high. Same fake PII "
                 "columns, same local masking.",
    },
}

# The two live-mode LLM task prompts, quoted from llm.py with the
# interpolated parts as {placeholders} (same convention as the
# cookbook UI).
MODEL_PICK_PROMPT = (
    "You are choosing the best clinical ML model for a dataset. "
    "Eligible models (all required features matched):\n"
    "- {model name}: {description} (mean match {score}; features: "
    "{matched columns})\n"
    "...\n"
    "Reply with exactly two lines:\n"
    "MODEL: <model name>\nJUSTIFICATION: <one sentence>"
)
IMAGE_PROMPT = (
    "What imaging modality is this, and what disease type (if any) "
    "is visible? Reply with exactly two lines:\n"
    "MODALITY: <modality>\nDISEASE: <disease type or none>"
)

app = FastAPI(title="agent-clinical-inference")

STYLE = """
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; color: #1a1a1a; background: #f6f6f4; }
  header { background: #173a2f; color: #fff; padding: 14px 20px; }
  header h1 { margin: 0; font-size: 18px; }
  header h1 a { color: #fff; text-decoration: none; }
  header p { margin: 4px 0 0; font-size: 13px; opacity: .8; }
  main { max-width: 960px; margin: 0 auto; padding: 20px 16px 40px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
  .card h2 { margin: 0 0 6px; font-size: 16px; }
  .card p { margin: 0 0 10px; font-size: 13px; line-height: 1.5; color: #444; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
  .panel h2 { margin: 0 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: #555; }
  .warn { background: #fdf6e3; border: 1px solid #e8d9a0; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 13px; line-height: 1.6; }
  .error { background: #fdecea; border: 1px solid #e5b4ae; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 13px; line-height: 1.6; }
  pre.output { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  button[type=submit] { padding: 9px 16px; background: #173a2f; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 8px; }
  button.run-big { padding: 14px 28px; font-size: 17px; font-weight: 700; border-radius: 8px; margin-top: 10px; }
  .run-panel { border: 2px solid #173a2f; }
  .or { text-align: center; color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; margin: 16px 0 4px; }
  input[type=file] { font-size: 14px; margin-top: 4px; }
  .back { font-size: 13px; color: #173a2f; text-decoration: none; }
  .back:hover { text-decoration: underline; }
  .hint { font-size: 12px; color: #888; margin-top: 8px; line-height: 1.5; }
  .meta { font-size: 13px; color: #555; line-height: 1.6; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f0f0ec; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  details summary { cursor: pointer; font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: #555; }
  details pre.output { margin-top: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
  th { color: #555; font-weight: 600; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
  .metric { background: #eef4f1; border: 1px solid #cfded6; border-radius: 8px; padding: 10px 12px; }
  .metric .value { font-size: 20px; font-weight: 700; color: #173a2f; }
  .metric .label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #555; }
  .badge { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #173a2f; background: #eef4f1; border: 1px solid #cfded6; border-radius: 10px; padding: 2px 8px; }
  .timeline { list-style: none; margin: 0; padding: 0; }
  .timeline li { border-left: 3px solid #cfded6; padding: 2px 0 12px 14px; position: relative; }
  .timeline .agent { font-weight: 600; font-size: 13px; }
  .timeline .decision { font-size: 13px; }
  .timeline .details { font-size: 13px; color: #444; line-height: 1.5; margin-top: 2px; }
  .timeline details { margin-top: 4px; }
  .timeline details summary { font-size: 11px; text-transform: none; letter-spacing: 0; color: #888; }
  .download { display: inline-block; background: #173a2f; color: #fff; text-decoration: none; font-size: 13px; padding: 7px 12px; border-radius: 6px; margin-top: 8px; }
"""

DISCLAIMER = (
    '<div class="warn"><strong>Not for clinical use.</strong> This is a '
    "research demo (arXiv:2507.18115) on synthetic mock data — no real "
    "patient data, no real PHI, no clinical validation. Uploaded CSVs "
    "are processed in a temporary directory and never stored; please "
    "don't upload anything real anyway.</div>"
)


def _page(title, body):
    return HTMLResponse(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)} — agent-clinical-inference</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        "<header><h1><a href=\"/\">agent-clinical-inference</a></h1>"
        "<p>Seven agents take raw medical data from ingestion to "
        "interpretable inference — run the whole pipeline right here.</p>"
        "</header>\n"
        f"<main>{body}</main>\n</body>\n</html>"
    )


def _system_prompt_section():
    return (
        '<details class="panel"><summary>System prompts</summary>'
        '<p class="meta" style="margin-top:10px"><strong>Conversational '
        "agent (Strands, <code>agent.py</code>)</strong> — the agent "
        "writes message text only; the deterministic state machine owns "
        "every pipeline transition:</p>"
        f'<pre class="output">{html.escape(SYSTEM_PROMPT)}</pre>'
        '<p class="meta" style="margin-top:12px"><strong>Model-pick '
        "prompt (live mode only, <code>llm.py</code>)</strong> — in "
        "mock mode the highest-scoring eligible model is picked "
        "deterministically instead:</p>"
        f'<pre class="output">{html.escape(MODEL_PICK_PROMPT)}</pre>'
        '<p class="meta" style="margin-top:12px"><strong>Image-header '
        "prompt (live mode only, <code>llm.py</code>)</strong>:</p>"
        f'<pre class="output">{html.escape(IMAGE_PROMPT)}</pre>'
        '<p class="hint">Raw PII never reaches any of these prompts: '
        "anonymization runs locally, before matching, modeling, or any "
        "LLM call.</p></details>"
    )


def _models_section():
    try:
        models = load_registry(default_registry_path())
    except Exception:  # keep the page up even if the registry breaks
        return ""
    rows = []
    for m in models:
        needs = (
            ", ".join(m.required_features)
            if m.required_features
            else f"{m.image_modality} / {m.disease_type}"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(m.name)}</code></td>"
            f"<td>{html.escape(m.task)}</td>"
            f"<td>{html.escape(needs)}</td>"
            f"<td><code>{html.escape(m.target)}</code></td>"
            "</tr>"
        )
    return (
        '<section class="panel"><h2>Registered models</h2>'
        "<table><tr><th>Model</th><th>Task</th><th>Needs these columns</th>"
        f"<th>Target</th></tr>{''.join(rows)}</table>"
        '<p class="hint">Your CSV is matched against this registry: every '
        "required feature must match one of your columns with similarity "
        "≥ 0.6 (synonyms count), one column per feature. The tabular "
        "models are trained on the fly from your anonymized data; the "
        "image model is a marked stub in v0.1.</p></section>"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    cards = []
    for key, ds in DATASETS.items():
        cards.append(
            '<article class="card">'
            f"<h2>{html.escape(ds['label'])}</h2>"
            f"<p>{html.escape(ds['blurb'])}</p>"
            '<form method="post" action="/run">'
            f'<input type="hidden" name="dataset" value="{key}">'
            '<button type="submit" class="run-big">Run pipeline</button>'
            "</form></article>"
        )
    body = (
        '<p class="meta" style="margin-top:0">You have a CSV of patient '
        "data and no ML team. This pipeline anonymizes it locally, "
        "matches it to a registered model, trains, and explains every "
        "prediction — with a typed audit journal for all seven agent "
        "steps. Pick a bundled dataset or bring your own CSV.</p>"
        # The run panel is the first thing on the page: nothing
        # executes until the visitor presses a Run button.
        '<section class="panel run-panel"><h2>Run the pipeline</h2>'
        f'<div class="cards">{"".join(cards)}</div>'
        '<p class="hint">Bundled data is synthetic (seeded RNG, 200 rows '
        "each). A run takes a few seconds: 150 rows train, 50 test.</p>"
        '<p class="or">— or —</p>'
        '<form method="post" action="/run" enctype="multipart/form-data">'
        '<label class="meta" for="csv-file"><strong>Upload your own '
        "CSV</strong></label><br>"
        '<input type="file" id="csv-file" name="file" accept=".csv"><br>'
        '<button type="submit" class="run-big">Run pipeline</button>'
        '<p class="hint">CSV only, up to 5 MB. Processed in a temporary '
        "directory and deleted when the run finishes — never stored, "
        "never sent anywhere. Identifier columns (name, email, phone, "
        "MRN, DOB…) are masked locally before anything else touches "
        "the data. Your columns need to match a registered model "
        "below, target column included.</p></form></section>"
        f"{DISCLAIMER}"
        f"{_models_section()}"
        f"{_system_prompt_section()}"
        '<section class="panel"><h2>Run it yourself</h2>'
        f'<p class="meta">Code: <a href="{GITHUB_URL}">agent-clinical-inference</a> '
        "on GitHub.<br>"
        "Locally: <code>make demo</code> (fall-risk) or "
        "<code>make demo-anxiety</code> — full pipeline, mock mode, no key. "
        "Tests: <code>make test</code>. "
        "This UI: <code>uvicorn web.app:app --reload</code>.</p>"
        "</section>"
    )
    return _page("Run the pipeline", body)


def _run(input_path, out_base):
    """Run the pipeline once; live only if a key is configured, with a
    mock fall back so the demo never dies on a network hiccup."""
    live = bool(os.environ.get("GEMINI_API_KEY"))
    try:
        result = run_pipeline(input_path, out_dir=out_base, live=live)
        return result, live, None
    except Exception:
        if not live:
            raise
        result = run_pipeline(input_path, out_dir=out_base, live=False)
        return result, False, (
            "Live mode (Gemini) failed, so this run fell back to the "
            "deterministic mock mode."
        )


def _partial_journal(out_base):
    """On failure the pipeline still writes journal.json into its run
    dir — recover it so the error page can show how far it got."""
    try:
        for journal_path in sorted(Path(out_base).glob("*/journal.json")):
            data = json.loads(journal_path.read_text())
            return data.get("entries", [])
    except Exception:
        pass
    return []


def _timeline(entries):
    items = []
    for e in entries:
        outputs = e.get("outputs") or {}
        outputs_html = ""
        if outputs:
            pretty = json.dumps(outputs, indent=2)
            outputs_html = (
                "<details><summary>outputs</summary>"
                f'<pre class="output">{html.escape(pretty)}</pre></details>'
            )
        items.append(
            "<li>"
            f'<span class="agent">Step {e.get("step")}: '
            f'{html.escape(str(e.get("agent", "")))}</span> '
            f'<span class="badge">{html.escape(str(e.get("state", "")))}</span>'
            f'<div class="decision">{html.escape(str(e.get("decision", "")))}</div>'
            f'<div class="details">{html.escape(str(e.get("details", "")))}</div>'
            f"{outputs_html}</li>"
        )
    return f'<ol class="timeline">{"".join(items)}</ol>'


def _results_page(result, *, source_label, live, note, predictions_bytes):
    entries = [vars(e) for e in result.journal.entries]
    matcher = next(
        (e for e in entries if e.get("agent") == "model_data_matcher"), {}
    )
    m_outputs = matcher.get("outputs") or {}
    mapping = m_outputs.get("mapping") or {}
    scores = m_outputs.get("scores") or {}
    mapping_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(feat))}</code></td>"
        f"<td><code>{html.escape(str(col))}</code></td>"
        f"<td>{scores.get(feat, '')}</td>"
        "</tr>"
        for feat, col in mapping.items()
    )
    metrics = result.metrics or {}
    metric_cards = "".join(
        '<div class="metric">'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="label">{html.escape(label)}</div></div>'
        for label, value in (
            ("Accuracy", metrics.get("accuracy", "—")),
            ("ROC-AUC", metrics.get("roc_auc", "—")),
            ("Train rows", metrics.get("n_train", "—")),
            ("Test rows", metrics.get("n_test", "—")),
        )
    )
    feature_rows = "".join(
        "<tr>"
        f"<td>{i}</td>"
        f"<td><code>{html.escape(str(f.get('feature', '')))}</code></td>"
        f"<td>{html.escape(str(f.get('importance', '')))}</td>"
        f"<td>{html.escape(str(f.get('method', '')))}</td>"
        "</tr>"
        for i, f in enumerate(result.top_features[:10], start=1)
    )
    download_html = ""
    if predictions_bytes:
        encoded = base64.b64encode(predictions_bytes).decode()
        n_rows = max(predictions_bytes.count(b"\n") - 1, 0)
        download_html = (
            f'<a class="download" href="data:text/csv;base64,{encoded}" '
            f'download="predictions.csv">Download predictions.csv '
            f"({n_rows} rows)</a>"
            '<p class="hint">The run itself lived in a temp directory '
            "and is already gone — this download is your copy.</p>"
        )
    mode = "live mode (Gemini)" if live else "mock mode (deterministic, no key)"
    note_html = f'<div class="warn">{html.escape(note)}</div>' if note else ""
    body = (
        '<p><a class="back" href="/">&larr; Run another dataset</a></p>'
        f"<h1 style=\"margin:0 0 6px;font-size:22px\">Run "
        f"{html.escape(result.run_id)} — {html.escape(result.state.value)}</h1>"
        f'<p class="meta">Source: {html.escape(source_label)} · {mode} · '
        f"model: <code>{html.escape(str(result.model_name))}</code></p>"
        f"{note_html}{DISCLAIMER}"
        '<section class="panel"><h2>Headline metrics</h2>'
        f'<div class="metrics">{metric_cards}</div>'
        '<p class="hint">HistGradientBoosting trained on the fly from '
        "the anonymized, preprocessed data (75/25 split). Small-data "
        "demo numbers, not a clinical claim.</p></section>"
        '<section class="panel"><h2>Matched model</h2>'
        f'<p class="meta"><strong>{html.escape(str(result.model_name))}</strong> — '
        f"{html.escape(str(matcher.get('details', '')))}</p>"
        "<table><tr><th>Required feature</th><th>Your column</th>"
        f"<th>Match score</th></tr>{mapping_rows}</table>"
        '<p class="hint">Every required feature must score ≥ 0.6; each '
        "of your columns is used at most once (greedy one-to-one).</p>"
        "</section>"
        '<section class="panel"><h2>Top explanations</h2>'
        "<table><tr><th>#</th><th>Feature</th><th>Importance</th>"
        f"<th>Method</th></tr>{feature_rows}</table></section>"
        '<section class="panel"><h2>Predictions</h2>'
        f"{download_html}</section>"
        '<section class="panel"><h2>Run journal — every agent step</h2>'
        f"{_timeline(entries)}</section>"
        f"{_system_prompt_section()}"
    )
    return _page(f"Run {result.run_id}", body)


def _error_page(message, entries):
    timeline_html = ""
    if entries:
        timeline_html = (
            '<section class="panel"><h2>How far it got</h2>'
            f"{_timeline(entries)}</section>"
        )
    body = (
        '<p><a class="back" href="/">&larr; Run another dataset</a></p>'
        "<h1 style=\"margin:0 0 6px;font-size:22px\">The pipeline "
        "stopped — cleanly</h1>"
        f'<div class="error">{html.escape(message)}</div>'
        '<p class="meta">That is by design: the state machine moves to '
        "FAILED and journals why, instead of guessing. The most common "
        "cause is headers that don't match any registered model above "
        "the 0.6 threshold — check the model list on the home page "
        "(your CSV needs the required columns <em>and</em> the target "
        "column).</p>"
        f"{timeline_html}"
    )
    return _page("Pipeline stopped", body)


@app.post("/run", response_class=HTMLResponse)
async def run(dataset: str = Form(default=None),
              file: UploadFile = File(default=None)):
    with tempfile.TemporaryDirectory(prefix="clinical-run-") as tmp:
        out_base = Path(tmp) / "runs"
        if file is not None and file.filename:
            if not file.filename.lower().endswith(".csv"):
                return _error_page(
                    "Only CSV uploads are supported here. "
                    f"Got: {file.filename}",
                    [],
                )
            content = await file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                return _error_page(
                    f"That file is {len(content) // 1024} KB — the demo "
                    "caps uploads at 5 MB. The bundled datasets are "
                    "200 rows; something in that range works best.",
                    [],
                )
            if not content.strip():
                return _error_page("That file is empty.", [])
            input_path = Path(tmp) / "upload.csv"
            input_path.write_bytes(content)
            source_label = f"uploaded CSV ({file.filename})"
        elif dataset in DATASETS:
            input_path = DATASETS[dataset]["path"]
            source_label = f"bundled dataset: {DATASETS[dataset]['label']}"
        else:
            return _error_page(
                "Pick a bundled dataset or choose a CSV file first.", []
            )
        try:
            result, live, note = _run(input_path, out_base)
        except Exception as exc:  # journal the failure, show it, never 500
            return _error_page(str(exc), _partial_journal(out_base))
        predictions_bytes = None
        pred_path = Path(result.run_dir) / "predictions.csv"
        if pred_path.exists():
            predictions_bytes = pred_path.read_bytes()
        return _results_page(
            result,
            source_label=source_label,
            live=live,
            note=note,
            predictions_bytes=predictions_bytes,
        )
