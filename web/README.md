# Clinical-inference web UI

The storefront: the full 7-agent pipeline, runnable in the browser.
Pick a bundled synthetic dataset (fall-risk or anxiety) or upload
your own CSV, and watch ingestion → anonymization → matching →
preprocessing → inference → explanation run end to end, with the
typed run journal as the audit trail. No install, no API key.

## Run it locally

```bash
pip install -r web/requirements.txt
uvicorn web.app:app --reload        # http://127.0.0.1:8000
```

(from the repo root — `web` is imported as a package; the app puts
`src/` on the path itself.)

## Test it

```bash
pip install httpx pytest
python3 -m pytest web -q
```

## How it works

- `app.py` — FastAPI app: home page (dataset picker + CSV upload),
  `POST /run`, `/health`. Server-rendered HTML with inline CSS
  (same look as the cookbook / scheduler / booking UIs); no build
  step, no frontend framework.
- Runs call the same `run_pipeline()` the CLI calls — pipeline
  behavior is untouched; the UI only wraps it.
- Every run executes in a fresh temp directory (`out_dir` under a
  `TemporaryDirectory`), so web runs never write into the repo's
  `runs/` and never see each other. Uploaded CSVs live only in that
  temp directory and are deleted when the request finishes.
- Mock mode by default: deterministic, offline, no key. If
  `GEMINI_API_KEY` is set in the environment, the two LLM steps
  (model pick, image headers) run live, with an automatic fall
  back to mock mode if a live call fails.
- The results page renders the run report readably: matched model
  + justification + feature mapping, accuracy / ROC-AUC, top-10
  explanations, the journal as a step timeline, and
  `predictions.csv` as a download (the temp run dir is already
  gone by then — the download is the visitor's copy).
- Failures are shown, not hidden: if the pipeline stops (e.g. no
  registered model matches the headers), the page shows the
  journaled reason and how far the run got.
- The agents' system prompts are displayed in a collapsible
  section (Strands agent prompt + the two live-mode LLM task
  prompts), per the standing rule that UIs show their prompts.

## Deploy

`render.yaml` at the repo root is a Render Blueprint (free tier,
`plan: free`): build installs `web/requirements.txt`, start runs
uvicorn on `$PORT`, health check is `/health`. Nothing to
configure in the dashboard — no keys, no database. Optionally set
`GEMINI_API_KEY` there to enable live mode.

Demo on synthetic data only — not for clinical use.
