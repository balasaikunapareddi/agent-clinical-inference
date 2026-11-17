.PHONY: install test demo demo-live demo-anxiety demo-anxiety-live demo-image chat

VENV = .venv/bin
PY = $(VENV)/python -m
export PYTHONPATH = src

install:
	python3 -m venv .venv
	$(VENV)/pip install -r requirements.txt

test:
	$(VENV)/pytest -q

demo: ## Full pipeline on the mock fall-risk CSV (mock mode, no key)
	$(PY) clinical_inference.cli --input data/fall_risk_mock.csv

demo-live: ## Same, with Gemini for the LLM steps (needs GEMINI_API_KEY)
	$(PY) clinical_inference.cli --input data/fall_risk_mock.csv --live

demo-anxiety: ## Full pipeline on the mock anxiety CSV (mock mode)
	$(PY) clinical_inference.cli --input data/anxiety_mock.csv

demo-anxiety-live: ## Anxiety CSV with Gemini for the LLM steps
	$(PY) clinical_inference.cli --input data/anxiety_mock.csv --live

demo-image: ## Image path (stub in v0.1 — clearly-marked mock result)
	$(PY) clinical_inference.cli --input data/colonoscopy_mock.png

demo-image-live: ## Image path with Gemini vision for modality/disease
	$(PY) clinical_inference.cli --input data/colonoscopy_mock.png --live

chat: ## Interactive Strands agent (needs GEMINI_API_KEY)
	$(PY) clinical_inference.agent

list-models:
	$(PY) clinical_inference.cli --list-models
