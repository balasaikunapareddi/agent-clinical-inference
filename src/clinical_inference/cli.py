"""CLI — one-command demos for the clinical inference pipeline.

    python -m clinical_inference.cli --input data/fall_risk_mock.csv
    python -m clinical_inference.cli --input data/anxiety_mock.csv --live
    python -m clinical_inference.cli --list-models
"""

from __future__ import annotations

import argparse
import sys

from clinical_inference.pipeline import run_pipeline
from clinical_inference.registry import default_registry_path, load_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agentic clinical data pipeline (arXiv:2507.18115)."
    )
    parser.add_argument(
        "--input", default="data/fall_risk_mock.csv",
        help="Input file: CSV, XLSX, JSON, ZIP or image.",
    )
    parser.add_argument(
        "--out", default="runs", help="Directory for run outputs."
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use Gemini (free tier) for the LLM steps. Needs GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Force a registered model by name (must be eligible).",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List the registered models and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        for m in load_registry(default_registry_path()):
            print(f"- {m.name} [{m.modality}] — {m.task}")
            print(f"  requires: {', '.join(m.required_features) or m.image_modality + '/' + m.disease_type}")
        return 0

    try:
        result = run_pipeline(
            args.input, out_dir=args.out, live=args.live, force_model=args.model
        )
    except Exception as exc:  # noqa: BLE001 — print a clean failure, exit 1
        print(f"Pipeline FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Run {result.run_id} — state: {result.state.value}")
    print(f"Run dir: {result.run_dir}")
    if result.model_name:
        print(f"Model: {result.model_name}")
    if result.metrics:
        print(
            f"Metrics: accuracy={result.metrics['accuracy']}, "
            f"ROC-AUC={result.metrics['roc_auc']} "
            f"(train={result.metrics['n_train']}, test={result.metrics['n_test']})"
        )
    if result.top_features:
        print("Top features:")
        for f in result.top_features[:5]:
            print(f"  - {f['feature']}: {f['importance']} ({f['method']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
