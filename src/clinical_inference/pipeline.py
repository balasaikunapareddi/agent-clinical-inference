"""Pipeline orchestrator — the deterministic state machine in action.

Runs INGEST -> ANONYMIZE -> EXTRACT -> MATCH -> PREPROCESS -> INFER
-> EXPLAIN -> DONE, logging every agent step to the typed run journal.
Any failure moves the machine to FAILED and is recorded, never hidden.
Policy gates run before MATCH (headers must be anonymized) and before
INFER (data must be anonymized AND matched).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from clinical_inference import anonymizer, features, ingestion, inference
from clinical_inference.journal import RunJournal
from clinical_inference.llm import LLMClient
from clinical_inference.matcher import match_models, pick_best
from clinical_inference.preprocessing import implement, recommend
from clinical_inference.registry import ModelSpec, default_registry_path, load_registry
from clinical_inference.state_machine import PipelineState, Policy, StateMachine


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    state: PipelineState
    model_name: str | None
    metrics: dict
    top_features: list[dict]
    journal: RunJournal


def _run_dir_for(out_dir: str | Path | None, run_id: str) -> Path:
    base = Path(out_dir) if out_dir else Path("runs")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    live: bool = False,
    registry_path: str | Path | None = None,
    force_model: str | None = None,
) -> RunResult:
    """Run the full 7-agent pipeline on one input file."""
    journal = RunJournal.new()
    sm = StateMachine()
    run_dir = _run_dir_for(out_dir, journal.run_id)
    context: dict = {"anonymized": False, "matched_model": None}
    llm = LLMClient(live=live)
    model_name: str | None = None
    metrics: dict = {}
    top_features: list[dict] = []

    try:
        # --- Agent 1: Ingestion Identifier ---
        detected = ingestion.detect_file(input_path)
        journal.log(
            "ingestion_identifier", sm.state.value, "File type detected",
            inputs={"path": str(input_path)},
            outputs=detected.to_dict(),
            details=f"Detected {detected.file_type.value} ({detected.mime}).",
        )
        if detected.file_type in (ingestion.FileType.PNG, ingestion.FileType.JPEG):
            return _run_image_path(
                input_path, run_dir, journal, sm, llm, detected
            )
        if detected.file_type is ingestion.FileType.UNKNOWN:
            raise ValueError(f"Unsupported file type for {input_path}.")
        df_raw = ingestion.load_table(input_path)
        sm.transition(PipelineState.ANONYMIZE)

        # --- Agent 2: Data Anonymizer ---
        df, audit = anonymizer.anonymize_dataframe(df_raw)
        context["anonymized"] = True
        # Fully-masked identifier columns (every value is "****") carry no
        # signal and must not reach matching or modeling: drop them here,
        # journaling the removal. The audit above already counted them.
        masked_cols = [
            str(c)
            for c in df.columns
            if df[c].notna().any()
            and df[c].dropna().astype(str).eq(anonymizer.MASK).all()
        ]
        if masked_cols:
            df = df.drop(columns=masked_cols)
        journal.log(
            "data_anonymizer", sm.state.value, "PII masked locally",
            inputs={"rows": len(df_raw), "columns": len(df_raw.columns)},
            outputs={**audit.to_dict(), "removed_pii_columns": masked_cols},
            details=(
                f"Masked {sum(audit.masked_by_type.values())} cells "
                f"({audit.masked_by_type or 'none found'}); schema preserved. "
                f"Removed fully-masked identifier columns: {masked_cols or 'none'}."
            ),
        )
        sm.transition(PipelineState.EXTRACT)

        # --- Agent 3: Feature Extraction ---
        headers = features.extract_headers(df)
        metas = features.describe_columns(df)
        journal.log(
            "feature_extraction", sm.state.value, "Headers extracted",
            inputs={"columns": len(headers)},
            outputs={"headers": headers},
            details=f"{len(headers)} tabular headers: {', '.join(headers)}.",
        )
        sm.transition(PipelineState.MATCH)

        # --- Agent 4: Model-Data Matcher ---
        Policy.require_anonymized_headers(context)
        models = load_registry(registry_path or default_registry_path())
        candidates = match_models(headers, models)
        if force_model:
            forced = [c for c in candidates if c.model.name == force_model]
            if not forced:
                raise ValueError(
                    f"Forced model '{force_model}' is not eligible for "
                    f"this data (headers: {headers})."
                )
            candidates = forced
        if not candidates:
            raise ValueError(
                "No registered model matches these headers above the 0.6 "
                f"threshold. Headers: {headers}."
            )
        best, justification = pick_best(
            candidates, llm if live else None
        )
        context["matched_model"] = best.model.name
        model_name = best.model.name
        journal.log(
            "model_data_matcher", sm.state.value, f"Model selected: {model_name}",
            inputs={"headers": headers},
            outputs={
                "model": model_name,
                "mapping": best.mapping,
                "scores": {k: round(v, 3) for k, v in best.scores.items()},
                "mean_score": round(best.mean_score, 3),
                "candidates": len(candidates),
            },
            details=justification,
        )
        sm.transition(PipelineState.PREPROCESS)

        # --- Agents 5 + 6: Preprocessing Recommender + Implementor ---
        plan = recommend(
            df, metas, best.model.target,
            file_size_bytes=Path(input_path).stat().st_size,
        )
        df_proc, applied = implement(df, plan)
        journal.log(
            "preprocessing_recommender", sm.state.value,
            "Column types + steps recommended",
            inputs={"target": best.model.target, "auto_mode": plan.auto_mode},
            outputs={
                "types": {p.column: p.col_type for p in plan.columns},
                "dropped": plan.dropped,
            },
            details="; ".join(plan.notes) or "No notes.",
        )
        journal.log(
            "preprocessing_implementor", sm.state.value, "Steps executed",
            outputs={"applied_steps": applied},
            details=f"{len(applied)} steps applied; shape now {df_proc.shape}.",
        )
        sm.transition(PipelineState.INFER)

        # --- Agent 7: Model Inference ---
        Policy.check_inference_allowed(context)
        model_features = [
            c for c in best.mapping.values() if c in df_proc.columns
        ]
        # One-hot expansion: mapped categorical columns become dummies.
        expanded = []
        for feat in model_features:
            if feat in df_proc.columns:
                expanded.append(feat)
            else:
                expanded.extend(
                    c for c in df_proc.columns if c.startswith(f"{feat}_")
                )
        result = inference.train_and_predict(
            df_proc, best.model.target, expanded, model_name
        )
        metrics = result.metrics
        top_features = result.top_features
        pred_path = run_dir / "predictions.csv"
        result.predictions.to_csv(pred_path, index=False)
        journal.log(
            "model_inference", sm.state.value, "GBM trained + predicted",
            inputs={"model": model_name, "features": expanded},
            outputs={"metrics": metrics, "predictions_csv": str(pred_path)},
            details=(
                f"HistGradientBoosting on {metrics['n_train']} train / "
                f"{metrics['n_test']} test rows; accuracy "
                f"{metrics['accuracy']}, ROC-AUC {metrics['roc_auc']}."
            ),
        )
        sm.transition(PipelineState.EXPLAIN)

        # --- Explainability (SHAP if installed, else permutation) ---
        journal.log(
            "model_inference", sm.state.value, "Top-10 features explained",
            outputs={"top_features": top_features},
            details=(
                "Method: "
                f"{top_features[0]['method'] if top_features else 'n/a'}; "
                "top feature: "
                f"{top_features[0]['feature'] if top_features else 'n/a'}."
            ),
        )
        sm.transition(PipelineState.DONE)

    except Exception as exc:  # noqa: BLE001 — journal the failure, then re-raise info
        sm.fail()
        journal.log(
            "pipeline", sm.state.value, f"FAILED: {exc}",
            details=str(exc),
        )
        journal.save(run_dir / "journal.json")
        (run_dir / "report.md").write_text(journal.report_md())
        raise

    journal.save(run_dir / "journal.json")
    (run_dir / "report.md").write_text(journal.report_md())
    return RunResult(
        run_id=journal.run_id,
        run_dir=run_dir,
        state=sm.state,
        model_name=model_name,
        metrics=metrics,
        top_features=top_features,
        journal=journal,
    )


def _run_image_path(input_path, run_dir, journal, sm, llm, detected) -> RunResult:
    """Image branch (stub in v0.1): headers via mock/Gemini vision, no
    real DETR — the result is explicitly marked MOCK."""
    sm.transition(PipelineState.ANONYMIZE)
    note = anonymizer.redact_image_note(str(input_path))
    journal.log(
        "data_anonymizer", sm.state.value, "Image redaction: STUB",
        inputs={"path": str(input_path)}, outputs=note, details=note["note"],
    )
    sm.transition(PipelineState.EXTRACT)
    headers = features.extract_image_headers(
        str(input_path), live=llm.live, llm=llm if llm.live else None
    )
    journal.log(
        "feature_extraction", sm.state.value, "Image headers extracted",
        outputs=headers,
        details=f"Modality: {headers.get('modality')}; disease: {headers.get('disease_type')}.",
    )
    sm.transition(PipelineState.MATCH)
    models = load_registry(default_registry_path())
    image_models = [
        m for m in models
        if m.modality == "image"
        and m.image_modality == headers.get("modality")
        and m.disease_type == headers.get("disease_type")
    ]
    if not image_models:
        raise ValueError(
            f"No registered image model for modality "
            f"{headers.get('modality')} / disease {headers.get('disease_type')}."
        )
    model_name = image_models[0].name
    journal.log(
        "model_data_matcher", sm.state.value, f"Image model selected: {model_name}",
        outputs={"model": model_name},
    )
    sm.transition(PipelineState.PREPROCESS)
    journal.log(
        "preprocessing_recommender", sm.state.value,
        "Model-specific preprocessing (stub)",
        details="Image preprocessing is model-specific per the paper and "
        "encapsulated in the model; no generic steps applied in v0.1.",
    )
    sm.transition(PipelineState.INFER)
    result = inference.infer_image_mock(headers)
    (run_dir / "image_result.json").write_text(
        __import__("json").dumps(result, indent=2)
    )
    journal.log(
        "model_inference", sm.state.value, "Image inference: MOCK result",
        outputs=result, details=result["note"],
    )
    sm.transition(PipelineState.EXPLAIN)
    journal.log(
        "model_inference", sm.state.value, "Explainability: not available (stub)",
        details="DETR attention maps need real weights; none in v0.1.",
    )
    sm.transition(PipelineState.DONE)
    journal.save(run_dir / "journal.json")
    (run_dir / "report.md").write_text(journal.report_md())
    return RunResult(
        run_id=journal.run_id,
        run_dir=run_dir,
        state=sm.state,
        model_name=model_name,
        metrics={},
        top_features=[],
        journal=journal,
    )
