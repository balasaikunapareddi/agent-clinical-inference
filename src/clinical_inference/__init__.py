"""agent-clinical-inference — an agentic clinical data pipeline.

Implements Shimgekar et al., "Agentic AI framework for End-to-End
Medical Data Inference" (arXiv:2507.18115): seven agents from ingestion
to inference, orchestrated here by a deterministic state machine
(Strands + LiteLLM wrap the pipeline for conversation only).
"""

from __future__ import annotations

from clinical_inference.pipeline import RunResult, run_pipeline

__all__ = ["run_pipeline", "RunResult"]
