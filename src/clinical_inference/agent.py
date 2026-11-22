"""Conversational layer — Strands Agents + LiteLLM (Gemini free tier).

The Strands agent wraps the deterministic pipeline: it can run the
pipeline, list registered models, and summarize a run. The agent
writes message text only — it never decides a pipeline transition;
the state machine inside run_pipeline owns those.

Requires strands-agents + litellm (installed in .venv) and
GEMINI_API_KEY for live model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_inference.prompts import SYSTEM_PROMPT


def build_agent():
    """Build the Strands agent. Raises a clear error if the Strands
    stack or the Gemini key is missing."""
    try:
        from strands import Agent, tool
        from strands.models.litellm import LiteLLMModel
    except ImportError as exc:
        raise ImportError(
            "Strands is not installed. Install it with: "
            "pip install strands-agents litellm"
        ) from exc
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY is not set. The conversational layer needs a "
            "Gemini key; the pipeline itself runs offline with `make demo`."
        )

    from clinical_inference.pipeline import run_pipeline
    from clinical_inference.registry import default_registry_path, load_registry

    @tool
    def run_clinical_pipeline(input_path: str, live: bool = False) -> str:
        """Run the full 7-agent clinical pipeline on a file.

        Args:
            input_path: Path to a CSV/XLSX/JSON/ZIP/image file.
            live: If true, use Gemini for the LLM steps (model pick,
                image headers). Default false (deterministic mock).
        """
        result = run_pipeline(input_path, live=live)
        return json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "state": result.state.value,
                "model": result.model_name,
                "metrics": result.metrics,
                "top_features": result.top_features[:5],
            },
            indent=2,
        )

    @tool
    def list_models() -> str:
        """List the registered models in the model database."""
        models = load_registry(default_registry_path())
        return json.dumps([m.to_dict() for m in models], indent=2)

    @tool
    def get_run_report(run_dir: str) -> str:
        """Read the markdown report of a completed run."""
        path = Path(run_dir) / "report.md"
        if not path.exists():
            return f"No report found in {run_dir}."
        return path.read_text()

    model = LiteLLMModel(model_id="gemini/gemini-3.5-flash-lite")
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[run_clinical_pipeline, list_models, get_run_report],
    )


class ClinicalInferenceAgent:
    """The clinical inference assistant, as one visible class.

    What it is: a Strands ``Agent`` on LiteLLM
    (``gemini/gemini-3.5-flash-lite``, needs GEMINI_API_KEY) that wraps
    the deterministic 7-agent pipeline. Its instructions are
    ``prompts.SYSTEM_PROMPT``; its tools are ``run_clinical_pipeline``
    (run the end-to-end pipeline on a file), ``list_models`` (the model
    registry), and ``get_run_report`` (read a completed run's report).
    The agent writes message text only — it never decides a pipeline
    transition; the state machine inside ``run_pipeline`` owns those,
    and it never sees raw patient data (the pipeline anonymizes first).

    ``run`` is one agent turn. Construction builds the underlying
    Strands agent via :func:`build_agent`, which raises a clear error
    if the Strands stack or the Gemini key is missing.
    """

    def __init__(self) -> None:
        self.agent = build_agent()

    def run(self, prompt: str):
        """One agent turn: send ``prompt``, return the Strands result."""
        return self.agent(prompt)


def main() -> None:
    """Interactive chat loop: `python -m clinical_inference.agent`."""
    agent = ClinicalInferenceAgent()
    print("Clinical inference assistant (Strands + Gemini). Type 'quit' to exit.")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        print(agent.run(user))


if __name__ == "__main__":
    main()
