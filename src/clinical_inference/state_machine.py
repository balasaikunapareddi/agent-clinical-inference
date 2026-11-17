"""Deterministic pipeline state machine + policy-as-code gates.

The state machine owns the flow (the paper's Google ADK orchestration,
replaced here with code — the LLM never decides a transition). States:

    INGEST -> ANONYMIZE -> EXTRACT -> MATCH -> PREPROCESS -> INFER
    -> EXPLAIN -> DONE
    any state -> FAILED

Policy gates (enforced in code, not in a prompt):
- Inference may only run on data that has been anonymized AND matched to
  a registered model. Raw PII never reaches a model call or an LLM call.
- Matching may only run on anonymized headers, never on raw column data.
"""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    INGEST = "INGEST"
    ANONYMIZE = "ANONYMIZE"
    EXTRACT = "EXTRACT"
    MATCH = "MATCH"
    PREPROCESS = "PREPROCESS"
    INFER = "INFER"
    EXPLAIN = "EXPLAIN"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.INGEST: {PipelineState.ANONYMIZE, PipelineState.FAILED},
    PipelineState.ANONYMIZE: {PipelineState.EXTRACT, PipelineState.FAILED},
    PipelineState.EXTRACT: {PipelineState.MATCH, PipelineState.FAILED},
    PipelineState.MATCH: {PipelineState.PREPROCESS, PipelineState.FAILED},
    PipelineState.PREPROCESS: {PipelineState.INFER, PipelineState.FAILED},
    PipelineState.INFER: {PipelineState.EXPLAIN, PipelineState.FAILED},
    PipelineState.EXPLAIN: {PipelineState.DONE, PipelineState.FAILED},
    PipelineState.DONE: set(),
    PipelineState.FAILED: set(),
}


class IllegalTransitionError(ValueError):
    """Raised when code tries to move the pipeline along an illegal edge."""


class PolicyViolationError(ValueError):
    """Raised when a policy gate blocks a step (e.g. inference on raw data)."""


class StateMachine:
    """Tiny deterministic state machine with a typed transition table."""

    def __init__(self) -> None:
        self.state = PipelineState.INGEST

    def transition(self, to: PipelineState) -> None:
        allowed = TRANSITIONS[self.state]
        if to not in allowed:
            raise IllegalTransitionError(
                f"Illegal transition: {self.state.value} -> {to.value}. "
                f"Allowed from {self.state.value}: "
                f"{sorted(s.value for s in allowed) or 'none (terminal)'}."
            )
        self.state = to

    def fail(self) -> None:
        if self.state not in (PipelineState.DONE, PipelineState.FAILED):
            self.state = PipelineState.FAILED


class Policy:
    """Policy-as-code gates. Each check raises PolicyViolationError."""

    @staticmethod
    def require_anonymized(context: dict) -> None:
        if not context.get("anonymized"):
            raise PolicyViolationError(
                "Inference blocked: data has not passed the anonymizer. "
                "Raw PII must never reach a model call."
            )

    @staticmethod
    def require_matched(context: dict) -> None:
        if not context.get("matched_model"):
            raise PolicyViolationError(
                "Inference blocked: no model was matched to this data. "
                "Run the Model-Data Matcher first."
            )

    @staticmethod
    def require_anonymized_headers(context: dict) -> None:
        if not context.get("anonymized"):
            raise PolicyViolationError(
                "Matching blocked: headers must come from anonymized data."
            )

    @staticmethod
    def check_inference_allowed(context: dict) -> None:
        """Both gates, in order. Called immediately before INFER."""
        Policy.require_anonymized(context)
        Policy.require_matched(context)
