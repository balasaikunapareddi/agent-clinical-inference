"""Tests for the deterministic state machine + policy gates."""

import pytest

from clinical_inference.state_machine import (
    IllegalTransitionError,
    PipelineState,
    Policy,
    PolicyViolationError,
    StateMachine,
)


def test_happy_path_transitions():
    sm = StateMachine()
    assert sm.state is PipelineState.INGEST
    for nxt in (
        PipelineState.ANONYMIZE,
        PipelineState.EXTRACT,
        PipelineState.MATCH,
        PipelineState.PREPROCESS,
        PipelineState.INFER,
        PipelineState.EXPLAIN,
        PipelineState.DONE,
    ):
        sm.transition(nxt)
    assert sm.state is PipelineState.DONE


def test_illegal_transition_raises():
    sm = StateMachine()
    with pytest.raises(IllegalTransitionError, match="Illegal transition"):
        sm.transition(PipelineState.INFER)  # cannot skip to INFER from INGEST
    with pytest.raises(IllegalTransitionError):
        sm.transition(PipelineState.DONE)


def test_fail_from_any_nonterminal_state():
    sm = StateMachine()
    sm.transition(PipelineState.ANONYMIZE)
    sm.fail()
    assert sm.state is PipelineState.FAILED


def test_done_is_terminal():
    sm = StateMachine()
    for nxt in (
        PipelineState.ANONYMIZE, PipelineState.EXTRACT, PipelineState.MATCH,
        PipelineState.PREPROCESS, PipelineState.INFER, PipelineState.EXPLAIN,
        PipelineState.DONE,
    ):
        sm.transition(nxt)
    with pytest.raises(IllegalTransitionError):
        sm.transition(PipelineState.FAILED)


def test_policy_blocks_inference_on_raw_data():
    with pytest.raises(PolicyViolationError, match="not passed the anonymizer"):
        Policy.check_inference_allowed({"anonymized": False, "matched_model": "x"})
    with pytest.raises(PolicyViolationError, match="no model was matched"):
        Policy.check_inference_allowed({"anonymized": True, "matched_model": None})


def test_policy_allows_inference_when_anonymized_and_matched():
    Policy.check_inference_allowed({"anonymized": True, "matched_model": "fall-risk-gbm"})


def test_policy_blocks_matching_on_raw_headers():
    with pytest.raises(PolicyViolationError, match="anonymized data"):
        Policy.require_anonymized_headers({"anonymized": False})
