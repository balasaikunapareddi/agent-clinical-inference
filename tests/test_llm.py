"""Tests for the LLM wrapper (mock + live-env handling).

The live path strips no_proxy/NO_PROXY (sandbox httpx quirk) but must
preserve GEMINI_API_KEY — a regression test for the bug where the env
was cleared before the cleaned copy was captured.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from clinical_inference.llm import LLMClient, _clean_env
from clinical_inference.matcher import MatchResult
from clinical_inference.registry import ModelSpec


def _candidate(name="fall-risk-gbm"):
    spec = ModelSpec(
        name=name, modality="tabular", task="test",
        description="test model", required_features=["age"], target="fell",
    )
    return MatchResult(model=spec, mapping={"age": "age"},
                       scores={"age": 1.0}, mean_score=1.0)


def test_clean_env_strips_no_proxy_keeps_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("no_proxy", "x")
    monkeypatch.setenv("NO_PROXY", "y")
    env = _clean_env()
    assert env["GEMINI_API_KEY"] == "test-key"
    assert "no_proxy" not in env and "NO_PROXY" not in env


def test_live_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        LLMClient(live=True)


def test_mock_choose_model_is_deterministic():
    client = LLMClient(live=False)
    best, justification = client.choose_model([_candidate()])
    assert best.model.name == "fall-risk-gbm"
    assert "Mock pick" in justification


def test_complete_preserves_api_key_and_strips_no_proxy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("no_proxy", "x")
    seen: dict = {}

    def fake_completion(**kwargs):
        seen.update(os.environ)
        resp = MagicMock()
        resp.choices = [
            MagicMock(message=MagicMock(
                content="MODEL: fall-risk-gbm\nJUSTIFICATION: best fit."))
        ]
        return resp

    with patch("litellm.completion", side_effect=fake_completion):
        client = LLMClient(live=True)
        best, justification = client.choose_model([_candidate()])

    assert seen.get("GEMINI_API_KEY") == "test-key"
    assert "no_proxy" not in seen and "NO_PROXY" not in seen
    assert os.environ.get("GEMINI_API_KEY") == "test-key"  # restored
    assert best.model.name == "fall-risk-gbm"
    assert justification == "best fit."


def test_mock_describe_image():
    client = LLMClient(live=False)
    out = client.describe_image("whatever.png")
    assert out["modality"] == "colonoscopy"
    assert "MOCK" in out["note"]
