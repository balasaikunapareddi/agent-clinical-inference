"""LLM wrapper — LiteLLM (Gemini free tier) for the two steps where the
paper uses an LLM: picking the best model among candidates, and the
multi-stage image classification (MedGemma substitute).

Mock mode is the default: deterministic, offline, no key. Live mode
uses litellm with gemini/gemini-3.5-flash-lite and GEMINI_API_KEY from
the environment only. The LLM writes message text / a choice with a
justification — it never decides a pipeline transition (the state
machine owns those).

Sandbox quirk: this sandbox's no_proxy contains IPv6 literals httpx
cannot parse, so live calls strip no_proxy/NO_PROXY from the child env.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

MODEL = "gemini/gemini-3.5-flash-lite"


def _clean_env() -> dict:
    env = dict(os.environ)
    env.pop("no_proxy", None)
    env.pop("NO_PROXY", None)
    return env


class LLMClient:
    def __init__(self, live: bool = False) -> None:
        self.live = live
        if live and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "Live mode needs GEMINI_API_KEY in the environment. "
                "Run without --live for the offline mock mode."
            )

    def _complete(self, prompt: str) -> str:
        if not self.live:
            raise RuntimeError("LLMClient is in mock mode; no live call made.")
        import litellm

        # Capture the cleaned env BEFORE clearing: _clean_env() reads
        # os.environ, so evaluating it after clear() would yield {}.
        clean_env = _clean_env()
        old_env = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(clean_env)
            resp = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            return resp.choices[0].message.content.strip()
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def choose_model(self, candidates) -> tuple:
        """Pick the best eligible model; returns (chosen, justification)."""
        if not self.live:
            best = candidates[0]
            justification = (
                f"Mock pick: '{best.model.name}' has the highest mean "
                f"header-match score ({best.mean_score:.2f}) among "
                f"{len(candidates)} eligible model(s)."
            )
            return best, justification
        lines = [
            f"- {c.model.name}: {c.model.description} "
            f"(mean match {c.mean_score:.2f}; features: "
            f"{', '.join(c.mapping.values())})"
            for c in candidates
        ]
        prompt = (
            "You are choosing the best clinical ML model for a dataset. "
            "Eligible models (all required features matched):\n"
            + "\n".join(lines)
            + "\nReply with exactly two lines:\n"
            "MODEL: <model name>\nJUSTIFICATION: <one sentence>"
        )
        text = self._complete(prompt)
        name = ""
        justification = ""
        for line in text.splitlines():
            if line.startswith("MODEL:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
        for c in candidates:
            if c.model.name == name:
                return c, justification or "Chosen by Gemini (no justification returned)."
        return candidates[0], (
            f"Gemini returned an unknown model name ({name!r}); fell back to "
            f"the highest-scoring candidate '{candidates[0].model.name}'."
        )

    def describe_image(self, path: str | Path) -> dict:
        """Multi-stage image headers (paper's MedGemma step): modality,
        then disease type. Mock returns the mock headers; live asks
        Gemini vision."""
        if not self.live:
            return {
                "modality": "colonoscopy",
                "disease_type": "polyp",
                "note": "MOCK: no real vision model in mock mode.",
            }
        import litellm

        data = base64.b64encode(Path(path).read_bytes()).decode()
        suffix = Path(path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        # Capture the cleaned env BEFORE clearing (see _complete).
        clean_env = _clean_env()
        old_env = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(clean_env)
            resp = litellm.completion(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What imaging modality is this, and what "
                                "disease type (if any) is visible? Reply with "
                                "exactly two lines:\nMODALITY: <modality>\n"
                                "DISEASE: <disease type or none>",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{data}"
                                },
                            },
                        ],
                    }
                ],
                timeout=90,
            )
            text = resp.choices[0].message.content.strip()
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        modality, disease = "unknown", "unknown"
        for line in text.splitlines():
            if line.startswith("MODALITY:"):
                modality = line.split(":", 1)[1].strip()
            elif line.startswith("DISEASE:"):
                disease = line.split(":", 1)[1].strip()
        return {"modality": modality, "disease_type": disease}
