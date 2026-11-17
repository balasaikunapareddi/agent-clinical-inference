"""Agent 4 — Model-Data Matcher (paper Table I: Ingestion_Feature_Matcher).

The paper: SapBERT embeddings (768-dim) + cosine similarity, threshold
0.6, greedy one-to-one assignment, LLM picks the best candidate.

This build: deterministic synonym map + token-overlap scoring by
default (offline, no download), same threshold 0.6, same greedy
one-to-one assignment, same LLM-picks-best step. Optional embedding
mode via sentence-transformers (SapBERT) when installed — an optional
path, never a hard dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clinical_inference.registry import ModelSpec

THRESHOLD = 0.6  # the paper's empirically-set threshold

# Token-level synonyms: clinical terms that mean the same thing in
# headers. Applied before token-overlap scoring.
SYNONYMS: dict[str, str] = {
    "sex": "gender",
    "walking": "gait",
    "walk": "gait",
    "velocity": "speed",
    "pressure": "pressure",
    "antidepressant": "antidepressant_use",
    "antidepressants": "antidepressant_use",
    "vomiting": "vomiting",
    "fall": "fall_history",
    "falls": "fall_history",
    "history": "history",
    "score": "score",
    "anxiety": "anxiety",
    "hope": "hope",
    "angle": "angle",
    "rom": "rom",
    "range": "rom",
    "motion": "rom",
    "stride": "stride_length",
    "length": "length",
    "foot": "foot",
    "ankle": "ankle",
    "step": "step",
    "time": "time",
    "variability": "variability",
    "patient": "patient",
    "mrn": "mrn",
}


def _tokens(name: str) -> set[str]:
    """Normalize a header into a canonical token set."""
    raw = re.split(r"[^a-z0-9]+", name.lower())
    out: set[str] = set()
    for tok in raw:
        if not tok:
            continue
        out.add(SYNONYMS.get(tok, tok))
    return out


def token_overlap(header: str, required: str) -> float:
    """Jaccard similarity over canonical token sets, in [0, 1]."""
    a, b = _tokens(header), _tokens(required)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return len(a & b) / len(a | b)


def _embedding_scores(
    headers: list[str], required: list[str]
) -> dict[tuple[str, str], float] | None:
    """Optional SapBERT path: cosine similarity over 768-dim embeddings.

    Returns None when sentence-transformers is not installed (the
    deterministic path is the default and always available).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None
    model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    h_emb = model.encode(headers, normalize_embeddings=True)
    r_emb = model.encode(required, normalize_embeddings=True)
    sims = cosine_similarity(r_emb, h_emb)
    return {
        (req, hdr): float(sims[i, j])
        for i, req in enumerate(required)
        for j, hdr in enumerate(headers)
    }


@dataclass
class MatchResult:
    model: ModelSpec
    mapping: dict[str, str]  # required feature -> dataset column
    scores: dict[str, float]  # required feature -> match score
    mean_score: float
    unmatched: list[str] = field(default_factory=list)


def match_features(
    required: list[str],
    columns: list[str],
    threshold: float = THRESHOLD,
    use_embeddings: bool = False,
) -> tuple[dict[str, str], dict[str, float], list[str]]:
    """Greedy one-to-one assignment (paper): iteratively pick the best
    available (required, column) pair; each column used at most once.
    A required feature is matched only when its score >= threshold."""
    if use_embeddings:
        emb = _embedding_scores(columns, required)
    else:
        emb = None

    def score(req: str, col: str) -> float:
        if emb is not None:
            return emb.get((req, col), 0.0)
        return token_overlap(col, req)

    pairs = sorted(
        ((score(req, col), req, col) for req in required for col in columns),
        key=lambda t: -t[0],
    )
    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}
    used_cols: set[str] = set()
    for s, req, col in pairs:
        if req in mapping or col in used_cols:
            continue
        if s >= threshold:
            mapping[req] = col
            scores[req] = s
            used_cols.add(col)
    unmatched = [req for req in required if req not in mapping]
    return mapping, scores, unmatched


def match_models(
    headers: list[str],
    models: list[ModelSpec],
    threshold: float = THRESHOLD,
    use_embeddings: bool = False,
) -> list[MatchResult]:
    """Eligible models: every required feature matched above threshold.
    Tabular models only — image models are matched on modality/disease
    by the image path, not on headers."""
    results: list[MatchResult] = []
    for model in models:
        if model.modality != "tabular":
            continue
        mapping, scores, unmatched = match_features(
            model.required_features, headers, threshold, use_embeddings
        )
        if not unmatched:
            mean = sum(scores.values()) / len(scores) if scores else 0.0
            results.append(
                MatchResult(
                    model=model,
                    mapping=mapping,
                    scores=scores,
                    mean_score=mean,
                    unmatched=unmatched,
                )
            )
    return sorted(results, key=lambda r: -r.mean_score)


def pick_best(
    candidates: list[MatchResult], llm=None
) -> tuple[MatchResult, str]:
    """The LLM picks the best candidate among the eligible ones, with a
    justification (paper). Mock mode: highest mean score, deterministic
    justification. --live: Gemini chooses and justifies."""
    if not candidates:
        raise ValueError("No eligible models to pick from.")
    if llm is not None:
        choice, justification = llm.choose_model(candidates)
        return choice, justification
    best = candidates[0]
    justification = (
        f"Mock pick: '{best.model.name}' has the highest mean header-match "
        f"score ({best.mean_score:.2f}) among {len(candidates)} eligible "
        f"model(s); all {len(best.mapping)} required features matched above "
        f"the {THRESHOLD} threshold."
    )
    return best, justification
