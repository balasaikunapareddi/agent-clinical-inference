"""Model registry — the paper's "model database" (Figure 2).

A curated JSON registry of mock models. Each entry lists the required
headers (features), the target, the modality and the algorithm, so the
Model-Data Matcher can check compatibility before any inference runs.
These are mock models for the v0.1 demo: the tabular ones are trained
on the fly from the uploaded (anonymized, preprocessed) data; the image
one is a clearly-marked stub until real DETR weights exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    name: str
    modality: str  # "tabular" | "image"
    task: str
    description: str
    required_features: list[str] = field(default_factory=list)
    target: str = ""
    algorithm: str = ""
    image_modality: str = ""  # for image models: e.g. "colonoscopy"
    disease_type: str = ""  # for image models: e.g. "polyp"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "modality": self.modality,
            "task": self.task,
            "description": self.description,
            "required_features": self.required_features,
            "target": self.target,
            "algorithm": self.algorithm,
            "image_modality": self.image_modality,
            "disease_type": self.disease_type,
        }


def load_registry(path: str | Path) -> list[ModelSpec]:
    data = json.loads(Path(path).read_text())
    return [ModelSpec(**entry) for entry in data["models"]]


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "model_registry.json"
