"""Generate the two synthetic mock datasets (clearly labeled MOCK).

No real PHI: every value is drawn from a seeded RNG. PII columns
(patient_name, email, phone, mrn, dob) are synthetic placeholders whose
only purpose is to exercise the anonymizer — they are masked and then
dropped before matching/modeling.

Design notes (the paper's exact column-typing rules):
- Continuous features are floats with many unique values, so they type
  as Numerical (unique > 0.8 * rows).
- Binary features are 0/1 ints or M/F strings (exactly 2 unique values).
- clinic_site is a short-string column with 3 unique values -> Categorical.
- Targets are generated from a logistic function of the real features,
  so the GBM has genuine signal (expected ROC-AUC ~0.8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N = 200

FIRST = ["Asha", "Ravi", "Meera", "Arjun", "Priya", "Kiran", "Anita", "Vikram"]
LAST = ["Sharma", "Patel", "Iyer", "Nair", "Reddy", "Khan", "Das", "Menon"]


def _pii(rng: np.random.Generator, n: int) -> dict:
    names = [
        f"{FIRST[i % len(FIRST)]} {LAST[(i * 3) % len(LAST)]}" for i in range(n)
    ]
    return {
        "patient_id": [f"PT-{1000 + i}" for i in range(n)],
        "patient_name": names,
        "email": [f"patient{i}@example.com" for i in range(n)],
        "phone": [f"555-01{rng.integers(10, 99):02d}-{rng.integers(1000, 9999)}" for i in range(n)],
        "mrn": [f"MRN-{rng.integers(100000, 999999)}" for i in range(n)],
        "dob": [
            f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(1940, 1960))}"
            for i in range(n)
        ],
    }


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def make_fall_risk() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = N
    age = rng.integers(65, 92, n).astype(float)
    gender = rng.choice(["M", "F"], n)
    gait_speed = np.round(rng.normal(0.95, 0.28, n), 4)
    stride_length = np.round(rng.normal(1.05, 0.22, n), 4)
    foot_pressure_mean = np.round(rng.normal(320, 85, n), 2)
    foot_angle = np.round(rng.normal(8.5, 4.2, n), 2)
    ankle_rom = np.round(rng.normal(28, 7.5, n), 2)
    step_time_variability = np.round(rng.normal(0.06, 0.03, n), 4)
    fall_history = rng.integers(0, 2, n)
    clinic_site = rng.choice(["Site A", "Site B", "Site C"], n)
    # Fall risk rises with age, slow gait, high step variability, prior falls.
    # Coefficients are strong enough for a clear demo signal (AUC ~0.8).
    z = (
        0.12 * (age - 78)
        - 4.5 * (gait_speed - 0.95)
        + 35 * (step_time_variability - 0.06)
        + 1.8 * fall_history
        - 1.0
    )
    fell = (rng.random(n) < _logistic(z)).astype(int)
    return pd.DataFrame(
        {
            **_pii(rng, n),
            "age": age,
            "gender": gender,
            "gait_speed": gait_speed,
            "stride_length": stride_length,
            "foot_pressure_mean": foot_pressure_mean,
            "foot_angle": foot_angle,
            "ankle_rom": ankle_rom,
            "step_time_variability": step_time_variability,
            "fall_history": fall_history,
            "clinic_site": clinic_site,
            "fell": fell,
        }
    )


def make_anxiety() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    n = N
    age = rng.integers(45, 89, n).astype(float)
    gender = rng.choice(["M", "F"], n)
    antidepressant_use = rng.integers(0, 2, n)
    vomiting = rng.integers(0, 2, n)
    pain_score = np.round(rng.uniform(0, 10, n), 2)
    fatigue_score = np.round(rng.uniform(0, 10, n), 2)
    sleep_hours = np.round(rng.normal(6.2, 1.4, n), 2)
    appetite_loss = rng.integers(0, 2, n)
    hope_score = np.round(rng.uniform(0, 10, n), 2)
    # Anxiety rises with pain, fatigue, vomiting, low hope, antidepressant use.
    # Coefficients are strong enough for a clear demo signal (AUC ~0.8).
    z = (
        0.7 * (pain_score - 5)
        + 0.6 * (fatigue_score - 5)
        + 1.6 * vomiting
        + 1.0 * antidepressant_use
        - 0.7 * (hope_score - 5)
        - 1.5
    )
    anxiety_high = (rng.random(n) < _logistic(z)).astype(int)
    low_hope = (hope_score < 4).astype(int)
    return pd.DataFrame(
        {
            **_pii(rng, n),
            "age": age,
            "gender": gender,
            "antidepressant_use": antidepressant_use,
            "vomiting": vomiting,
            "pain_score": pain_score,
            "fatigue_score": fatigue_score,
            "sleep_hours": sleep_hours,
            "appetite_loss": appetite_loss,
            "hope_score": hope_score,
            "anxiety_high": anxiety_high,
            "low_hope": low_hope,
        }
    )


if __name__ == "__main__":
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    fall = make_fall_risk()
    anx = make_anxiety()
    fall.to_csv(f"{out_dir}/fall_risk_mock.csv", index=False)
    anx.to_csv(f"{out_dir}/anxiety_mock.csv", index=False)
    print(f"fall_risk_mock.csv: {fall.shape}, fell rate {fall['fell'].mean():.2f}")
    print(f"anxiety_mock.csv: {anx.shape}, anxiety rate {anx['anxiety_high'].mean():.2f}")
