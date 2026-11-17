"""Agent 2 — Data Anonymizer (paper Table I: Ingestion_Anonymizer).

The paper uses Google Cloud DLP. This build uses LOCAL deterministic
PII detection instead — regex for emails, phones, SSNs, MRNs and DOBs,
plus column-name-based masking for names and identifiers. That is a
deliberate fix for the paper's own limitation (Section V / VI): the
paper's cloud anonymization sends sensitive data to external servers,
which breaks data sovereignty where cloud access is restricted. Here
nothing leaves the machine.

Masking uses the paper's fixed-length placeholder "****" and preserves
the schema (same columns, same row count). The audit log records COUNTS
by PII type only — never the values that were masked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

MASK = "****"

# --- Deterministic PII patterns (regex, applied to every string cell) ---
PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "dob": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    # Explicit "MRN: 123456" style identifiers in free text.
    "mrn_text": re.compile(r"\bMRN[-:\s]*\d{5,10}\b", re.IGNORECASE),
}

# Columns whose VALUES are identifiers by name — masked wholesale.
# (Deterministic column-name detection; the paper's DLP uses NER here.
#  See README limitations: no free-text name NER in v0.1.)
NAME_LIKE = {
    "name", "patient_name", "full_name", "first_name", "last_name",
    "patient", "contact_name", "guardian_name",
}
ID_LIKE = {
    "mrn", "medical_record_number", "record_number", "patient_id",
    "ssn", "social_security", "email", "phone", "phone_number",
    "dob", "date_of_birth", "address", "street_address",
}


def _norm_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _is_name_like(col: str) -> bool:
    return _norm_col(col) in NAME_LIKE


def _is_id_like(col: str) -> bool:
    return _norm_col(col) in ID_LIKE


@dataclass
class AnonymizeAudit:
    masked_by_type: dict[str, int] = field(default_factory=dict)
    columns_masked: list[str] = field(default_factory=list)
    rows: int = 0
    columns: int = 0

    def to_dict(self) -> dict:
        return {
            "masked_by_type": self.masked_by_type,
            "columns_masked": self.columns_masked,
            "rows": self.rows,
            "columns": self.columns,
            "total_masked_cells": sum(self.masked_by_type.values()),
        }


def anonymize_text(text: str, audit: dict[str, int]) -> str:
    """Mask every PII pattern in a free-text value; count by type."""
    if not isinstance(text, str):
        return text
    out = text
    for pii_type, pattern in PATTERNS.items():
        matches = pattern.findall(out)
        if matches:
            audit[pii_type] = audit.get(pii_type, 0) + len(matches)
            out = pattern.sub(MASK, out)
    return out


def anonymize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, AnonymizeAudit]:
    """Mask PII in a DataFrame, preserving the schema.

    Returns (masked_df, audit). Audit counts masked cells by PII type;
    it never contains the masked values.
    """
    audit_counts: dict[str, int] = {}
    columns_masked: list[str] = []
    out = df.copy()

    for col in df.columns:
        norm = _norm_col(str(col))
        if _is_name_like(str(col)) or _is_id_like(str(col)):
            # Whole identifier column: every non-null value is PII.
            pii_type = "name" if _is_name_like(str(col)) else "identifier"
            n = int(out[col].notna().sum())
            if n:
                audit_counts[pii_type] = audit_counts.get(pii_type, 0) + n
                columns_masked.append(str(col))
            out[col] = out[col].apply(lambda v: MASK if pd.notna(v) else v)
            continue
        # Free-text / string columns: regex-scan each cell.
        if df[col].dtype == object or str(df[col].dtype) in ("string", "str"):
            masked_any = False
            for idx, value in out[col].items():
                if isinstance(value, str):
                    new = anonymize_text(value, audit_counts)
                    if new != value:
                        out.at[idx, col] = new
                        masked_any = True
            if masked_any and str(col) not in columns_masked:
                columns_masked.append(str(col))

    audit = AnonymizeAudit(
        masked_by_type=audit_counts,
        columns_masked=columns_masked,
        rows=len(df),
        columns=len(df.columns),
    )
    return out, audit


def redact_image_note(path: str) -> dict:
    """Image redaction is a stub in v0.1: images are never sent to any
    model or service, and no opaque overlay is applied. The pipeline
    records that the image path was not anonymized, so downstream
    steps cannot mistake a raw image for a redacted one."""
    return {
        "path": path,
        "redacted": False,
        "note": "STUB v0.1: image PII redaction (opaque overlays per the paper) "
        "is not implemented. Images are kept local and never sent to a model.",
    }
