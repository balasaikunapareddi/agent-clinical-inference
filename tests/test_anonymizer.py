"""Tests for Agent 2 — Data Anonymizer."""

import pandas as pd

from clinical_inference.anonymizer import (
    MASK,
    anonymize_dataframe,
    anonymize_text,
)


def test_email_masked_in_text():
    audit: dict = {}
    out = anonymize_text("Contact jane.doe@example.com for help", audit)
    assert "jane.doe@example.com" not in out
    assert MASK in out
    assert audit["email"] == 1


def test_phone_and_ssn_masked():
    audit: dict = {}
    out = anonymize_text("Call 555-123-4567, SSN 123-45-6789", audit)
    assert "555-123-4567" not in out and "123-45-6789" not in out
    assert audit["phone"] == 1 and audit["ssn"] == 1


def test_dob_masked():
    audit: dict = {}
    out = anonymize_text("Born 03/14/1952, noted.", audit)
    assert "03/14/1952" not in out
    assert audit["dob"] == 1


def test_dataframe_name_and_id_columns_masked():
    df = pd.DataFrame(
        {
            "patient_name": ["Asha Sharma", "Ravi Patel"],
            "mrn": ["MRN-123456", "MRN-654321"],
            "age": [70, 80],
            "notes": ["see jane@example.com", "no pii here"],
        }
    )
    masked, audit = anonymize_dataframe(df)
    # Schema preserved: same columns, same rows.
    assert list(masked.columns) == list(df.columns)
    assert len(masked) == 2
    assert (masked["patient_name"] == MASK).all()
    assert (masked["mrn"] == MASK).all()
    assert masked["age"].tolist() == [70, 80]  # non-PII untouched
    assert "jane@example.com" not in masked["notes"].iloc[0]
    assert "no pii here" in masked["notes"].iloc[1]
    assert audit.masked_by_type["name"] == 2
    assert audit.masked_by_type["identifier"] == 2
    assert audit.masked_by_type["email"] == 1


def test_audit_never_contains_values():
    df = pd.DataFrame({"email": ["secret@example.com"], "age": [70]})
    _, audit = anonymize_dataframe(df)
    blob = str(audit.to_dict())
    assert "secret@example.com" not in blob
    assert audit.masked_by_type["identifier"] == 1


def test_no_pii_no_masking():
    df = pd.DataFrame({"age": [70, 80], "gait_speed": [0.9, 1.1]})
    masked, audit = anonymize_dataframe(df)
    assert masked.equals(df)
    assert audit.masked_by_type == {}
