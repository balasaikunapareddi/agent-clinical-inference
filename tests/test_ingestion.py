"""Tests for Agent 1 — Ingestion Identifier."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from clinical_inference.ingestion import (
    FileType,
    detect_bytes,
    detect_file,
    load_table,
)


def test_detect_csv_by_content():
    assert detect_bytes(b"a,b,c\n1,2,3\n", "data.csv") is FileType.CSV


def test_detect_json_by_content():
    assert detect_bytes(b'{"a": 1}', "data.bin") is FileType.JSON


def test_detect_png_by_magic():
    assert detect_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10, "img.bin") is FileType.PNG


def test_detect_jpeg_by_magic():
    assert detect_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10, "img.bin") is FileType.JPEG


def test_detect_xlsx_is_not_plain_zip(tmp_path):
    # A real XLSX: build one with pandas, check it is XLSX not ZIP.
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "sheet.xlsx"
    df.to_excel(path, index=False)
    assert detect_file(path).file_type is FileType.XLSX


def test_detect_zip_and_recursive_unpack(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("inner.csv", "a,b\n1,2\n")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("nested.zip", inner.getvalue())
        zf.writestr("top.json", '{"x": 1}')
    detected = detect_file(outer)
    assert detected.file_type is FileType.ZIP
    names = {n.path for n in detected.nested}
    assert "nested.zip" in names and "top.json" in names
    nested = next(n for n in detected.nested if n.path == "nested.zip")
    assert nested.nested[0].file_type is FileType.CSV


def test_detect_unknown():
    assert detect_bytes(b"\x00\x01\x02\x03\x04", "weird.bin") is FileType.UNKNOWN


def test_load_table_csv(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    df = load_table(path)
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_load_table_rejects_unknown(tmp_path):
    path = tmp_path / "t.bin"
    path.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError, match="Cannot load"):
        load_table(path)
