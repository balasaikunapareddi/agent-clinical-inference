"""Agent 1 — Ingestion Identifier (paper Table I: Ingestion_Classifier).

The paper uses Magika (a deep-learning MIME-type detector). This build
uses a deterministic magic-bytes + extension sniffer (stdlib-first):
same job, no model download, no network, fully reproducible. Magika is
noted in the README as the paper's choice and why the local sniffer
replaces it.

Supports CSV, XLSX, JSON, ZIP (recursive unpack, like the paper) and
common image formats (PNG/JPEG — routed to the image path, stub in v0.1).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

MAX_ZIP_DEPTH = 3


class FileType(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    ZIP = "zip"
    PNG = "png"
    JPEG = "jpeg"
    UNKNOWN = "unknown"


# Magic bytes (prefixes) for the formats we detect.
MAGIC: list[tuple[bytes, FileType]] = [
    (b"\x89PNG\r\n\x1a\n", FileType.PNG),
    (b"\xff\xd8\xff", FileType.JPEG),
    (b"PK\x03\x04", FileType.ZIP),  # also XLSX — disambiguated below
    (b"PK\x05\x06", FileType.ZIP),  # empty zip
    (b"PK\x07\x08", FileType.ZIP),
]


@dataclass
class DetectedFile:
    path: str
    file_type: FileType
    mime: str
    size_bytes: int
    nested: list["DetectedFile"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "file_type": self.file_type.value,
            "mime": self.mime,
            "size_bytes": self.size_bytes,
            "nested": [n.to_dict() for n in self.nested],
        }


MIME: dict[FileType, str] = {
    FileType.CSV: "text/csv",
    FileType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    FileType.JSON: "application/json",
    FileType.ZIP: "application/zip",
    FileType.PNG: "image/png",
    FileType.JPEG: "image/jpeg",
    FileType.UNKNOWN: "application/octet-stream",
}


def _sniff_xlsx(data: bytes) -> bool:
    """An XLSX is a ZIP whose entries include the Open XML workbook."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        return "xl/workbook.xml" in names or "[Content_Types].xml" in names
    except zipfile.BadZipFile:
        return False


def detect_bytes(data: bytes, filename: str = "") -> FileType:
    """Magic-bytes first, extension as a tiebreaker (paper: structural
    and byte-level content; extension alone is never trusted)."""
    head = data[:16]
    for magic, ftype in MAGIC:
        if head.startswith(magic):
            if ftype is FileType.ZIP:
                ext = Path(filename).suffix.lower()
                if ext in (".xlsx", ".xlsm") or _sniff_xlsx(data):
                    return FileType.XLSX
                return FileType.ZIP
            return ftype
    stripped = data.lstrip()
    if stripped[:1] in (b"{", b"["):
        return FileType.JSON
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return FileType.CSV
    if ext == ".json":
        return FileType.JSON
    if ext in (".xlsx", ".xlsm"):
        return FileType.XLSX
    if ext == ".zip":
        return FileType.ZIP
    if ext == ".png":
        return FileType.PNG
    if ext in (".jpg", ".jpeg"):
        return FileType.JPEG
    # Last resort: decodable text with commas/newlines looks like a CSV.
    try:
        text = data[:4096].decode("utf-8")
        if "," in text and "\n" in text:
            return FileType.CSV
    except UnicodeDecodeError:
        pass
    return FileType.UNKNOWN


def detect_file(path: str | Path) -> DetectedFile:
    """Classify one file on disk; ZIPs are recursively unpacked in
    memory (paper: in-memory recursive unpacking of nested archives)."""
    path = Path(path)
    data = path.read_bytes()
    ftype = detect_bytes(data, path.name)
    detected = DetectedFile(
        path=str(path), file_type=ftype, mime=MIME[ftype], size_bytes=len(data)
    )
    if ftype is FileType.ZIP:
        detected.nested = _unpack_zip(data, depth=0)
    return detected


def _unpack_zip(data: bytes, depth: int) -> list[DetectedFile]:
    if depth >= MAX_ZIP_DEPTH:
        return []
    found: list[DetectedFile] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = zf.read(info.filename)
                ftype = detect_bytes(inner, info.filename)
                entry = DetectedFile(
                    path=info.filename,
                    file_type=ftype,
                    mime=MIME[ftype],
                    size_bytes=len(inner),
                )
                if ftype is FileType.ZIP:
                    entry.nested = _unpack_zip(inner, depth + 1)
                found.append(entry)
    except zipfile.BadZipFile:
        pass
    return found


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a detected tabular file into a DataFrame."""
    path = Path(path)
    data = path.read_bytes()
    ftype = detect_bytes(data, path.name)
    if ftype is FileType.CSV:
        return pd.read_csv(path)
    if ftype is FileType.JSON:
        return pd.read_json(path)
    if ftype is FileType.XLSX:
        return pd.read_excel(path)
    raise ValueError(f"Cannot load {path} as a table: detected type {ftype.value}")
