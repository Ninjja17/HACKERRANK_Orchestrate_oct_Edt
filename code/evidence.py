"""Evidence access layer: maps `file_id` -> parsed content by evidence type.

`dataset/evidence_files.csv` only gives `file_id -> file_path, origin,
captured_at`; it does not say which case/user a file concerns. Content must
be read and matched to identifiers at evaluation time (no dataset
shortcuts), file by file, through the readers below.

Parsing is dispatched strictly by file extension (`.sql`, `.json`, `.log`,
`.txt`, image extensions) so each evidence type goes through a dedicated,
predictable reader instead of a generic "read bytes and hope" path.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from config import EVIDENCE_DIR
from data_loader import load_evidence_files
from schemas import EvidenceFileRow

TEXT_EXTENSIONS = {".log", ".txt"}
JSON_EXTENSIONS = {".json"}
SQL_EXTENSIONS = {".sql"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class EvidenceNotFoundError(RuntimeError):
    """Raised when a referenced `file_id` has no matching manifest entry."""


def build_evidence_index() -> dict[str, EvidenceFileRow]:
    """Return `file_id -> EvidenceFileRow` for every manifest entry."""
    return {row.file_id: row for row in load_evidence_files()}


def resolve_evidence_path(evidence_row: EvidenceFileRow) -> Path:
    path = EVIDENCE_DIR.parent / evidence_row.file_path
    if not path.is_file():
        raise EvidenceNotFoundError(f"Evidence file missing on disk: {path}")
    return path


def read_text_evidence(path: Path) -> str:
    """Read a `.log` / `.txt` evidence file as plain text."""
    return path.read_text(encoding="utf-8")


def read_json_evidence(path: Path) -> Any:
    """Read a `.json` evidence file."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_sql_evidence(path: Path) -> list[dict[str, Any]]:
    """Execute a SQLite-compatible `.sql` snapshot export and return its rows.

    Loads the script into a throwaway in-memory SQLite database, then
    returns every row of every table as a list of dicts tagged with the
    source table name, so callers can search across tables without knowing
    the schema in advance.

    TODO(next step): decide which tables/columns matter per data category
    in `deletion_rules.csv` and extract targeted identifier matches instead
    of dumping every row.
    """
    script = path.read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(script)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [r[0] for r in cursor.fetchall()]
        rows: list[dict[str, Any]] = []
        for table in tables:
            for row in conn.execute(f"SELECT * FROM {table}"):  # noqa: S608 - table names from trusted schema DDL
                record = dict(row)
                record["_table"] = table
                rows.append(record)
        return rows
    finally:
        conn.close()


def read_pdf_evidence(path: Path) -> str:
    """Extract plain text from a `.pdf` evidence document, page by page.

    Returns an empty string (logged) if `pypdf` is unavailable or extraction
    fails, so a missing optional dependency degrades gracefully instead of
    crashing the whole run.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logging.getLogger(__name__).warning("pypdf not installed; skipping PDF evidence %s", path)
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:  # noqa: BLE001 - a malformed PDF must not abort the run
        logging.getLogger(__name__).warning("Failed to extract text from PDF %s", path, exc_info=True)
        return ""


def read_image_evidence_path(path: Path) -> Path:
    """Return the path to an image evidence file for multimodal inspection.

    Images are not yet indexed for identifier matching (no vision model is
    wired in); they remain available on disk but are excluded from
    grounded evidence citations until multimodal support is added.
    """
    return path


def load_evidence_content(evidence_row: EvidenceFileRow) -> Any:
    """Dispatch a manifest row to the reader matching its file extension."""
    path = resolve_evidence_path(evidence_row)
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return read_text_evidence(path)
    if suffix in JSON_EXTENSIONS:
        return read_json_evidence(path)
    if suffix in SQL_EXTENSIONS:
        return read_sql_evidence(path)
    if suffix in PDF_EXTENSIONS:
        return read_pdf_evidence(path)
    if suffix in IMAGE_EXTENSIONS:
        return read_image_evidence_path(path)
    raise ValueError(f"Unsupported evidence file extension: {suffix} ({path})")
