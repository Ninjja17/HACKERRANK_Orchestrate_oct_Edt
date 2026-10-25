"""Strict, fail-fast CSV loaders for every dataset file.

Each loader reads one specific file path (no globbing / discovery) and
validates every row against its Pydantic schema. A malformed row raises
immediately with the file path and row number so data problems surface at
load time rather than silently producing wrong answers downstream.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config import (
    DELETION_REQUESTS_CSV,
    DELETION_RULES_CSV,
    EVIDENCE_FILES_CSV,
    SYSTEM_ACTIVITY_CSV,
    USER_AUDIT_LOG_CSV,
    USER_IDENTITIES_CSV,
)
from schemas import (
    DeletionRequestRow,
    DeletionRuleRow,
    EvidenceFileRow,
    SystemActivityRow,
    UserAuditLogRow,
    UserIdentityRow,
)

M = TypeVar("M", bound=BaseModel)


class DatasetLoadError(RuntimeError):
    """Raised when a dataset file is missing or contains an invalid row."""


def _load_csv(path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        raise DatasetLoadError(f"Expected dataset file not found: {path}")

    rows: list[M] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, raw_row in enumerate(reader, start=2):  # header is line 1
            try:
                rows.append(model.model_validate(raw_row))
            except ValidationError as exc:
                raise DatasetLoadError(
                    f"Invalid row in {path.name} at line {line_no}: {exc}"
                ) from exc
    return rows


def load_deletion_requests() -> list[DeletionRequestRow]:
    return _load_csv(DELETION_REQUESTS_CSV, DeletionRequestRow)


def load_user_identities() -> list[UserIdentityRow]:
    return _load_csv(USER_IDENTITIES_CSV, UserIdentityRow)


def load_user_audit_log() -> list[UserAuditLogRow]:
    return _load_csv(USER_AUDIT_LOG_CSV, UserAuditLogRow)


def load_system_activity() -> list[SystemActivityRow]:
    return _load_csv(SYSTEM_ACTIVITY_CSV, SystemActivityRow)


def load_deletion_rules() -> list[DeletionRuleRow]:
    return _load_csv(DELETION_RULES_CSV, DeletionRuleRow)


def load_evidence_files() -> list[EvidenceFileRow]:
    return _load_csv(EVIDENCE_FILES_CSV, EvidenceFileRow)
