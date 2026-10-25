"""Builds the evidence/activity graph and retrieves grounded context per case.

This is the "model-driven routing" data layer: it does not decide anything
about a case's deletion status. It only (a) parses every evidence file and
every row of `system_activity.csv` / `user_audit_log.csv` into a uniform
`GraphEntry`, (b) indexes entries by normalized identifier and by
`resource_id`, and (c) for one case's resolved identifiers, retrieves the
entries that are actually linked to that subject, at or before `audit_at`.

The LLM then interprets the raw retrieved fields itself (FOUND/NOT_FOUND,
`account_status`, `enabled`, `restore_allowed`, etc.) — this module never
pre-decides a record's state.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from data_loader import load_evidence_files, load_system_activity
from evidence import load_evidence_content, resolve_evidence_path
from identity import SubjectIdentity
from normalize import normalize_email, normalize_phone, parse_iso8601
from schemas import EvidenceFileRow

logger = logging.getLogger(__name__)

KNOWN_SYSTEMS = {
    "account_service",
    "search_service",
    "analytics_platform",
    "support_platform",
    "export_service",
    "integration_service",
    "backup_service",
    "security_service",
    "profile_service",
    "notification_service",
    "email_delivery_service",
}

TABLE_TO_SYSTEM = {
    "customer_profiles": "account_service",
    "customer_search_documents": "search_service",
    "customer_analytics_profiles": "analytics_platform",
    "support_records": "support_platform",
    "customer_exports": "export_service",
    "connector_subscriptions": "integration_service",
    "backup_restore_entries": "backup_service",
    "security_audit_records": "security_service",
}

TABLE_TO_CATEGORY = {
    "customer_profiles": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "support_records": "SUPPORT_MESSAGE",
    "customer_exports": "USER_EXPORT",
    "connector_subscriptions": "DATA_FLOW",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "security_audit_records": "SECURITY_AUDIT_LOG",
}

COLLECTION_TO_CATEGORY = {
    "customer_profiles": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "customer_analytics_reports": "ANALYTICS_PROFILE",
    "support_records": "SUPPORT_MESSAGE",
    "customer_exports": "USER_EXPORT",
    "connector_subscriptions": "DATA_FLOW",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "security_audit_records": "SECURITY_AUDIT_LOG",
    "security_audit_ledger": "SECURITY_AUDIT_LOG",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RESOURCE_ID_RE = re.compile(
    r"\b(?:account|search|support|export|data_flow|backup|analytics|audit_record)_[A-Za-z0-9]+\b",
    re.IGNORECASE,
)
_USER_ID_RE = re.compile(r"\buser_\d+\b", re.IGNORECASE)
_LEADING_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
_KV_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')


def _looks_like_system(key: str) -> bool:
    return key in KNOWN_SYSTEMS or key.endswith("_service") or key.endswith("_platform")


@dataclass(frozen=True)
class GraphEntry:
    """One normalized, citable record from evidence, activity, or audit history."""

    evidence_id: str
    source: str  # "evidence_file" | "system_activity" | "user_audit_log"
    captured_at: datetime
    system_id: str | None
    resource_id: str | None
    category: str | None
    identifiers: frozenset[str]
    raw: dict[str, Any]


def flatten_record(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Shallow-ish flatten of a JSON record into dotted keys.

    Nested dicts are merged in with dotted key paths (e.g. `profile.email`).
    Lists of dicts (e.g. `recent_activity`) are summarized by count only,
    since they are not identifier-bearing and would otherwise explode
    prompt size.
    """
    out: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_record(value, prefix=f"{path}."))
        elif isinstance(value, list):
            if value and all(not isinstance(x, (dict, list)) for x in value):
                out[path] = value
            elif value:
                out[f"{path}_count"] = len(value)
        else:
            out[path] = value
    return out


def extract_identifiers(flat: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key, value in flat.items():
        if value is None:
            continue
        key_l = key.lower()
        val = str(value).strip()
        if not val or val.lower() in {"null", "none"}:
            continue
        if key_l.endswith("hash"):
            out.add(val.lower())
        elif "email" in key_l:
            out.add(normalize_email(val))
        elif "phone" in key_l:
            digits = normalize_phone(val)
            if len(digits) >= 7:
                out.add(digits)
        elif key_l in {"customer_reference", "subject_reference", "user_id"} or key_l == "lookup_key":
            if val.lower().startswith("user_"):
                out.add(val.lower())
            elif "@" in val:
                out.add(normalize_email(val))
    return out


def _iter_record_lists(
    obj: Any,
    ancestor_system: str | None,
    ancestor_captured_at: Any,
):
    """Recursively yield (list_of_record_dicts, system_id, captured_at) tuples."""
    if isinstance(obj, dict):
        captured_at = obj.get("captured_at") or obj.get("exported_at") or ancestor_captured_at
        for key, value in obj.items():
            system_here = key if _looks_like_system(key) else ancestor_system
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                yield value, system_here, captured_at
            if isinstance(value, (dict, list)):
                yield from _iter_record_lists(value, system_here, captured_at)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_record_lists(item, ancestor_system, ancestor_captured_at)


def parse_json_entries(evidence_row: EvidenceFileRow, content: Any) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for record_list, ancestor_system, ancestor_captured_at in _iter_record_lists(
        content, None, content.get("exported_at") if isinstance(content, dict) else None
    ):
        for item in record_list:
            if not isinstance(item, dict):
                continue
            flat = flatten_record(item)
            resource_id = flat.get("resource_id")
            system_id = flat.get("system") or flat.get("data_store") or flat.get("service") or ancestor_system
            captured_raw = (
                flat.get("recorded_at")
                or flat.get("observed_at")
                or flat.get("captured_at")
                or flat.get("updated_at")
                or ancestor_captured_at
                or evidence_row.captured_at
            )
            identifiers = extract_identifiers(flat)
            if not resource_id and not identifiers:
                continue
            category = flat.get("record_type") or flat.get("collection") or COLLECTION_TO_CATEGORY.get(str(system_id))
            try:
                captured_at = parse_iso8601(captured_raw)
            except ValueError:
                captured_at = evidence_row.captured_at
            entries.append(
                GraphEntry(
                    evidence_id=evidence_row.file_id,
                    source="evidence_file",
                    captured_at=captured_at,
                    system_id=str(system_id) if system_id else None,
                    resource_id=str(resource_id) if resource_id else None,
                    category=str(category) if category else None,
                    identifiers=frozenset(identifiers),
                    raw=flat,
                )
            )
    return entries


def parse_sql_entries(evidence_row: EvidenceFileRow, rows: list[dict[str, Any]]) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for row in rows:
        table = row.get("_table", "")
        flat = {k: v for k, v in row.items() if k != "_table"}
        resource_id = flat.get("resource_id")
        identifiers = extract_identifiers(flat)
        if not resource_id and not identifiers:
            continue
        captured_raw = flat.get("updated_at") or evidence_row.captured_at
        try:
            captured_at = parse_iso8601(captured_raw)
        except ValueError:
            captured_at = evidence_row.captured_at
        entries.append(
            GraphEntry(
                evidence_id=evidence_row.file_id,
                source="evidence_file",
                captured_at=captured_at,
                system_id=TABLE_TO_SYSTEM.get(table),
                resource_id=str(resource_id) if resource_id else None,
                category=TABLE_TO_CATEGORY.get(table),
                identifiers=frozenset(identifiers),
                raw=flat,
            )
        )
    return entries


def _parse_text_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    kv: dict[str, Any] = {}

    if stripped.count("|") >= 4:
        parts = [p.strip() for p in stripped.split("|")]
        if len(parts) >= 6:
            kv.update(
                {
                    "recorded_at": parts[0],
                    "system": parts[1],
                    "resource_id": parts[2],
                    "record_type": parts[3],
                    "state": parts[4],
                    "identifier": parts[5],
                }
            )

    for key, value in _KV_RE.findall(stripped):
        kv.setdefault(key, value.strip('"'))

    if "resource_id" not in kv:
        rid_match = _RESOURCE_ID_RE.search(stripped)
        if rid_match:
            kv["resource_id"] = rid_match.group(0)

    if "system" not in kv:
        for candidate in KNOWN_SYSTEMS:
            if candidate in stripped:
                kv["system"] = candidate
                break

    emails = set(_EMAIL_RE.findall(stripped))
    user_ids = set(m.group(0).lower() for m in _USER_ID_RE.finditer(stripped))
    if emails:
        kv.setdefault("identifier", next(iter(emails)))
    identifiers = extract_identifiers(kv) | {normalize_email(e) for e in emails} | user_ids

    if not identifiers and "resource_id" not in kv:
        return None

    kv["text"] = stripped
    kv["_identifiers"] = identifiers
    return kv


def parse_text_entries(evidence_row: EvidenceFileRow, content: str) -> list[GraphEntry]:
    entries: list[GraphEntry] = []
    for line in content.splitlines():
        ts_match = _LEADING_TS_RE.match(line.strip())
        parsed = _parse_text_line(line)
        if parsed is None:
            continue
        identifiers = parsed.pop("_identifiers")
        captured_raw = parsed.get("recorded_at") or (ts_match.group(1) if ts_match else None) or evidence_row.captured_at
        try:
            captured_at = parse_iso8601(captured_raw)
        except ValueError:
            captured_at = evidence_row.captured_at
        system_id = parsed.get("system")
        entries.append(
            GraphEntry(
                evidence_id=evidence_row.file_id,
                source="evidence_file",
                captured_at=captured_at,
                system_id=system_id,
                resource_id=parsed.get("resource_id"),
                category=parsed.get("record_type") or COLLECTION_TO_CATEGORY.get(str(system_id)),
                identifiers=frozenset(identifiers),
                raw=parsed,
            )
        )
    return entries


def parse_evidence_file(evidence_row: EvidenceFileRow) -> list[GraphEntry]:
    try:
        path = resolve_evidence_path(evidence_row)
    except Exception:  # noqa: BLE001
        logger.warning("Evidence file could not be resolved: %s", evidence_row.file_id, exc_info=True)
        return []

    suffix = path.suffix.lower()
    try:
        content = load_evidence_content(evidence_row)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to read evidence file %s", evidence_row.file_id, exc_info=True)
        return []

    if suffix == ".sql":
        return parse_sql_entries(evidence_row, content)
    if suffix == ".json":
        return parse_json_entries(evidence_row, content)
    if suffix in (".log", ".txt"):
        return parse_text_entries(evidence_row, content)
    if suffix == ".pdf":
        return parse_text_entries(evidence_row, content) if content else []
    # Images are not yet indexed (no vision pipeline wired in).
    return []


@dataclass
class EvidenceGraph:
    by_identifier: dict[str, list[GraphEntry]]
    by_resource_id: dict[str, list[GraphEntry]]
    evidence_manifest: dict[str, EvidenceFileRow]


def _index_entry(graph_by_id: dict[str, list[GraphEntry]], graph_by_resource: dict[str, list[GraphEntry]], entry: GraphEntry) -> None:
    for identifier in entry.identifiers:
        graph_by_id.setdefault(identifier, []).append(entry)
    if entry.resource_id:
        graph_by_resource.setdefault(entry.resource_id, []).append(entry)


@lru_cache(maxsize=1)
def build_evidence_graph() -> EvidenceGraph:
    """Parse every evidence file and system_activity row once, and index them.

    Cached for the lifetime of the process: dataset files do not change
    mid-run, so this cost is paid exactly once regardless of case count.
    """
    by_identifier: dict[str, list[GraphEntry]] = {}
    by_resource_id: dict[str, list[GraphEntry]] = {}

    manifest = {row.file_id: row for row in load_evidence_files()}
    for evidence_row in manifest.values():
        for entry in parse_evidence_file(evidence_row):
            _index_entry(by_identifier, by_resource_id, entry)

    for activity in load_system_activity():
        entry = GraphEntry(
            evidence_id=activity.activity_id,
            source="system_activity",
            captured_at=activity.event_time,
            system_id=activity.source_system,
            resource_id=activity.resource_id,
            category=activity.event_type,
            identifiers=frozenset(),
            raw=activity.model_dump(mode="json"),
        )
        _index_entry(by_identifier, by_resource_id, entry)

    logger.info(
        "Evidence graph built: %s evidence files, %s identifiers indexed, %s resource_ids indexed",
        len(manifest),
        len(by_identifier),
        len(by_resource_id),
    )
    return EvidenceGraph(by_identifier=by_identifier, by_resource_id=by_resource_id, evidence_manifest=manifest)


@dataclass
class CaseContext:
    entries: list[GraphEntry]
    known_evidence_ids: dict[str, datetime]


def gather_case_context(
    subject: SubjectIdentity,
    audit_at: datetime,
    graph: EvidenceGraph,
    *,
    max_entries: int = 400,
) -> CaseContext:
    """Retrieve every graph entry linked to `subject`, at or before `audit_at`."""
    seed_identifiers = subject.all_identifiers()

    hop1: dict[tuple[str, int], GraphEntry] = {}
    for identifier in seed_identifiers:
        for entry in graph.by_identifier.get(identifier, []):
            if entry.captured_at <= audit_at:
                hop1[(entry.evidence_id, id(entry))] = entry

    discovered_resource_ids = {e.resource_id for e in hop1.values() if e.resource_id}

    hop2: dict[tuple[str, int], GraphEntry] = dict(hop1)
    for resource_id in discovered_resource_ids:
        for entry in graph.by_resource_id.get(resource_id, []):
            if entry.captured_at <= audit_at:
                hop2[(entry.evidence_id, id(entry))] = entry

    entries = sorted(hop2.values(), key=lambda e: e.captured_at)
    if len(entries) > max_entries:
        logger.warning("Truncating case context from %s to %s entries", len(entries), max_entries)
        entries = entries[-max_entries:]

    known_evidence_ids = {e.evidence_id: e.captured_at for e in entries}
    return CaseContext(entries=entries, known_evidence_ids=known_evidence_ids)
