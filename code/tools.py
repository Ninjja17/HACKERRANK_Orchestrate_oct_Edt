"""Agent tools: bounded, callable functions the LLM invokes to gather
evidence dynamically, instead of being handed a pre-assembled context.

Each tool is a small, pure function over a shared `AgentContext` (datasets
loaded once at startup). `TOOL_SCHEMAS` describes them in OpenAI
function-calling format for `llm_client.complete_with_tools`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from identity import SubjectIdentity, resolve_subject
from normalize import parse_iso8601
from retrieval import EvidenceGraph, GraphEntry, gather_case_context
from schemas import DeletionRequestRow, DeletionRuleRow, OutputRow, UserAuditLogRow, UserIdentityRow
from validation import NeedsRetryError, enforce_output_contract

# Cap on evidence items returned per `query_historical_logs` call, so a
# single tool result can never balloon the conversation; the agent is
# nudged to narrow with `resource_id` instead of relying on one giant dump.
MAX_EVIDENCE_PER_CALL = 60


@dataclass
class AgentContext:
    """Datasets loaded once and shared read-only across every case/tool call."""

    requests_by_id: dict[str, DeletionRequestRow]
    identities: list[UserIdentityRow]
    audit_log: list[UserAuditLogRow]
    deletion_rules: list[DeletionRuleRow]
    graph: EvidenceGraph


def _identity_to_dict(subject: SubjectIdentity) -> dict[str, Any]:
    return {
        "status": subject.status,
        "input_email": subject.input_email,
        "user_ids": list(subject.user_ids),
        "current_identities": [
            {
                "user_id": u.user_id,
                "name": u.name,
                "email": u.email,
                "phone": u.phone,
                "email_hash": u.email_hash,
                "updated_at": u.updated_at.isoformat(),
            }
            for u in subject.current_identities
        ],
        "historical_emails": sorted(subject.historical_emails),
        "historical_phones": sorted(subject.historical_phones),
    }


def _entry_to_dict(entry: GraphEntry) -> dict[str, Any]:
    return {
        "evidence_id": entry.evidence_id,
        "source": entry.source,
        "captured_at": entry.captured_at.isoformat(),
        "system_id": entry.system_id,
        "resource_id": entry.resource_id,
        "category": entry.category,
        "fields": entry.raw,
    }


def tool_read_deletion_request(ctx: AgentContext, case_id: str) -> dict[str, Any]:
    """Return the raw, unstructured deletion request for `case_id`."""
    request = ctx.requests_by_id.get(case_id)
    if request is None:
        return {"error": f"Unknown case_id {case_id!r}"}
    return request.model_dump(mode="json")


def tool_query_historical_logs(
    ctx: AgentContext,
    case_id: str,
    max_timestamp: str | None = None,
    resource_id: str | None = None,
) -> dict[str, Any]:
    """Resolve identifiers and return evidence/activity at or before audit_at.

    `max_timestamp`, if given, can only narrow the cutoff further — it is
    always clamped to the case's own `audit_at`, so later evidence can
    never be returned regardless of what is requested.
    """
    request = ctx.requests_by_id.get(case_id)
    if request is None:
        return {"error": f"Unknown case_id {case_id!r}"}

    cutoff = request.audit_at
    if max_timestamp:
        try:
            cutoff = min(cutoff, parse_iso8601(max_timestamp))
        except ValueError:
            pass  # ignore an unparsable override; keep the hard audit_at cap

    subject = resolve_subject(request.email, ctx.identities, ctx.audit_log, cutoff)
    case_context = gather_case_context(subject, cutoff, ctx.graph)

    entries = case_context.entries
    if resource_id:
        entries = [e for e in entries if e.resource_id == resource_id]

    total = len(entries)
    truncated = total > MAX_EVIDENCE_PER_CALL
    shown = entries[-MAX_EVIDENCE_PER_CALL:] if truncated else entries

    return {
        "identity": _identity_to_dict(subject),
        "evidence": [_entry_to_dict(e) for e in shown],
        "total_matching": total,
        "truncated": truncated,
        "hint": (
            "Call again with resource_id=<one of the resource_id values above> "
            "to see the full history for a single resource."
            if truncated
            else None
        ),
    }


def tool_fetch_deletion_rules(ctx: AgentContext) -> list[dict[str, str]]:
    """Return the compliance rule table: data_category, condition, required_action."""
    return [
        {
            "data_category": r.data_category,
            "condition": r.condition,
            "required_action": r.required_action,
        }
        for r in ctx.deletion_rules
    ]


def tool_submit_case_decision(
    status: str,
    steps_json: str,
    *,
    case_id: str,
    known_evidence_ids: dict[str, datetime],
    audit_at: datetime,
    allow_retry: bool,
) -> tuple[OutputRow | None, dict[str, Any]]:
    """Validate and, if the output contract is satisfied, accept the case decision.

    This is the deterministic safety boundary: it is the ONLY place a case
    decision can be accepted, and it fails closed on every contract
    violation instead of trusting the model's compliance:

    1. Allowed phases: FINDING, REMEDIATION, VERIFICATION (anything else is
       rejected/dropped).
    2. Allowed FINDING operations/states: REPORT_REMAINING_RECORD/PRESENT,
       REPORT_EXPOSED_PII/PRESENT, REPORT_ACTIVE_DATA_FLOW/ACTIVE,
       REPORT_RESTORE_RISK/ACTIVE.
    3. Allowed REMEDIATION operations/states: DISABLE_DATA_FLOW/DISABLED,
       BLOCK_RESTORE/BLOCKED, DELETE_RECORD/ABSENT, REDACT_RECORD/REDACTED,
       ANONYMIZE_RECORD/ANONYMIZED.
    4. Allowed VERIFICATION operations/states: VERIFY_RECORD_ABSENT/ABSENT,
       VERIFY_RECORD_PRESENT/PRESENT, VERIFY_RECORD_REDACTED/REDACTED,
       VERIFY_RECORD_ANONYMIZED/ANONYMIZED, VERIFY_DATA_FLOW_DISABLED/DISABLED,
       VERIFY_RESTORE_BLOCKED/BLOCKED.
       (2-4 are enforced by deriving phase/state from `operation` via
       `schemas.OPERATION_PHASE`/`OPERATION_STATE` — there is no other way
       to produce a phase/state than through one of these exact pairs.)
    5. Mandatory sorting: phase order -> operation order -> system_id
       ascending -> resource_id ascending, then step_no renumbered
       sequentially from 1 (`ordering.order_steps`).
    6. Evidence rules: NOT_APPLICABLE forces steps_json to `[]`; COMPLETE
       requires every step to be VERIFICATION with a valid, non-empty
       evidence_id; INCOMPLETE requires FINDING steps to cite valid
       evidence and forces REMEDIATION/future-VERIFICATION evidence_id to
       `""`.

    Returns `(output_row, tool_result)`. `output_row` is non-None only when
    the submission was accepted — the caller should then end the agent
    loop for this case. Otherwise `tool_result["error"]` is the exact
    violation string to feed back to the model so it can self-correct.
    """
    try:
        steps = json.loads(steps_json or "[]")
        if not isinstance(steps, list):
            raise ValueError("steps_json must decode to a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        return None, {"accepted": False, "error": f"steps_json was not a valid JSON array: {exc}"}

    try:
        output_row = enforce_output_contract(
            case_id=case_id,
            model_status=status,
            model_steps=steps,
            known_evidence_ids=known_evidence_ids,
            audit_at=audit_at,
            allow_retry=allow_retry,
        )
        return output_row, {"accepted": True}
    except NeedsRetryError as exc:
        return None, {"accepted": False, "error": str(exc)}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit_case_decision",
            "description": (
                "Submit the final decision for the current case. The request, compliance rules, and all "
                "retrieved evidence were already provided in the first message — call this exactly once, "
                "reasoning only over that pre-loaded context. If it violates the output contract, you will "
                "get back the specific violation and must call this again with a corrected submission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]},
                    "steps_json": {
                        "type": "string",
                        "description": (
                            "A JSON-encoded array of step objects, each with exactly: system_id, resource_id, "
                            "operation, evidence_id. Do not include phase/state/step_no. Use \"[]\" for NOT_APPLICABLE."
                        ),
                    },
                },
                "required": ["status", "steps_json"],
            },
        },
    },
]


TOOL_SCHEMAS_BATCH: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit_batch_decisions",
            "description": (
                "Submit the final decision for EVERY case in this batch in a single call. "
                "`results` must be a JSON-encoded array with exactly one object per case shown above "
                "(same case_id values, any order), each with exactly: case_id, status, steps_json. "
                "There is no retry round-trip in this mode -- call this exactly once with your best, "
                "final answer for every case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "string",
                        "description": (
                            'JSON-encoded array of objects, one per case: {"case_id": str, '
                            '"status": "COMPLETE"|"INCOMPLETE"|"NOT_APPLICABLE", '
                            '"steps_json": "<JSON-encoded array of step objects, as a string>"}.'
                        ),
                    },
                },
                "required": ["results"],
            },
        },
    },
]


def _parse_batch_steps_json(steps_json: Any) -> list[Any]:
    if not isinstance(steps_json, str):
        steps_json = json.dumps(steps_json) if steps_json is not None else "[]"
    try:
        steps = json.loads(steps_json or "[]")
        if not isinstance(steps, list):
            raise ValueError("steps_json must decode to a JSON array")
    except (json.JSONDecodeError, ValueError):
        steps = []
    return steps


def parse_batch_results(results_json: Any, case_ids: list[str]) -> dict[str, dict[str, Any] | None]:
    """Parse a (possibly malformed) batched `results` payload into one raw
    `{case_id, status, steps_json}` item per requested `case_id`, or `None`
    where the model omitted that case or the array itself didn't parse.

    `results_json` is normally the JSON-encoded string the schema asks
    for, but some models hand back an already-parsed array instead of a
    string despite the schema -- accept either.
    """
    if isinstance(results_json, list):
        parsed: Any = results_json
    else:
        try:
            parsed = json.loads(results_json or "[]")
        except (json.JSONDecodeError, TypeError):
            parsed = []
    if not isinstance(parsed, list):
        parsed = []

    by_case_id: dict[str, Any] = {}
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("case_id"), str):
            by_case_id[item["case_id"]] = item

    return {case_id: by_case_id.get(case_id) for case_id in case_ids}


def resolve_batch_case(
    item: dict[str, Any] | None,
    *,
    case_id: str,
    known_evidence_ids: dict[str, datetime],
    audit_at: datetime,
    allow_retry: bool,
) -> tuple[OutputRow | None, str | None]:
    """Resolve one case's raw batch item (from `parse_batch_results`) to an
    `OutputRow`.

    Returns `(output_row, None)` on success. With `allow_retry=True`, a
    retryable violation (or a case missing from the model's results
    entirely) returns `(None, violation_message)` instead of raising, so
    the caller can collect every failing case_id in the batch before
    deciding whether to issue one micro-retry call for just those cases.
    With `allow_retry=False` this never returns `(None, ...)` -- every
    violation is deterministically resolved, same fail-closed guarantee as
    the single-case path (`tools.tool_submit_case_decision`).
    """
    if item is None:
        if allow_retry:
            return None, f"No result object was returned for case_id {case_id!r}."
        return (
            enforce_output_contract(
                case_id=case_id,
                model_status="INCOMPLETE",
                model_steps=[],
                known_evidence_ids=known_evidence_ids,
                audit_at=audit_at,
                allow_retry=False,
            ),
            None,
        )

    status = item.get("status")
    steps = _parse_batch_steps_json(item.get("steps_json"))
    try:
        output_row = enforce_output_contract(
            case_id=case_id,
            model_status=status,
            model_steps=steps,
            known_evidence_ids=known_evidence_ids,
            audit_at=audit_at,
            allow_retry=allow_retry,
        )
        return output_row, None
    except NeedsRetryError as exc:
        return None, str(exc)


def tool_submit_batch_decisions(
    results_json: Any,
    *,
    case_ids: list[str],
    known_evidence_ids_by_case: dict[str, dict[str, datetime]],
    audit_at_by_case: dict[str, datetime],
) -> dict[str, OutputRow]:
    """Validate a batched decision array with NO retry round-trip -- every
    per-case violation is fixed by a deterministic local fallback via
    `enforce_output_contract(..., allow_retry=False)`. Thin wrapper around
    `parse_batch_results` + `resolve_batch_case`; kept for the final,
    guaranteed-terminating pass (see `agent.run_agent_for_batch` for the
    micro-retry pass that runs before this one).
    """
    parsed_by_case = parse_batch_results(results_json, case_ids)
    rows: dict[str, OutputRow] = {}
    for case_id in case_ids:
        output_row, _ = resolve_batch_case(
            parsed_by_case[case_id],
            case_id=case_id,
            known_evidence_ids=known_evidence_ids_by_case[case_id],
            audit_at=audit_at_by_case[case_id],
            allow_retry=False,
        )
        rows[case_id] = output_row  # type: ignore[assignment]  # allow_retry=False never returns None
    return rows


def dispatch_tool(ctx: AgentContext, name: str, arguments: dict[str, Any]) -> Any:
    """Route a non-terminal tool call by name.

    Not currently used in the single-call architecture (the only tool the
    model is offered is `submit_case_decision`, handled directly by the
    agent loop) — kept for any future case that needs an on-demand,
    mid-conversation lookup.
    """
    if name == "query_historical_logs":
        return tool_query_historical_logs(ctx, **arguments)
    return {"error": f"Unknown tool {name!r}"}

