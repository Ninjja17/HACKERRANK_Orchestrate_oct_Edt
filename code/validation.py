"""Strict output-enforcement layer.

Intercepts the model's proposed `deletion_status` + steps and fails closed
on every contract violation instead of trusting the model's compliance:

- Unknown/missing operation, system_id, or resource_id -> step is dropped.
- A FINDING step without a valid evidence_id (one of the evidence IDs
  actually shown to the model for this case, timestamped at or before
  `audit_at`) -> step is dropped.
- Any structural violation, while `allow_retry=True`, raises `NeedsRetryError`
  so the caller can re-prompt the model with the exact reason before
  falling back to a corrected/degraded result.
- `NOT_APPLICABLE` always forces `steps=[]`.
- `COMPLETE` requires every surviving step to be `VERIFICATION` with a
  non-empty evidence_id; otherwise the case is downgraded to `INCOMPLETE`
  (never the reverse) per the "unsafe states never completed" policy.
- Once final status is `INCOMPLETE`, every `VERIFICATION` step's
  evidence_id is forced to `""` (a future check, not yet performed).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ordering import order_steps
from schemas import OPERATION_PHASE, OPERATION_STATE, OutputRow, Step

logger = logging.getLogger(__name__)

ALLOWED_STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}


class NeedsRetryError(RuntimeError):
    """Raised to signal the model should be re-prompted with this feedback."""


def _coerce_raw_step(
    raw: dict[str, Any],
    known_evidence_ids: dict[str, datetime],
    audit_at: datetime,
    warnings: list[str],
) -> dict[str, Any] | None:
    operation = raw.get("operation")
    if operation not in OPERATION_PHASE:
        warnings.append(f"unknown operation {operation!r} was dropped")
        return None

    system_id = str(raw.get("system_id") or "").strip()
    resource_id = str(raw.get("resource_id") or "").strip()
    if not system_id or not resource_id:
        warnings.append(f"{operation} step missing system_id/resource_id was dropped")
        return None

    evidence_id = str(raw.get("evidence_id") or "").strip()
    phase = OPERATION_PHASE[operation]
    state = OPERATION_STATE[operation]

    if phase == "FINDING":
        captured_at = known_evidence_ids.get(evidence_id)
        if not evidence_id or captured_at is None or captured_at > audit_at:
            warnings.append(
                f"FINDING {operation} on {system_id}/{resource_id} cited evidence_id {evidence_id!r}, "
                "which is not among the evidence shown for this case at or before audit_at; step was dropped"
            )
            return None
    elif phase == "REMEDIATION":
        evidence_id = ""  # remediation is always proposed/future work
    elif phase == "VERIFICATION" and evidence_id:
        captured_at = known_evidence_ids.get(evidence_id)
        if captured_at is None or captured_at > audit_at:
            warnings.append(
                f"VERIFICATION {operation} on {system_id}/{resource_id} cited evidence_id {evidence_id!r}, "
                "which is not among the evidence shown for this case at or before audit_at; evidence_id cleared"
            )
            evidence_id = ""

    return {
        "system_id": system_id,
        "resource_id": resource_id,
        "operation": operation,
        "phase": phase,
        "state": state,
        "evidence_id": evidence_id,
    }


def enforce_output_contract(
    *,
    case_id: str,
    model_status: str,
    model_steps: list[dict[str, Any]],
    known_evidence_ids: dict[str, datetime],
    audit_at: datetime,
    allow_retry: bool,
) -> OutputRow:
    if model_status not in ALLOWED_STATUSES:
        if allow_retry:
            raise NeedsRetryError(
                f"deletion_status must be exactly one of {sorted(ALLOWED_STATUSES)}, got {model_status!r}"
            )
        logger.warning("case %s: invalid deletion_status %r; falling back to INCOMPLETE", case_id, model_status)
        model_status = "INCOMPLETE"

    if model_status == "NOT_APPLICABLE":
        if model_steps:
            logger.info("case %s: NOT_APPLICABLE with proposed steps; forcing steps=[]", case_id)
        return OutputRow(case_id=case_id, deletion_status="NOT_APPLICABLE", steps=[])

    warnings: list[str] = []
    coerced = [
        c for raw in model_steps if (c := _coerce_raw_step(raw, known_evidence_ids, audit_at, warnings)) is not None
    ]

    if allow_retry and warnings:
        raise NeedsRetryError("Fix the following and resend the full JSON object: " + "; ".join(warnings))

    if model_status == "COMPLETE":
        unproven = [c for c in coerced if c["phase"] != "VERIFICATION" or not c["evidence_id"]]
        if unproven or not coerced:
            if allow_retry:
                raise NeedsRetryError(
                    "COMPLETE requires at least one VERIFICATION step, every one of them with a valid, "
                    "non-empty evidence_id proving the state at audit_at. Remove any FINDING/REMEDIATION "
                    "steps or evidence-less VERIFICATION steps, or change deletion_status to INCOMPLETE."
                )
            logger.warning("case %s: COMPLETE claim is not fully evidence-backed; downgrading to INCOMPLETE", case_id)
            model_status = "INCOMPLETE"

    if model_status == "INCOMPLETE":
        if not coerced and allow_retry:
            raise NeedsRetryError(
                "INCOMPLETE requires at least one FINDING step citing evidence, plus the remediation "
                "and verification steps needed to resolve it."
            )
        for c in coerced:
            if c["phase"] == "VERIFICATION":
                c["evidence_id"] = ""  # a future check, not yet performed

    for warning in warnings:
        logger.warning("case %s: %s", case_id, warning)

    steps = [
        Step(
            step_no=1,
            phase=c["phase"],
            system_id=c["system_id"],
            resource_id=c["resource_id"],
            operation=c["operation"],
            state=c["state"],
            evidence_id=c["evidence_id"],
        )
        for c in coerced
    ]
    return OutputRow(case_id=case_id, deletion_status=model_status, steps=order_steps(steps))
