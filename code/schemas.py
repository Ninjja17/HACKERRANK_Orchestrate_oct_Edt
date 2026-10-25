"""Strict Pydantic schemas for every dataset file and the produced output.

Following the "detailed foundation" approach: every field is typed, every
enum is a closed `Literal`, and invalid rows raise immediately instead of
being silently coerced or skipped.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ReportedStatus = Literal["COMPLETE", "PARTIALLY_COMPLETE", "NOT_STARTED"]
DeletionStatus = Literal["COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"]
Phase = Literal["FINDING", "REMEDIATION", "VERIFICATION"]

Operation = Literal[
    "REPORT_REMAINING_RECORD",
    "REPORT_EXPOSED_PII",
    "REPORT_ACTIVE_DATA_FLOW",
    "REPORT_RESTORE_RISK",
    "DISABLE_DATA_FLOW",
    "BLOCK_RESTORE",
    "DELETE_RECORD",
    "REDACT_RECORD",
    "ANONYMIZE_RECORD",
    "VERIFY_RECORD_ABSENT",
    "VERIFY_RECORD_PRESENT",
    "VERIFY_RECORD_REDACTED",
    "VERIFY_RECORD_ANONYMIZED",
    "VERIFY_DATA_FLOW_DISABLED",
    "VERIFY_RESTORE_BLOCKED",
]

State = Literal[
    "PRESENT",
    "ABSENT",
    "ACTIVE",
    "DISABLED",
    "BLOCKED",
    "REDACTED",
    "ANONYMIZED",
]

# Each operation only ever produces one valid state (per problem_statement.md).
OPERATION_STATE: dict[str, str] = {
    "REPORT_REMAINING_RECORD": "PRESENT",
    "REPORT_EXPOSED_PII": "PRESENT",
    "REPORT_ACTIVE_DATA_FLOW": "ACTIVE",
    "REPORT_RESTORE_RISK": "ACTIVE",
    "DISABLE_DATA_FLOW": "DISABLED",
    "BLOCK_RESTORE": "BLOCKED",
    "DELETE_RECORD": "ABSENT",
    "REDACT_RECORD": "REDACTED",
    "ANONYMIZE_RECORD": "ANONYMIZED",
    "VERIFY_RECORD_ABSENT": "ABSENT",
    "VERIFY_RECORD_PRESENT": "PRESENT",
    "VERIFY_RECORD_REDACTED": "REDACTED",
    "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED",
    "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
    "VERIFY_RESTORE_BLOCKED": "BLOCKED",
}

OPERATION_PHASE: dict[str, Phase] = {
    "REPORT_REMAINING_RECORD": "FINDING",
    "REPORT_EXPOSED_PII": "FINDING",
    "REPORT_ACTIVE_DATA_FLOW": "FINDING",
    "REPORT_RESTORE_RISK": "FINDING",
    "DISABLE_DATA_FLOW": "REMEDIATION",
    "BLOCK_RESTORE": "REMEDIATION",
    "DELETE_RECORD": "REMEDIATION",
    "REDACT_RECORD": "REMEDIATION",
    "ANONYMIZE_RECORD": "REMEDIATION",
    "VERIFY_RECORD_ABSENT": "VERIFICATION",
    "VERIFY_RECORD_PRESENT": "VERIFICATION",
    "VERIFY_RECORD_REDACTED": "VERIFICATION",
    "VERIFY_RECORD_ANONYMIZED": "VERIFICATION",
    "VERIFY_DATA_FLOW_DISABLED": "VERIFICATION",
    "VERIFY_RESTORE_BLOCKED": "VERIFICATION",
}


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields instead of ignoring them."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


# --- dataset/deletion_requests.csv -----------------------------------------
class DeletionRequestRow(StrictModel):
    case_id: str
    email: str
    requested_at: datetime
    reported_status: ReportedStatus
    reported_at: datetime
    audit_at: datetime
    request_text: str


# --- dataset/user_identities.csv -------------------------------------------
class UserIdentityRow(StrictModel):
    user_id: str
    name: str
    email: str
    phone: str
    email_hash: str
    updated_at: datetime


# --- dataset/user_audit_log.csv --------------------------------------------
class UserAuditLogRow(StrictModel):
    audit_id: str
    user_id: str
    event_time: datetime
    field_name: str
    old_value: str
    new_value: str


# --- dataset/system_activity.csv -------------------------------------------
class SystemActivityRow(StrictModel):
    activity_id: str
    event_time: datetime
    event_type: str
    source_system: str
    target_system: str
    resource_id: str
    reported_result: str
    details: str


# --- dataset/deletion_rules.csv --------------------------------------------
class DeletionRuleRow(StrictModel):
    rule_id: str
    data_category: str
    condition: str
    required_action: Literal["DELETE", "REDACT", "ANONYMIZE", "RETAIN"]


# --- dataset/evidence_files.csv --------------------------------------------
class EvidenceFileRow(StrictModel):
    file_id: str
    file_path: str
    origin: str
    captured_at: datetime


# --- produced output: output.csv -------------------------------------------
class Step(StrictModel):
    step_no: int
    phase: Phase
    system_id: str
    resource_id: str
    operation: Operation
    state: State
    evidence_id: str

    @field_validator("step_no")
    @classmethod
    def _step_no_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step_no must start at 1")
        return v

    @field_validator("state")
    @classmethod
    def _state_matches_operation(cls, v: str, info) -> str:
        operation = info.data.get("operation")
        expected = OPERATION_STATE.get(operation) if operation else None
        if expected is not None and v != expected:
            raise ValueError(f"operation {operation} requires state {expected}, got {v}")
        return v

    @field_validator("phase")
    @classmethod
    def _phase_matches_operation(cls, v: str, info) -> str:
        operation = info.data.get("operation")
        expected = OPERATION_PHASE.get(operation) if operation else None
        if expected is not None and v != expected:
            raise ValueError(f"operation {operation} belongs to phase {expected}, got {v}")
        return v


class OutputRow(StrictModel):
    case_id: str
    deletion_status: DeletionStatus
    steps: list[Step]

    @field_validator("steps")
    @classmethod
    def _steps_numbered_consecutively(cls, v: list[Step]) -> list[Step]:
        for idx, step in enumerate(v, start=1):
            if step.step_no != idx:
                raise ValueError(f"steps must be numbered consecutively from 1, got {step.step_no} at position {idx}")
        return v

    @field_validator("steps")
    @classmethod
    def _not_applicable_has_no_steps(cls, v: list[Step], info) -> list[Step]:
        status = info.data.get("deletion_status")
        if status == "NOT_APPLICABLE" and len(v) != 0:
            raise ValueError("NOT_APPLICABLE rows must have an empty steps array")
        return v
