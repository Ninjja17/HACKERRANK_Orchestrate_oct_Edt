"""Identity resolution: map a request email to the subject's full identifier set.

Edge-case policy (ambiguous identifiers): if the request email maps to zero
or more than one `user_id`, this module does not guess — it returns an
`UNRESOLVED` or `AMBIGUOUS` `SubjectIdentity` and still exposes every
identifier it could gather, so the retrieval/prompt layer can search
broadly while the model is instructed to report conservatively rather than
claim a confident match.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from data_loader import load_user_audit_log, load_user_identities
from normalize import normalize_email, normalize_phone
from schemas import UserAuditLogRow, UserIdentityRow

ResolutionStatus = Literal["RESOLVED", "AMBIGUOUS", "UNRESOLVED"]


@dataclass(frozen=True)
class SubjectIdentity:
    """All identifiers known for one subject, current and historical."""

    input_email: str
    status: ResolutionStatus
    user_ids: tuple[str, ...]
    current_identities: tuple[UserIdentityRow, ...]
    historical_emails: frozenset[str]
    historical_phones: frozenset[str]
    audit_trail: tuple[UserAuditLogRow, ...]

    def all_identifiers(self) -> set[str]:
        ids: set[str] = {self.input_email}
        ids.update(self.historical_emails)
        ids.update(self.historical_phones)
        ids.update(uid.lower() for uid in self.user_ids)
        for identity in self.current_identities:
            ids.add(identity.email.strip().lower())
            phone_digits = "".join(ch for ch in identity.phone if ch.isdigit())
            if len(phone_digits) >= 7:
                ids.add(phone_digits)
            if identity.email_hash:
                ids.add(identity.email_hash.strip().lower())
        return ids


def build_identity_indexes() -> tuple[list[UserIdentityRow], list[UserAuditLogRow]]:
    """Load the two identity dataset files once for reuse across cases."""
    return load_user_identities(), load_user_audit_log()


def resolve_subject(
    email: str,
    identities: list[UserIdentityRow],
    audit_log: list[UserAuditLogRow],
    audit_at: datetime,
) -> SubjectIdentity:
    """Resolve the request `email` to a subject and their known identifiers.

    Matches the current identity snapshot (case-insensitive) and the
    historical email changes in `audit_log` up to `audit_at`. Never follows
    identity changes that happen after `audit_at`.
    """
    normalized = normalize_email(email)

    current_matches = [u for u in identities if u.email.strip().lower() == normalized]
    history_user_ids = {
        row.user_id
        for row in audit_log
        if row.field_name.lower() == "email"
        and row.event_time <= audit_at
        and normalized in {row.old_value.strip().lower(), row.new_value.strip().lower()}
    }

    all_user_ids = sorted({u.user_id for u in current_matches} | history_user_ids)

    if not all_user_ids:
        return SubjectIdentity(
            input_email=normalized,
            status="UNRESOLVED",
            user_ids=(),
            current_identities=(),
            historical_emails=frozenset(),
            historical_phones=frozenset(),
            audit_trail=(),
        )

    current_identities = [u for u in identities if u.user_id in all_user_ids]
    relevant_audit = sorted(
        (row for row in audit_log if row.user_id in all_user_ids and row.event_time <= audit_at),
        key=lambda r: r.event_time,
    )

    historical_emails = {
        value.strip().lower()
        for row in relevant_audit
        if row.field_name.lower() == "email"
        for value in (row.old_value, row.new_value)
        if value
    }
    historical_phones = {
        normalize_phone(value)
        for row in relevant_audit
        if row.field_name.lower() == "phone"
        for value in (row.old_value, row.new_value)
        if value and len(normalize_phone(value)) >= 7
    }

    status: ResolutionStatus = "RESOLVED" if len(all_user_ids) == 1 else "AMBIGUOUS"

    return SubjectIdentity(
        input_email=normalized,
        status=status,
        user_ids=tuple(all_user_ids),
        current_identities=tuple(current_identities),
        historical_emails=frozenset(historical_emails),
        historical_phones=frozenset(historical_phones),
        audit_trail=tuple(relevant_audit),
    )
