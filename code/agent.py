"""The bounded agent loop for one case — single-call architecture.

All deterministic data (request, deletion rules, and the subject's bounded
evidence) is pre-fetched in Python and packaged into the first message by
`prompts.build_initial_message`. The happy path is exactly ONE API call:
the model reasons over that pre-loaded context and calls
`submit_case_decision` directly. The loop still bounds a small number of
retry turns for the case where `submit_case_decision` is rejected by the
validation layer and the model must resubmit a corrected decision.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from llm_client import LLMClient, ModelResponseError
from normalize import parse_iso8601
from prompts import (
    AGENT_SYSTEM_PROMPT,
    BATCH_AGENT_SYSTEM_PROMPT,
    build_batch_message,
    build_batch_retry_message,
    build_initial_message,
)
from retrieval import gather_case_context
from retry import RetryExhaustedError
from identity import resolve_subject
from schemas import OutputRow
from tools import (
    MAX_EVIDENCE_PER_CALL,
    TOOL_SCHEMAS,
    TOOL_SCHEMAS_BATCH,
    AgentContext,
    parse_batch_results,
    resolve_batch_case,
    tool_submit_case_decision,
)
from validation import enforce_output_contract

logger = logging.getLogger(__name__)

MAX_TURNS = 5  # the happy path uses 1; the rest is headroom for a few rejected-submission retries
MAX_SUBMIT_ATTEMPTS = 4
# Cap on how much of one tool result is written to the console per call, so a
# single large evidence dump can't overwhelm the terminal. The step-by-step
# sequence of calls/decisions is still logged in full either way.
MAX_LOG_PAYLOAD_CHARS = 8000


def _log_payload(label: str, payload: str) -> str:
    if len(payload) <= MAX_LOG_PAYLOAD_CHARS:
        return payload
    return f"{payload[:MAX_LOG_PAYLOAD_CHARS]}... [{label} truncated for console, {len(payload)} chars total]"


def _prefetch_evidence(case_id: str, ctx: AgentContext) -> dict:
    """Deterministically resolve identity + gather bounded evidence, in Python.

    Mirrors what `tools.tool_query_historical_logs` used to do on-demand;
    now it always runs exactly once, before the first (and ideally only)
    model call, so there is no round trip for the model to request it.
    """
    request = ctx.requests_by_id[case_id]
    subject = resolve_subject(request.email, ctx.identities, ctx.audit_log, request.audit_at)
    case_context = gather_case_context(subject, request.audit_at, ctx.graph, max_entries=MAX_EVIDENCE_PER_CALL)

    identity_dict = {
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
    entries = [
        {
            "evidence_id": e.evidence_id,
            "source": e.source,
            "captured_at": e.captured_at.isoformat(),
            "system_id": e.system_id,
            "resource_id": e.resource_id,
            "category": e.category,
            "fields": e.raw,
        }
        for e in case_context.entries[:MAX_EVIDENCE_PER_CALL]
    ]
    return {
        "identity": identity_dict,
        "evidence": entries,
        "total_matching": len(case_context.entries),
        "truncated": len(case_context.entries) > MAX_EVIDENCE_PER_CALL,
    }


def run_agent_for_case(case_id: str, ctx: AgentContext, llm_client: LLMClient) -> OutputRow:
    request = ctx.requests_by_id[case_id]
    audit_at = request.audit_at

    evidence_context = _prefetch_evidence(case_id, ctx)
    known_evidence_ids: dict[str, datetime] = {
        e["evidence_id"]: parse_iso8601(e["captured_at"]) for e in evidence_context["evidence"]
    }

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_message(request, ctx.deletion_rules, evidence_context)},
    ]

    submit_attempts = 0
    last_status = "INCOMPLETE"
    last_steps: list = []
    first_attempted_status: str | None = None  # the model's very first proposed status

    for turn in range(1, MAX_TURNS + 1):
        try:
            message = llm_client.complete_with_tools(messages=messages, tools=TOOL_SCHEMAS)
        except (ModelResponseError, RetryExhaustedError) as exc:
            logger.warning("case %s: turn %s LLM call failed: %s", case_id, turn, exc)
            break

        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.info("case %s | turn %s | agent reasoning: %s", case_id, turn, _log_payload("reasoning", message.content))

        if not message.tool_calls:
            logger.info("case %s | turn %s | no tool call; nudging the agent to submit", case_id, turn)
            messages.append({"role": "user", "content": "Call submit_case_decision now with your decision."})
            continue

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"
            logger.info("case %s | turn %s | TOOL CALL -> %s(%s)", case_id, turn, name, raw_arguments)
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}

            if name != "submit_case_decision":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Unknown tool {name!r}; call submit_case_decision instead."}),
                    }
                )
                continue

            submit_attempts += 1
            allow_retry = turn < MAX_TURNS and submit_attempts < MAX_SUBMIT_ATTEMPTS
            last_status = arguments.get("status")
            last_steps_json = arguments.get("steps_json") or "[]"
            if first_attempted_status is None:
                first_attempted_status = last_status

            # Deterministic guardrail, independent of model reliability: once
            # the model's first attempt commits to COMPLETE/INCOMPLETE (i.e.
            # it judged request_text to authorize deletion), a later retry
            # must not flip to NOT_APPLICABLE just because it struggled to
            # produce a valid COMPLETE/INCOMPLETE submission — that would be
            # giving up, not a genuine re-read of request_text.
            if last_status == "NOT_APPLICABLE" and first_attempted_status != "NOT_APPLICABLE":
                if allow_retry:
                    logger.info(
                        "case %s | turn %s | submit_case_decision REJECTED (attempt %s/%s): "
                        "NOT_APPLICABLE contradicts the first attempt's status %r",
                        case_id,
                        turn,
                        submit_attempts,
                        MAX_SUBMIT_ATTEMPTS,
                        first_attempted_status,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "accepted": False,
                                    "error": (
                                        f"Your first attempt proposed {first_attempted_status!r}, meaning you "
                                        "already judged request_text to authorize deletion. NOT_APPLICABLE "
                                        "cannot be used now just because a COMPLETE/INCOMPLETE submission was "
                                        "rejected. Fix the rejected submission instead: resend COMPLETE or "
                                        "INCOMPLETE with corrected steps."
                                    ),
                                }
                            ),
                        }
                    )
                    continue
                logger.warning(
                    "case %s: forcing INCOMPLETE fallback — model flipped to NOT_APPLICABLE after a %r first attempt",
                    case_id,
                    first_attempted_status,
                )
                last_status, last_steps = "INCOMPLETE", []

            output_row, tool_result = tool_submit_case_decision(
                last_status,
                last_steps_json,
                case_id=case_id,
                known_evidence_ids=known_evidence_ids,
                audit_at=audit_at,
                allow_retry=allow_retry,
            )
            if output_row is not None:
                logger.info(
                    "case %s | turn %s | submit_case_decision ACCEPTED: status=%s, %s step(s)",
                    case_id,
                    turn,
                    output_row.deletion_status,
                    len(output_row.steps),
                )
                return output_row
            logger.info(
                "case %s | turn %s | submit_case_decision REJECTED (attempt %s/%s): %s",
                case_id,
                turn,
                submit_attempts,
                MAX_SUBMIT_ATTEMPTS,
                tool_result.get("error"),
            )
            try:
                last_steps = json.loads(last_steps_json)
            except json.JSONDecodeError:
                last_steps = []
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result)})

    logger.warning("case %s: agent loop ended without an accepted submission; falling back", case_id)
    return enforce_output_contract(
        case_id=case_id,
        model_status=last_status,
        model_steps=last_steps,
        known_evidence_ids=known_evidence_ids,
        audit_at=audit_at,
        allow_retry=False,
    )


def run_agent_for_batch(case_ids: list[str], ctx: AgentContext, llm_client: LLMClient) -> dict[str, OutputRow]:
    """Decide a whole batch of cases with ONE LLM call, plus a bounded
    micro-retry for only the specific cases that violated the output
    contract on that first pass.

    Collapses 518 cases into ~ceil(518/BATCH_SIZE) requests so the
    client-side requests-per-minute throttle -- not round-trip count --
    is the throughput limiter. Unlike the original no-retry batch design,
    a case that fails validation does NOT immediately get a deterministic
    fallback: it is collected and re-sent, alone with the others that also
    failed, in exactly one more LLM call with the specific violation
    feedback attached (`prompts.build_batch_retry_message`). Whatever that
    second pass returns is final -- resolved with `allow_retry=False`,
    which never raises, so the loop still terminates in at most 2 calls
    per batch regardless of model behavior.
    """
    cases_payload: list[dict] = []
    known_evidence_ids_by_case: dict[str, dict[str, datetime]] = {}
    audit_at_by_case: dict[str, datetime] = {}

    for case_id in case_ids:
        request = ctx.requests_by_id[case_id]
        evidence_context = _prefetch_evidence(case_id, ctx)
        known_evidence_ids_by_case[case_id] = {
            e["evidence_id"]: parse_iso8601(e["captured_at"]) for e in evidence_context["evidence"]
        }
        audit_at_by_case[case_id] = request.audit_at
        cases_payload.append(
            {
                "case_id": case_id,
                "request": request.model_dump(mode="json"),
                "identity": evidence_context["identity"],
                "evidence": evidence_context["evidence"],
                "total_matching": evidence_context["total_matching"],
                "truncated": evidence_context["truncated"],
            }
        )
    cases_payload_by_id = {case["case_id"]: case for case in cases_payload}

    def _fallback_all() -> dict[str, OutputRow]:
        return {
            case_id: enforce_output_contract(
                case_id=case_id,
                model_status="INCOMPLETE",
                model_steps=[],
                known_evidence_ids=known_evidence_ids_by_case[case_id],
                audit_at=audit_at_by_case[case_id],
                allow_retry=False,
            )
            for case_id in case_ids
        }

    def _call_batch(system_prompt: str, user_message: str) -> dict[str, dict | None]:
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        message = llm_client.complete_with_tools(messages=messages, tools=TOOL_SCHEMAS_BATCH)
        results_json: Any = "[]"
        for tool_call in message.tool_calls or []:
            if tool_call.function.name == "submit_batch_decisions":
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                results_json = arguments.get("results") or "[]"
                break
        else:
            logger.warning("batch %s: model did not call submit_batch_decisions this pass", case_ids)
        logger.info("batch %s | TOOL CALL -> submit_batch_decisions(%s)", case_ids, _log_payload("batch results", results_json))
        return parse_batch_results(results_json, list(cases_payload_by_id))

    # --- Pass 1: the whole batch, one call -----------------------------
    try:
        parsed_pass_1 = _call_batch(BATCH_AGENT_SYSTEM_PROMPT, build_batch_message(cases_payload, ctx.deletion_rules))
    except (ModelResponseError, RetryExhaustedError) as exc:
        logger.warning("batch %s: LLM call failed: %s; falling back all cases to INCOMPLETE", case_ids, exc)
        return _fallback_all()

    rows: dict[str, OutputRow] = {}
    violations: dict[str, str] = {}
    for case_id in case_ids:
        output_row, violation = resolve_batch_case(
            parsed_pass_1[case_id],
            case_id=case_id,
            known_evidence_ids=known_evidence_ids_by_case[case_id],
            audit_at=audit_at_by_case[case_id],
            allow_retry=True,
        )
        if output_row is not None:
            rows[case_id] = output_row
        else:
            violations[case_id] = violation or "output contract violation"

    if not violations:
        return rows

    # --- Pass 2 (micro-retry): only the cases that failed, once -------
    retry_case_ids = list(violations)
    logger.info("batch %s: micro-retry for %s case(s) that failed validation: %s", case_ids, len(retry_case_ids), retry_case_ids)
    retry_payload = [cases_payload_by_id[cid] for cid in retry_case_ids]
    try:
        parsed_pass_2 = _call_batch(
            BATCH_AGENT_SYSTEM_PROMPT,
            build_batch_retry_message(retry_payload, ctx.deletion_rules, violations),
        )
    except (ModelResponseError, RetryExhaustedError) as exc:
        logger.warning("batch %s: micro-retry LLM call failed: %s; falling back those cases to INCOMPLETE", case_ids, exc)
        parsed_pass_2 = {cid: None for cid in retry_case_ids}

    for case_id in retry_case_ids:
        output_row, _ = resolve_batch_case(
            parsed_pass_2.get(case_id),
            case_id=case_id,
            known_evidence_ids=known_evidence_ids_by_case[case_id],
            audit_at=audit_at_by_case[case_id],
            allow_retry=False,
        )
        rows[case_id] = output_row  # type: ignore[assignment]  # allow_retry=False never returns None

    return rows
