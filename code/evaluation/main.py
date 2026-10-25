"""Evaluation driver: quantitative accuracy against the 32 known-answer
sample cases, plus 3 participant-authored custom test cases.

Run with: python3 code/evaluation/main.py (from the repository root, or
anywhere -- the repo's code/ directory is added to sys.path below).

Writes machine-readable results to code/evaluation/eval_results.json; the
human-readable summary in evaluation_report.md Section 5 is transcribed
from this run's output.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from agent import run_agent_for_batch  # noqa: E402
from config import EVALUATION_DIR, SAMPLE_DELETION_REQUESTS_CSV, SAMPLE_OUTPUT_CSV  # noqa: E402
from data_loader import _load_csv, load_deletion_rules  # noqa: E402
from identity import build_identity_indexes  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from normalize import parse_iso8601  # noqa: E402
from retrieval import build_evidence_graph  # noqa: E402
from schemas import DeletionRequestRow  # noqa: E402
from tools import AgentContext  # noqa: E402

BATCH_SIZE = 10


def load_sample_golden() -> dict[str, dict]:
    golden: dict[str, dict] = {}
    with SAMPLE_OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            golden[row["case_id"]] = {
                "deletion_status": row["deletion_status"],
                "steps": json.loads(row["steps_json"]),
            }
    return golden


def step_key(step: dict) -> tuple:
    return (step.get("system_id"), step.get("resource_id"), step.get("operation"))


def run_batches(ctx: AgentContext, llm_client: LLMClient, case_ids: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    batches = [case_ids[i : i + BATCH_SIZE] for i in range(0, len(case_ids), BATCH_SIZE)]
    for batch in batches:
        batch_results = run_agent_for_batch(batch, ctx, llm_client)
        for case_id, row in batch_results.items():
            results[case_id] = row.model_dump(mode="json")
    return results


def score_against_golden(predicted: dict[str, dict], golden: dict[str, dict]) -> dict:
    n = len(golden)
    case_correct = 0
    sequence_correct = 0
    step_field_total = 0
    step_field_correct = 0
    per_case = []
    for case_id, gold in golden.items():
        pred = predicted.get(case_id)
        if pred is None:
            per_case.append({"case_id": case_id, "status_match": False, "sequence_match": False, "note": "missing prediction"})
            continue
        status_match = pred["deletion_status"] == gold["deletion_status"]
        case_correct += int(status_match)

        gold_ops = [s["operation"] for s in gold["steps"]]
        pred_ops = [s["operation"] for s in pred["steps"]]
        sequence_match = gold_ops == pred_ops
        sequence_correct += int(sequence_match)

        for i in range(min(len(gold["steps"]), len(pred["steps"]))):
            g, p = gold["steps"][i], pred["steps"][i]
            step_field_total += 1
            if step_key(g) == step_key(p):
                step_field_correct += 1

        per_case.append(
            {
                "case_id": case_id,
                "status_match": status_match,
                "gold_status": gold["deletion_status"],
                "pred_status": pred["deletion_status"],
                "sequence_match": sequence_match,
                "gold_step_count": len(gold["steps"]),
                "pred_step_count": len(pred["steps"]),
            }
        )

    return {
        "n": n,
        "case_accuracy": case_correct / n,
        "sequence_accuracy": sequence_correct / n,
        "step_field_accuracy": (step_field_correct / step_field_total) if step_field_total else None,
        "step_field_total": step_field_total,
        "per_case": per_case,
    }


def build_custom_cases() -> list[DeletionRequestRow]:
    """Three participant-authored cases probing specific policy edges, not
    present anywhere in the dataset.

    1. A non-deletion request using deletion-adjacent language (marketing
       opt-out while keeping the account) -- must be NOT_APPLICABLE.
    2. A deletion request for an email with no matching identity at all --
       the Ambiguous Identifiers Policy forbids a confident COMPLETE here.
    3. A real, otherwise-COMPLETE sample case (case_01 / amina.elmasri) with
       `audit_at` moved to BEFORE evidence_54 (the failover-replica capture
       that proves deletion, captured 2026-09-02T23:00:00Z) -- the agent
       must not be able to verify absence using evidence that didn't exist
       yet at that audit_at, so it cannot still be COMPLETE at this cutoff.
    """
    return [
        DeletionRequestRow(
            case_id="custom_01",
            email="amina.elmasri@gmail.com",
            requested_at=parse_iso8601("2026-09-10T09:00:00Z"),
            reported_status="NOT_STARTED",
            reported_at=parse_iso8601("2026-09-10T09:05:00Z"),
            audit_at=parse_iso8601("2026-09-20T00:00:00Z"),
            request_text=(
                "Please unsubscribe me from all marketing emails and promotional "
                "notifications. I still want to log in and use my account normally."
            ),
        ),
        DeletionRequestRow(
            case_id="custom_02",
            email="totally.unknown.person@nowhere.example",
            requested_at=parse_iso8601("2026-09-10T09:00:00Z"),
            reported_status="NOT_STARTED",
            reported_at=parse_iso8601("2026-09-10T09:05:00Z"),
            audit_at=parse_iso8601("2026-09-20T00:00:00Z"),
            request_text="Please delete all of my personal data from your systems immediately.",
        ),
        DeletionRequestRow(
            case_id="custom_03",
            email="amina.elmasri@gmail.com",
            requested_at=parse_iso8601("2026-08-18T09:14:00Z"),
            reported_status="COMPLETE",
            reported_at=parse_iso8601("2026-08-20T16:42:00Z"),
            audit_at=parse_iso8601("2026-09-01T00:00:00Z"),  # BEFORE evidence_54 (2026-09-02T23:00:00Z)
            request_text=(
                "Hi, I haven't used this account in years and would like it closed. "
                "Please remove the profile and the personal details attached to it."
            ),
        ),
    ]


def expected_custom_outcomes() -> dict[str, str]:
    return {
        "custom_01": "NOT_APPLICABLE (marketing opt-out, not a deletion request)",
        "custom_02": "Must NOT be a confident COMPLETE (unresolved identity -- no matching user)",
        "custom_03": "Must NOT be COMPLETE via evidence_54 (captured after this case's audit_at)",
    }


def main() -> None:
    identities, audit_log = build_identity_indexes()
    graph = build_evidence_graph()
    deletion_rules = load_deletion_rules()
    llm_client = LLMClient()

    # --- Part 1: quantitative accuracy against the 32 known-answer sample cases
    sample_requests = _load_csv(SAMPLE_DELETION_REQUESTS_CSV, DeletionRequestRow)
    sample_ctx = AgentContext(
        requests_by_id={r.case_id: r for r in sample_requests},
        identities=identities,
        audit_log=audit_log,
        deletion_rules=deletion_rules,
        graph=graph,
    )
    sample_predicted = run_batches(sample_ctx, llm_client, sorted(sample_ctx.requests_by_id))
    golden = load_sample_golden()
    scores = score_against_golden(sample_predicted, golden)

    # --- Part 2: 3 custom test cases
    custom_cases = build_custom_cases()
    custom_ctx = AgentContext(
        requests_by_id={r.case_id: r for r in custom_cases},
        identities=identities,
        audit_log=audit_log,
        deletion_rules=deletion_rules,
        graph=graph,
    )
    custom_predicted = run_batches(custom_ctx, llm_client, sorted(custom_ctx.requests_by_id))
    expected = expected_custom_outcomes()
    custom_results = {
        case_id: {"expected": expected[case_id], "predicted_status": custom_predicted[case_id]["deletion_status"], "predicted_steps": custom_predicted[case_id]["steps"]}
        for case_id in sorted(custom_predicted)
    }

    output = {
        "sample_eval": scores,
        "custom_cases": custom_results,
    }
    out_path = EVALUATION_DIR / "eval_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
