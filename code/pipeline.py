"""Orchestrates one full run: load data once -> run the bounded tool-calling
agent (`agent.py`) over every case -> write `output.csv`.

Model-driven routing: there is one evaluation entry point,
`agent.run_agent_for_case`, called identically for every request. It is
the LLM — not hardcoded per-case or per-dataset branching — that decides
which tools to call, how to interpret what they return, and the final
status/step plan. This module only loads each dataset file once into a
shared `AgentContext`.

Cases run concurrently (`ThreadPoolExecutor`, bounded by
`CONCURRENCY_SETTINGS.max_workers`) since each case is independent and
`LLMClient` is thread-safe (shared rate limiter + usage log). Concurrency
can only close the gap between the configured requests-per-minute cap and
actual achieved throughput — it cannot exceed that cap, since the limiter
is global/shared across every worker. Results are collected keyed by
`case_id` and only sorted into final order after every case has finished,
so output ordering never depends on completion order.

`build_context()` and `run()` are split so `main.py` can own the
`LLMClient` and read its `usage_log` after the run (even a partial one)
to produce `evaluation/usage_report.md`.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from agent import run_agent_for_batch
from config import BATCH_SIZE, CASE_LIMIT, CONCURRENCY_SETTINGS
from data_loader import load_deletion_requests, load_deletion_rules
from identity import build_identity_indexes
from llm_client import LLMClient
from retrieval import build_evidence_graph
from schemas import OutputRow
from tools import AgentContext

logger = logging.getLogger(__name__)


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_context() -> AgentContext:
    """Load every dataset file once into the shared, read-only agent context."""
    requests = load_deletion_requests()
    deletion_rules = load_deletion_rules()
    identities, audit_log = build_identity_indexes()
    graph = build_evidence_graph()
    requests_by_id = {r.case_id: r for r in requests}
    if CASE_LIMIT:
        limited_ids = sorted(requests_by_id)[:CASE_LIMIT]
        logger.warning("ORCHESTRATOR_CASE_LIMIT set: running only %s/%s cases (debug only)", len(limited_ids), len(requests_by_id))
        requests_by_id = {cid: requests_by_id[cid] for cid in limited_ids}
    return AgentContext(
        requests_by_id=requests_by_id,
        identities=identities,
        audit_log=audit_log,
        deletion_rules=deletion_rules,
        graph=graph,
    )


def run(ctx: AgentContext, llm_client: LLMClient) -> list[OutputRow]:
    """Run the agent over every case, batched (`BATCH_SIZE` cases per LLM
    call), concurrently across batches; return rows sorted by `case_id`.

    A progress bar (tqdm) tracks overall case completion on the console;
    the full per-batch reasoning/tool-call trace still goes to
    `run_main.log` via the file logging handler configured in `main.py`.
    """
    case_ids = sorted(ctx.requests_by_id)
    total = len(case_ids)
    batches = _chunk(case_ids, BATCH_SIZE)
    results_by_case: dict[str, OutputRow] = {}

    with ThreadPoolExecutor(max_workers=CONCURRENCY_SETTINGS.max_workers) as executor:
        futures = {executor.submit(run_agent_for_batch, batch, ctx, llm_client): batch for batch in batches}
        with tqdm(total=total, desc="Auditing cases", unit="case") as progress:
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_results = future.result()
                except Exception:
                    logger.exception("batch %s: unhandled error in agent loop", batch)
                    raise
                results_by_case.update(batch_results)
                for case_id in batch:
                    output_row = results_by_case[case_id]
                    logger.info(
                        "=== %s — decided %s with %s step(s) (%s/%s complete) ===",
                        case_id,
                        output_row.deletion_status,
                        len(output_row.steps),
                        len(results_by_case),
                        total,
                    )
                progress.update(len(batch))

    # Deterministic final ordering: sorted by case_id, independent of
    # whatever order the concurrent batches actually finished in.
    return [results_by_case[case_id] for case_id in case_ids]
