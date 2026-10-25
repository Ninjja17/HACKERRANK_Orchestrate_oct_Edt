"""Generates `code/evaluation/usage_report.md` from one run's token usage.

Aggregates `llm_client.ModelUsage` entries by (provider, model_name) and
estimates cost using the configurable rates in `config.PRICING_SETTINGS`.
Costs are estimates, clearly labeled as such, since real billing rates
vary by provider/contract and change over time.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import PRICING_SETTINGS
from llm_client import ModelUsage


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * PRICING_SETTINGS.price_per_1m_input_tokens
        + output_tokens / 1_000_000 * PRICING_SETTINGS.price_per_1m_output_tokens
    )


def build_usage_report(usage_log: list[ModelUsage], case_count: int) -> str:
    by_model: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    for usage in usage_log:
        bucket = by_model[(usage.provider, usage.model_name)]
        bucket["calls"] += 1
        bucket["input_tokens"] += usage.input_tokens
        bucket["output_tokens"] += usage.output_tokens

    total_calls = sum(v["calls"] for v in by_model.values())
    total_input = sum(v["input_tokens"] for v in by_model.values())
    total_output = sum(v["output_tokens"] for v in by_model.values())
    total_tokens = total_input + total_output
    total_cost = _estimate_cost(total_input, total_output)
    avg_tokens = total_tokens / case_count if case_count else 0.0
    avg_cost = total_cost / case_count if case_count else 0.0

    lines = [
        "# Usage Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if not usage_log:
        lines += [
            "> **Placeholder:** no model calls have been recorded yet (this file is "
            "regenerated automatically by `code/main.py` after every run — run it "
            "with a real `ORCHESTRATOR_API_KEY` in `code/.env` to populate real numbers).",
            "",
        ]
    lines += [
        f"Run summary for the model calls that produced `output.csv` ({case_count} cases).",
        "",
        "## Per-model totals",
        "",
        "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Estimated cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (provider, model_name), v in sorted(by_model.items()):
        model_tokens = v["input_tokens"] + v["output_tokens"]
        model_cost = _estimate_cost(v["input_tokens"], v["output_tokens"])
        lines.append(
            f"| {provider} | {model_name} | {v['calls']} | {v['input_tokens']} | "
            f"{v['output_tokens']} | {model_tokens} | ${model_cost:.4f} |"
        )
    if not by_model:
        lines.append("| *(no model calls recorded)* | | 0 | 0 | 0 | 0 | $0.0000 |")

    lines += [
        "",
        "## Overall totals",
        "",
        f"- Total model calls: {total_calls}",
        f"- Total input (prompt) tokens: {total_input}",
        f"- Total output (completion) tokens: {total_output}",
        f"- Total tokens: {total_tokens}",
        f"- Cases processed: {case_count}",
        f"- Average tokens per request (case): {avg_tokens:.1f}",
        f"- Average estimated cost per request (case): ${avg_cost:.4f}",
        f"- **Estimated total run cost: ${total_cost:.4f}**",
        "",
        (
            f"Cost estimates use configured rates of "
            f"${PRICING_SETTINGS.price_per_1m_input_tokens:.2f}/1M input tokens and "
            f"${PRICING_SETTINGS.price_per_1m_output_tokens:.2f}/1M output tokens "
            "(`ORCHESTRATOR_PRICE_PER_1M_INPUT` / `ORCHESTRATOR_PRICE_PER_1M_OUTPUT` in "
            "`code/.env`). Verify against the provider's actual billing rate for the "
            "model(s) used before reporting final costs."
        ),
        "",
    ]
    return "\n".join(lines)


def write_usage_report(usage_log: list[ModelUsage], case_count: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_usage_report(usage_log, case_count), encoding="utf-8")
