"""Central configuration: file paths and model/runtime settings.

All paths are resolved relative to the repository root so the solution runs
the same way regardless of the current working directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# repo_root/code/config.py -> parents[1] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"

# Load code/.env (if present) before any os.environ.get() below runs, so a
# local, uncommitted .env file is enough to configure the model/API key —
# no secrets are ever hardcoded here.
load_dotenv(CODE_DIR / ".env")
EVIDENCE_DIR = DATASET_DIR / "evidence"
EVALUATION_DIR = CODE_DIR / "evaluation"

# Participant-facing dataset files (strict, specific paths; no discovery/globbing).
DELETION_REQUESTS_CSV = DATASET_DIR / "deletion_requests.csv"
SAMPLE_DELETION_REQUESTS_CSV = DATASET_DIR / "sample_deletion_requests.csv"
SAMPLE_OUTPUT_CSV = DATASET_DIR / "sample_output.csv"
USER_IDENTITIES_CSV = DATASET_DIR / "user_identities.csv"
USER_AUDIT_LOG_CSV = DATASET_DIR / "user_audit_log.csv"
SYSTEM_ACTIVITY_CSV = DATASET_DIR / "system_activity.csv"
DELETION_RULES_CSV = DATASET_DIR / "deletion_rules.csv"
EVIDENCE_FILES_CSV = DATASET_DIR / "evidence_files.csv"
BLANK_OUTPUT_TEMPLATE_CSV = DATASET_DIR / "output.csv"

# Final submission output must be written at the repository root.
OUTPUT_CSV = REPO_ROOT / "output.csv"

EXPECTED_CASE_COUNT = 518

# Optional debug knob: trim the run to the first N case_ids (sorted) for a
# quick speed/behavior check before committing to the full 518-case run.
# Unset (or 0) in a real submission run — output_writer enforces exactly
# EXPECTED_CASE_COUNT rows, so a limited run cannot be the final output.
CASE_LIMIT = int(os.environ.get("ORCHESTRATOR_CASE_LIMIT", "0")) or None

# Batch mode: how many cases are packed into a single LLM call. Collapses
# 518 cases into ~ceil(518/BATCH_SIZE) requests so the client-side
# requests-per-minute throttle (not round-trip count) is the only real
# throughput limiter. No per-case retry round-trip in this mode — invalid
# per-case submissions fall back deterministically (see validation.py).
BATCH_SIZE = int(os.environ.get("ORCHESTRATOR_BATCH_SIZE", "10"))

OUTPUT_COLUMNS = ("case_id", "deletion_status", "steps_json")

DELETION_STATUSES = ("COMPLETE", "INCOMPLETE", "NOT_APPLICABLE")

STEP_PHASES = ("FINDING", "REMEDIATION", "VERIFICATION")

# Canonical operation order within each phase, used for deterministic step
# ordering once the model has produced findings/remediation/verification.
PHASE_OPERATION_ORDER: dict[str, tuple[str, ...]] = {
    "FINDING": (
        "REPORT_REMAINING_RECORD",
        "REPORT_EXPOSED_PII",
        "REPORT_ACTIVE_DATA_FLOW",
        "REPORT_RESTORE_RISK",
    ),
    "REMEDIATION": (
        "DISABLE_DATA_FLOW",
        "BLOCK_RESTORE",
        "DELETE_RECORD",
        "REDACT_RECORD",
        "ANONYMIZE_RECORD",
    ),
    "VERIFICATION": (
        "VERIFY_RECORD_ABSENT",
        "VERIFY_RECORD_PRESENT",
        "VERIFY_RECORD_REDACTED",
        "VERIFY_RECORD_ANONYMIZED",
        "VERIFY_DATA_FLOW_DISABLED",
        "VERIFY_RESTORE_BLOCKED",
    ),
}


@dataclass(frozen=True)
class ModelSettings:
    """Model/provider configuration, read from the environment.

    No secrets are hardcoded. Set these via a local, uncommitted `.env` file
    or real environment variables before running `main.py`.
    """

    provider: str = os.environ.get("ORCHESTRATOR_MODEL_PROVIDER", "openai")
    model_name: str = os.environ.get("ORCHESTRATOR_MODEL_NAME", "gpt-4.1")
    api_key: str | None = os.environ.get("ORCHESTRATOR_API_KEY")
    api_base: str | None = os.environ.get("ORCHESTRATOR_API_BASE")
    temperature: float = float(os.environ.get("ORCHESTRATOR_TEMPERATURE", "0"))
    # Higher default than the single-case architecture needed: one batch
    # response now encodes ~BATCH_SIZE cases worth of steps_json in one call.
    max_output_tokens: int = int(os.environ.get("ORCHESTRATOR_MAX_OUTPUT_TOKENS", "8192"))
    request_timeout_s: float = float(os.environ.get("ORCHESTRATOR_TIMEOUT_S", "60"))
    # Proactive client-side throttle (0 disables it) — set below a provider's
    # requests-per-minute quota (e.g. a free tier) to avoid burning retry
    # attempts on predictable 429s.
    max_requests_per_minute: int = int(os.environ.get("ORCHESTRATOR_MAX_REQUESTS_PER_MINUTE", "12"))


@dataclass(frozen=True)
class RetrySettings:
    """Exponential backoff settings for all model/API calls."""

    max_attempts: int = int(os.environ.get("ORCHESTRATOR_RETRY_MAX_ATTEMPTS", "5"))
    base_delay_s: float = float(os.environ.get("ORCHESTRATOR_RETRY_BASE_DELAY_S", "1.0"))
    max_delay_s: float = float(os.environ.get("ORCHESTRATOR_RETRY_MAX_DELAY_S", "30.0"))
    jitter_s: float = float(os.environ.get("ORCHESTRATOR_RETRY_JITTER_S", "0.5"))


@dataclass(frozen=True)
class PricingSettings:
    """Estimated $ price per 1M tokens, for the usage/cost report only.

    These are rough defaults for the gpt-4.1 family and may not match your
    actual billing rate. Override with real values via `code/.env` —
    `evaluation/usage_report.md` labels every cost figure as an estimate.
    """

    price_per_1m_input_tokens: float = float(os.environ.get("ORCHESTRATOR_PRICE_PER_1M_INPUT", "2.00"))
    price_per_1m_output_tokens: float = float(os.environ.get("ORCHESTRATOR_PRICE_PER_1M_OUTPUT", "8.00"))


@dataclass(frozen=True)
class ConcurrencySettings:
    """How many cases run concurrently (ThreadPoolExecutor workers).

    Concurrency cannot exceed what `ModelSettings.max_requests_per_minute`
    allows account-wide — it only closes the gap between that cap and
    actual achieved throughput (sequential overhead, per-call latency). If
    you upgrade to a paid tier with a much higher RPM quota, raise both
    settings together for a real speedup.
    """

    max_workers: int = int(os.environ.get("ORCHESTRATOR_MAX_CONCURRENT_WORKERS", "10"))


MODEL_SETTINGS = ModelSettings()
RETRY_SETTINGS = RetrySettings()
PRICING_SETTINGS = PricingSettings()
CONCURRENCY_SETTINGS = ConcurrencySettings()
