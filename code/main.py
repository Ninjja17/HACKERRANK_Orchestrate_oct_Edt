"""Entry point: `python3 code/main.py`.

Loads and strictly validates the dataset, runs the model-driven audit
agent (a bounded, concurrent tool-calling ReAct loop, see `agent.py` and
`pipeline.run`) over every case in `dataset/deletion_requests.csv`, writes
the final `output.csv` at the repository root, and writes
`code/evaluation/usage_report.md` summarizing token usage and estimated
cost for the run (written even on a partial/failed run, from whatever
usage was recorded before the failure).

Console output is kept to a progress bar + warnings/errors; the full
per-turn, per-tool-call reasoning trace for every case is always written
in full to `run_main.log` at the repository root.
"""
from __future__ import annotations

import logging
import sys

from config import EVALUATION_DIR, OUTPUT_CSV, REPO_ROOT
from data_loader import DatasetLoadError
from llm_client import LLMClient
from output_writer import write_output
from pipeline import build_context, run
from usage_report import write_usage_report

_file_handler = logging.FileHandler(REPO_ROOT / "run_main.log", mode="w", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)  # keep the console clear for tqdm's progress bar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[_file_handler, _console_handler],
)
logger = logging.getLogger(__name__)

USAGE_REPORT_PATH = EVALUATION_DIR / "usage_report.md"


def main() -> int:
    try:
        ctx = build_context()
    except DatasetLoadError:
        logger.exception("Dataset failed strict validation; fix the data or the schema and re-run.")
        return 1

    try:
        llm_client = LLMClient()
    except RuntimeError:
        logger.exception("Could not start the model client (check code/.env).")
        return 3

    output_rows: list = []
    try:
        output_rows = run(ctx, llm_client)
    finally:
        write_usage_report(llm_client.usage_log, max(len(output_rows), 1), USAGE_REPORT_PATH)
        logger.info("Wrote usage report to %s", USAGE_REPORT_PATH)

    write_output(output_rows, OUTPUT_CSV)
    logger.info("Wrote %s output rows to %s", len(output_rows), OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
