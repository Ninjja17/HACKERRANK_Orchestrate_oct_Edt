"""Validated CSV writer for the final `output.csv` submission file."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from config import EXPECTED_CASE_COUNT, OUTPUT_COLUMNS
from schemas import OutputRow


class OutputValidationError(RuntimeError):
    """Raised when the collected output rows violate the submission contract."""


def validate_output_rows(rows: list[OutputRow]) -> None:
    if len(rows) != EXPECTED_CASE_COUNT:
        raise OutputValidationError(
            f"Expected {EXPECTED_CASE_COUNT} output rows, got {len(rows)}"
        )
    case_ids = [row.case_id for row in rows]
    if case_ids != sorted(case_ids):
        raise OutputValidationError("Output rows must be sorted by case_id")
    if len(set(case_ids)) != len(case_ids):
        raise OutputValidationError("Duplicate case_id values found in output rows")


def write_output(rows: list[OutputRow], destination: Path) -> None:
    """Validate and write `rows` to `destination` as the final `output.csv`."""
    validate_output_rows(rows)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            steps_json = json.dumps([step.model_dump(mode="json") for step in row.steps])
            writer.writerow([row.case_id, row.deletion_status, steps_json])
