"""Deterministic final ordering for a case's step list.

The model proposes which findings/remediation/verification apply; this
module is the single source of truth for final `step_no` assignment and
ordering, so submission ordering never depends on model output order.
Kept separate from `pipeline.py` / `validation.py` to avoid a circular
import between them.
"""
from __future__ import annotations

from config import PHASE_OPERATION_ORDER
from schemas import Step


def order_steps(steps: list[Step]) -> list[Step]:
    """Re-number `steps` by phase, then table operation order, then ids."""

    def sort_key(step: Step) -> tuple[int, int, str, str]:
        phase_rank = ("FINDING", "REMEDIATION", "VERIFICATION").index(step.phase)
        op_order = PHASE_OPERATION_ORDER[step.phase]
        op_rank = op_order.index(step.operation) if step.operation in op_order else len(op_order)
        return (phase_rank, op_rank, step.system_id, step.resource_id)

    ordered = sorted(steps, key=sort_key)
    return [step.model_copy(update={"step_no": idx}) for idx, step in enumerate(ordered, start=1)]
