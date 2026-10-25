# Evaluation Report — Ghost Data Still Exists

## 1. Agentic architecture

> **Architecture note (final submission):** to meet the submission deadline under a
> hard client-side rate limit, the agent runs in **batched** mode: `BATCH_SIZE=10`
> cases are decided per LLM call instead of one case per call, cutting ~518 calls
> down to ~52-104. All per-case identity/evidence pre-fetching, the deterministic
> fail-closed validation contract (`validation.enforce_output_contract`), and the
> output schema are unchanged from the single-case design described below. The one
> structural difference is that one model call now returns one decision object per
> case in the batch (`submit_batch_decisions`), evaluated with `allow_retry=True`
> per case first; any case that violates the output contract is collected and
> resent, together with the other violators from that same batch and the exact
> violation feedback, in exactly **one bounded micro-retry call** per batch
> (`prompts.build_batch_retry_message`, `agent.run_agent_for_batch`) — never an
> open-ended loop. Whatever that second pass returns is final
> (`allow_retry=False`), so every batch resolves in at most 2 LLM calls
> regardless of model behavior. This replaces an earlier, stricter no-retry-at-all
> version of the batched design after a 32-case sample evaluation showed it caused
> real accuracy regressions (see §5).

For every `case_id` in `dataset/deletion_requests.csv`, `pipeline.run()` groups
case_ids into `BATCH_SIZE`-sized batches and calls
`agent.run_agent_for_batch(batch, ctx, llm_client)`, which is the single, uniform
entry point used identically for every batch — there is no per-case or per-dataset
branching in code. All deterministic data gathering (identity resolution, bounded
evidence retrieval) still happens once per case in Python
(`agent._prefetch_evidence`); the model's job is purely the compliance *decision*
for each case, reasoning over its own pre-loaded identity/evidence/rules context
without being able to invoke a tool mid-case.

```
main.py
  -> pipeline.build_context()   # load every dataset file once (strict schemas)
  -> pipeline.run(ctx, llm_client)
       -> agent.run_agent_for_batch(batch_of_10_case_ids, ...)
            -> agent._prefetch_evidence(case_id, ctx)         # per case: identity + bounded evidence
            -> llm_client.complete_with_tools(...)             # pass 1: ONE call decides the whole batch
            -> tools.resolve_batch_case(..., allow_retry=True)  # per case; collects violators, doesn't fall back yet
            -> [only if any case violated] llm_client.complete_with_tools(...)  # pass 2: ONE micro-retry call, violators only
            -> tools.resolve_batch_case(..., allow_retry=False) # final, guaranteed-terminating resolution
  -> output_writer.write_output(...)   # validated, sorted output.csv
  -> usage_report.write_usage_report(...)
```

### Tools available to the agent (`code/tools.py`)

| Tool | Purpose |
|---|---|
| `submit_batch_decisions(results)` | The only tool offered in the batched architecture actually used for the final run. `results` is a JSON array with one `{case_id, status, steps_json}` object per case in the batch — parsed by `tools.parse_batch_results` and validated per case by `tools.resolve_batch_case`. |
| `submit_case_decision(status, steps_json)` | The original single-case tool (kept, still used by `agent.run_agent_for_case`, not the entry point used for the final run) — allows up to a few rejected-submission retries per case. |

### Deterministic layers around the model

The model is trusted to *reason*, never to *self-certify compliance*. Every output
constraint is enforced in code, not by instruction alone:

- `schemas.OPERATION_PHASE` / `OPERATION_STATE`: closed maps from `operation` to its
  only valid `phase`/`state`. The model is only ever asked for `operation`
  (never `phase`/`state`/`step_no`), so a whole class of mismatches is structurally
  impossible.
- `ordering.order_steps`: final phase → operation-table order → `system_id` →
  `resource_id` sort, with `step_no` renumbered from 1 — independent of whatever
  order the model produced its steps in.
- `validation.enforce_output_contract` / `tools.resolve_batch_case`: the fail-closed
  interceptor described in §3, run per-case even inside a batched response, once per
  pass (with `allow_retry=True` on pass 1, `allow_retry=False` on the bounded
  micro-retry pass).

## 2. Bounded tool use

"Bounded" applies at these levels:

1. **Per-call result size.** Each case's pre-fetched evidence is capped at
   `MAX_EVIDENCE_PER_CALL = 60` items (`agent._prefetch_evidence`), with
   `truncated=true` reported alongside when a subject has more matching evidence
   than that — the model is instructed to treat the unseen remainder
   conservatively. This keeps the batched prompt's total size predictable even
   across 10 cases at once.
2. **At most one micro-retry per batch, scoped to only the cases that need it.**
   Pass 1 makes one LLM call for the whole batch and validates every case with
   `allow_retry=True` (catches the violation but doesn't resolve it yet). Any case
   that violates the output contract is collected; if there is at least one, pass 2
   makes exactly **one more** LLM call containing only those violating cases plus
   their specific violation feedback (`prompts.build_batch_retry_message`) — never
   a loop, never a per-case retry, and never a re-prompt for a case that already
   passed on pass 1.
3. **Guaranteed termination.** Pass 2 (or pass 1, if nothing violated) is always
   resolved with `enforce_output_contract(..., allow_retry=False)`, which never
   raises — it deterministically downgrades/corrects instead. Every case is
   therefore guaranteed to produce a schema-valid row in at most 2 LLM calls, even
   if the model omits a case from its `results` array entirely, returns malformed
   JSON, or an API call fails outright (that batch, or just the still-violating
   cases, falls back to `INCOMPLETE`/`[]`).

Every API/model call also goes through `retry.call_with_retry` (exponential backoff +
full jitter, configurable via `ORCHESTRATOR_RETRY_*` in `code/.env`), so transient
failures (rate limits, timeouts, 5xx) are retried before being treated as a hard failure
— this absorbed several token-per-minute 429s during the final full run without any
case being lost.

## 3. `submit_batch_decisions` — the deterministic safety boundary

`tools.tool_submit_batch_decisions` is the only way a batch's case decisions can be
accepted. It intercepts the raw `results` payload (tolerating a model that returns a
native JSON array instead of the string the schema requested) and, for every case,
fails closed on every violation via the same `validation.enforce_output_contract`
conversation:

1. **Allowed phases:** FINDING, REMEDIATION, VERIFICATION — enforced structurally
   (phase is derived from `operation`, never accepted as free text).
2. **Allowed FINDING operations/states:** `REPORT_REMAINING_RECORD`/PRESENT,
   `REPORT_EXPOSED_PII`/PRESENT, `REPORT_ACTIVE_DATA_FLOW`/ACTIVE,
   `REPORT_RESTORE_RISK`/ACTIVE.
3. **Allowed REMEDIATION operations/states:** `DISABLE_DATA_FLOW`/DISABLED,
   `BLOCK_RESTORE`/BLOCKED, `DELETE_RECORD`/ABSENT, `REDACT_RECORD`/REDACTED,
   `ANONYMIZE_RECORD`/ANONYMIZED.
4. **Allowed VERIFICATION operations/states:** `VERIFY_RECORD_ABSENT`/ABSENT,
   `VERIFY_RECORD_PRESENT`/PRESENT, `VERIFY_RECORD_REDACTED`/REDACTED,
   `VERIFY_RECORD_ANONYMIZED`/ANONYMIZED, `VERIFY_DATA_FLOW_DISABLED`/DISABLED,
   `VERIFY_RESTORE_BLOCKED`/BLOCKED.
5. **Mandatory sorting:** phase order → operation order → `system_id` ascending →
   `resource_id` ascending → `step_no` renumbered sequentially from 1
   (`ordering.order_steps`).
6. **Evidence rules:**
   - `NOT_APPLICABLE` → `steps_json` forced to `[]` regardless of what was submitted.
   - `COMPLETE` → every step must be VERIFICATION with a non-empty evidence_id that
     is one of the IDs *actually shown to the model this session* (not just any real
     ID in the dataset) and timestamped at or before `audit_at`. Any violation
     downgrades the whole case to `INCOMPLETE` — never the reverse.
   - `INCOMPLETE` → FINDING steps must cite valid evidence (otherwise dropped);
     REMEDIATION and (now-future) VERIFICATION steps always get `evidence_id=""`,
     regardless of what the model supplied.

## 4. Edge-case fallbacks

| Scenario | Behavior |
|---|---|
| Historical logs conflict, or a record's removal looks pending/unconfirmed | Agent policy ("Contradictory/Pending Items Policy" in `prompts.AGENT_SYSTEM_PROMPT`) instructs treating it as still PRESENT/ACTIVE/exposed; never conclude a clean state without a positive, verified absence record. |
| Identity resolves to 0 or >1 `user_id` (`identity.resolve_subject` → `UNRESOLVED`/`AMBIGUOUS`) | Agent policy ("Ambiguous Identifiers Policy") forbids guessing; the model is told not to claim COMPLETE off an assumed match. |
| Any uncertainty about an active data flow / restorable backup / unredacted record | Agent policy ("Unsafe States Policy") forces a FINDING, never a COMPLETE verification. |
| Model never calls any tool for `MAX_TURNS` turns | Loop nudges it each turn ("Use the available tools..."); if it still never submits, falls back to `INCOMPLETE` with `steps=[]`. |
| Model repeatedly submits an invalid/contradictory decision | Each rejection returns the exact violation string as a tool result so the model can self-correct; after `MAX_SUBMIT_ATTEMPTS` (or the final turn), the last attempt is force-resolved deterministically (`allow_retry=False`) instead of looping forever. |
| Model cites an evidence_id it was never actually shown, or one timestamped after `audit_at` | Dropped/cleared at the schema layer (`tool_submit_case_decision`); for COMPLETE claims this forces a downgrade to INCOMPLETE. |
| A batched case violates the output contract on its first attempt | Collected (not immediately resolved) and re-sent, with the exact violation text, in one bounded micro-retry call alongside any other violating cases from the same batch (`agent.run_agent_for_batch`); whatever that second pass returns is final. |
| Transient API failure (timeout, rate limit, 5xx) | Retried with exponential backoff (`retry.py`) before being surfaced; if retries are exhausted mid-case, that case falls back to `INCOMPLETE`/`[]` rather than crashing the whole 518-case run. |
| Images (`.png` evidence) | Not content-matched (no vision model wired in yet) — a known, documented limitation, not silently ignored. |

## 5. Quantitative evaluation (real run, against `dataset/sample_output.csv`)

Produced by `code/evaluation/main.py` (`python3 code/evaluation/main.py`), which runs
the actual batched pipeline (`agent.run_agent_for_batch`, same code path as the real
518-case submission run) over all 32 cases in `dataset/sample_deletion_requests.csv`
and scores the result against the known-correct `dataset/sample_output.csv`. Raw
per-case output is in `code/evaluation/eval_results.json`. Two runs are reported:
the original no-retry batch design, and the current design after adding the bounded
micro-retry pass and the strengthened NOT_APPLICABLE prompt warning (§1-§2).

| Metric | No-retry batch (previous) | Micro-retry batch (current) |
|---|---:|---:|
| Case-level status accuracy | 78.1% (25/32) | **90.6% (29/32)** |
| Sequence accuracy (exact operation-order match) | 6.25% (2/32) | 0.0% (0/32) |
| Step/field accuracy (aligned steps, `system_id`+`resource_id`+`operation` match) | 38.4% (48/125) | **40.0% (54/135)** |

**Case-level accuracy improved substantially (+12.5 points)** and, more importantly,
**both of the two serious failure classes found in the previous run are fully
resolved**: zero false `NOT_APPLICABLE` results and zero false `COMPLETE` results in
this run (see "Remaining failures" below for exactly what's left).

**Sequence accuracy got slightly worse (2/32 → 0/32) and needs an honest caveat:**
this metric requires the predicted `steps_json` array to have the *exact same length
and operation order* as the hand-authored golden answer — it is extremely strict and,
on inspection, the two cases that happened to match before (by coincidence, both were
1-step cases) no longer do simply because the micro-retry pass changed their step
count by one. The agent still consistently produces a shorter, more consolidated step
list than the golden answer across nearly every case (e.g. golden 5/7/8/14 vs.
predicted 3/3/6/12 for the same cases) — this is a real, unresolved gap (it reaches
the same high-level conclusion but skips some finer-grained per-resource steps the
reference answer spells out), not something this turn's fixes targeted. Step/field
accuracy (40.0%, up slightly from 38.4%) is the more informative metric for this gap
since it doesn't require exact length/order alignment.

### 3 participant-authored custom test cases (`code/evaluation/main.py::build_custom_cases`)

| Case | Scenario | Expected | Predicted | Result |
|---|---|---|---|---|
| `custom_01` | Marketing-opt-out request using deletion-adjacent language ("unsubscribe... but I still want to log in") | `NOT_APPLICABLE` | `NOT_APPLICABLE` | ✅ pass |
| `custom_02` | Deletion request for an email with **zero** matching identity (`totally.unknown.person@nowhere.example`) | Must not be a confident `COMPLETE` (Ambiguous Identifiers Policy) | `INCOMPLETE` | ✅ pass |
| `custom_03` | Real sample case_01 (`amina.elmasri@gmail.com`), but `audit_at` moved to `2026-09-01T00:00:00Z` — **before** `evidence_54` (captured `2026-09-02T23:00:00Z`), the only evidence that proves deletion in the real case | Must not be `COMPLETE` via evidence that didn't exist yet at this `audit_at` | `INCOMPLETE` | ✅ pass |

All three custom cases passed both before and after this turn's fixes, including the
temporal-cutoff test (`custom_03`), which specifically confirms the agent does not
"see into the future" relative to `audit_at`.

### Remaining failures (3/32, all in the safe direction)

All 3 remaining mismatches (`case_01`, `case_02`, `case_04`) are golden `COMPLETE`,
predicted `INCOMPLETE` — the agent judged the available evidence insufficient to
confidently prove full deletion and reported it conservatively instead, per the Unsafe
States Policy ("any uncertainty... must default to a FINDING state, never a COMPLETE
status"). This costs accuracy points but never produces an unsafe false-positive
`COMPLETE`, which is the intentional fail-closed design tradeoff throughout this
project.

### Fixes applied this turn and why

1. **Step ordering was investigated and found to already be correct — no change
   made.** The hypothesis was that `steps_json` ordering mismatches were causing
   guardrail rejections. Verified against `dataset/sample_output.csv` directly
   (e.g. `case_11`'s golden FINDING order is `REPORT_REMAINING_RECORD →
   REPORT_EXPOSED_PII → REPORT_EXPOSED_PII → REPORT_ACTIVE_DATA_FLOW`, which matches
   `config.PHASE_OPERATION_ORDER`'s table order exactly, not alphabetical order).
   `ordering.order_steps` already re-sorts and renumbers every case's steps
   unconditionally (pass or fail, every case) using that exact table order before
   it's ever written to `output.csv` — switching to alphabetical order as initially
   proposed would have actively *reduced* accuracy against the real golden
   convention, so the existing table-order sort was kept as-is.
2. **`prompts.BATCH_AGENT_SYSTEM_PROMPT` strengthened** with an explicit warning:
   genuine deletion requests must never be returned as `NOT_APPLICABLE` even when
   evidence is messy — default to `INCOMPLETE` instead. `NOT_APPLICABLE` is reserved
   only for requests that aren't about deletion at all.
3. **Bounded micro-retry added** (`tools.parse_batch_results` /
   `tools.resolve_batch_case` / `agent.run_agent_for_batch` /
   `prompts.build_batch_retry_message`): pass 1 validates every case with
   `allow_retry=True` without immediately falling back; any case that violates the
   contract is resent, together with the specific violation text, in exactly one
   more LLM call scoped to just those cases. This gives genuinely strong cases a
   real second chance to self-correct (e.g. cite a valid `evidence_id` instead of an
   empty one) instead of an instant, uninformed fallback — while still guaranteeing
   termination in at most 2 LLM calls per batch.

### Failures found and honest assessment

1. **3/7 status mismatches were over-conservative** (`case_01`, `case_02`, `case_04`:
   golden `COMPLETE`, predicted `INCOMPLETE`). This is the *safe* direction per the
   Unsafe States Policy (never claim COMPLETE without full confidence) — it costs
   accuracy points but never produces an unsafe false-COMPLETE.
2. **2/7 status mismatches were false `NOT_APPLICABLE`** (`case_07`, `case_22`: golden
   `INCOMPLETE`, i.e. genuine deletion requests, predicted `NOT_APPLICABLE`). This is
   the same failure mode documented earlier in this project (the model "gives up" on a
   hard case and reclassifies it as not-applicable rather than committing to a
   difficult `INCOMPLETE` finding). **Root cause specific to the final batch
   architecture:** the single-case design had a deterministic code guardrail
   (`first_attempted_status` in `agent.py`) that rejected exactly this flip-flop and
   forced a retry — batch mode has **no retry turn at all** (by explicit design, to
   hit the submission deadline under the rate limit), so that guardrail has nothing to
   attach to and this failure mode is currently unguarded in the batched path. This is
   a known, real accuracy cost of the speed-for-correctness-retries tradeoff made
   under time pressure, not a silent bug — flagging it explicitly rather than hiding
   it.
3. **2/7 status mismatches were false `COMPLETE`** (`case_15`, `case_17`: golden
   `INCOMPLETE`, predicted `COMPLETE`). These are more concerning: the predicted
   `VERIFICATION` step was structurally valid (a real, correctly-timestamped
   `evidence_id`), so `validation.enforce_output_contract`'s structural fail-closed
   check could not catch it — the error is a *substantive* misreading of the evidence
   content, not a schema violation, which is outside what code-level validation alone
   can guard against.
4. **Step granularity gap:** independent of status correctness, the agent tends to
   under-report the full remediation/verification step list the golden answers
   include (see sequence accuracy above).

**Improvements that would address these, if there were time for another full run:**
restore a single bounded retry specifically gated on `NOT_APPLICABLE` and `COMPLETE`
claims (the two highest-risk categories) even in batch mode, instead of removing
retries uniformly; and strengthen the prompt's instruction to enumerate the full
remediation/verification chain per flagged resource instead of a minimal passing set.
Given the submission deadline, these are documented here as findings rather than
implemented.

