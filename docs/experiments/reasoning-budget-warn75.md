# A mid-budget warning window for thinking-budget truncation

*2026-08-30 - qwen4exp runtime fork, patches 0016-0020 (series `patches/qwen4exp-mtp/`)*

## TL;DR

The fork's reasoning ("thinking") budget forced the end of the reasoning phase
only at exhaustion, which meant the "wrap up" nudge arrived with roughly zero
tokens left to act on it. We added a warning window at 75% of the budget: a
mid-conversation message is injected when the model crosses the threshold, the
countdown pauses while the model acknowledges it, and a residual convergence
window remains before the forced end. In a one-hour real agent session
(48 requests, budget 17408): 48/48 natural ends, one warning injected at
exactly 75% (13056/17408) with convergence 1.3 s later, zero exhausted
budgets. The empirical expectation that "the first heavy task always burns
the whole budget, even at 16384" did not survive contact with the warning.

## The problem

Server-side budget truncation is cache-friendly (the cap lives in a request
field, not in the system prompt), and at exhaustion the fork already injected
a final message asking the model to wrap up. But an end-of-budget message is
structurally too late: the model has nothing left to spend on closing its
reasoning cleanly. Literature pointers that shaped the design: TALE (ACL25)
and s1 (arXiv:2501.19393) - mid-generation intervention steers token budgets
effectively, while a priori budgets do not.

## Design

- `warn_at = budget x (1 - ratio)`, default ratio 0.75, CLI
  `--reasoning-budget-warn-ratio` / env `LLAMA_ARG_THINK_BUDGET_WARN_RATIO`.
- On crossing `warn_at`, the sampler injects a user-role message (customizable
  via `reasoning_budget_message`) and enters a `WARN_FORCING` state: the token
  countdown is suspended for the length of the message, then resumed
  (`warn complete, resuming countdown`). The forced-end sequence remains as
  the safety net at true exhaustion.
- The warn state survives context clones (speculative-decoding parallelism)
  and re-arms per request.
- Guard `warn_at >= 2`: for tiny budgets the message is always deliverable
  (the never-fire band is explicit and tested - requesting a warn window
  smaller than the message delivery cannot silently no-op).
- Public surface is backward compatible: new parameters appended with
  defaults; the request field remains `thinking_token_budget`.

## Validation

- Unit suite `tests/test-reasoning-budget.cpp`: 11/11 (6 base, 4 warn-window
  scenarios including exhausted-after-warn and clone-carries-state, 1 UTF-8
  boundary). Re-run as the merge gate on main.
- Independent pre-deploy review fixed 3 MAJORs (positional-initializer struct
  breakage, never-fire band, CLI flag).
- Real-world session (agent client, one hour, coding task): 48 requests at
  budget 17408 -> 48 natural ends, 1 warning at exactly 75% with convergence
  in 1.3 s, 0 exhausted, prompt-cache similarity 0.95-0.999 throughout, no
  unexpected cold prefills.

## Open question

The resend seam at the injection point: the forced message ends with a
newline, and if the model samples a bare "Ċ" the re-tokenization can fuse the
seams (a potential LCP cut at the junction). No symptom in the measured
session (similarity stayed 0.99+ after the warned request), but a dedicated
measurement is still open.

## Files

`common/reasoning-budget.{h,cpp}`, `common/sampling.cpp`,
`common/speculative.{h,cpp}`, `common/{arg.cpp,common.h}`,
`tools/server/{server-common,server-context,server-task}.{h,cpp}` - patches
0016-0020 of the qwen4exp series; also on branch `qwen4exp-conv-ring-slots`
(merged to the fork's main, commit `62416acd3`).
