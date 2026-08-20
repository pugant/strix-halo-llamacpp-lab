# Upstream llama.cpp patch — ARCHIVED (never sent)

**Status: topic CLOSED 18/08/2026 by user decision.** The PR to ggml-org/llama.cpp
was never sent: the repo's AGENTS.md forbids the agent from writing commit
messages / PR descriptions / pushing / creating PRs (ban risk for the
contributor), and the user chose not to draft the required texts personally.

## What it contains

`0001-server-reasoning-budget-forced-newline.patch` — fix for the round-trip of
the forced sequence at the end of the reasoning budget on **upstream master
3dc7285** (18/08): forced sequence `'\n' + message + ('\n' if message is
non-empty) + end_tag`, tokenized as a single string, in the
`reasoning_budget_message` handler of `tools/server/server-schema.cpp` (+ helper
`first_reasoning_end_tag`). Port of our fork patches 0005+0009 to upstream.

## Verification performed (before archiving)

- Bug confirmed in the code: token concatenation without a newline, line ~421.
- e2e A/B (Qwen3.8-2B-Q4_K_M + 27B template via `--chat-template-file`):
  main = 116/170 tokens re-evaluated on resend (DIVERGED), patch = 17/170
  (EXACT-HIT), natural baseline 20/173; same with a wrap-up message (S2).
- Details: `docs/superpowers/plans/2026-08-18-upstream-roundtrip-verify.md`;
  PR factsheet: `docs/upstream-pr/2026-08-18-reasoning-budget-roundtrip-factsheet.md`;
  script: `scripts/upstream-roundtrip-test.py`. (All artifacts from the original
  workspace, not included in this repo.)

## If it is reopened

Re-clone (shallow), apply this patch, build, re-run the verification script;
then commit/push/PR **by the human contributor**, with AI disclosure in the PR
template. Note: upstream evolves — re-check that the handler is still in
`server-schema.cpp` and that the bug has not already been fixed.
