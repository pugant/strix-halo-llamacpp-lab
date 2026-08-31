# Reasoning Pressure (soft steering of the reasoning budget) — Design Spec

**Date:** 2026-08-17 · **Thread:** ThinkingCap at the root (user redirect: anti-overthinking, NOT a budget floor)
**Production context:** ROCmFPX fork ckpt6 (patches 0001-0009), Qwen3.8-27B STRIX_LEAN, MTP n6, llm-service
**Reference literature report:** `docs/research/2026-08-17-thinkingcap-root-literature.md` (not included in this repo)

## 1. Problem and goal

Qwen3.8-27B has no strong stop signal in its weights: on dense tasks the reasoning
explodes and ends up TRUNCATED by the hard cap (A/B/B' data: 70-90% exhausted at level low).
The declarative line (L0) is NO-GO (anchors to full usage: 70→80→90%).

**Goal:** make the model close the reasoning *on its own and well* within the
budget, via soft inference-time steering in the sampler — the L3 family (guided
decoding) simplified and training-free, without a predictor.

**Success metric:** % exhausted (cuts) from 70-90% → **<30-40%** at parity of
perceived quality (A/B gate). The per-level budget (1024/2048/4096/8192 from pi) stays
the reference.

**Explicit NON-goals:** budget floor / force-continue "Wait" (the model does not
under-think — user decision 17/08); interventions on the weights (L4); hidden-states
predictor (Phase C paper-grade); reducing tokens on short tasks (which today close well).

## 2. Chosen design — Approach 1: extension of the `reasoning-budget` sampler

All the logic lives in the existing sampler (`common/reasoning-budget.cpp`, patch 0010
on branch `spec-cache-soft-wrap` = ckpt6 code). No engine work: the sampler is already
in the target's chain → MTP/cache interaction inherited from the forced-end in production.

### 2.1 Extended state machine

```
IDLE → (start_tag) → COUNTING → (natural end_tag) → DONE   [unchanged]
                         │ used ≥ pressure_start×budget (and UTF-8 complete):
                         │   FORCING_NOTA (deterministic token-per-token notice,
                         │                same accept/apply logic as FORCING,
                         │                at end of sequence ↩ COUNTING, not DONE)
                         │ COUNTING + used ≥ squeeze_from (DERIVED condition,
                         │ NOT a state: see §2.2)
                         ▼
                      [squeeze active in apply] → natural close or argmax → DONE
                         │ budget=0 (final net UNCHANGED):
                         ▼
                      FORCING (today's forced-end 0005 + soft-wrap 0008/0009)
```

- **SQUEEZE is not a state**: it is a derived condition evaluated in `apply()` when
  `state == COUNTING` and `used ≥ squeeze_from`. All the COUNTING transitions
  (end_matcher → DONE, remaining--, WAITING_UTF8/FORCING at 0, re-arm) stay the
  same — no duplication.
- `FORCING_NOTA` is a new state that reuses FORCING's accept/apply logic
  (sequence = note_tokens, -inf masking) but returns to COUNTING at the end of the
  sequence.
- The note is injected ONCE per think block: a `note_injected` flag cleared by the
  re-arm on a new start_tag (together with recomputing the thresholds).
- **UTF-8 guard on the note trigger**: the existing WAITING_UTF8 covers ONLY
  exhaustion; the note trigger (75%) has the same need → if the last accepted
  token is an incomplete UTF-8 piece, the injection is deferred to the first complete
  token (flag `note_pending`, same pattern as the existing WAITING_UTF8).
- **Token accounting for the note**: the forced tokens of the note do NOT decrement
  `remaining` (like the forced-end tokens today) → `used = budget - remaining` measures
  only the tokens thought by the model.

### 2.2 Squeeze formula (condition in `apply`, COUNTING state)

```cpp
// thresholds in double, computed at init (and at re-arm):
squeeze_from = min(pressure_start * budget + grace, budget - 1);
// in apply, if state==COUNTING && used >= squeeze_from:
double x = (double(used) - squeeze_from) / (budget - squeeze_from);
x = std::min(std::max(x, 0.0), 1.0);        // clamp [0,1] (degenerate windows)
const double boost = x * x * max_boost;     // quadratic ramp
// boost applied to the end token EXPECTED BY THE MATCHER:
const llama_token tok = end_tokens[end_matcher.pos];  // usually == end_tokens[0]
// scan cur_p for the id `tok` (as the masking loop does) and:
cur_p[i].logit += boost;                    // ONLY that token
```

- `end_tokens` is the tokenized SEQUENCE of the end tag (the matcher does not assume a
  single token): the boost goes to the next expected one
  (`end_tokens[end_matcher.pos]`);
  for Qwen3.8 `</think>` is typically 1 special token, but the code does not assume that.
- Double clamp (`squeeze_from` and `x`) covers small budgets: with budget=64 → note at
  48, squeeze from 63 (1-token window: effectively note → cap; degraded behavior
  accepted and documented).
- Quadratic: at half the window only +Δ/4; with `max_boost` ~9 the end becomes argmax
  in the last ~5-8% of the budget. If it does not close → FORCING at budget 0 (final
  net).
- **Activation only with an explicit budget**: `reasoning_budget_tokens >= 0` BEFORE
  the `INT_MAX` filler (sampling.cpp:305): with unlimited budget + grammar_lazy the
  steering is inert (no threshold at 1.6e9 nor int32 overflow — thresholds in double).

### 2.3 Contextual note

Forced sequence, tokenized ONCE at init, deterministic:

```
\n\n[Budget notice: wrap up the reasoning and give the final answer]\n\n
```

- Visible in the extracted `reasoning_content` (transparency; short, English,
  square brackets, system style). NOT filtered: the client resend must recompose it
  identically for the exact cache match (filtering would break the round-trip).
- Customizable per-request (`reasoning_pressure_notice`).

## 3. Parameters and wiring

Per-request (0006/0008 pattern), falling back to server defaults:

| OAI body key | Default | Meaning |
|---|---|---|
| `reasoning_pressure_start` | `0.75` | fraction of the budget at which the note fires |
| `reasoning_pressure_grace` | `200` | tokens of grace post-note before the squeeze |
| `reasoning_pressure_boost` | `9.0` | max_boost of the quadratic ramp |
| `reasoning_pressure_notice` | text §2.3 | injected note |

- Activation: `reasoning_pressure_start > 0` AND an EXPLICIT budget
  (`reasoning_budget_tokens >= 0` before the INT_MAX filler of sampling.cpp:305 — the
  check lives where the rbudget sampler is built, so that INT_MAX never activates
  thresholds). With `start=0` → bit-for-bit today's behavior.
- **Explicit rollout**: with the 0.75 default, steering is ACTIVE for every request
  with an explicit budget once 0010 is applied. For the A/B it is the ON arm; the
  production default is decided at GO (image switch).
- Server CLI/env defaults: `--reasoning-pressure-start/-grace/-boost/-notice` +
  `LLM_REASONING_PRESSURE_START/GRACE/BOOST/NOTICE` (`LLM_REASONING_BUDGET` pattern).
- Touch points: `common/reasoning-budget.{cpp,h}` (state machine, extended init with
  the new parameters, complete clone/reset), `common/common.h` (sampling params
  fields), `tools/server/server-common.cpp` (per-request OAI body, lines ~1178-1195),
  `tools/server/server-task.cpp` (tokenize note + pass-through, zone ~496-525),
  `tools/server/server-context.cpp` (chat_params if needed for the slot path),
  `common/common.cpp`/`common/arg.cpp` (CLI/env defaults).
- Durable patch: `patches/reasoning-pressure/0010-reasoning-pressure.patch` —
  **0010 CONTINUES the numbering of the `spec-cache-soft-wrap` branch series**
  (0001-0009 in `patches/spec-cache-trailing-rollback/`), separate dir for the
  one-patch-per-feature rule; rebuilding ckpt6+0010 = series 0001-0010.
- Patch 0010 does NOT modify the behavior of 0005-0009 (forced-end, soft-wrap,
  alias): with steering disabled the output is identical (T3 demonstrates this).

## 4. Interactions to preserve (and verify in tests)

- **MTP + clone/rollback (critical point)**: note and boost act in `apply`
  downstream of the exact verify; the MTP path clones the sampler and RESTORES it on
  partial reject (server-context.cpp:3570 save / :3631 restore). The existing
  `clone()` does not even copy `force_pos` (latent gap, harmless with the forced-end at
  budget 0 where rollbacks are rare, but EXPOSED by the note that lives at 75% of the
  budget): 0010 MUST extend `clone()` to ALL state fields (existing force_pos +
  note_injected/note_pending/
  note_pos + thresholds + `remaining` (the base of `used`, today reset to `budget` by
  clone) + `start/end_matcher.pos` (the second is the TARGET of the boost; today both
  zeroed by clone)) and `reset()` to clear them. The enumeration is normative, not
  exhaustive: "all fields" applies to future fields too. Without this, a mid-note
  rollback re-injects/corrupts the sequence and breaks the round-trip.
- **Chain order**: the rbudget apply runs BEFORE the grammar and sampler chain
  (sampling.cpp:631): masking and boost see the complete candidate set — this is what
  makes the mechanism work; to be asserted with an inline code comment.
- **Cache round-trip**: the note is a deterministic sequence in the reasoning → the
  client resend contains it → exact match. T1 verification mandatory (t1 pattern of
  the 2026-08-15 plan, `scripts/diff-prompt-cache.py` — script not included in this
  repo).
- **Soft-wrap**: if the model closes after note/squeeze → natural DONE, zero
  exhausted; the 0008 wrap message remains only for the forced-end case.
- **Multi-block re-arm**: a new `<think>` clears note/squeeze for that block.
- **UTF-8**: dedicated guard on the note trigger (§2.1, `note_pending`) — NOT
  inherited from WAITING_UTF8 which covers only exhaustion.

## 5. Experiment and GO/NO-GO gate

### 5.1 Technical sanity (test container :1235, dedicated GPU pre-authorized)
- **T1 round-trip**: dense task with a budget that triggers note+close → resend
  content+reasoning verbatim → NO cold fallback (exact hit). Includes the
  mid-note MTP-rollback case (log -lv 5: look for restore/checkpoint around the
  injection). Tool: existing t1/t3 patterns.
- **T2 local effectiveness**: 3-4 dense tasks with budget 2048 AND with budget 1024
  (the narrowest production window: 56 squeeze tokens), -lv 5 → verify in the
  logs: note emitted at 75%, natural close within grace or in squeeze, zero
  chopped sentences, short MTP rounds confined to the squeeze window.
- **T3 non-regression**: `reasoning_pressure_start=0` → output bit-identical to
  ckpt6 (same seed/prompt, content comparison).

### 5.2 A/B on real tasks (quality gate)
- Arms: OFF (ckpt6) vs ON (ckpt6+0010), same tasks/levels as the A/B/B' manifest
  (27 reusable tasks), extraction with `scripts/ab-bprime-extract.py` adapted
  (script not included in this repo).
- Primary metrics: % exhausted per level, mean/sd reasoning tokens, wall time per
  turn.
- Quality gate: blind comparison on a sample of 10-15 tasks at level low (maximum
  pressure on the cuts): ON ≥ OFF; zero closing artifacts (chopped sentences, known
  loops).
- **Production GO**: exhausted <30-40% + quality gate + T1-T3 green → llm-service
  image switch with user consensus. **NO-GO**: document (curve too weak or degraded
  quality) — the current hard cap + soft-wrap stay.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The model ignores the note (no early closing) | T2 measures effectiveness before the A/B; parameterized boost/slope |
| Degraded quality from forced early closing | blind A/B gate; late quadratic ramp |
| Cache round-trip broken by the note | T1 mandatory pre-A/B; deterministic note |
| Unexpected MTP interaction in the squeeze window | verify sanity in T2 (short rounds expected only there); forced-end inheritance |
| "Too many Wait" style plateau/degradation (ICLR26) | ONE single note per block, never repeated |
| Anchoring effect (like L0) | no number declared to the model: the note mentions the behavior, not the remaining budget |

## 7. Out of scope

Floor/force-continue Wait; hidden-states predictor (Phase C); weight interventions;
steering on short tasks; modification of the existing forced-end/soft-wrap.
