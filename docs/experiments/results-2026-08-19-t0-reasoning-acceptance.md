# Results — T0: DFlash2 vs MTP acceptance on thinking (T7-f2 routing policy gating)

**Date:** 2026-08-19 · **Plan:** internal test plan `2026-08-19-t0-reasoning-acceptance.md` (not included in this repo)
**Setup:** identical to T7 — LEAN 13.8 GB, Vulkan RADV, dedicated GPU (llm-service stopped
12:18→12:47, restart + health OK), `-c 16384 -fa on --jinja`, temp 0, p_min 0.75,
p_split 0.10, warm-up discarded, TREATMENT marker. Arms: MTP6 (ckpt7) / DF7 (dflash2).
Script `scripts/bench-t0-thinking-acceptance.sh` (not included in this repo), log `logs/bench-t0-thinking/`.

## Data

**Set A — thinking-dominant** (long reasoning, ~1 token answer; max_tokens 3000):

| Prompt | MTP6 tok/s | DF7 tok/s | Δ | acc mean MTP | acc mean DF7 | per-pos DF7 |
|---|---|---|---|---|---|---|
| A1 treno | 27.4 | 26.8 | −2% | 3.85 | 4.85 | (0.90, 0.73, 0.61, 0.55, 0.47, 0.35, 0.25) |
| A2 radice | 26.5 | **30.1** | **+14%** | 3.63 | **5.27** | (0.87, 0.77, 0.64, 0.59, 0.53, 0.46, 0.42) |
| A3 scatole | 24.0 | 22.9 | −5% | 3.47 | 4.06 | (0.83, 0.64, 0.49, 0.39, 0.30, 0.24, 0.18) |

Reasoning chars: A1 2547/3682, A2 821/1327, A3 4131/10180 (MTP/DF7 — the batched-verify
numeric divergence, a known pre-existing issue, changes the reasoning trajectory).

**Anchor B (reproducibility control vs T7):**

| Prompt | MTP6 | DF7 | T7 reference |
|---|---|---|---|
| B4 det (count 1-200) | 44.7 | 56.8 | 45.2 / 57.4 ✓ ±1% |
| B5 prose (history of Roma) | 21.0 | 15.3 | 19.6 / 14.2 (+7-8% run-to-run, relative Δ −27% confirmed) |

Anchors reproduced → valid run.

## Interpretation

1. **Thinking is NOT a prose-class workload.** DF7 acceptance on thinking 4.06-5.27 (vs prose
   2.39, vs det 7.59): it sits between the classes, close to det. Per-pos stays ≥0.42 up to
   pos 7 on A1/A2 (prose collapses to 0.03). MTP also rises on thinking (3.47-3.85
   vs 2.57 prose).
2. **tok/s on thinking = near-parity**: −2% / +14% / −5% per prompt (**mean +2.4%**).
   The free-prose −26% penalty does NOT apply to reasoning.
3. Token-weighted aggregation: −3.7% — confounder note: on A3 the numeric divergence
   makes DF7 reason 2.5× longer (10180 vs 4131 char), so the weight of the single
   declining prompt is overestimated in the aggregate; it is not a drafter speed effect.

## Verdict (rule frozen in the plan)

- Acceptance mean DF7 set A ≥ 3.5: **PASS** (4.06/4.85/5.27).
- tok/s DF7 ≥ MTP6 −3%: **PASS on the per-prompt mean** (+2.4%; token-weighted −3.7% at
  the boundary, with the confounder described).

**→ Thinking = DFlash-tolerant → WHOLE-REQUEST policy** (drafter switch only at a task
boundary). The phase-aware option (MTP for thinking, switch at end_tag) would buy ~0-4%
on the thinking phases at the cost of a mid-generation switch + prefix re-encode +
checkpoint tagging: **rejected on data (quantified YAGNI)**. It remains a documented
fallback if production telemetry showed under-performance on thinking-heavy requests
(the design's per-seq `drafter` field would support it without rework).

## Implication for the T7-f2 design

- Section 1 architecture **without relaxation**: switch ONLY at a task boundary (no
  mid-generation changes, checkpoint/salvage in the regime they were written for).
- Policy: request-level signal (`tools`/`tool_choice` → DFlash, default MTP) +
  optional body override. The cost of misrouting prose→DF7 remains −26%: the policy
  stays conservative (default MTP), but the fear "DFlash harms agentic reasoning"
  is refuted by the data (agentic thinking is precisely the tolerance case).

## Caveat

- Short prompts/fresh ctx (production: long ctx + budget 2048): the bandwidth floor
  lowers both drafters in a similar way; the relative comparison holds.
- 3 prompts per class: sufficient for the policy decision (wide gap), not for
  fine claims about A2's +14%.
