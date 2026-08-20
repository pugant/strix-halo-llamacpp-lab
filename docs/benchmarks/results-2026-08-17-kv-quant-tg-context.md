# Asymmetric q8_0 KV vs f16 — tg by context (NO-GO)

**Date:** 2026-08-17 16:45-16:51 · dedicated GPU · image `vulkan-fork-ckpt6` ·
Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN · MTP n6 · greedy · 128 tok/rep · warm-up discarded,
mean±sd over 2 reps (sd from the plan 3→2 for speed; direction confirmed by the individual reps).

| ctx | A: KV f16 | C: q8_0 + draft q4_0 | Δ C vs A |
|---|---|---|---|
| ~1k | 32.11 ± 0.12 | 32.08 ± 0.60 | on par |
| ~12k | **24.83 ± 0.08** | 20.93 ± 0.58 | **−15.7%** (≈7σ) |
| ~24k | **21.72 ± 0.22** | 21.03 ± 0.41 | **−3.2%** (≈3σ) |

Prompt: real prose (workspace docs) + final instruction; subsequent reps with the prompt
cache active (cached = 1023/12635/25241). Smoke output identical across arms.

## Verdict: NO-GO (plan criterion: needed C ≥ A at 1k ✓ BUT C > A at 24k ✗)

Quantized KV does NOT help on this stack (Vulkan fork backend, dense 27B): it worsens or
ties everywhere. The bottleneck at long ctx is NOT just KV read bandwidth — if it were,
q8_0 (half the bytes) would have gained. Law `tg=floor(ctx)×boost(acc)`: the −32% drop
from 1k to 24k must be sought elsewhere (hypotheses to verify in sub-thread 3.2:
speculative checkpoint/ring overhead, verify cost with large KV; also MTP acceptance not
measured in this run — the container logs were removed at the end of the run).

Note: the fork doc "optimal TurboQuant config for qwen35-a3b" concerned the qwen35 MoE
arch (Qwopus) — this result on the dense 3.8 does NOT contradict it.

**Consequences:** the "KV q8_0 on Vulkan experiment" thread CLOSED. Next active
sub-thread: **3.2 ring/checkpoint tuning** (flag sweep at ctx 24k, zero code).
Raw: /tmp/bench-kv-{A,C}.json (ephemeral), log /tmp/bench-kv-orchestrator.log.
