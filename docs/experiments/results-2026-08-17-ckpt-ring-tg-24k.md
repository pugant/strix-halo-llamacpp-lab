# Checkpoint/ring cost and MTP boost vs context (3.2)

**Date:** 2026-08-17 16:55-17:05 · dedicated GPU · same image/model/greedy as 3.1 ·
same prompt for all arms (fixed seed; NOT 1:1 comparable with 3.1, corpus
changed). Warm-up discarded, mean±sd over 2 reps.

| Arm | 1k | 24k | Interpretation |
|---|---|---|---|
| T1 base (MTP n6, default) | 28.89 ± 0.01 | **23.47 ± 0.03** | baseline |
| T2 `-cpent -1` (no ckpt prefill) | — | 24.34 ± 0.27 | **+3.7%** (~3σ, small) |
| T3 `-ctxcp 4` (ring 4) | — | 22.83 ± 1.0 | −2.7% NOT significant (sd ±1.0) |
| T4 without MTP | 9.33 ± 0.3 | 7.66 ± 0.06 | pure floor |

## Conclusions

1. **Checkpoint/ring: dead lever.** Creating checkpoints in prefill costs ≤4%; ring 32 vs
   4 is indistinguishable. The fork's defaults are already optimal on the tg side. (And
   `-cpent -1` in production would break the 0007 checkpoint-salvage for a ~null gain.)
   Sub-thread 3.2 CLOSED.
2. **MTP boost CONSTANT with context**: boost(1k) = 28.89/9.33 = **3.09×**, boost(24k) =
   23.47/7.66 = **3.06×**. Acceptance does NOT degrade at long ctx: MTP remains the best
   and most robust lever. (Baseline data also for thread 2: prose penalizes ABSOLUTE
   acceptance, not the one relative to context.)
3. **The 1k→24k drop (−19%) is ALL in the floor** (9.33→7.66 without MTP; with MTP
   28.89→23.47 = the same −19%: the multiplicative boost amplifies the floor, it does not
   cause it). And 3.1 already ruled out the "KV q8_0" route (the Vulkan dequant kernel
   costs more than the bandwidth saved). → The remaining floor margin is in **per-token
   overhead** (kernel launch/dispatch) and attention with large KV: this points to
   sub-threads **3.3 (HIP/Vulkan graphs)** and **3.4 (fused verifier/kernels)**.

Raw: /tmp/bench-kv-T{1..4}*.json, log /tmp/bench-ckpt-orchestrator.log (ephemeral).
