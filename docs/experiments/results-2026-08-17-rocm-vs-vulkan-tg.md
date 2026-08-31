# ROCm (hipGraphs) vs Vulkan — tg at 1k/24k (sub-thread 3.3)

**Date:** 2026-08-17 17:12-17:15 · dedicated GPU · Qwen3.8-27B STRIX_LEAN · MTP n6 · greedy ·
warm-up discarded, mean±sd over 2 reps. ROCm: image `dflash` (charlie main 13/08,
USE_CUDA_GRAPH→hipGraph by default, dense without mul_mat_id → graphs viable),
env `HSA_OVERRIDE_GFX_VERSION=11.5.1` + unified memory. Vulkan: T1 from the 3.2 run (16:55).

| ctx | Vulkan (T1) | ROCm+graphs (R1) | Δ |
|---|---|---|---|
| 1k | 28.89 ± 0.01 | 29.33 ± 0.39 | +1.5% (~1σ) |
| 24k | 23.47 ± 0.03 | 23.76 ± 0.15 | +1.2% (~2σ) |

Caveat: the prompt corpus changed between runs (new .md files in docs/ → different glob:
1275/25215 vs 1242/25459 tokens, ~1%): a small confounder, direction unchanged.

## Verdict: 3.3 CLOSED — the backend is not the lever

ROCm (with graphs) and Vulkan are **equivalent** at 1k and 24k on this model/protocol.
Per-token launch overhead is not the bottleneck (or graphs do not change enough on
gfx1151). NB: the historical "+53% tg dense→ROCm" (run 2026-08-15) does NOT reproduce
here — different protocol (context/workload/flags of that bench); to be clarified only
if needed, today's figure is the relevant one for llm-service.

## Overall picture of topic 3 after 3.1/3.2/3.3 (all with dedicated GPU)

- Floor at 24k ≈ 23.5 tok/s with MTP (constant 3.06× boost) ≈ **UMA bandwidth limit**:
  13.8 GB weights + per-token KV-read dominate; KV q8_0 worsens (dequant kernel),
  checkpoint/ring neutral, backend neutral.
- Real remaining margins: **3.4 kernels** (fused verifier MMVQ / fused attention) and
  **thread 2** (prose acceptance → boost). The rest is physics (LPDDR5X ~270 GB/s).

Raw: /tmp/bench-kv-R1-rocm.json, log /tmp/bench-rocm-orchestrator.log (ephemeral).
