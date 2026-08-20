# ROCm (hipGraphs) vs Vulkan — tg a 1k/24k (sub-filone 3.3)

**Data:** 2026-08-17 17:12-17:15 · GPU dedicata · Qwen3.8-27B STRIX_LEAN · MTP n6 · greedy ·
warm-up scartato, mean±sd su 2 rep. ROCm: immagine `dflash` (main charlie 13/08,
USE_CUDA_GRAPH→hipGraph di default, denso senza mul_mat_id → graphs percorribili),
env `HSA_OVERRIDE_GFX_VERSION=11.5.1` + unified memory. Vulkan: T1 del run 3.2 (16:55).

| ctx | Vulkan (T1) | ROCm+graphs (R1) | Δ |
|---|---|---|---|
| 1k | 28.89 ± 0.01 | 29.33 ± 0.39 | +1.5% (~1σ) |
| 24k | 23.47 ± 0.03 | 23.76 ± 0.15 | +1.2% (~2σ) |

Caveat: il corpus prompt è variato tra i run (nuovi .md in docs/ → glob diverso:
1275/25215 vs 1242/25459 token, ~1%): confondente piccola, direzione invariata.

## Verdetto: 3.3 CHIUSO — il backend non è la leva

ROCm (con graphs) e Vulkan sono **equivalenti** a 1k e 24k su questo modello/protocollo.
L'overhead launch per-token non è il collo (o i graphs non cambiano abbastanza su gfx1151).
NB: il "+53% tg densi→ROCm" storico (run 2026-08-15) NON si riproduce qui — protocollo
diverso (contesto/workload/flag di quel bench); da chiarire solo se servisse, il dato
odiorno è quello rilevante per llm-service.

## Quadro complessivo del topic 3 dopo 3.1/3.2/3.3 (tutti con GPU dedicata)

- Floor a 24k ≈ 23.5 tok/s con MTP (boost 3.06× costante) ≈ **limite banda UMA**:
  pesi 13.8 GB + KV-read per token dominano; KV q8_0 peggiora (kernel dequant),
  checkpoint/ring neutri, backend neutro.
- Margini residui reali: **3.4 kernels** (fused verifier MMVQ / fused attention) e
  **filone 2** (acceptance prosa → boost). Il resto è fisica (LPDDR5X ~270 GB/s).

Raw: /tmp/bench-kv-R1-rocm.json, log /tmp/bench-rocm-orchestrator.log (effimeri).
