# Phase A route 1 — in-house ppl: STRIX_LEAN vs Q4_K_M vs BF16

**Date:** 2026-08-17 17:52-18:01 · dedicated GPU · `llama-perplexity` (dflash/ROCm),
`-c 512 -b 512 -fa on -ngl 999`. Arms: local BF16 · current STRIX_LEAN (grug imatrix,
on disk) · in-house generated Q4_K_M (99 s, CPU). Identical corpora for all:
wikitext-2-en 150k char (~73 chunks) · Italian technical 51k char (30 chunks, plans from 15-16/08).

| Arm | bpw/size | PPL en | Δ vs BF16 | PPL it | Δ vs BF16 |
|---|---|---|---|---|---|
| BF16 | 16 / 50.9 GB | 6.6409 ± 0.121 | — | 11.7156 ± 0.388 | — |
| **STRIX_LEAN (grug imatrix)** | 4.38 / 13.8 GB | 6.8226 ± 0.124 | **+2.74%** | 12.1168 ± 0.402 | **+3.42%** |
| **Q4_K_M (no imatrix)** | ~4.8 / 16.8 GB | 6.6747 ± 0.122 | +0.51% | 12.0991 ± 0.404 | **+3.27%** |

## Interpretations

1. **First direct Δ for OUR GGUF**: +2.74% en (much better than the +4.16% KAT-Coder,
   a different MoE arch and WITHOUT imatrix — the proxy is no longer needed).
2. **MEMORY QUALITY GAP CLOSED** ("LEAN vs K-quant, same measurement"): on English Q4_K_M
   wins (+0.51 vs +2.74%, ~2σ); **but on Italian they are ON PAR** (+3.42 vs +3.27%,
   overlapping bars ±0.4). Q4_K_M degrades much more when switching to it (+0.51→+3.27%),
   the LEAN stays stable (the grug imatrix covers the domain) — probably a multilingual
   calibration effect of the corpus.
3. **On the real domain (Italian-agentic) the LEAN in production is already at quality
   parity with Q4_K_M, with 18% less size/bandwidth** → the quality budget is well spent.
4. Phase B (imatrix v2): the plan's threshold was "Δ≥+2.5% → worth it" — we are at the
   boundary (2.74/3.42%). Realistically recoverable margin: reduce the en gap without
   losing it, with a mixed corpus (grug + Italian technical + code). Cost ~45-60 min.
   Optional.

Artifacts: /tmp/ppl-*.log, /tmp/pplcorpus/ (ephemeral) · Q4_K_M: ~/llmodels/QWEN3.8/tmp-Q4_K_M.gguf
(16.8 GB — keep only if Phase B is done, otherwise clean up along with the BF16).
