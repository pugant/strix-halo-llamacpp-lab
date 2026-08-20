# KV q8_0 asimmetrica vs f16 — tg per contesto (NO-GO)

**Data:** 2026-08-17 16:45-16:51 · GPU dedicata · immagine `vulkan-fork-ckpt6` ·
Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN · MTP n6 · greedy · 128 tok/rep · warm-up scartato,
mean±sd su 2 rep (sd del piano 3→2 per velocità; direzione confermata dai singoli rep).

| ctx | A: KV f16 | C: q8_0 + draft q4_0 | Δ C vs A |
|---|---|---|---|
| ~1k | 32.11 ± 0.12 | 32.08 ± 0.60 | pari |
| ~12k | **24.83 ± 0.08** | 20.93 ± 0.58 | **−15.7%** (≈7σ) |
| ~24k | **21.72 ± 0.22** | 21.03 ± 0.41 | **−3.2%** (≈3σ) |

Prompt: prosa reale (docs del workspace) + istruzione finale; rep successivi con prompt
cache attiva (cached = 1023/12635/25241). Output smoke identico tra bracci.

## Verdetto: NO-GO (criterio del piano: serviva C ≥ A a 1k ✓ MA C > A a 24k ✗)

KV quantizzata NON aiuta su questo stack (backend Vulkan fork, 27B denso): peggiora o
pareggia ovunque. Il collo a ctx lungo NON è la sola banda di lettura KV — se lo fosse,
q8_0 (metà byte) avrebbe guadagnato. Legge `tg=floor(ctx)×boost(acc)`: il calo −32%
da 1k a 24k va cercato altrove (ipotesi da verificare nel sub-filone 3.2: overhead
checkpoint/ring speculativi, costo verify con KV grande; anche acceptance MTP non
misurata in questo run — i log del container sono stati rimossi a fine corsa).

Nota: la doc fork "config ottimale TurboQuant per qwen35-a3b" riguardava arch MoE
qwen35 (Qwopus) — questo risultato su denso 3.8 NON la contraddice.

**Conseguenze:** thread "esperimento KV q8_0 su Vulkan" CHIUSO. Prossimo sub-filone
attivo: **3.2 tuning ring/checkpoint** (sweep flag a ctx 24k, zero codice).
Raw: /tmp/bench-kv-{A,C}.json (effimeri), log /tmp/bench-kv-orchestrator.log.
