# Fase A strada 1 — ppl in-house: STRIX_LEAN vs Q4_K_M vs BF16

**Data:** 2026-08-17 17:52-18:01 · GPU dedicata · `llama-perplexity` (dflash/ROCm),
`-c 512 -b 512 -fa on -ngl 999`. Bracci: BF16 locale · STRIX_LEAN attuale (imatrix
grug, su disco) · Q4_K_M generato in-house (99 s, CPU). Corpus identici per tutti:
wikitext-2-en 150k char (~73 chunk) · italiano tecnico 51k char (30 chunk, piani 15-16/08).

| Braccio | bpw/size | PPL en | Δ vs BF16 | PPL it | Δ vs BF16 |
|---|---|---|---|---|---|
| BF16 | 16 / 50.9 GB | 6.6409 ± 0.121 | — | 11.7156 ± 0.388 | — |
| **STRIX_LEAN (grug imatrix)** | 4.38 / 13.8 GB | 6.8226 ± 0.124 | **+2.74%** | 12.1168 ± 0.402 | **+3.42%** |
| **Q4_K_M (no imatrix)** | ~4.8 / 16.8 GB | 6.6747 ± 0.122 | +0.51% | 12.0991 ± 0.404 | **+3.27%** |

## Letture

1. **Primo Δ diretto del NOSTRO GGUF**: +2.74% en (molto meglio del +4.16% KAT-Coder,
   arch MoE diversa e SENZA imatrix — proxy non più necessario).
2. **GAP MEMORIA QUALITY CHIUSO** ("LEAN vs K-quant stessa misura"): su inglese Q4_K_M
   vince (+0.51 vs +2.74%, ~2σ); **ma su italiano sono PARI** (+3.42 vs +3.27%, barre
   sovrapposte ±0.4). Il Q4_K_M degrada molto di più passando a it (+0.51→+3.27%),
   il LEAN resta stabile (l'imatrix grug copre il dominio) — probabilmente effetto
   calibrazione multilingue del corpus.
3. **Sul dominio reale (italiano-agentic) il LEAN in produzione è già a parità qualità
   col Q4_K_M, con il 18% in meno di size/banda** → il budget qualità è ben speso.
4. Fase B (imatrix v2): soglia del piano era "Δ≥+2.5% → conviene" — siamo al confine
   (2.74/3.42%). Margine realisticamente recuperabile: ridurre il gap en senza perdere
   it, con corpus mix (grug + italiano tecnico + code). Costo ~45-60 min. Opzionale.

Artefatti: /tmp/ppl-*.log, /tmp/pplcorpus/ (effimeri) · Q4_K_M: ~/llmodels/QWEN3.8/tmp-Q4_K_M.gguf
(16.8 GB — da tenere solo se si fa Fase B, altrimenti cleanup col BF16).
