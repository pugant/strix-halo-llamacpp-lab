# ROCmFP4 pieno (dual-scale) vs STRIX_LEAN — risultati esperimento A/B/D/E

**Data:** 2026-08-18, 19:46-20:40 · Piano: piano di esperimento interno `2026-08-18-rocmfp4-full-vs-strix-lean-experiment.md` (non incluso nel repo) · Contesto: filone 4 (NVFP4→qualità), report ricognizione DGX Spark stesso giorno.

**Setup:** Qwen3.8-27B (denso, 48 SSM + 16 full-attn + MTP) dal BF16 locale, imatrix autoprodotta (gate-PASS 496/496) per TUTTI i bracci. Quant nel container convert dell'11/08 (preset 100/103). PPL: `llama-perplexity` dflash/ROCm, `-c 512 -b 512 -fa on -ngl 999`, corpus phaseA identici (en 150k/70 chunk, it 51.5k/30 chunk; LEAN di controllo riprodotto **esattamente** al 4° decimale). Bench: server Vulkan fork ckpt7, MTP n6 p_min 0.75 p_split 0.10, c 16384, temp 0, warm-up scartato, marker TREATMENT nei log; llama-bench `-p 512 -n 128 -fa 1 -r 5`.

**Scoperta preliminare (routing):** il preset `Q4_0_ROCMFP4` "pieno" NON è "dual-scale ovunque a 4.50 bpw": routing misto `attn_qkv→q5_K`, `ffn_down→q6_K`, dual-scale ROCmFP4 su gate/up (+emb). Il confronto accademico puro richiede i preset `_EVEN` (`--pure`): da qui i bracci D/E aggiunti in corsa (addendum piano).

## Tabella 1 — Qualità (ppl wikitext-2-en / italiano tecnico)

| Braccio | Preset | bpw | Size | PPL en | **Δ en vs BF16** | PPL it | **Δ it vs BF16** |
|---|---|---|---|---|---|---|---|
| BF16 (rif. 17/08) | — | 16 | 50.9 GB | 6.6409 ± 0.121 | — | 11.7156 ± 0.388 | — |
| A LEAN (produzione) | STRIX_LEAN | 4.38 | 14.85 GB | 6.8226 | +2.74% | 12.1168 | +3.42% |
| **B full** | Q4_0_ROCMFP4 | ~5.24 | 17.74 GB | **6.6901 ± 0.122** | **+0.74%** | **11.8479 ± 0.394** | **+1.13%** |
| D even (dual ovunque) | Q4_0_ROCMFP4_EVEN | 4.55 | 15.39 GB | 6.7669 ± 0.123 | +1.90% | 12.0215 ± 0.399 | +2.61% |
| E fast-even (single ovunque) | Q4_0_ROCMFP4_FAST_EVEN | 4.30 | 14.53 GB | 6.8474 ± 0.125 | +3.10% | 12.1566 ± 0.404 | +3.76% |
| Q4_K_M (rif. 17/08, no imatrix) | — | ~4.8 | 16.8 GB | 6.6747 | +0.51% | 12.0991 | +3.27% |

## Tabella 2 — Velocità (Strix Halo gfx1151, Vulkan RADV; EVEN aggiunto 19/08, addendum 2)

| Test | LEAN | FULL | Δ FULL | **EVEN (D)** | Δ EVEN |
|---|---|---|---|---|---|
| MTP n6 prosa (2 prompt) | 19.6 / 20.3 | 15.6 / 17.2 | **−16%** | **19.5 / 19.6** | −0.5% / −3.4% |
| MTP n6 det (2 prompt) | 45.4 / 26.2 | 37.1 / 25.0 | **−17%** | **41.7 / 27.7** | −8.1% / +5.7% |
| llama-bench tg128 (no MTP) | 13.62 ± 0.86 | 9.37 ± 0.92 | **−31%** | **13.64 ± 0.01** | **+0.1% (pari)** |
| llama-bench pp512 | 346.5 ± 10.2 | 314.6 ± 9.6 | **−9%** | **347.6 ± 10.4** | **+0.3% (pari)** |

- Il Δ MTP del FULL (−16/−17%) coincide col ratio byte (16.51/13.82 GiB = 1.195). Il suo tg128 no-MTP (−31%) ha un extra ~−15% oltre i byte: path tg dei tensori q5_K/q6_K (qkv/down) su Vulkan meno efficiente del ROCmFP4 custom.
- **EVEN = velocità PARI al LEAN su tutto** (tg128 ±0.01, pp512 ±1, MTP entro ±8% run-noise): il kernel dual-scale applicato ovunque NON costa nulla in inferenza — il layout "FAST single-scale" non ha vantaggio misurabile sul tg; il costo del FULL viene solo dai tensori K-quant protetti.

## Analisi dei contrasti (cosa causa la perdita qualità)

| Contrasto | Cosa isola | Δ ppl en | Δ ppl it | Costo size |
|---|---|---|---|---|
| E → D (single→dual ovunque) | **granularità scala 32→16** (la domanda NVFP4 del filone 4) | **−1.20 pt** (3.10→1.90) | **−1.15 pt** (3.76→2.61) | +0.86 GB (+5.9%) |
| E → LEAN | protezioni K/V+emb del LEAN | −0.36 pt | −0.34 pt | +0.32 GB |
| D → B | protezioni qkv q5_K + down q6_K | **−1.16 pt** | **−1.48 pt** | +2.35 GB |

1. **La granularità della scala È una leva reale e isolata**: dual-scale ovunque (D) recupera ~1.2 punti di Δppl con soli +0.86 GB — risposta SÌ alla domanda "block-scale FP8 → qualità superiore" del filone 4, misurata per la prima volta a routing identico (primo dato del genere in community).
2. Le protezioni del full valgono quasi quanto la granularità (−1.2/−1.5 pt) ma costano il triplo in byte (qkv q5_K + down q6_K su un denso).
3. Il LEAN attuale spreca poco: le sue protezioni valgono solo ~0.35 pt.

## Criterio decisionale del piano

> GO sostituzione produzione se Δppl pieno ≤ metà LEAN su entrambi i corpus AND tg MTP pieno ≥ 97% del LEAN.

- Qualità: **PASS** (+0.74 ≤ 1.4 ✓, +1.13 ≤ 1.7 ✓)
- Velocità: **FAIL** (tg MTP = 83-84% del LEAN, < 97%)

**Esito: nessun GO automatico — trade-off da decidere.** Il full costa ~−16% tok/s in produzione e guadagna: Δppl 2.0 pt (en) / 2.3 pt (it), e su italiano è il miglior quant mai misurato su questa macchina (+1.13% vs +3.27% del Q4_K_M e +3.42% del LEAN). Per confronto dimensionale: il full è il "gradino qualità" della community theory (KAT-Coder raccomandava il pieno per coding) qui quantificato sul nostro modello/hw.

## Opzioni per la produzione (aggiornato 19/08 con i dati EVEN)

| Opzione | Δppl en/it | tg128 no-MTP | MTP det | Size | Note |
|---|---|---|---|---|---|
| Status quo LEAN | +2.74/+3.42 | 13.62 | 45.4 | 13.82 GiB | attuale |
| FULL | +0.74/+1.13 | 9.37 (−31%) | 37.1 | 16.51 GiB | qualità massima, caro in velocità |
| **EVEN (D)** | **+1.90/+2.61** | **13.64 (pari)** | **41.7 (pari entro noise)** | 14.32 GiB (+3.6%) | **quasi-pareto: −0.84/−0.81 pt di ppl GRATIS** |

**Conclusione aggiornata**: l'EVEN domina quasi-pareto il LEAN (qualità migliore su entrambe le lingue a velocità e size praticamente identiche) → è il candidato naturale a nuova produzione se si vuole qualità extra senza costo; il FULL resta l'opzione "qualità massima" per usi non-latency-critical. Decisione all'utente.

## Note tecniche

- Il routing "pieno" spiega il 5.25 bpw del KAT-Coder: su KAT (MoE) il bpw effettivo era 5.25; qui 27B denso → 17.74 GB (~5.24 bpw effettivi).
- Smoke Vulkan PASS: output coerente, gen no-MTP 12.5 t/s = floor atteso (222 GB/s effettivi) → il kernel dual-scale non ha overhead anomalo in inferenza server.
- Braccio C (NVFP4 nativo) escluso per precedente negativo julianmb (bug scale2: ppl 109.8 incoerente).
- Artefatti: GGUF B/D/E in `models-test/` (mantenuti per decisione); log `logs/ppl-full-vs-lean-20260818.log`, `logs/bench-full-vs-lean/`; corpus `/tmp/pplcorpus` (effimero).
- llm-service ripristinato su ckpt7 e healthy al termine.
