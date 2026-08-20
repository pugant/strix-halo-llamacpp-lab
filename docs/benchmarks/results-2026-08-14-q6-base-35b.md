# Risultati — Qwen3.6-35B-A3B base → Q6_0_ROCMFPX (pipeline 14/08 sera)

**Data:** 2026-08-14 · GPU esclusiva · `llama-bench -ngl 999 -fa 1 -p 512 -n 128`
Immagine bench: `docker-llm-service:vulkan-fork` (fork charlie12345/ROCmFPX, build b2f5829,
backend **Vulkan RADV**) · Fonte BF16: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (shard verificati
byte-per-byte vs HF) · Quantize: `docker-llm-service:latest`, preset `Q6_0_ROCMFPX` (type 110,
6.50 bpw) **CON imatrix** (accettata), 178 s, 753 tensori (nextn/MTP inclusi).

## Il nuovo file

| Metrica | Valore |
|---|---|
| Output | `Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX.gguf` — **27.39 GiB** (6.62 BPW effettivi) |
| tg128 Vulkan RADV | **49.16 ± 0.36** tok/s |
| pp512 Vulkan RADV | **1120.38 ± 10.36** tok/s |
| SHA256 (sanificato) | `2251b28cd9f3e2edaa37d91fff3ae3172acc9a8f91eaaf4f167254e22295caca` |

Sanity check: il Qwopus Q6 (finetune stessa arch/quant, 27.38 GiB) dava 49.48/1076.91 →
il base Q6 è perfettamente in linea (±1% tg, +4% pp).

## Tabella comparativa — tutte le versioni Qwen3.6-35B-A3B provate (14/08, stessa metodologia)

| Versione | Quant | Size | Backend | pp512 | tg128 |
|---|---|---:|---|---:|---:|
| **base (questa pipeline)** | **Q6_0_ROCMFPX** | **27.39 GiB** | **Vulkan RADV** | **1120.38** | **49.16** |
| Qwopus3.6-35B (finetune abliterated) | Q6_0_ROCMFPX | 27.38 GiB | Vulkan RADV | 1076.91 | 49.48 |
| Qwopus3.6-35B (finetune abliterated) | Q6_0_ROCMFPX | 27.38 GiB | ROCm | 520.56 ⚠️ | 51.29 |
| base | Q4_0_ROCMFP4_STRIX_LEAN | 17.73 GiB | Vulkan RADV (fork) | 1164.67 | **81.57** |
| base | Q4_0_ROCMFP4_STRIX_LEAN | 17.73 GiB | ROCm | **1420.72** | 71.23 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | Vulkan RADV | 1008.09 | 57.93 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | ROCm | 1359.39 | 50.76 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | Vulkan AMDVLK | 663.02 | 55.84 |

⚠️ pp 520 su ROCm: staging Q6 non ottimizzato su build ROCm 00d5452 — per il Q6 il backend di
riferimento è **Vulkan** (per questo il publish dichiara i numeri RADV).

### Lettura

- Il Q6 base è il **tier qualità** (~Q6_K-class): costa -40% tg vs fp4_LEAN su Vulkan (81.57→49.16)
  e -15% vs Q5_K_M UD (57.93→49.16), a fronte di 6.62 BPW.
- pp del Q6 (1120) è superiore al Q5_K_M (1008) su RADV: a parità di backend il routing Q6 del
  fork pagherà meno sul prefill.
- Il Q6 con MTP (config validata n-max 4): prosa ~57.9 (+17%), deterministico ~73.7 (+49%)
  [server-timing, misura su Qwopus Q6 — stessa arch/quant].

## Sweep MTP n-max sul BASE Q6 (2026-08-15, piano 2026-08-15-q6-base-mtp-nmax-test.md — non incluso nel repo)

Server-timing, 2 prompt × 2 run, ctx 16k, Vulkan RADV (vulkan-fork b2f5829), script
`scripts/mtp-nmax-test.sh` (non incluso nel repo). Flag: `--spec-type draft-mtp --spec-draft-ngl all
--spec-draft-p-min 0.0 --spec-draft-p-split 0.10` + n-max variato.

| n-max | P1 prosa (2 run, media) | P2 det (media) | acceptance/pos (ultimo task) |
|---|---:|---:|---|
| 4 | 44.7 (46.6/42.8) | 62.8 | (0.869, 0.756, 0.606, 0.512) |
| 3 | 54.0 (55.1/52.9) | **67.3** | det (0.874, 0.769, 0.643) |
| **2** | **58.2** (56.5/59.8) | 62.4 | det (0.822, 0.660) |

**Conclusione: sul BASE il n-max 4 è dominato.** Il pos-4 accetta solo ~0.51 e l'overhead
draft non è ripagato: prosa -23% vs n-max 2. Ottimale per contesto misto (coding agent):
**n-max 3** (det 67.3, prosa 54.0); per puro decode prosa: n-max 2 (58.2). Diverso dal
Qwopus (n-max 4 → 57.9 prosa il 14/08): acceptance distribution dipende dal finetune.
Nota: varianza run-to-run prosa ±3 tok/s (sampling temp default 1.0) — differenze >8 tok/s
sono segnale, sotto è rumore.

## Note pipeline

- Download: 66.2 GiB BF16 via wget -c (~12-27 MB/s). **Incidente evitato**: doppio wget concorrente
  sullo shard2 (script detached della sessione precedente + task di questa sessione) → kill di
  entrambi, cancellazione shard2 parziale, download pulito singolo. Verifica finale byte-per-byte
  contro `repo_info(files_metadata=True)`.
- Sanitize metadata: eseguito nel container `docker-llm-service-convert` (gguf-py fork ROCmFPX;
  l'host python3.12 con llama.cpp-pflash NON conosce i tipi 102/110 → ValueError). Config
  `scripts/configs/qwen36-35b-q6-metadata.json` (non incluso nel repo).
- Smoke post-sanitize: load Vulkan OK (pp16 202.88 / tg8 48.27) ✓.
