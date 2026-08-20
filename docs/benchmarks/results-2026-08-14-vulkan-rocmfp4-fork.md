# Risultati — ROCmFP4 su backend Vulkan (fork build) vs ROCm

**Data:** 2026-08-14 · GPU esclusiva · llama-bench `-ngl 999 -fa 1 -p 512 -n 128`
Immagini: `docker-llm-service:latest` (ROCm, fork b00d5452) · `docker-llm-service:vulkan-fork`
(fork charlie12345/ROCmFPX main, build Vulkan-only b2f5829, da `Dockerfile.vulkan-rocmfpx` dei
toolboxes, ICD RADV mesa)

## Tabella — Qwen3.6-35B-A3B, stesso GGUF `Q4_0_ROCMFP4_STRIX_LEAN` (17.73 GiB)

| Backend | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| ROCm (produzione attuale) | **1420.7** | 71.2 |
| **Vulkan RADV (fork)** | 1164.7 | **81.6** |

Delta: **RADV tg +14.5%**, ROCm pp +22%.

## Quadro completo aggiornato (tutte le misure 14/08)

| Stack | Quant/formato | Size | pp512 | tg128 |
|---|---|---|---|---|
| ROCm | ROCmFP4 STRIX_LEAN | 17.7 GiB | **1420.7** | 71.2 |
| **Vulkan RADV fork** | **ROCmFP4 STRIX_LEAN** | 17.7 GiB | 1164.7 | **81.6** |
| Vulkan RADV | UD-Q5_K_M_MTP | 25.2 GiB | 1008.1 | 57.9 |
| ROCm | UD-Q5_K_M_MTP | 25.2 GiB | 1359.4 | 50.8 |

## Conclusioni

1. **Confermato in-house il claim plunderstruck (78-90 t/s)**: il nostro 81.6 cade esattamente nel
   range. Lo staging Vulkan dei tipi ROCmFPX è reale e performante.
2. **Cade l'affermazione "Vulkan non supporta i formati ROCmFP4"**: era vera per llama.cpp vanilla
   (`invalid ggml type 101`), falsa per il fork charlie12345/ROCmFPX compilato Vulkan-only.
3. **Il "meglio dei due mondi" (RADV + fp4) esiste**: tg record 81.6 plain (+14.5% vs ROCm stesso
   file). ROCm resta ahead su pp (+22%) e ha MTP/DFlash collaudati.
4. Implicazione pratica: per carichi decode-bound (chat), lo stack Vulkan-fork + ROCmFP4 è ora il
   più veloce; per prefill-bound (RAG/agenti con prompt lunghi) resta ROCm. Da valutare MTP/DFlash
   sul backend Vulkan fork (il fork main ha DFLASH — mem: il nostro runtime :dflash è derivato
   proprio da questo).

## Note riproduzione

- Immagine: `cd <toolboxes-dir>/workspace/docker/amd-strix-halo-toolboxes/toolboxes && docker build
  -f Dockerfile.vulkan-rocmfpx -t docker-llm-service:vulkan-fork .`
  (adjust `<toolboxes-dir>` to your local checkout of kyuz0/amd-strix-halo-toolboxes)
- Run: `docker run --rm --device /dev/dri --group-add render -v ~/llmodels/models:/llmodels:ro
  docker-llm-service:vulkan-fork llama-bench -m <gguf> -ngl 999 -fa 1 -p 512 -n 128`
  (ICD RADV è il default nell'immagine; nessun VK_ICD_FILENAMES necessario)
- Il Dockerfile dei toolboxes risolve due trappole: usa lld+clang (niente symlink `ld` pendente
  della base vulkan-backup) e `LLAMA_BUILD_WEBUI=OFF` + `LLAMA_USE_PREBUILT_WEBUI=OFF`
  (asset WebUI non scaricabili in build).

---

## Appendice — Q6_0_ROCMFPX (Qwopus3.6-35B-A3B-v1-MTP, 27.38 GiB, build rcmorano)

Stesso giorno, stessa metodologia (GPU esclusiva, tg128/pp512, fa 1, ngl 999):

| Backend | pp512 | tg128 |
|---|---|---|
| Vulkan RADV (vulkan-fork b2f5829) | **1076.9** | 49.5 |
| ROCm (:latest 00d5452) | 520.6 ⚠️ | 51.3 |

Note:
- Su ROCm (build più vecchia) il Q6_0_ROCMFPX "experimental staging" ha **pp dimezzato** (520 vs
  1077 Vulkan): il layout Q6 non è ottimizzato lì — per il Q6 il backend giusto è Vulkan.
- tg Q6 (49.5) ≤ Q5_K_M UD (57.9) a size simile: l'ipotesi "Q6 ROCmFPX domina il Q5 ovunque" è
  **falsa** su tg. Il valore del Q6 è solo la qualità (~Q6_K lossless).
- Costo del salto qualità su RADV: fp4_LEAN 81.6 → Q6 49.5 (**-39%**).

## Ripristino llm-service (14/08 sera)

Container `llm-service` su `docker-llm-service:vulkan-fork` (RADV) con ROCmFP4 STRIX_LEAN e flag
completi replicati da start-llama-server.sh: `--jinja --parallel 4 --top-p 0.95 --top-k 20
--temperature 1.0 --reasoning on --reasoning-budget 16384` (senza quest'ultimo il default è
INT32_MAX → thinking senza limite, osservato runaway ~20k token). Backup del vecchio container
dflash: `llm-service-dflash-backup`.

## Appendice 2 — Q6_0_ROCMFPX + MTP (draft-mtp, n-max 4, RADV)

Config validata dal fork (regression guard qwen35-a3b): `--spec-type draft-mtp --spec-draft-ngl all
--spec-draft-n-max 4 --spec-draft-p-min 0.0 --spec-draft-p-split 0.10`. Test corto server 2-prompt
(metodologia bench-oneoff, ctx 16k):

| Metrica | Valore |
|---|---|
| tg prosa | **57.9 t/s** (vs 49.5 plain → +17%) |
| tg deterministico | **73.7 t/s** (+49%) |
| acceptance pos-1 | 0.81-0.89 |
| mean accepted length | 3.19-3.63 token (n-max 4 giustificato: pos 3-4 accettano 46-58%) |

Caveat: numeri server-timing vs llama-bench plain (non confronto rigoroso).
