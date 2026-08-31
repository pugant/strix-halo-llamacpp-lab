# Results — ROCmFP4 on the Vulkan backend (fork build) vs ROCm

**Date:** 2026-08-14 · exclusive GPU · llama-bench `-ngl 999 -fa 1 -p 512 -n 128`
Images: `docker-llm-service:latest` (ROCm, fork b00d5452) · `docker-llm-service:vulkan-fork`
(charlie12345/ROCmFPX fork main, Vulkan-only build b2f5829, from the toolboxes'
`Dockerfile.vulkan-rocmfpx`, mesa RADV ICD)

## Table — Qwen3.6-35B-A3B, same GGUF `Q4_0_ROCMFP4_STRIX_LEAN` (17.73 GiB)

| Backend | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| ROCm (current production) | **1420.7** | 71.2 |
| **Vulkan RADV (fork)** | 1164.7 | **81.6** |

Delta: **RADV tg +14.5%**, ROCm pp +22%.

## Updated full picture (all measurements from 14/08)

| Stack | Quant/format | Size | pp512 | tg128 |
|---|---|---|---|---|
| ROCm | ROCmFP4 STRIX_LEAN | 17.7 GiB | **1420.7** | 71.2 |
| **Vulkan RADV fork** | **ROCmFP4 STRIX_LEAN** | 17.7 GiB | 1164.7 | **81.6** |
| Vulkan RADV | UD-Q5_K_M_MTP | 25.2 GiB | 1008.1 | 57.9 |
| ROCm | UD-Q5_K_M_MTP | 25.2 GiB | 1359.4 | 50.8 |

## Conclusions

1. **plunderstruck's claim (78-90 t/s) confirmed in-house**: our 81.6 falls exactly within the
   range. The Vulkan staging of ROCmFPX types is real and performant.
2. **The claim "Vulkan does not support ROCmFP4 formats" falls**: it was true for vanilla
   llama.cpp (`invalid ggml type 101`), false for the charlie12345/ROCmFPX fork compiled
   Vulkan-only.
3. **The "best of both worlds" (RADV + fp4) exists**: record plain tg 81.6 (+14.5% vs ROCm,
   same file). ROCm stays ahead on pp (+22%) and has battle-tested MTP/DFlash.
4. Practical implication: for decode-bound workloads (chat), the Vulkan-fork + ROCmFP4 stack is
   now the fastest; for prefill-bound (RAG/agents with long prompts) ROCm remains. MTP/DFlash
   on the Vulkan fork backend still to be evaluated (the fork main has DFLASH — note: our
   :dflash runtime derives exactly from this).

## Reproduction notes

- Image: `cd <toolboxes-dir>/workspace/docker/amd-strix-halo-toolboxes/toolboxes && docker build
  -f Dockerfile.vulkan-rocmfpx -t docker-llm-service:vulkan-fork .`
  (adjust `<toolboxes-dir>` to your local checkout of kyuz0/amd-strix-halo-toolboxes)
- Run: `docker run --rm --device /dev/dri --group-add render -v ~/llmodels/models:/llmodels:ro
  docker-llm-service:vulkan-fork llama-bench -m <gguf> -ngl 999 -fa 1 -p 512 -n 128`
  (the RADV ICD is the default in the image; no VK_ICD_FILENAMES needed)
- The toolboxes' Dockerfile solves two traps: it uses lld+clang (no dangling `ld` symlink from
  the vulkan-backup base) and `LLAMA_BUILD_WEBUI=OFF` + `LLAMA_USE_PREBUILT_WEBUI=OFF`
  (WebUI assets not downloadable at build time).

---

## Appendix — Q6_0_ROCMFPX (Qwopus3.6-35B-A3B-v1-MTP, 27.38 GiB, build rcmorano)

Same day, same methodology (exclusive GPU, tg128/pp512, fa 1, ngl 999):

| Backend | pp512 | tg128 |
|---|---|---|
| Vulkan RADV (vulkan-fork b2f5829) | **1076.9** | 49.5 |
| ROCm (:latest 00d5452) | 520.6 ⚠️ | 51.3 |

Notes:
- On ROCm (older build) the "experimental staging" Q6_0_ROCMFPX has **half the pp** (520 vs
  1077 Vulkan): the Q6 layout is not optimized there — for Q6 the right backend is Vulkan.
- Q6 tg (49.5) ≤ UD Q5_K_M (57.9) at similar size: the hypothesis "Q6 ROCmFPX dominates Q5
  everywhere" is **false** on tg. The Q6's value is quality only (~lossless Q6_K).
- Cost of the quality jump on RADV: fp4_LEAN 81.6 → Q6 49.5 (**-39%**).

## llm-service restore (14/08 evening)

`llm-service` container on `docker-llm-service:vulkan-fork` (RADV) with ROCmFP4 STRIX_LEAN and
the full flags replicated from start-llama-server.sh: `--jinja --parallel 4 --top-p 0.95 --top-k 20
--temperature 1.0 --reasoning on --reasoning-budget 16384` (without the latter the default is
INT32_MAX → unbounded thinking, observed a ~20k token runaway). Backup of the old dflash
container: `llm-service-dflash-backup`.

## Appendix 2 — Q6_0_ROCMFPX + MTP (draft-mtp, n-max 4, RADV)

Config validated by the fork (qwen35-a3b regression guard): `--spec-type draft-mtp --spec-draft-ngl all
--spec-draft-n-max 4 --spec-draft-p-min 0.0 --spec-draft-p-split 0.10`. Short 2-prompt server
test (bench-oneoff methodology, ctx 16k):

| Metric | Value |
|---|---|
| tg prose | **57.9 t/s** (vs 49.5 plain → +17%) |
| tg deterministic | **73.7 t/s** (+49%) |
| acceptance pos-1 | 0.81-0.89 |
| mean accepted length | 3.19-3.63 token (n-max 4 justified: pos 3-4 accept 46-58%) |

Caveat: server-timing numbers vs plain llama-bench (not a rigorous comparison).
