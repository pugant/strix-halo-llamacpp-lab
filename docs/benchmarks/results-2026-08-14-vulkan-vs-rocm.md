# Results — In-house Vulkan vs ROCm validation (Qwen3.6-35B-A3B)

**Date:** 2026-08-14 · exclusive GPU (llm-service stopped during the runs) · llama-bench `-ngl 999 -fa 1 -p 512 -n 128`
Images: `docker-llm-service:latest` (ROCm, build 00d5452) · `docker-llm-service:vulkan-backup` (Vulkan, build 3f7c79d7b)

## Summary table — Qwen3.6-35B-A3B, same machine, same session

| Backend | Quant | Size | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|---|---|
| **ROCm** | **Q4_0_ROCMFP4_STRIX_LEAN** (production) | 17.73 GiB | **1420.72 ± 14** | **71.23 ± 0.5** |
| ROCm | UD-Q5_K_M_MTP | 25.22 GiB | 1359.39 ± 12 | 50.76 ± 0.2 |
| **Vulkan RADV** (mesa) | UD-Q5_K_M_MTP | 25.22 GiB | 1008.09 ± 14 | **57.93 ± 0.7** |
| Vulkan AMDVLK | UD-Q5_K_M_MTP | 25.22 GiB | 663.02 ± 6 | 55.84 ± 0.7 |

## Interpretation of the data

1. **Backend-only check (exact same UD-Q5_K_M_MTP GGUF):** RADV beats ROCm on tg128 by
   **+14.1%** (57.93 vs 50.76) → the community claim "~+18% tg for Vulkan RADV" **replicates
   in-house** within methodological tolerance. On pp512 ROCm wins **+34.8%** (1359 vs 1008) →
   this also confirms the community (ROCm/AMDVLK win pp).
2. **Stack-vs-stack comparison (what matters in production):** our ROCmFP4 stack wins on both
   fronts: tg128 **+22.9%** (71.23 vs 57.93) and pp512 **+40.9%** (1420 vs 1008). The reason
   is the format: ROCmFP4 weighs 17.7 GiB vs the Q5_K_M's 25.2 GiB → less LPDDR5X bandwidth
   consumed at equal perceptual quality.
3. **RADV >> AMDVLK**, as per the community: here the difference is huge on pp512 (+52%, 1008
   vs 663) and slight on tg (+3.7%). AMDVLK should be considered obsolete.

## Operational notes (to reproduce)

- RADV ICD in the vulkan-backup container: `-e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`
  (⚠️ the exact file name has the `.x86_64` suffix; with the wrong name ggml-vulkan **silently
  falls back to the CPU backend** without any error — always check the `ggml_vulkan: 0 = ... (radv)`
  line in the output).
- ROCm requires `--device /dev/kfd --device /dev/dri --group-add video --group-add render` +
  `HSA_OVERRIDE_GFX_VERSION=11.5.1`; Vulkan only needs `/dev/dri` + `render`.
- Raw files: `bench-qwen36-35b-rocmfp4.txt`, `bench-qwen36-35b-udq5km-rocm.txt`,
  `bench-qwen36-35b-udq5km-vulkan-radv.txt`, `bench-qwen36-35b-udq5km-vulkan-amdlvk.txt`.
- The historical production baseline "63 tok/s plain" (server-timing, grug design 11/08) is
  contemporary with and consistent with today's 71.2 llama-bench (server-timing < llama-bench).

## Conclusion

The community picture is **confirmed at equal quant** (RADV +14% tg), but the stack choice
remains **ROCmFP4**: it wins tg and pp thanks to the lighter fp4 format, plus MTP and capacity.
Vulkan RADV remains plan B (userspace driver, no need for the ROCmFPX fork for standard quants).
