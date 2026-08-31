# Nemotron 3.5 Lightning 30B-A3B → ROCmFP4-STRIX_LEAN — Bench Results

**Date:** 2026-08-12
**Model:** NVIDIA-Nemotron-3.5-Lightning-30B-A3B (`nemotron_h_moe`, 31.58B params, 3.5B active/token)
**Architecture:** Hybrid Mamba-2 + MoE (128 experts, 6 per token) + Attention (52 layers)
**Toolchain:** fork charlie12345/ROCmFPX commit a7ddbe7, ROCm 7.2.4, gfx1151

## System config

| Component | Value |
|---|---|
| APU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (Strix Halo) |
| GPU | Radeon 8060S Graphics, gfx1151, 126976 MiB (unified memory) |
| Kernel | 7.0.0-28-generic |
| Power profile | balanced |
| VRAM partition | 512 MB (UMA mandatory: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) |

## ROCmFP4-STRIX_LEAN vs Q4_K_M comparison

Both quantized from the same fork BF16 (401 tensors, converted from nvidia safetensors with the fork's converter). Bench via `llama-bench -ngl 999 -fa on -p 512 -n 128`.

| Metric | ROCmFP4-STRIX_LEAN | Q4_K_M | Δ ROCmFP4 |
|---|---|---|---|
| **tg128 (tok/s)** | **84.28 ± 0.66** | 63.64 ± 0.93 | **+32%** |
| **pp512 (tok/s)** | **1051 ± 5.88** | 813 ± 55.46 | **+29%** |
| Size | 15.72 GiB | 22.82 GiB | **−31%** |
| BPW | 4.28 | 6.21 | |
| Fallback tensors | 1/401 | 134/401 | |

**Verdict:** ROCmFP4-STRIX_LEAN is the optimal format for Nemotron on Strix Halo: +32% tg128, +29% pp512, and 31% smaller than Q4_K_M.

### Technical notes

1. **Tensor fallback:** Q4_K_M has 134/401 tensors in fallback (Mamba/SSM tensors not quantizable to Q4_K_M due to incompatible shapes → they fall back to a larger type). ROCmFP4 has only 1 tensor in fallback. This contributes to ROCmFP4's advantage: it natively quantizes the SSM tensors (`ssm_in.weight`, `ssm_out.weight`) to `q4_0_rocmfp4_fast`.

2. **`-fit off` mandatory for nemotron_h_moe:** `llama-server` with the default fit (`-fit on`) hangs on "fitting params to device memory" (infinite loop for the 128-expert MoE). Always use `-fit off`. `llama-bench` does not have this problem.

3. **Slow BF16 loading (~8 min):** graph_reserve for 52 mixed Mamba/MoE/Attention layers is slow. ROCmFP4 (16 GB) loads in ~14s. BF16 (60 GB) takes ~8 min.

4. **Mamba-on-HIP validated:** the `fused Gated Delta Net` (Mamba-2) kernels run on gfx1151 without asserts. Inference produces coherent text.

5. **MTP not embedded:** the fork's converter does not convert the nextn tensors (MTP layer). The model works as a base without embedded speculative decoding. For MTP, a separate drafter is needed (not tested in this session).

## Serving command (ROCmFP4)

```bash
docker run --rm -d --name nemotron \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  -p 1234:1234 -v ~/llmodels/models:/llmodels \
  docker-llm-service:nemotron llama-server \
    -m /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
    -ngl 999 -c 32768 -fa on --jinja --host 0.0.0.0 --port 1234 -fit off
```

## Comparison with other models on Strix Halo

| Model | Format | tg128 | Size |
|---|---|---|---|
| **Nemotron 30B-A3B** | **ROCmFP4-STRIX_LEAN** | **84.28** ⭐ | 15.72 GiB |
| grug-35b-v2 | ROCmFP4-STRIX_LEAN | 70.92 | 17.32 GiB |
| Ornith-1.0-35B | ROCmFP4-STRIX_LEAN | 66.68 | 17.32 GiB |
| Qwen3.6-27B | ROCmFP4-STRIX_LEAN | 13.68 (32.7 with MTP) | 13.82 GiB |

Nemotron ROCmFP4 is the **fastest ever tested** on Strix Halo (+19% vs grug, the previous record).
