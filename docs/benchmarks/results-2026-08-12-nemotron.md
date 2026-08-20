# Nemotron 3.5 Lightning 30B-A3B → ROCmFP4-STRIX_LEAN — Risultati Bench

**Data:** 2026-08-12
**Modello:** NVIDIA-Nemotron-3.5-Lightning-30B-A3B (`nemotron_h_moe`, 31.58B params, 3.5B attivi/token)
**Architettura:** Hybrid Mamba-2 + MoE (128 esperti, 6 per token) + Attention (52 layer)
**Toolchain:** fork charlie12345/ROCmFPX commit a7ddbe7, ROCm 7.2.4, gfx1151

## Config sistema

| Componente | Valore |
|---|---|
| APU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (Strix Halo) |
| GPU | Radeon 8060S Graphics, gfx1151, 126976 MiB (unified memory) |
| Kernel | 7.0.0-28-generic |
| Power profile | balanced |
| VRAM partition | 512 MB (UMA obbligatoria: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) |

## Confronto ROCmFP4-STRIX_LEAN vs Q4_K_M

Entrambi quantizzati dallo stesso BF16 fork (401 tensori, convertito da safetensors nvidia col converter del fork). Bench via `llama-bench -ngl 999 -fa on -p 512 -n 128`.

| Metrico | ROCmFP4-STRIX_LEAN | Q4_K_M | Δ ROCmFP4 |
|---|---|---|---|
| **tg128 (tok/s)** | **84.28 ± 0.66** | 63.64 ± 0.93 | **+32%** |
| **pp512 (tok/s)** | **1051 ± 5.88** | 813 ± 55.46 | **+29%** |
| Size | 15.72 GiB | 22.82 GiB | **−31%** |
| BPW | 4.28 | 6.21 | |
| Tensori fallback | 1/401 | 134/401 | |

**Verdetto:** ROCmFP4-STRIX_LEAN è il formato ottimale per Nemotron su Strix Halo: +32% tg128, +29% pp512, e 31% più piccolo del Q4_K_M.

### Note tecniche

1. **Fallback tensori:** Q4_K_M ha 134/401 tensori in fallback (tensori Mamba/SSM non quantizzabili a Q4_K_M per shape incompatibili → cadono su tipo superiore). ROCmFP4 ha solo 1 tensore in fallback. Questo contribuisce al vantaggio di ROCmFP4: quantizza nativamente i tensori SSM (`ssm_in.weight`, `ssm_out.weight`) a `q4_0_rocmfp4_fast`.

2. **`-fit off` obbligatorio per nemotron_h_moe:** `llama-server` col fit di default (`-fit on`) si blocca su "fitting params to device memory" (loop infinito per MoE 128 esperti). Usare sempre `-fit off`. `llama-bench` non ha questo problema.

3. **Caricamento BF16 lento (~8 min):** il graph_reserve per 52 layer misti Mamba/MoE/Attention è lento. Il ROCmFP4 (16 GB) carica in ~14s. Il BF16 (60 GB) richiede ~8 min.

4. **Mamba-on-HIP validato:** i kernel `fused Gated Delta Net` (Mamba-2) girano su gfx1151 senza assert. Inference produce testo coerente.

5. **MTP non embedded:** il converter del fork non converte i tensori nextn (MTP layer). Il modello funziona come base senza speculative decoding embedded. Per MTP, serve drafter separato (non testato in questa session).

## Comando serving (ROCmFP4)

```bash
docker run --rm -d --name nemotron \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  -p 1234:1234 -v ~/llmodels/models:/llmodels \
  docker-llm-service:nemotron llama-server \
    -m /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
    -ngl 999 -c 32768 -fa on --jinja --host 0.0.0.0 --port 1234 -fit off
```

## Confronto con altri modelli su Strix Halo

| Modello | Format | tg128 | Size |
|---|---|---|---|
| **Nemotron 30B-A3B** | **ROCmFP4-STRIX_LEAN** | **84.28** ⭐ | 15.72 GiB |
| grug-35b-v2 | ROCmFP4-STRIX_LEAN | 70.92 | 17.32 GiB |
| Ornith-1.0-35B | ROCmFP4-STRIX_LEAN | 66.68 | 17.32 GiB |
| Qwen3.6-27B | ROCmFP4-STRIX_LEAN | 13.68 (32.7 con MTP) | 13.82 GiB |

Nemotron ROCmFP4 è il **più veloce mai testato** su Strix Halo (+19% vs grug, il precedente record).
