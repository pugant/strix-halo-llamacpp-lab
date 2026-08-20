# Benchmarks grug-35b-v2 ROCmFP4-STRIX_LEAN (2026-08-11)

## Risultati finali — Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X)

| Metrica | ROCmFP4-STRIX_LEAN | Q4_K_M (baseline) | Δ |
|---|---|---|---|
| Size | **17.31 GiB** (18.60 GB decimali) | 19.70 GiB | **-12%** ⭐ |
| Plain tg128 | **70.92 ± 0.66 tok/s** | 61.18 ± 0.57 tok/s | **+16%** ⭐ |
| Plain pp512 | **1418.50 ± 10.87 tok/s** | 1370.25 ± 13.76 tok/s | +3.5% |
| BPW | 4.29 | ~4.8 (Q4_K_M media) | — |

**Verdetto**: ROCmFP4-STRIX_LEAN domina il Q4_K_M su size (-12%) e tg128 (+16%). Confronto con Qwen3.6-35B-A3B produzione (63 tok/s): grug-ROCmFP4 fa **+12%** (70.92 vs 63 tok/s).

## Setup runtime
- Container: `docker-llm-service` (fork ROCmFPX `charlie12345/ROCmFPX@00d5452`, ROCm 7.2.4, gfx1151)
- Hardware: AMD Ryzen AI Max+ 395 / Radeon 8060S, 128 GB unified LPDDR5X
- Comando: `llama-bench -m <model>.gguf -p 512 -n 128 -ngl 999 -fa on --mmap 0`

## Smoke test (sidecar porta 8083, `-c 32768`, `--reasoning on --reasoning-budget 8192`)

- **Coding Fibonacci iterativo**: ✅ PASS — `def fibonacci(n): ...` sintatticamente corretto, no spam, 70.66 tok/s
- **Tool-call JSON (Roma + Milano)**: ✅ PASS — 2 tool-call distinti `get_weather` con arguments JSON ben formati (`{"city":"Roma"}`, `{"city":"Milano"}`), 70.79 tok/s
- **Single-token spam**: ✅ assente — conversazioni normali, niente loop di `/` (problema noto DeltaNet su context-shift evitato)

## Audit
- **SHA256 output**: `3012eb6c3483f88130750ba015a7e31708220e901f2429a76f67620ef483782b`
- **Path output**: `~/llmodels/models/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf`
- **Tensore totali**: 733
- **Quant time**: 4 min 18 sec (16 threads, MoE 40 layer)
- **Imatrix gen time**: 13 min 44 sec (256 chunks, 16 threads CPU-only)

## Comando serving produzione (spec §9, validato in smoke test)
```bash
llama-server -m /llmodels/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf \
  --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234 \
  --reasoning on --reasoning-budget 8192 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

(Senza `--spec-type draft-mtp`: grug non ha MTP, e allineamento a pipeline MoE plain.)

## Dettagli quantizzazione
- **Preset**: `Q4_0_ROCMFP4_STRIX_LEAN` (type 106)
- **BPW finale**: 4.29
- **Protezione**: attention K/V (`attn_qkv`/`attn_v` → `q4_0_rocmfp4`) + Q5_K token embeddings
- **Expert** (`ffn_*_exps`, 256 expert MoE): `q4_0_rocmfp4_fast` (max speed)
- **Shared expert** (`ffn_*_shexp`): `q4_0_rocmfp4_fast`
- **Architettura riconosciuta**: `qwen35moe` (Qwen3.5-VL-MoE con Gated DeltaNet + attention ibrida)

## Imatrix (calibrata su ProCreations/grug-think-v3-10k)
- **Chat template**: applicato con jinja2 puro, `normalize_messages()` per convertire `tool_calls.arguments` da stringa JSON a dict (mismatch dataset↔template Qwen3.5)
- **Sample**: 250 conversazioni random seed=42 su 10000
- **Calibration size**: 4.95 MB plain text (best practice 1-5 MB)
- **Diversità fonti**: 6/6 fonti rappresentate (smith_tool 61, smith_ticks 55, nebius 47, toolace 37, hermes 31, glaive 19)
- **Render fallback**: 0/250 (tutte renderizzate col template corretto)

## Lezioni apprese
1. **gguf-py custom obbligatorio**: il converter del fork ROCmFPX richiede `MODEL_ARCH.DFLASH` (non presente in gguf-py vanilla PyPI né nel commit `00d5452`, ma nel main HEAD). La derivata `docker-llm-service-convert` deve installare gguf-py dal main HEAD del fork.
2. **MoE Qwen3.5 = MoE 35B-A3B-class**: 3B params attivi → ceiling LPDDR5X ~180 tok/s. Plain speed 70 tok/s senza MTP, simile al Qwen3.6-35B-A3B produzione (63 tok/s).
3. **Download parallelo**: BF16 (18 shard) seriale cronometrato a 0.8 MB/s; 4 shard paralleli a ~20 MB/s totali (×7 accelerazione). `wget -c` preserva progressi.
4. **hf_xet stuck** (problema già noto dalle note interne): confermato su shard 3.9 GB, risolto con `wget -c`.
5. **Verify before claim**: in plan review iter 2, avevo dichiarato "verificato ✓" senza testare; reviewer adversarial lo ha beccato. Ho consolidato la disciplina del test empirico.

## Follow-up
- **Pubblicazione HF**: spec dedicata in `docs/design/2026-08-11-hf-publish-grug-ornith-design.md`
- **FASE 2 Ornith**: completata (modello in `~/llmodels/models/ORNITH/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf`, 17.32 GiB)
- **MTP su Ornith**: non attivato per allineamento a pipeline MoE plain. Test futuro: attivare `--spec-type draft-mtp` per misurare boost (layer `blk.40` presente).
