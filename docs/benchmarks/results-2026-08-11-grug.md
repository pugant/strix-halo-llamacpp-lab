# Benchmarks grug-35b-v2 ROCmFP4-STRIX_LEAN (2026-08-11)

## Final results — Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X)

| Metric | ROCmFP4-STRIX_LEAN | Q4_K_M (baseline) | Δ |
|---|---|---|---|
| Size | **17.31 GiB** (18.60 GB decimal) | 19.70 GiB | **-12%** ⭐ |
| Plain tg128 | **70.92 ± 0.66 tok/s** | 61.18 ± 0.57 tok/s | **+16%** ⭐ |
| Plain pp512 | **1418.50 ± 10.87 tok/s** | 1370.25 ± 13.76 tok/s | +3.5% |
| BPW | 4.29 | ~4.8 (Q4_K_M average) | — |

**Verdict**: ROCmFP4-STRIX_LEAN dominates Q4_K_M on size (-12%) and tg128 (+16%). Compared to the production Qwen3.6-35B-A3B (63 tok/s): grug-ROCmFP4 is **+12%** (70.92 vs 63 tok/s).

## Runtime setup
- Container: `docker-llm-service` (ROCmFPX fork `charlie12345/ROCmFPX@00d5452`, ROCm 7.2.4, gfx1151)
- Hardware: AMD Ryzen AI Max+ 395 / Radeon 8060S, 128 GB unified LPDDR5X
- Command: `llama-bench -m <model>.gguf -p 512 -n 128 -ngl 999 -fa on --mmap 0`

## Smoke test (sidecar port 8083, `-c 32768`, `--reasoning on --reasoning-budget 8192`)

- **Iterative Fibonacci coding**: ✅ PASS — `def fibonacci(n): ...` syntactically correct, no spam, 70.66 tok/s
- **Tool-call JSON (Roma + Milano)**: ✅ PASS — 2 distinct `get_weather` tool-calls with well-formed JSON arguments (`{"city":"Roma"}`, `{"city":"Milano"}`), 70.79 tok/s
- **Single-token spam**: ✅ absent — normal conversations, no `/` loops (known DeltaNet context-shift issue avoided)

## Audit
- **SHA256 output**: `3012eb6c3483f88130750ba015a7e31708220e901f2429a76f67620ef483782b`
- **Output path**: `~/llmodels/models/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf`
- **Total tensors**: 733
- **Quant time**: 4 min 18 sec (16 threads, 40 MoE layers)
- **Imatrix gen time**: 13 min 44 sec (256 chunks, 16 threads CPU-only)

## Production serving command (spec §9, validated in smoke test)
```bash
llama-server -m /llmodels/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf \
  --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234 \
  --reasoning on --reasoning-budget 8192 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

(Without `--spec-type draft-mtp`: grug has no MTP, and this stays aligned with the plain MoE pipeline.)

## Quantization details
- **Preset**: `Q4_0_ROCMFP4_STRIX_LEAN` (type 106)
- **Final BPW**: 4.29
- **Protection**: attention K/V (`attn_qkv`/`attn_v` → `q4_0_rocmfp4`) + Q5_K token embeddings
- **Experts** (`ffn_*_exps`, 256 MoE experts): `q4_0_rocmfp4_fast` (max speed)
- **Shared expert** (`ffn_*_shexp`): `q4_0_rocmfp4_fast`
- **Recognized architecture**: `qwen35moe` (Qwen3.5-VL-MoE with Gated DeltaNet + hybrid attention)

## Imatrix (calibrated on ProCreations/grug-think-v3-10k)
- **Chat template**: applied with pure jinja2, `normalize_messages()` to convert `tool_calls.arguments` from a JSON string to a dict (dataset↔Qwen3.5 template mismatch)
- **Sample**: 250 random conversations seed=42 out of 10000
- **Calibration size**: 4.95 MB plain text (best practice 1-5 MB)
- **Source diversity**: 6/6 sources represented (smith_tool 61, smith_ticks 55, nebius 47, toolace 37, hermes 31, glaive 19)
- **Render fallback**: 0/250 (all rendered with the correct template)

## Lessons learned
1. **Custom gguf-py mandatory**: the ROCmFPX fork's converter requires `MODEL_ARCH.DFLASH` (not present in vanilla PyPI gguf-py nor in commit `00d5452`, but in main HEAD). The `docker-llm-service-convert` derivative must install gguf-py from the fork's main HEAD.
2. **Qwen3.5 MoE = 35B-A3B-class MoE**: 3B active params → LPDDR5X ceiling ~180 tok/s. Plain speed 70 tok/s without MTP, similar to the production Qwen3.6-35B-A3B (63 tok/s).
3. **Parallel download**: serial BF16 (18 shards) timed at 0.8 MB/s; 4 parallel shards at ~20 MB/s total (×7 speedup). `wget -c` preserves progress.
4. **hf_xet stuck** (issue already known from internal notes): confirmed on a 3.9 GB shard, fixed with `wget -c`.
5. **Verify before claim**: in plan review iter 2, I had declared "verified ✓" without testing; an adversarial reviewer caught it. I consolidated the discipline of empirical testing.

## Follow-up
- **HF publication**: dedicated spec in `docs/design/2026-08-11-hf-publish-grug-ornith-design.md`
- **PHASE 2 Ornith**: completed (model at `~/llmodels/models/ORNITH/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf`, 17.32 GiB)
- **MTP on Ornith**: not enabled, to stay aligned with the plain MoE pipeline. Future test: enable `--spec-type draft-mtp` to measure the boost (`blk.40` layer present).
