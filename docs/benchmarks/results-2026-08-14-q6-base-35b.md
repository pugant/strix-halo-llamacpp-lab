# Results — Qwen3.6-35B-A3B base → Q6_0_ROCMFPX (08/14 evening pipeline)

**Date:** 2026-08-14 · exclusive GPU · `llama-bench -ngl 999 -fa 1 -p 512 -n 128`
Bench image: `docker-llm-service:vulkan-fork` (fork charlie12345/ROCmFPX, build b2f5829,
backend **Vulkan RADV**) · BF16 source: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (shards verified
byte-per-byte vs HF) · Quantize: `docker-llm-service:latest`, preset `Q6_0_ROCMFPX` (type 110,
6.50 bpw) **WITH imatrix** (accepted), 178 s, 753 tensors (nextn/MTP included).

## The new file

| Metric | Value |
|---|---|
| Output | `Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX.gguf` — **27.39 GiB** (6.62 effective BPW) |
| tg128 Vulkan RADV | **49.16 ± 0.36** tok/s |
| pp512 Vulkan RADV | **1120.38 ± 10.36** tok/s |
| SHA256 (sanitized) | `2251b28cd9f3e2edaa37d91fff3ae3172acc9a8f91eaaf4f167254e22295caca` |

Sanity check: the Qwopus Q6 (finetune, same arch/quant, 27.38 GiB) gave 49.48/1076.91 →
the base Q6 is perfectly in line (±1% tg, +4% pp).

## Comparative table — all Qwen3.6-35B-A3B versions tested (14/08, same methodology)

| Version | Quant | Size | Backend | pp512 | tg128 |
|---|---|---:|---|---:|---:|
| **base (this pipeline)** | **Q6_0_ROCMFPX** | **27.39 GiB** | **Vulkan RADV** | **1120.38** | **49.16** |
| Qwopus3.6-35B (abliterated finetune) | Q6_0_ROCMFPX | 27.38 GiB | Vulkan RADV | 1076.91 | 49.48 |
| Qwopus3.6-35B (abliterated finetune) | Q6_0_ROCMFPX | 27.38 GiB | ROCm | 520.56 ⚠️ | 51.29 |
| base | Q4_0_ROCMFP4_STRIX_LEAN | 17.73 GiB | Vulkan RADV (fork) | 1164.67 | **81.57** |
| base | Q4_0_ROCMFP4_STRIX_LEAN | 17.73 GiB | ROCm | **1420.72** | 71.23 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | Vulkan RADV | 1008.09 | 57.93 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | ROCm | 1359.39 | 50.76 |
| base | UD-Q5_K_M_MTP (unsloth) | 25.22 GiB | Vulkan AMDVLK | 663.02 | 55.84 |

⚠️ pp 520 on ROCm: the Q6 staging is not optimized on the ROCm 00d5452 build — for Q6 the
reference backend is **Vulkan** (which is why the publish declares the RADV numbers).

### Interpretation

- The base Q6 is the **quality tier** (~Q6_K-class): it costs -40% tg vs fp4_LEAN on Vulkan
  (81.57→49.16) and -15% vs UD Q5_K_M (57.93→49.16), against 6.62 BPW.
- The Q6's pp (1120) is higher than the Q5_K_M's (1008) on RADV: on the same backend, the fork's
  Q6 routing will cost less on prefill.
- The Q6 with MTP (validated config n-max 4): prose ~57.9 (+17%), deterministic ~73.7 (+49%)
  [server-timing, measured on Qwopus Q6 — same arch/quant].

## MTP n-max sweep on the BASE Q6 (2026-08-15, plan 2026-08-15-q6-base-mtp-nmax-test.md — not included in this repo)

Server-timing, 2 prompts × 2 runs, ctx 16k, Vulkan RADV (vulkan-fork b2f5829), script
`scripts/mtp-nmax-test.sh` (not included in this repo). Flags: `--spec-type draft-mtp --spec-draft-ngl all
--spec-draft-p-min 0.0 --spec-draft-p-split 0.10` + n-max varied.

| n-max | P1 prose (2 runs, average) | P2 det (average) | acceptance/pos (last task) |
|---|---:|---:|---|
| 4 | 44.7 (46.6/42.8) | 62.8 | (0.869, 0.756, 0.606, 0.512) |
| 3 | 54.0 (55.1/52.9) | **67.3** | det (0.874, 0.769, 0.643) |
| **2** | **58.2** (56.5/59.8) | 62.4 | det (0.822, 0.660) |

**Conclusion: on the BASE, n-max 4 is dominated.** pos-4 accepts only ~0.51 and the draft
overhead is not repaid: prose -23% vs n-max 2. Optimal for mixed context (coding agent):
**n-max 3** (det 67.3, prose 54.0); for pure prose decode: n-max 2 (58.2). Different from
Qwopus (n-max 4 → 57.9 prose on 14/08): the acceptance distribution depends on the finetune.
Note: run-to-run prose variance ±3 tok/s (default sampling temp 1.0) — differences >8 tok/s
are signal, below that is noise.

## Pipeline notes

- Download: 66.2 GiB BF16 via wget -c (~12-27 MB/s). **Incident avoided**: double concurrent
  wget on shard2 (detached script from the previous session + this session's task) → killed
  both, deleted the partial shard2, single clean download. Final byte-per-byte verification
  against `repo_info(files_metadata=True)`.
- Metadata sanitize: run in the `docker-llm-service-convert` container (ROCmFPX fork gguf-py;
  the host python3.12 with llama.cpp-pflash does NOT know types 102/110 → ValueError). Config
  `scripts/configs/qwen36-35b-q6-metadata.json` (not included in this repo).
- Post-sanitize smoke: Vulkan load OK (pp16 202.88 / tg8 48.27) ✓.
