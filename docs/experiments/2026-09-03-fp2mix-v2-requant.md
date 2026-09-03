# FP2MIX v2: the ROCmFP2 build re-quantized (PLE restored, trunk upgraded)

2026-09-03. The 2-bit FP2MIX build of Qwen3.8-Flash-Next had one broken piece and one
honest compromise: the PLE n-gram table at Q2 had lost the recall that makes the
architecture work (Dante 8k ppl 37.90 vs the LEAN's 1.18), and its dense trunk was
2-bit throughout. The v2 rebuild fixes the first and upgrades the second — same disk
footprint class, same RAM boundary, validated end to end on the lab's
[bare-metal Strix Halo](../../BARE-METAL.md).

## The recipe (372 of 1224 tensors changed)

- `per_layer_token_embd` → **Q5_1**, served disk-resident via `--ple-disk` (35.76 GiB
  never loaded — blocks are read on demand from the GGUF itself; the v1/T25 path)
- 358 dense Q2 tensors → **Q4_0_ROCMFP4** (all attention/projection/indexer/PLE-projection
  weights; the `ssm_*` band stays 2-bit)
- 13 `ffn_*_exps` tensors in blocks {40,41,42,44,46} → **Q3_0_ROCMFPX** (+1.89 GiB surgery)

File: 75.87 GiB (was 53.12). Type counts: FP2 235 · FP4 582 · Q3 13 · Q5_1 1, plus the
untouched F32/F16/Q5_K/Q6_K/FAST tensors — 1224 total, byte-exact against the recipe.

## Gates (all measured, same protocols as every note here)

| Gate | Result | Reference |
|---|---|---|
| Build (size + type counts) | 75.87 GiB, 9/9 counts exact | — |
| ppl EN (`qwen38-calibration`, 8k) | **3.2689 ± 0.015** | v1 7.2928; LEAN ≈ 3.36 |
| ppl Dante 8k (holdout, tracking) | **2.0676 ± 0.011** | v1 37.9027 (×18.3 worse); LEAN 1.1776 |
| tg128, same-harness llama-bench (Vulkan, `--ple-disk`) | **32.18 ± 0.46 t/s** = 0.945 × v1 | v1 34.05 ± 0.38 |
| RAM boundary (ctx 131072, MTP n6, mmproj, KV q5_1) | **55.94 GiB resident** (Δfree method), 0 OOM, PLE marker present | gate ≤ 58 GiB |

Reading: the v1's damage was not quantization noise — it was PLE recall loss. Q5_1
restores it fully (Dante ×18.3), and the FP4/Q3 trunk costs −5.5% tg128 while bringing
EN ppl within ~2% of the current LEAN's (3.27 vs 3.1961; below the pre-09-02 LEAN's 3.3601). A 76 GiB file that holds ~56 GiB of RAM: the delta is
the disk-resident PLE.

## Where it runs

Production stays on the ROCmFP4 LEAN (operator choice — the v2 is a published artifact
for the FP2MIX line, not the lab's serving build). The v2 ships in the model repo under
`ROCmFP2-STRIX_LEAN/` replacing the v1 shards in place: if you run the 2-bit build,
**re-download both shards** — the filenames are unchanged, the bytes are not. Serve it
with `--ple-disk --ple-cache-mib 4096 --no-mmap`; do not be alarmed by the file size —
with the n-gram offload the RAM boundary is the one measured above, not the file size.

## Artifacts

- Mapping script with self-test and size guard: `rocmfpx/` snapshot does not carry it
  (workspace tooling); the tt-file recipe is fully described above.
- Raw logs and the exact gate protocols live in the lab's working tree under
  `logs/t28/` (quantize 18.1 min, 0 silent fallbacks; ppl runs; boundary sampling).

---
*Thread index: [`README.md`](README.md). Related: the FP2MIX-64GB honest-frontier note
(T27), [BARE-METAL.md](../../BARE-METAL.md) for the machine behind every number.*
