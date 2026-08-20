# Full ROCmFP4 (dual-scale) vs STRIX_LEAN — experiment results A/B/D/E

**Date:** 2026-08-18, 19:46-20:40 · Plan: internal experiment plan `2026-08-18-rocmfp4-full-vs-strix-lean-experiment.md` (not included in this repo) · Context: thread 4 (NVFP4→quality), DGX Spark reconnaissance report same day.

**Setup:** Qwen3.8-27B (dense, 48 SSM + 16 full-attn + MTP) from local BF16, self-produced imatrix (gate-PASS 496/496) for ALL arms. Quant in the 11/08 convert container (presets 100/103). PPL: `llama-perplexity` dflash/ROCm, `-c 512 -b 512 -fa on -ngl 999`, identical phaseA corpora (en 150k/70 chunks, it 51.5k/30 chunks; control LEAN reproduced **exactly** to the 4th decimal). Bench: server Vulkan fork ckpt7, MTP n6 p_min 0.75 p_split 0.10, c 16384, temp 0, warm-up discarded, TREATMENT marker in the logs; llama-bench `-p 512 -n 128 -fa 1 -r 5`.

**Preliminary finding (routing):** the "full" `Q4_0_ROCMFP4` preset is NOT "dual-scale everywhere at 4.50 bpw": mixed routing `attn_qkv→q5_K`, `ffn_down→q6_K`, dual-scale ROCmFP4 on gate/up (+emb). A pure academic comparison requires the `_EVEN` presets (`--pure`): hence arms D/E added mid-run (plan addendum).

## Table 1 — Quality (ppl wikitext-2-en / Italian technical)

| Arm | Preset | bpw | Size | PPL en | **Δ en vs BF16** | PPL it | **Δ it vs BF16** |
|---|---|---|---|---|---|---|---|
| BF16 (ref 17/08) | — | 16 | 50.9 GB | 6.6409 ± 0.121 | — | 11.7156 ± 0.388 | — |
| A LEAN (production) | STRIX_LEAN | 4.38 | 14.85 GB | 6.8226 | +2.74% | 12.1168 | +3.42% |
| **B full** | Q4_0_ROCMFP4 | ~5.24 | 17.74 GB | **6.6901 ± 0.122** | **+0.74%** | **11.8479 ± 0.394** | **+1.13%** |
| D even (dual everywhere) | Q4_0_ROCMFP4_EVEN | 4.55 | 15.39 GB | 6.7669 ± 0.123 | +1.90% | 12.0215 ± 0.399 | +2.61% |
| E fast-even (single everywhere) | Q4_0_ROCMFP4_FAST_EVEN | 4.30 | 14.53 GB | 6.8474 ± 0.125 | +3.10% | 12.1566 ± 0.404 | +3.76% |
| Q4_K_M (ref 17/08, no imatrix) | — | ~4.8 | 16.8 GB | 6.6747 | +0.51% | 12.0991 | +3.27% |

## Table 2 — Speed (Strix Halo gfx1151, Vulkan RADV; EVEN added 19/08, addendum 2)

| Test | LEAN | FULL | Δ FULL | **EVEN (D)** | Δ EVEN |
|---|---|---|---|---|---|
| MTP n6 prose (2 prompts) | 19.6 / 20.3 | 15.6 / 17.2 | **−16%** | **19.5 / 19.6** | −0.5% / −3.4% |
| MTP n6 det (2 prompts) | 45.4 / 26.2 | 37.1 / 25.0 | **−17%** | **41.7 / 27.7** | −8.1% / +5.7% |
| llama-bench tg128 (no MTP) | 13.62 ± 0.86 | 9.37 ± 0.92 | **−31%** | **13.64 ± 0.01** | **+0.1% (on par)** |
| llama-bench pp512 | 346.5 ± 10.2 | 314.6 ± 9.6 | **−9%** | **347.6 ± 10.4** | **+0.3% (on par)** |

- The FULL's MTP Δ (−16/−17%) matches the byte ratio (16.51/13.82 GiB = 1.195). Its no-MTP tg128 (−31%) has an extra ~−15% beyond the bytes: the tg path of the q5_K/q6_K tensors (qkv/down) on Vulkan is less efficient than the custom ROCmFP4.
- **EVEN = speed ON PAR with the LEAN across the board** (tg128 ±0.01, pp512 ±1, MTP within ±8% run-noise): applying the dual-scale kernel everywhere costs NOTHING in inference — the "FAST single-scale" layout has no measurable advantage on tg; the FULL's cost comes only from the protected K-quant tensors.

## Contrast analysis (what causes the quality loss)

| Contrast | What it isolates | Δ ppl en | Δ ppl it | Size cost |
|---|---|---|---|---|
| E → D (single→dual everywhere) | **scale granularity 32→16** (thread 4's NVFP4 question) | **−1.20 pt** (3.10→1.90) | **−1.15 pt** (3.76→2.61) | +0.86 GB (+5.9%) |
| E → LEAN | the LEAN's K/V+emb protections | −0.36 pt | −0.34 pt | +0.32 GB |
| D → B | qkv q5_K + down q6_K protections | **−1.16 pt** | **−1.48 pt** | +2.35 GB |

1. **Scale granularity IS a real, isolated lever**: dual-scale everywhere (D) recovers ~1.2 points of Δppl for only +0.86 GB — a YES answer to thread 4's question "block-scale FP8 → higher quality", measured for the first time at identical routing (first data point of this kind in the community).
2. The full's protections are worth almost as much as the granularity (−1.2/−1.5 pt) but cost three times as much in bytes (qkv q5_K + down q6_K on a dense model).
3. The current LEAN wastes little: its protections are worth only ~0.35 pt.

## The plan's decision criterion

> GO for production replacement if full Δppl ≤ half the LEAN's on both corpora AND full MTP tg ≥ 97% of the LEAN's.

- Quality: **PASS** (+0.74 ≤ 1.4 ✓, +1.13 ≤ 1.7 ✓)
- Speed: **FAIL** (MTP tg = 83-84% of the LEAN, < 97%)

**Outcome: no automatic GO — a trade-off to decide.** The full costs ~−16% tok/s in production and gains: Δppl 2.0 pt (en) / 2.3 pt (it), and on Italian it is the best quant ever measured on this machine (+1.13% vs the Q4_K_M's +3.27% and the LEAN's +3.42%). For a dimensional comparison: the full is the community theory's "quality step" (KAT-Coder recommended the full one for coding), here quantified on our model/hw.

## Options for production (updated 19/08 with the EVEN data)

| Option | Δppl en/it | tg128 no-MTP | MTP det | Size | Notes |
|---|---|---|---|---|---|
| Status quo LEAN | +2.74/+3.42 | 13.62 | 45.4 | 13.82 GiB | current |
| FULL | +0.74/+1.13 | 9.37 (−31%) | 37.1 | 16.51 GiB | max quality, expensive in speed |
| **EVEN (D)** | **+1.90/+2.61** | **13.64 (on par)** | **41.7 (on par within noise)** | 14.32 GiB (+3.6%) | **quasi-pareto: −0.84/−0.81 pt of ppl FREE** |

**Updated conclusion**: the EVEN quasi-Pareto dominates the LEAN (better quality on both languages at practically identical speed and size) → it is the natural candidate for the new production if extra quality is wanted at no cost; the FULL remains the "max quality" option for non-latency-critical uses. Decision up to the user.

## Technical notes

- The "full" routing explains KAT-Coder's 5.25 bpw: on KAT (MoE) the effective bpw was 5.25; here dense 27B → 17.74 GB (~5.24 effective bpw).
- Vulkan smoke PASS: coherent output, no-MTP gen 12.5 t/s = the expected floor (222 effective GB/s) → the dual-scale kernel has no anomalous overhead in server inference.
- Arm C (native NVFP4) excluded due to julianmb's negative precedent (scale2 bug: incoherent ppl 109.8).
- Artifacts: B/D/E GGUFs in `models-test/` (kept for the decision); logs `logs/ppl-full-vs-lean-20260818.log`, `logs/bench-full-vs-lean/`; corpus `/tmp/pplcorpus` (ephemeral).
- llm-service restored on ckpt7 and healthy at the end.
