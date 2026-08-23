# ROCmFP3 (Q3_0_ROCMFPX) on Qwen3.8-27B: not a true 3-bit on hybrid gated-deltanet architectures

**Research note — August 2026 (thread T5).** How much quality do we lose going from
the production ROCmFP4-STRIX_LEAN preset (4.38 bpw) down to ROCmFP3 — and do the
(supposedly) fewer weight bytes buy decode speed back? On this model the answer is
structural before it is empirical: the quant routing protects all attention tensors
with K-quants, and Qwen3.8's 48 gated-deltanet layers carry a *fused* `attn_qkv` —
the most numerous attention tensor in the model — so the "3.50 bpw nominal" preset
lands at 4.44/5.72 **effective** bpw. Both arms are *larger* than the FP4 baseline.
There were no bytes to save; the measurements below close the thread anyway.

## 1. The question

T5 asked two things. **Quality:** how much ppl is lost moving from the production
`Q4_0_ROCMFP4_STRIX_LEAN` (4.38 bpw) to the Q3_0_ROCMFPX preset? **Speed:** does the
smaller preset pay back in tg? The motivation was bandwidth arithmetic: at 24k ctx on
this dense 27B, weights are ≈ 90% of the bandwidth budget, so a genuine ~20% cut in
weight bytes would translate almost linearly into decode time.

## 2. Method

- **Arms** — `Q3_0_ROCMFPX` (base) and `Q3_0_ROCMFPX_AGENT`, both quantized in the
  convert container from the same local BF16 GGUF and the same self-produced
  496-entry imatrix as the 18/08 preset comparison
  ([results-2026-08-18-rocmfp4-full-vs-strix-lean.md](../benchmarks/results-2026-08-18-rocmfp4-full-vs-strix-lean.md)).
- **Quality gate** — `llama-perplexity` on wikitext-2-en (150k tokens) + Italian
  technical (51.5k), `-c 512 -b 512 -fa on`; BF16 references 6.6409 / 11.7156. The
  LEAN control was re-run in the same session and reproduced within sigma.
- **Backend for ppl** — Vulkan: a fork-image gap (the older dflash image lacks GGML
  type 104) kept the ROCm-side binary from reading the new types.
- **Speed** — `llama-bench` tg128 / pp512, single session (`-r 5`), vs the LEAN
  control run in the same session.
- **Hardware** — AMD Strix Halo: Radeon 8060S iGPU (gfx1151, RDNA 3.5), 128 GB
  unified LPDDR5X, RADV.

## 3. Results

| Arm | eff. bpw | Size | PPL en (Δ vs BF16) | PPL it (Δ) | tg128 (Δ vs LEAN) | pp512 (Δ) |
|---|---|---|---|---|---|---|
| STRIX_LEAN baseline | 4.38 | 13.82 GiB | 6.8226 (+2.74%) | 12.1168 (+3.42%) | 13.06 ± 1.34 | 340.96 ± 6.09 |
| Q3_0_ROCMFPX base | 4.44 | 14.125 GiB | 6.9943 (+5.32%) | 12.3121 (+5.09%) | 11.02 ± 1.47 (−15.6%) | 264.89 ± 5.16 (−22.3%) |
| Q3_0_ROCMFPX_AGENT | 5.72 | 18.198 GiB | 6.7665 (+1.89%) | 11.9541 (+2.04%) | 8.92 ± 1.01 (−31.7%) | 298.74 ± 10.63 (−12.4%) |

## 4. Findings

1. **Structural — no bytes to save.** The fork's routing protects **all** attention
   tensors with K-quants, and qwen35's 48 gated-deltanet layers carry FUSED
   `attn_qkv` — the most numerous attention tensor — so the 3-bit core covers a
   smaller share of the weights than on a vanilla transformer. The "3.50 bpw
   nominal" preset lands at 4.44 (base) / 5.72 (agent) effective bpw, both above
   the LEAN's 4.38. The premise of the thread — fewer weight bytes — falls before
   any benchmark runs.
2. **Speed — the K-quant path is a tax, not a saving.** Base tg128 is −15.6% at a
   size ratio of 1.022 (14.125/13.82 GiB): with bytes essentially equal, the whole
   loss is the cost of running the K-quant-protected tensors on their path instead
   of the custom ROCmFPX kernels. The agent arm is the inverse size ratio (1.317)
   plus that same path cost: −31.7%. **pp512 anomaly:** agent (−12.4%) beats base
   (−22.3%) despite being 29% larger, suggesting the Q6_0_ROCMFPX MMQ path is more
   prefill-efficient than the Q3_0_ROCMFPX+Q5_K mix (single `-r 5` run — do not
   generalize).
3. **Quality matches EFFECTIVE bpw, not the nominal preset.** Agent (+1.89/+2.04%)
   sits at the level of the 18/08 EVEN preset (+1.90/+2.61% at 4.55 bpw); base
   (+5.32/+5.09%) is clearly below FAST_EVEN (+3.10/+3.76% at 4.30). At matched
   effective bpw the FP4 presets dominate the Q3 mix on both corpora — the 3-bit
   core buys nothing the FP4 dual-scale layout does not buy cheaper.

## 5. Verdict — NO-GO for production

The base arm is **dominated by STRIX_LEAN on every axis** (larger, worse ppl on
both corpora, slower on both metrics). The agent arm trades 31.7% tg128 and 31.7%
size for 0.85–1.38 pt of ppl — worse quality-per-byte and quality-per-second than
the existing FP4 options (LEAN, EVEN). Thread closed. The planned multi-ctx MTP
step became moot: tg128 at short context — exactly where a weight-byte advantage
would peak — is already −15.6%.

## 6. Caveats

ppl was measured on the **Vulkan** backend (fork-image gap, see method); the LEAN
control reproduced within sigma (+0.34%/+0.49% offset vs the 18/08 comparison), so
verdicts are unchanged at backend parity. llama-bench numbers are a single session.

## Pointers

- Weights + full model card: <https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q3_0_ROCMFPX>
- Imatrix: [pugant/Qwen3.8-27B-imatrix](https://huggingface.co/pugant/Qwen3.8-27B-imatrix)
- FP4 preset comparison this note leans on:
  [`docs/benchmarks/results-2026-08-18-rocmfp4-full-vs-strix-lean.md`](../benchmarks/results-2026-08-18-rocmfp4-full-vs-strix-lean.md)
- Full experiment protocol (Italian): lab workspace records.

---

*Negative results are results: the weights are published so nobody (ourselves
included) has to re-derive them.*
