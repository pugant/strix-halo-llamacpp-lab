# Round budget: the speculative round decomposed to the millisecond, and the 3 bpw door closed

**Research note — August 2026 (thread T10).** Companion to
[per-round-drafter-switching.md](per-round-drafter-switching.md) (T9, which closed per-round
drafter switching) and to [2026-08-23-rocmfp3-quality-speed.md](2026-08-23-rocmfp3-quality-speed.md)
(T5, the other point on the compression quality curve): this one asks how much decode throughput
is still recoverable on the production quant — by compressing weights towards ~3 bpw, or by
kernel work on the speculative round — and closes the line by answering, mechanically and to the
millisecond, where the speculative round actually spends its time.

Everything was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU,
gfx1151, 128 GB unified LPDDR5X), both backends of the fork (Vulkan/RADV and HIP/ROCm images),
target model `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB; arch `qwen35`, 17 full-attention +
48 gated-deltanet layers), temp 0, ctx window 16384 (`-c 16384 -fa on --jinja`), p_min 0.75,
parallel 1, single stream, dedicated GPU window, warm-up discarded (1 warm-up + 5 replicas per
cell — the section 9 sweep uses 3; the long-context arm decodes at a real KV of ~13.8k tokens, the
short arms at 130-200 tokens).

**TL;DR**

- **The question.** Two levers for tg on the LEAN quant: weight compression to ~3 bpw (T2) and
  kernel work on the speculative round (T1/T3). The success criterion was twofold — +10% tg
  measured, OR a physical ceiling documented with quantified margins. The second branch is the one
  that happened, realized in full. No production changes.
- **A contradiction resolved.** The historical "+53% ROCm over Vulkan for dense models" (Aug 15)
  does not reproduce under the unified protocol: plain ROCm 13.70/13.69 vs Vulkan 14.06/14.01
  tok/s amounts to parity (−2.6%/−2.3%) — a protocol/build artifact of its day. Backend choice
  rides on the speculative arms, where Vulkan wins by +23%/+29% (det/prose).
- **The plain path is already near-optimal.** Direct microbenchmarks: 236.6 GB/s pure read,
  208.5 GB/s device-to-device copy; plain decode runs at 87-89% of the read roof. The recoverable
  margin is not in the plain round — it is in the speculative round.
- **The central deliverable.** The speculative round is decomposed mechanically and the accounting
  identity closes: 136.43 ms accounted against 136.42 measured (mtp6-det ROCm). Structural:
  weights+KV+SSM floor 63.45, draft-chain floor 12.25, verify per-row KV-read 16.14, GDN RS
  snapshots 7.58, conv-rollback 0.42. Software: head-call 12.30, verify 14.46, plain residual 9.83.
- **"Logits bandwidth" is falsified in the code.** On Vulkan, with ≤16 columns every target
  mul_mat takes the VEC pipeline and the shader shares each dequantized weight across all batch
  columns: the verify batch does not re-read the 0.675 GB output head per row. The residual verify
  overhead is kernel efficiency at width 5-7 columns, not bandwidth. On HIP the MMVQ threshold
  is 8 rows.
- **n6 confirmed optimal.** det n8 is −16.2% tok/s; positions beyond 6 accept ~0.5 or less; on
  prose, p_min truncates the chain regardless of n_max — rising n is pure cost.
- **Compression is NO-GO on perplexity.** MIX at 3.01 bpw costs +27.04%/+31.62% ppl (en/it), V2
  at 3.10 bpw +27.19%/+30.99%, against a gate of ≤ +5.48%/+6.84% — roughly 5×/4.5× over. The
  kernels are healthy (uniform per-chunk degradation, no NaN): quantization noise, not a broken
  kernel. The ~3 bpw class is not viable on this dense model.
- **The ceiling ladder.** 43.0 tok/s measured → ~48.6 with the verify software recovered → 60.6
  with all software → 69.8-79.2 pure-bandwidth ideal. The one concrete intervention (batched
  dispatch for 2-9 columns) has a best case of +3.4 tok/s — exactly the 2σ gate — so its expected
  value is below the gate. Discarded.

## 1. The question and the outcome

T10 asked: on `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` running MTP n6, how much decode throughput is
still recoverable, and by which lever — (a) compressing the weights towards ~3 bpw to cut the
bandwidth floor (T2), or (b) kernel work on the speculative round (T1/T3)? The pre-registered
success criterion was twofold: **"+10% tg measured on a standard workload, OR a physical ceiling
documented with quantified margins."** The outcome is the second branch, realized in full; the
first branch failed by two independent routes:

- The compression lever — worth a computed +16-24% tg (F4 §2.2) — died at the quality gate: both
  iterations of the ~3 bpw candidate fail the perplexity gate by ~5×/4.5× (section 7). NO-GO.
- The kernel lever was evaluated to completion (F6 §3) and closed as documentation-only: the best
  case of the only concrete candidate is +3.4 tok/s, exactly the 2σ gate — expected value below it
  (section 9).

The production configuration is confirmed unchanged: MTP n6 on the Vulkan backend (F0 §5, F2 §4).
Intermediate gates: G-F5-ppl FAIL 2/2; G-F5-smoke and G-F5-speed MOOT (the candidate was dead at
quality before any speed run); G-F6 PASS at 28.3% ≥ 10% — the verify residual is real and
localized to the verify pass (F4 §4) — and the residual was then explained by reading the code,
with no kernel work at all.

## 2. A contradiction resolved: the +53% ROCm reading does not reproduce

Two historical readings were in conflict: Aug 15 said ROCm beats Vulkan by +53% on dense models;
Aug 17 said parity. Under the unified protocol (same T10 payloads, the same two images the
historical readings came from, same 24/08 session) the answer is **parity**: plain ROCm 13.70/13.69
vs Vulkan 14.06/14.01 tok/s, i.e. −2.6%/−2.3%; in ms/round, 73.3 vs 71.4-71.7 (+2.2-2.7%) (F0 §5).
The Aug 15 +53% reading is classified as a **protocol/build artifact of its day** — different
build, phase, cache and parallelism — not a property of the ROCm backend on this architecture. The
historical ROCm spec reference also does not survive the stricter protocol: the old n6 det 41.9
(n_predict 600, parallel 4) reads 34.75 here (F0 §6.3).

Two consequences. Operationally, the backend must be chosen on the **speculative** arms — Vulkan
wins mtp6 by +23%/+29% (det/prose, tok/s; F0 §5) — not on the plain arm; the "dense models → ROCm"
rule does not apply to this LEAN model in unified memory. Methodologically, older cross-protocol
percentages should be treated as unverified until re-run under the unified protocol. Declared
caveat (F0 §6.1): the two images are different builds (the ROCm image predates the type-104 and
DFlash2 porting work; the Vulkan image post-dates it), so even the unified-protocol comparison
remains cross-build — but it is the same image pair in which the contradiction was born, so the
contradiction is resolved under the conditions that produced it.

## 3. Measured bandwidth: the plain path is already near-optimal

The roofline inputs were measured directly, not assumed (`t10-bandwidth.sh`, torch on
rocm/pytorch:rocm7.2.1, 1 GiB × 50, idle GPU — F0 §3):

| measurement | loop GB/s | per-op mean (min-max) | role |
|---|---:|---|---|
| pure read (READ_ONLY_SUM) | **236.6** | 237.5 (225.0-239.5) | physical roof of the weight stream |
| device-to-device copy | **208.5** | 209.0 (203.9-210.4) | practical roof of write-back operations |
| plain decode @ctx1024 | 206.4 (ROCm) / 211.3 (Vulkan) | — | efficiency bound, not a gate input |

Plain decode runs at **87-89% of the pure-read roof** (206.4/236.6 = 87.2%, 211.3/236.6 = 89.3%):
the plain round's software margin is ~25-30 GB/s-equivalent, ~10 ms/round, and at the D2D
roof the plain residual collapses to 0.8-1.3 ms — most of it is structural write-back, not
recoverable by batching fixes (F4 §1). The speculative round, by contrast, is systematically less
byte-efficient than plain: 110.0-149.8 vs 204.7-210.3 real GB/s-equivalent (54-72% of plain,
within-backend; F0 §7c) — its tg gain comes from tokens committed per round, not from round
efficiency. That asymmetry is the whole thread in one line: **the recoverable margin lives in the
speculative round.** (The short-context payloads carry a real KV of only ~0.01 GB — context
130-200 tokens — so their floor is 63.45 ms at 236.6 GB/s, not the 68.22 ms of the @16k
scenario; F0 §3.)

## 4. The central deliverable: where the milliseconds of the round go

This section decomposes the speculative decode round mechanically. Cell: mtp6-det, ROCm build,
**136.42 ms/round measured** (F0 §1; k = 3.96 head-calls/round, 4.74 tok/round). Floors evaluated
at the measured read roof of 236.6 GB/s; the KV-read term at the long-context arm (~13.8k). The
identity closes: the components sum to 136.43.

| component | ms/round | nature and source |
|---|---:|---|
| weights + real-KV + SSM state floor (1×) | **63.45** | 15.012 GB ÷ 236.6 GB/s (F2 §2.2) — structural |
| draft-chain floor (k=3.96 × 0.731 GB) | **12.25** | MTP head-call = nextn 0.056 + shared head 0.675 GB ⇒ 3.09 ms/call @236.6; one `llama_decode(ctx_dft)` per token (`common/speculative.cpp:2109-2306`), head shared with the target (`qwen35.cpp:530/624/627-630`) — structural |
| verify per-row KV-read (k × 4.08 @~13.8k ctx) | **16.14** | flash attention: every query row reads the entire K+V (17 full-attn layers) — F6 §1.5, regime A (code-level attribution; see caveat) |
| GDN RS snapshot (k × 0.453 GB) | **7.58** | 3× the 0.151 GB SSM state per token: op write + read + write in the RS ring (`gated_delta_net.comp:121` token loop, snapshots `:173-181`; ring copy `delta-net-base.cpp:585-603`) — structural (RS mechanism) |
| conv-rollback RS | **0.42** | K copies/layer/round (`delta-net-base.cpp:497-522`, F6 §1.4) — structural |
| software head-call / draft | **12.30** | measured draft chain 24.54 (internal timer `dur(g)`, F2 §2.1) − floor 12.25 ≈ **3.1 ms/head-call above floor** |
| software verify | **14.46** | measured verify-extra +38.60 (F2 §2.2) − structural 24.14 ≈ **3.6 ms/row** |
| plain software residual | **9.83** | plain 73.28 − floor 63.45 (F2 §2.3); collapses to 0.8-1.3 ms at the D2D roof ⇒ mostly write-back (F4 §1) |
| **TOTAL** | **136.43** | measured 136.42 (F0 §1) — the identity closes |

Two declared annotations, kept from the source reports:

- **The KV-read row is the least solid.** It is a code-level attribution (F6 §1.4-1.5); the direct
  differential measurement of the verify pass across the same context delta (F2 §2.2) gave only
  +5.08 ms ≈ 1.25× the plain cost, i.e. a "KV read once per pass" reading. If read-once holds, the
  16.1 ms term migrates from structural to software (verify software ~30.6 instead of ~14.5). The
  thread's conclusion — the software of the speculative round is the only remaining tg lever — is
  unchanged under either reading; only the structural/software split inside verify-extra moves.
- **Declared mix** (F4 §1): wall-clock from the Vulkan build (which has no internal timer) +
  decomposition from the ROCm build (the only one printing `dur(b,g,a)`).

A cross-check that the overhead belongs to the target's round and not to the drafter: the much
lighter DFlash2 drafter (1.14 GB, ~0.79 calls/round) still lands at 121.0 ms/round with a
residual of +53.8 ms over its floor, on the same payload — 5.2 ms/round *faster* than the MTP6
Vulkan cell (126.2), nowhere near floor (F2 §2.2 B4, §2.3).

## 5. The "logits bandwidth" hypothesis, falsified in the code

The obvious structural story for verify cost — each of the k+1 logits rows re-reads the 0.675 GB
output head, so a batch of B rows costs B×0.675 GB — is false on this stack, and as a code fact,
not speculation (F6 §1.1-1.2):

- The verify batch receives **1 logits row for the sampled token plus 1 for every drafted token**
  (k+1 rows; `server-context.cpp:498-501`), decoded by a single `llama_decode(ctx_tgt, batch)`
  (`server-context.cpp:3920`). The output projection runs **only on the logits rows**
  (`get_rows` + `build_lora_mm`, `qwen35.cpp:219-227`).
- On Vulkan, with ≤16 columns **every** target mul_mat — FFN, attention, GDN projections, output
  head — dispatches to the VEC pipeline (`ggml-vulkan.cpp:9535-9537`; threshold
  `mul_mat_vec_max_cols = 16` at `:324`). The `mul_mat_vec.comp` shader accumulates into
  `temp[NUM_COLS][NUM_ROWS]`: each dequantized weight element is read **once** and its dot product
  accumulated against **all** batch columns — weights are shared across the rows of the batch.
- Therefore the extra bandwidth per verify row is ~0 on every mul_mat (the logits tensor itself is
  ~6.9 MB ≈ 0.03 ms — F6 §1.5); the per-column cost is compute only. The data agrees: the per-k
  slope is neither zero nor the ~2.85 ms/position a weight re-read would predict
  (fitted 12.85; F6 §2.3).
- On HIP the analogous threshold is 8 rows (MMVQ; `mmvq.cu:464-508`): at n6 the verify batch is 7
  rows and also reads weights once; at n8 it is 9 rows and switches to the MMQ kernel class — a
  switch this thread never measured on HIP (declared, F6 §1.3).

Consequence: the residual verify overhead is **kernel efficiency at width 5-7 columns plus
per-row components** (GDN snapshots, KV-read, sampler) — not bandwidth. That reframing is what
killed the kernel lever's best case in section 9.

## 6. n-max curves: n6 confirmed optimal

MTP n-max sweep on Vulkan, same session as F0 (n6 re-run agrees with F0's 42.86 within 0.5%;
F2 §4):

| n_max | tg det ± σ | ms/round det | tok/round det | tg prose ± σ |
|---:|---:|---:|---:|---:|
| **6** | **43.09 ± 1.63** | 125.6 ± 1.9 | 5.41 ± 0.28 | **26.69 ± 0.89** |
| 8 | 36.11 ± 0.66 | 171.4 ± 4.8 | 6.19 ± 0.23 | 24.99 ± 1.83 |
| 10 | 37.62 ± 1.12 | 186.1 ± 2.1 | 7.00 ± 0.24 | 21.53 ± 0.98 |

Acceptance per draft position (means over 5 replicas): n6 = 0.97 0.92 0.82 0.75 0.74 0.62; beyond
position 6 the chain accepts ~0.5 or less (n8 positions 7-8: 0.50/0.49; n10 positions 7-10:
0.55/0.50/0.42/0.28). The full draft+verify cost is paid for positions that commit half the
time: Δ(n6→n8) = −6.98 tok/s (−16.2%) for +45.8 ms/round; at n8 a marginal committed token costs
~59 ms against 23.2 ms/token average at n6. The pre-registered sanity check "tg non-decreasing
in n" FAILS at n8 — saturation is documented between 6 and 8. On prose the picture is sharper:
p_min 0.75 truncates the chain at k≈1.6-2.1 whatever n_max is (tok/round flat at 2.67→2.71), so
rising n is pure cost there, and the curve is monotonically decreasing. **n6 is optimal on both
payloads; the production configuration stands.**

## 7. Compression to ~3 bpw: NO-GO on perplexity

The quality gate (protocol verbatim from T5: `llama-perplexity -c 512 -b 512 -fa on`,
wikitext-2-en + Italian technical; gate = 2× the LEAN delta of the historical ROCm reference of
17/08, +2.74%/+3.42% vs BF16, hence ≤ +5.48%/+6.84% — the within-build control re-sampled here
reads +3.09%/+3.93%; F5 tab. 1):

| arm | preset | eff. bpw | size | PPL en | Δ en | PPL it | Δ it | verdict |
|---|---|---:|---|---|---|---|---|---|
| BF16 reference (17/08) | — | 16 | 50.9 GB | 6.6409 ± 0.121 | — | 11.7156 ± 0.388 | — | — |
| LEAN control (this build) | Q4_0_ROCMFP4_STRIX_LEAN | 4.38 | 13.82 GiB | 6.8458 ± 0.125 | +3.09% | 12.1762 ± 0.406 | +3.93% | gate reference |
| **GATE** | | | | | **≤ +5.48%** | | **≤ +6.84%** | |
| MIX (iteration 1) | Q2_3_ROCMFPX_MIX | 3.01 | 10.29 GB | 8.4367 ± 0.166 | **+27.04%** | 15.4202 ± 0.546 | **+31.62%** | **FAIL** |
| MIX_V2 (iteration 2) | Q2_3_ROCMFPX_MIX_V2 | 3.10 | 10.60 GB | 8.4465 ± 0.167 | **+27.19%** | 15.3462 ± 0.544 | **+30.99%** | **FAIL** |

Both iterations overshoot the gate by ~5× (en) and ~4.5× (it). The control re-sampled on the
same build reproduces the T5 values to the fourth decimal (6.8458/12.1762), so the gate is
validated. Three readings close the lever:

- **The kernels are healthy; the quantization is simply too aggressive.** Per-chunk ppl ratio
  MIX/LEAN = 1.2483 ± 0.0105, uniform across 70 chunks (range 1.228-1.292), no NaN/Inf — the
  signature of homogeneous quantization noise, not of a broken kernel (unlike T5's type-104
  episode). The V2/MIX ratio is 0.9985: raising the 48 fused `attn_qkv` tensors (2.517 B params)
  from 3.5 to 4.5 bpw changes nothing. The degradation is dominated by the FFN block at Q2 — 17.38
  B params = 63.6% of the model at 2.50 bpw, down from 4.25 bpw (the FAST category of LEAN; F5,
  analysis 1-2).
- **The bpw→ppl curve on this model is steep.** 4.44 bpw → +5.32/+5.09 (T5, borderline pass);
  3.10 bpw → +27.2/+31.0. Between 4.4 and 3.1 bpw the perplexity explodes: the ~3 bpw class is
  not viable on this dense model with any reasonable category-based mix, and an imatrix-guided
  variant (the planned F5b) could not close a ~21-point gap on en — the FFN would still sit
  at ~2-3 bpw over the vast majority of its 17.38 B parameters (F5, analysis 3).
- **Contrast with Escha-W2.** The external HF GGUF (2.469 bpw, 10.15 GB) claims quality comparable
  to FP8 on benchmarks, without reporting ppl. Benchmark-parity without ppl ≠ ppl-parity: T10 is
  the first direct ppl evidence on the 2/3-bit class for this model, and it is negative.

One observational note, not a gate: in the ppl run itself (batch 512, compute-bound) MIX runs at
1.80 s/pass vs 1.59 for LEAN — smaller but slower per unit of work at high batch. No direct tg
implication (decode is bandwidth-bound), but a hint that the Q2 kernels are not more efficient
than the ROCmFP4 ones even per byte (F5, analysis 4).

## 8. The ceiling ladder

LEAN, mtp6-det, Vulkan (k = 4.68, 5.40 tok/round). The recalculated identity: **125.62 ms/round =
structural floor 89.15 (71.0%) + software residual 36.47 (29.0%)** (F6 §1.5) — the code study
moved ~11.2 ms from "unexplained" into the floor (GDN snapshots ×3 + conv-rollback + the 0.604 GB
GDN plain baseline), correcting F4's recovery ceilings downward.

| ceiling | round ms | tok/s | source / formula |
|---|---:|---:|---|
| **measured today** | 125.62 | **43.0** | F0 §1 (42.86) / identity F4 §2.2 |
| best case of intervention (a) (width-dependent excess at n6 zeroed) | 117.1 | 45.3 | F6 §3, computed on the section 9 sweep cell (126.55 ms) — equals the 2σ gate, EV below → discarded |
| verify software recovered only (~14.5 ms/round, ROCm split) | ~111.2 | **~48.6** | 125.62 − 14.46; 5.40 × 1000/111.16 |
| all software residual recovered (36.47) | 89.15 | **60.6** | F6 §1.5; was 69.3 under the pre-F6 floor (−11.2 ms correction) |
| pure-bandwidth ideal @16k, drafter excluded | 68.22 (read) / 77.42 (D2D) | **79.2 / 69.8** | F1 §7 / F4 §2.1 (at 236.6 and 208.5 GB/s) |

Consistency note (declared in the source): the tempting combination "F2 floor 63.45 + draft-chain
floor 14.46 (k=4.68, the Vulkan cell) + corrected residual 36.5 = 114.4 → 47.2 tok/s" is **not** a
closed identity — it omits the ~11.2 ms of GDN/conv structural traffic that F6 moved into the
floor (the correct identity is 89.15 + 36.47 = 125.62); the coherent intermediate ceiling for
"verify software only" is ~48.6. Note the coincidence to not be fooled by: the Vulkan draft-chain
floor is 14.46 ms — the same digits as the ROCm verify-software component of section 4.
The practical ceiling therefore depends entirely on how much verify software is recoverable: the
defensible interval is **43 (today) → 60.6 (all software zeroed)**, with the 69.8-79.2 ideal
reachable only by also changing the drafter (14.46 ms of head-call floor) and the RS mechanism.
The F5a ceilings of F4 §2.2 (e.g. 50.8 det VK) remain unverified theoretical bounds — that
candidate died at the ppl gate before any speed bench.

## 9. The one remaining lever, precisely scoped — and why we did not pull it

An instrumented batch sweep (same build throughout, det payload, 72-token ctx, warm-up + 3
replicas; F6 §2.1):

| n_max | tok/s ± σ | ms/round ± σ | k | verify rows (k+1) |
|---:|---:|---:|---:|---:|
| plain | 14.03 ± 0.01 | 71.30 | — | 1 |
| 1 | 23.63 ± 0.01 | 83.34 ± 0.05 | 0.969 | 1.97 |
| 2 | 30.85 ± 0.39 | 91.51 ± 0.21 | 1.838 | 2.84 |
| 3 | 36.76 ± 0.26 | 99.03 ± 0.22 | 2.683 | 3.68 |
| 4 | 40.72 ± 0.58 | 106.58 ± 0.30 | 3.441 | 4.44 |
| 6 | 41.95 ± 2.05 | 126.55 ± 2.45 | 4.563 | 5.56 |
| 8 | 42.46 ± 0.63 | 144.74 ± 0.83 | 5.761 | 6.76 |

Least-squares fits (18 replicas): ms/round = 73.33 + 8.823·n_max (R² 0.9956); per-k
67.40 + 12.846·k (R² 0.9746); the quadratic in k reaches R² 0.9901. The marginal cost per unit
of k climbs from ~9 to ~15-18 ms as k goes from ~1.4 to ~5.8 (consecutive segments: 9.40, 8.90,
9.97, 17.79, 15.19); the width-dependent excess over a flat-slope counterfactual is **+9.4
ms/round at n6** (+16.3 at n8). No structural component grows superlinearly (GDN and KV are
linear per row, the head-call is linear per call), so the rising part is software — compatible
with VEC pipeline efficiency degrading as NUM_COLS grows, exactly the mechanism section 5
isolates (F6 §2.2-2.3).

So the only remaining tg lever is the **software overhead of the speculative round**: ~14.5
ms/round in the verify pass (ROCm split, ~3.6 ms/row) up to ~36.5 ms/round of total residual on
Vulkan (draft + verify + orchestration; F6 §1.5/§4). The Vulkan verify-row software is bounded at
5.9-9.0 ms/row against a floor of 1.93, and its non-KV part converges with the ROCm estimate
(5.6 ms/row) — two builds, one number (F6 §2.3). Declared confounder: p_min-aborted head-calls
are not counted in the k or head-call counters (at n8, acceptance 0.889 ⇒ ~1 abort/round is
plausible), which deflates
the apparent per-k slope; not separable without the internal timer on the Vulkan build.

The one concrete intervention — a batched-dispatch switch for 2-9 columns — was evaluated and
**discarded on expected value** (F6 §3): full recovery of the width-dependent excess at n6 would
take the round from 126.55 to 117.1 ms/round = 45.3 tok/s, i.e. **+3.4 tok/s = exactly the 2σ
gate** — the best
case equals the gate, it does not clear it; a 50% recovery gives +1.7, below it; the probability
of clearing the gate is ~30-40%; the cost is 2-3 h of GPU time plus numeric drift (a `.f32acc`
variant, and an acceptance gate because greedy outputs can shift). And the candidate's original
theoretical basis — bandwidth — is falsified by section 5, leaving only kernel-efficiency upside
whose direction is not guaranteed: the VEC threshold of 16 columns exists precisely because the
vector path wins at small widths.

## 10. What is reusable from here

- **Exact round counting: R = 255 − A.** Every decode round commits its accepted drafts plus one
  extra token, and the first token comes from the prefill, so 256 = 1 + A + R; A is the
  per-request accepted-draft counter.
  Validated directly against the cumulative draft-call counter (6/6 requests on the det cells).
  This replaces `ms_tok × mean_acc_len`, which overestimates ms/round by 7-22% because "mean
  acceptance length" averages only over verify rounds (F0 §2).
- **The fork's internal draft timer `dur(b,g,a)`** (ROCm build only,
  `common/speculative.cpp:3811`): the measured draft-chain vs target-pass split. The Vulkan build
  does not print it — the declared wall-clock/decomposition mix of this thread (F2 §1).
- **The measured bandwidth pair for any future roofline:** 236.6 GB/s pure read, 208.5 GB/s D2D
  (torch, 1 GiB × 50, idle GPU; F0 §3).
- **The correct per-position floors:** head-call 0.731 GB/call (0.675 shared head + 0.056 nextn);
  GDN RS snapshot 0.453 GB/token (3× the 0.151 GB state); verify per-row KV-read = context ×
  69,632 B per query row (0.02 ms/row at 72 tokens, 4.08 at 13.85k); byte/round @16k exactly
  16.1417 GB = weights 14.845 + KV 1.1409 + SSM 0.1510 + conv 0.0047 (F0 §1, F6 §1.4-1.5).
- One cheap instrument is still missing, suggested and not committed: a counter for p_min-aborted
  head-calls, which would close the section 9 confounder without an internal timer on the Vulkan
  build.

## 11. Limitations

- **Cross-build comparisons, declared.** The plain ROCm vs Vulkan comparison spans two different
  images (the ROCm image predates type-104 and the DFlash2 porting; the Vulkan image post-dates
  it). Unified protocol, not unified build — though the same image pair in which the +53%
  contradiction was originally observed (F0 §5, §6.1).
- **Declared mix of builds.** Wall-clock numbers come from the Vulkan build (no internal timer);
  the draft/verify decomposition comes from the ROCm build. The Vulkan cells are
  differential-only (F2 §3, F4 §1).
- **The per-row KV-read term (16.14 ms) is a code-level attribution** — the weakest row of the
  central table. The direct differential measurement suggested a read-once reading (+5.08 ms ≈
  1.25× plain). Under read-once the term migrates structural→software (verify software ~30.6
  instead of ~14.5); the thread's conclusion is unchanged either way (F6 §1.5; F2 §2.2).
- **Backend kernel classes differ.** The decomposition coefficients are valid for the Vulkan VEC
  pipeline; on HIP (gfx1151) the MMVQ→MMQ switch happens above 8 rows, so the ROCm n8 marginal
  would include a kernel-class change never measured here. The ROCm verify-extra of 38.60 ms was
  measured at n6 (7 rows, MMVQ) (F6 §1.3).
- **Head-call abort confounder.** p_min-aborted head-calls are not counted in k/G; at n8 (~1
  abort/round plausible) this deflates the apparent per-k slope and cannot be separated without
  an internal timer on the Vulkan build (F6 §2.3).
- **No GPU tracer exists in any available image** (verified): inside the verify residual, the
  split between launch gaps/CPU and kernel inefficiency is not measurable; the decomposition is
  differential + code reading (F2 §1, §3).
- **k is a per-request average** (head-calls over rounds), not a per-round value; the marginal
  cost of positions beyond the mean is not decomposable with these counters (F2 §3).
- **The F5a ceilings are unverified bounds:** the compression candidate failed the ppl gate
  before any speed bench, so its projected ceilings (e.g. 50.8 det VK) were never tested.
- Single machine, single model, single slot; two payload classes (deterministic counting, prose)
  plus one long-context padded variant.

## 12. Reproducibility

- **Code.** Fork branch `t10` @ `8ce2f6f29` (base `6d55f6c7d`): three patches — `Q2_3_ROCMFPX_MIX`
  preset routing, the ftype name, and the V2 variant — preserved as a patch series in
  `patches/t10/` of this repository. **Research patches, not production**: the preset failed the
  quality gate.
- **Images.** `docker-llm-service:t10-vulkan` (the only runtime exercising GGML type 107 Q2; its
  Q2/Q3 kernels verified numerically healthy in F5) for the sweep and the ppl gate; the benchmark
  arms of sections 2-6 ran on `docker-llm-service:vulkan-fork-dflash2` (Vulkan) and
  `docker-llm-service:dflash` (ROCm), ports 8193/8194, containers removed after each run.
- **The research GGUFs are NOT shipped.** `Qwen3.8-27B-Q2_3_ROCMFPX_MIX.gguf` (10.29 GB, ftype
  118) and `..._V2.gguf` (10.60 GB, ftype 120) were deleted in the post-run cleanup. They are
  regenerable from the BF16 with the preset from the patch series in minutes (the V2 requant
  measured 172 s);
  the per-tensor composition of all 866 tensors was verified in the run logs
  (`logs/t10/f5-composition{,-v2}.txt` in the workspace, 0 routing errors).
- **Scripts** (preserved in the lab workspace `scripts/`): `t10-bench.sh` (parametric wrapper:
  image/port/spec/ctx/nmax/model), `t10-bandwidth.sh`, `t10-roofline.py`, `t10-ppl-f5.sh`
  (reusable ppl gate), `t10-parse-composition.py` (stdlib GGUF parser), `t10-payloads.json`,
  `t10-payloads-long.json`, and `t10-loop-detector.py` (built and fixture-tested, never
  exercised — the smoke gate was declared MOOT).
- **Primary reports.** The six Italian benchmark reports
  `docs/benchmarks/results-2026-08-24-t10-{f0-baseline,f1-roofline,f2-profiles,f4-margini,f5-mix,f6-verify}.md`
  and the logs under `logs/t10/` live in the lab workspace; every number in this note is cited
  to them.
- **Hardware and flags.** AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151, 128 GB
  unified LPDDR5X), target `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx window
  16384, p_min 0.75, parallel 1, single stream, dedicated GPU window, warm-up discarded.

Pointers: the companion notes are [per-round-drafter-switching.md](per-round-drafter-switching.md)
(T9) and [2026-08-23-rocmfp3-quality-speed.md](2026-08-23-rocmfp3-quality-speed.md) (T5, the
4.44 bpw point of the compression curve); the plan and binding spec
(`docs/superpowers/plans/2026-08-24-t10-round-budget.md`,
`docs/superpowers/specs/2026-08-24-t10-round-budget-design.md`) are in the lab workspace.

---

*Nothing shipped and no gate cleared — but the round now closes its books: of the 136.42
milliseconds measured, 136.43 are accounted for by name.*
