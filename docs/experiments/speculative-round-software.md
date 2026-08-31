# Round software: the speculative round's ~38 ms residue attacked with both remaining levers, and both falsified

**Research note — August 2026 (thread T11).** Companion to
[speculative-round-budget.md](speculative-round-budget.md) (T10, which decomposed the
speculative round to the millisecond and left its software residue as the only untried lever),
and to the two notes that closed every other speculative-round redesign:
[per-round-drafter-switching.md](per-round-drafter-switching.md) (T9, per-round drafter choice)
and [dual-drafter-synergy.md](dual-drafter-synergy.md) (T8, same-round cooperation). T11 takes
the two software interventions T10 identified but never executed, builds both, measures both
A/B within-build — and closes the line with two falsifications of pre-registered expectations,
one per lever, in opposite phases of the round.

Everything ran on one machine: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151,
128 GB unified LPDDR5X), Vulkan (RADV) build of the fork, target model
`Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx window 16384 (`-fa on`), p_min
0.75, p_split 0.10, n_max 6, parallel 1, single stream, dedicated GPU window, warm-up
discarded. Every A/B is **within-build by construction** — one docker image per experiment, the
intervention enabled by a single environment variable — over an interleaved 4-boot protocol
(OFF/ON/OFF/ON, never simultaneous), 10 replicas per arm per cell (the T10 payload pair:
deterministic counting and free prose, n_predict 256).

**TL;DR**

- **The question.** T10's accounting identity for the mtp6 round on Vulkan (sweep cell,
  126.55 ms/round): structural floor 88.6 ms + **software residue ~38.0 ms (30%)** (T10 called this the software
  residual) — draft chain, verify pass, orchestration. Two levers existed to attack it, both specified but never
  executed: (b) collapse the MTP draft chain into one fused graph (1 build+submit+sync+readback
  per round instead of N), and (a) divert the verify batch's quantized mul_mats from the
  VEC/DMMV path to the batched MMQ pipelines. The gate: +10% tg (2σ) on det or prose,
  OR a documented verdict with quantified expected value per lever. The second branch is what
  happened.
- **Feasibility first, zero GPU (G-F0 GREEN).** All ops a fused draft graph needs are native
  on this fork's Vulkan backend — argmax included, with an in-codebase precedent (the DFlash
  markov head chains get_rows → argmax → get_rows in-graph). The pipelined-submit variant
  turned out to be, in fact, subsumed by the fused one. The verify switch needs zero new
  shaders — the batched pipelines already exist for both LEAN tensor types (422 of type
  101, 82 of type 100) — and the change is one dispatch point and one env var.
- **Lever (b) falsified.** `SPEC_DRAFT_FUSED=1`: det **−4.34 tok/s (−10.2%)**, prose **−4.87
  (−18.6%)**. Mechanism measured with the internal per-phase timer: the fused draft phase is
  **flat at ~38 ms/round regardless of accepted length**, while the per-token loop it replaces
  scales with k (25.9 ms det / 14.9 ms prose) — ggml has no data-dependent control flow, so the
  p_min early-stop stays host-side and the fused graph pays the full n_max=6 chain every round.
  The pre-registered +9-14 ms/round recovery reversed sign.
- **Lever (a) falsified, hard.** `GGML_VK_VERIFY_SWITCH=1` (forced f32acc mode): det **−23.38
  tok/s (−55.3%)**, prose **−14.05 (−55.6%)**. 98.7-98.9% of the regression sits in the verify
  residual (×2.60 det / ×2.45 prose): the batched MMQ pipeline is ~2.5-2.6× *slower* than
  VEC/DMMV at 2-9 columns on this GPU/driver. This falsifies T10's hypothesis that the
  width-dependent verify excess was a dispatch artifact recoverable by switching: the VEC path
  is already near-optimal, and the excess is the real cost of the computation at that width.
  The pre-registered +9.4 ms/round recovery (best case +3.4 tok/s) reversed sign at ~17× the expected magnitude.
- **The combined arm was skipped by pre-registered rule**: with both levers measured negative,
  no combination can reach +10% — the combined A/B would have had a predetermined outcome.
- **Conclusion.** The ~38 ms/round software residue is **structural on Vulkan/RADV for this
  model** within these bounds. Moving it requires going outside them: device-side
  early-stop (control flow in the backend), cheaper fused graphs, or different driver/hardware.
  Both interventions remain in branch `t11` behind default-off, inert flags — reusable, not
  merged.
- **Instrumentation delivered** (the durable output): the `dur(b,g,a)` draft timer works on
  Vulkan builds too (correcting T10's "ROCm build only"), a non-gated `draft_aborted` counter
  closes T10's declared per-k confounder, and the ggml INFO→TRACE log remap is documented with
  a diagnostic boot log — boot markers need `-lv` ≥ 4, WARNs are visible at default.

## 1. The question and the outcome

T10 closed its books with an identity: of the 126.55 ms of the mtp6-det Vulkan round (sweep
cell), 88.6 ms are structural floor (weights 1×, GDN RS snapshots, conv rollback, drafter
k×0.731 GB) and **~38.0 ms (30%) are software** — the draft chain's per-call overhead, the
verify pass's width-dependent excess, orchestration (T10 F6 §4; the spec's §1 table). T10
evaluated exactly one intervention on paper — the batched-dispatch switch — and discarded it on
expected value (best case exactly at the 2σ gate). The head-call software component (~9-14
ms/round on Vulkan) had never been evaluated as an intervention at all. T11 was opened to close
both:

- **(b) fused draft chain** — collapse the MTP draft chain (N sequential 1-row
  `llama_decode(ctx_dft)` calls per round) into a single chained graph: one build, one submit,
  one sync, one readback per round.
- **(a) verify dispatch switch** — divert quantized mul_mats at 2-9 columns (the verify batch)
  from the VEC/DMMV path to the batched MMQ branch of the same dispatcher.

The pre-registered success criterion was T10-shaped: **+10% tg measured (2σ) on det or prose,
OR a documented verdict with quantified expected value per lever.** The outcome is the second
branch, reached the hard way: both levers were implemented, both were measured, and both
*falsified their pre-registered expectation with the sign reversed* — lever (b) at −10.2%/−18.6%,
lever (a) at −55.3%/−55.6%. Gate registry of the thread:

```
G-F0 GREEN  (zero-GPU feasibility: b1 feasible + a implementable)        -> build phase
G-F1 FAIL   (fused draft chain:  dtg -10.2% / -18.6%, both cells negative)
G-F2 FAIL   (verify dispatch:    dtg -55.3% / -55.6%, both cells negative)
combined arm: SKIPPED by pre-registered rule (both levers NO-GO)
THREAD GATE: +10% (2 sigma) NOT reached -> T11 closed NO-GO, EV documented per lever
```

No production changes: both flags are default-off and inert, and the standing rule — no merge
without a gate PASS — kept everything in the research branch.

## 2. Feasibility first: what zero GPU established (F0)

Before touching the GPU, a code-reading phase settled feasibility with file:line anchors on the
real clone (branch `t11`, base `6d55f6c7d`), and its gate G-F0 came out GREEN: variant (b1)
feasible at effort M, lever (a) implementable at effort S. Four findings carried the whole
thread:

**Every op a fused draft graph needs is native on this fork's Vulkan backend.** The fused chain
requires `get_rows` (embedding lookup by argmax index), `argmax`, `set_rows` (drafter KV
writes), and `soft_max` (max-prob for p_min) — all verified inside
`ggml_backend_vk_device_supports_op`, with the attention/rope/norm/concat ops already in
production on the trunk. The specific worry that drove earlier hesitation — "argmax is missing
on Vulkan, so each step would fall back to CPU with a graph split and sync" — is **false on
this fork**: the op has a dedicated tree-reduction shader (`argmax.comp`) and a native
dispatch. The fused graph stays mono-backend, zero splits, zero intermediate copies. There is
even an in-codebase precedent for exactly the required pattern: the DFlash markov head
(`src/models/dflash.cpp:369-390`) already chains `get_rows(w1, prev)` → logits → `prev =
ggml_argmax(...)` in-graph, repeated for its block-diffusion steps; `ggml_argmax` is also used
by the backend sampler chains.

**The pipelined-submit variant is subsumed by the fused one.** The tempting lighter
alternative — keep N graphs, drop the intermediate syncs, one final sync — dies on a structural
data dependency, not on backend limits: graph compute is *already asynchronous* on gfx1151/RADV
(the backend synchronizes at graph end only on Intel DG1 or with `GGML_VK_DISABLE_ASYNC`), and
the per-step host sync is forced by the greedy sampling itself, which reads each step's logits
before the next step's inputs exist. The inputs of step k+1 are f(outputs of step k) consumed
on the host — so concatenated submits are impossible while the greedy stays host-side, and any
implementation that works must move argmax and the hidden-state routing into the graph — which
is (b1)'s definition. The weakened fallback b2′ (pre-build the next graph while the current one
flies) has its gain bounded by min(build time, GPU step time) and shares the family verdict.

**The verify switch needs zero new shaders.** The Vulkan dispatcher has exactly one dispatch
point (`ggml-vulkan.cpp:9535-9539`): today every quantized mul_mat with 2-16 columns and batch 1
goes to the VEC/DMMV family (threshold `mul_mat_vec_max_cols = 16`); the verify batch at n6 is
5-7 rows, inside that 2-16 range. The batched branch it would divert to already has pipelines for
**both** LEAN tensor types in every shader family of the backend — verified against the
production GGUF header itself: 422 tensors of type 101 (`Q4_0_ROCMFP4_FAST`: output head, FFN
down/up, attn_gate, ...) and 82 of type 100 (`Q4_0_ROCMFP4`: attn_qkv, attn_v — the size-biased
K/V part of the recipe), and verify touches both. On HIP the analogous crossover is wired
(MMVQ up to 8 rows, then MMQ); on Vulkan no such hard-coded threshold exists, and lever (a) simply extends the
batched branch down to 2-9 columns. Two spec corrections came out of this phase and were
applied before implementation: the quantized operand the switch tests is **src0** (the
weights — src1 is the F32 activations), and the `.f32acc` numeric recipe is an AND (integer-dot
disabled AND a non-f16acc selector outcome), not an OR.

**The pre-registered expectations F1/F2 would test.** From the spec: lever (b) was expected to
recover ~9-14 ms/round of the head-call software residue; lever (a) to recover the +9.4
ms/round width-dependent verify excess (T10's accounting), best case +3.4 tok/s — exactly the 2σ
gate, with a declared ~30-40% probability of clearing it. Both expectations are recorded
before any measurement; both are what the experiments falsified.

## 3. Experiment 1 — the fused draft chain (`SPEC_DRAFT_FUSED`)

**Setup.** Image `docker-llm-service:t11-vulkan-f1` (`81eb386179ec`, branch @`866d92dcb`) for
both arms — within-build by construction. The flag is env-only: arm ON = `SPEC_DRAFT_FUSED=1`,
arm OFF = no extra env; same binary, same model, same spec config (draft-mtp, n_max 6, p_min
0.75, p_split 0.10, draft-ngl all). Interleaved 4 boots OFF/ON/OFF/ON, alternating ports,
warm-up + 5 replicas per boot per cell, both cells per boot: 10 replicas per arm per cell.
Marker gates were verified **before** reading any number: the constructor marker
(`SPEC_DRAFT_FUSED active: draft sampling kept on the host`) and the engaged marker appear
exactly once per ON container, zero occurrences in any OFF directory, and zero
`outside the fused envelope` warnings anywhere — the fused path never left its declared
envelope. Env hygiene: `GGML_VK_PERF_LOGGER` and `GGML_VK_DISABLE_ASYNC` verified absent.

| cell | arm | tg (tok/s) | Δtg (ON−OFF) | ms/round |
|---|---|---:|---:|---:|
| mtp6-det | OFF | 42.43 ± 1.67 | | 127.3 |
| mtp6-det | ON | 38.09 ± 0.52 | **−4.34 (−10.2%)** | 141.2 (+13.9) |
| mtp6-prose | OFF | 26.15 ± 0.80 | | 102.0 |
| mtp6-prose | ON | 21.28 ± 0.55 | **−4.87 (−18.6%)** | 124.5 (+22.5) |

Gate G-F1 (Δtg ≥ 2·SE AND acceptance within SE AND content control): det Δtg −4.34 against a
required ≥ +1.11 — FAIL; prose −4.87 against ≥ +0.61 — FAIL. **G-F1 FAIL on both cells, with
the sign the pre-registration did not expect.**

**The mechanism, from the internal per-phase timer.** The fork's `dur(b,g,a)` draft timer (see
section 7) gives the draft phase per round, as a per-request delta of the cumulative
`statistics draft-mtp:` log lines:

| cell | arm | draft phase ms/round | post-accept ms/round |
|---|---|---:|---:|
| det (k = 4.69) | OFF | 25.86 ± 1.36 | 0.022 |
| det | ON | **38.35 ± 0.40** | 0.021 |
| prose (k = 1.94) | OFF | 14.90 ± 0.78 | 0.016 |
| prose | ON | **37.73 ± 1.06** | 0.016 |

The pattern is the whole story: OFF scales with k (25.9 ms at k = 4.69, 14.9 at k = 1.94 —
p_min truncates the per-token chain early); ON is **~flat at ≈ 38 ms/round on both cells** (coincidentally the same number as the
residue).
The fused graph pays the complete n_max = 6 chain every round, regardless of how much of it
will be accepted or aborted, because the p_min early-stop is host-side — ggml has no
data-dependent control flow, so nothing inside the graph can switch later steps off. The
draft-phase delta (+12.5 det / +22.8 prose ms/round) covers ~90-101% of the total slowdown:
**all of the regression is in the draft phase**. The saving of N−1 submits/syncs/readbacks is
more than cancelled by computing the entire chain without early-stop. On prose the effect is
amplified (−18.6% vs −10.2%) because the OFF chains there are short (k = 1.94), so the fused
graph's fixed ~38 ms cost weighs proportionally more.

**The declared confound, measured and dismissed.** The ON arm samples the draft tokens on the
host (the fused graph returns argmaxes, the sampler runs outside), expected ~0.5 ms/round. The
measured post-accept accumulation phase is 0.021 ms/round in the ON arm — unchanged from OFF. The
confound is irrelevant at two orders of magnitude below the effect.

**Acceptance and content.** Det acceptance drops −0.02738 (0.9363 → 0.9089), beyond the SE
gate — the host-side sampling of the chain produces slightly worse *draft* tokens at equal
committed work (rounds per request R unchanged at 47.4 → 47.6; draft call tokens G +6.4 per request). Prose acceptance
is unchanged within SE. Content control: within-arm cross-boot fingerprints are **10/10
identical on all four boot-pair combinations** — the machine is deterministic per
configuration, so the cross-arm divergence is attributable to the flag, not to noise. Cross-arm
on det, 4/5 replicas are char-identical over the full window (the first post-warm-up replica
phase-diverges at token 103, the machine's known rep-phase pattern; declared, minimum coverage
24.5%): in the reproducible window the fused path commits **identical tokens** to OFF. On
prose the cross-arm control is structurally null (0/5 identical windows; divergence from
character 0 in 4/5 replicas): with ~96 rounds and acceptance 0.85, any numeric drift re-composes
the verify batches from the first round — declared drift, not noise, and the reason the prose
content gate is FAIL by design. The smoke signal from the pre-A/B boot (−10% on det) is
confirmed exactly at −10.2%; prose amplifies to −18.6%.

## 4. Experiment 2 — the verify dispatch switch (`GGML_VK_VERIFY_SWITCH`, forced f32acc)

**Setup.** Image `docker-llm-service:t11-vulkan` (`41301f59806a`, branch @`1df4db2aa`,
containing **both** flags; `SPEC_DRAFT_FUSED` never activated in this experiment) for both
arms. Arm ON = `GGML_VK_VERIFY_SWITCH=1`, resolved to mode f32acc on every ON arm (the mmq
mode of the switch was never activated). Same protocol as F1: interleaved 4 boots, 10 replicas
per arm per cell, marker gates before numbers.

| cell | arm | tg (tok/s) | Δtg (ON−OFF) |
|---|---|---:|---:|
| mtp6-det | OFF | 42.29 ± 1.67 | |
| mtp6-det | ON | 18.91 ± 0.29 | **−23.38 (−55.3%)** |
| mtp6-prose | OFF | 25.28 ± 1.69 | |
| mtp6-prose | ON | 11.23 ± 0.25 | **−14.05 (−55.6%)** |

Gate G-F2: det Δtg −23.38 against ≥ +1.07 — FAIL; prose −14.05 against ≥ +1.08 — FAIL. The
null outcome was pre-accepted (~30-40% probability of passing); what happened is not a null
but a large negative with the mechanism identified.

**The mechanism, localized to the phase the switch touches.** The `dur(b,g,a)` decomposition
plus residual (verify = total JSON time − draft-timer phases, per request):

| cell | arm | total ms/round | draft ms/round | verify residual ms/round |
|---|---|---:|---:|---:|
| det | OFF | 128.0 ± 1.5 | 25.76 ± 1.37 | **102.2 ± 0.7** |
| det | ON | 293.1 ± 6.1 | 27.79 ± 0.48 | **265.3 ± 5.8** |
| prose | OFF | 106.0 ± 6.4 | 15.55 ± 1.29 | **90.4 ± 5.3** |
| prose | ON | 238.1 ± 6.2 | 17.06 ± 0.43 | **221.1 ± 5.9** |

Delta ON−OFF: det +165.2 total = +2.03 draft + **+163.1 verify residual (×2.60)**; prose
+132.2 total = +1.51 draft + **+130.7 verify residual (×2.45)**. That is **98.7-98.9% of the
regression sitting exactly in the phase the switch acts on** (quantized mul_mats at 2-9
columns = the verify batch), while the draft phase — 1-column mul_mats, not diverted — is
unchanged to second order, consistent with its k moving 4.69 → 4.79 / 1.94 → 2.01. The
conclusion is direct: on GFX1151/RADV with ROCmFP4 weights, the batched MMQ `.f32acc` pipeline
is **~2.5-2.6× slower than the default VEC/DMMV path at 2-9 columns** — the batched pipelines
are saddled with a layout/quantization scheme designed for large N, wasted at single-digit
column counts.

**What this falsifies.** T10's accounting attributed +9.4 ms/round at n6 to a width-dependent
verify excess and hypothesized degraded VEC efficiency at 5-7 columns, recoverable by
switching to the batched pipelines. The switch was built, forced to the f32acc numeric mode (the
cleanest comparison — accumulation stays F32 like the VEC path), and measured: the excess is
**not a dispatch artifact**. The VEC/DMMV path is already near-optimal at these widths; the
excess is the real cost of the computation at that width on this hardware/driver. The
pre-registered expectation (+9.4 ms/round recovery, best case +3.4 tok/s) reversed sign at ~17×
the expected magnitude: 163.1 / 9.4 ≈ 17.

**Acceptance and content.** Det acceptance +0.00687 (0.9363 → 0.9432) — ON slightly *better*,
but beyond the SE gate (the gate demands invariance; the f32acc accumulation-order drift moves
logits, and acceptance is measured on the diverted path's logits). Prose −0.02632, also beyond
SE. Content: within-arm cross-boot fingerprints 20/20 identical; cross-arm on det 4/5
char-identical full-window (first replica phase-diverges at 103 — same position as F1),
confirming that f32acc's drift is accumulation-order only: in the reproducible window the
diverted path commits identical tokens. Prose 0/5 (divergence from character 0 in 4/5
replicas) — the same structural pattern as F1, there caused by host-side sampling, here by
f32 accumulation. The OFF arms of this experiment also re-validated the baseline: the OFF
smoke run's fingerprints are char-exact against the t0 baseline across three builds
(t0/F1/F2), so the OFF path is build-stable and the gate was informative.

The two levers perturb complementary phases — (b) regressed the draft phase, (a) the verify
residual — and neither recovered anything.

## 5. The combined arm: skipped by pre-registered rule

The plan's final task was the combined arm (a)+(b) against the same cells. It was **not run**,
by the branch rule written before the experiments: with both levers measured NO-GO (G-F1 FAIL,
G-F2 FAIL, both with net regression), no combination of two interventions that each make the
round slower can reach a +10% gate — the combined A/B would have been an experiment with a
predetermined outcome. The thread went straight to closure.

## 6. What the two falsifications establish

**Verdict: T11 NO-GO**, on the second branch of the gate, with the expected value measured per
lever:

| lever | measured EV (det / prose) | gate required |
|---|---|---|
| (b) `SPEC_DRAFT_FUSED` | **−10.2% / −18.6%** | +10% (2σ) |
| (a) `GGML_VK_VERIFY_SWITCH` (f32acc) | **−55.3% / −55.6%** | +10% (2σ) |

No lever remains inside T11's scope: the pipelined-submit variant was closed in F0,
subsumed by the fused one; its weakened pre-build fallback shares the family verdict;
the levers excluded by the spec stay excluded (KV quantization — measured NO-GO earlier on
this exact setup; dual-drafter in every form — closed by T8/T9; weight compression — closed by
T5/T10; GDN RS intervention — structural, required by the rollback mechanism).

Two durable conclusions, both negative and both settling the question:

1. **The VEC/DMMV dispatch for 2-9 columns is already near-optimal on this backend.** The
   batched MMQ `.f32acc` alternative is ~2.5-2.6× worse at those widths on GFX1151/RADV with
   ROCmFP4 weights. The width-dependent excess in T10's accounting is the price of the verify
   computation at that width, not an unpaid dispatch debt.
2. **The per-token draft loop beats an N-step fused graph on Vulkan.** A loop that pays N
   submits+syncs+readbacks but stops at p_min outperforms a chained graph that pays one
   submit but must evaluate the full n_max chain: flat ~38 ms/round versus k-scaled
   14.9-25.9. ggml has no data-dependent control flow, so "fuse to save syncs" is
   counterproductive exactly as long as the early-stop stays host-side.

**The structural verdict.** The ~38 ms/round software residue (30% of the round) is structural on
Vulkan/RADV for this model, within the bounds T11 was allowed to touch. What would move
it, in increasing order of departure: device-side early-stop (control flow in the backend, so
a fused graph could stop at p_min), materially cheaper fused graphs, or a different
driver/hardware generation. Both interventions remain in branch `t11` behind default-off,
inert flags — `SPEC_DRAFT_FUSED` (`common/speculative.cpp`) and `GGML_VK_VERIFY_SWITCH` with
its f32acc/mmq modes (`ggml-vulkan.cpp`) — reusable for future studies (the switch's mmq mode
was never exercised; the fused graph is the natural target once a backend early-stop exists),
with no production merge.

## 7. Instrumentation that entered the codebase

The thread's most reusable output is three instruments, one of them correcting an earlier
belief of this series:

- **The `dur(b,g,a)` draft timer works on Vulkan builds too.** T10 declared it "ROCm build
  only" — false, and the correction matters for anyone decomposing rounds on this stack. It is
  a host wall-clock around `impl->draft()` (`common/speculative.cpp:3997`, via
  `common_time_meas`; `gen_perf` is hardcoded true at `:202`), active on any backend. It covers
  the **draft phase only**: b = draft state refresh, g = draft generation, a = post-accept
  accumulation. The verify phase is *not* in the timer and is obtained as a residual: total
  JSON time − (b+g+a) per request. The `statistics draft-mtp:` lines are present in the
  Vulkan logs of every F1/F2 run — both causal decompositions of this note come from Vulkan
  builds, no ROCm image involved.
- **The `draft_aborted` counter — the instrument T10 asked for and did not have.** Non-gated
  (printed in OFF arms too), it counts p_min-aborted head-calls, closing the confounder T10
  declared when fitting per-k slopes (aborted calls were invisible in the k/G counters and
  deflated the apparent slope). Measured per request: det 14.6 (OFF) → 13.8/13.2 (ON arms of
  F1/F2), prose 86.0 → 89.2/85.8 — small, opposite-sign variations across cells, consistent
  with p_min evaluating slightly different logits under numeric drift; no anomalies, no
  envelope violations.
- **The ggml INFO→TRACE remap, documented with evidence.** The boot marker
  `vk-verify-switch active (mode=...)` is a `GGML_LOG_INFO`; the llama.cpp default log
  callback remaps INFO to LOG_LEVEL_TRACE (`common/log.cpp:435-446`) and the server's default
  threshold is 3 — so the marker is **suppressed at default verbosity**. WARN lines are
  visible at default, which turns their *absence* into a valid negative signal (zero
  `no batched pipeline` warnings in the ON arms = the diverted dispatch always found its
  pipeline). Boot markers of this kind need `-lv` ≥ 4; a diagnostic boot log preserved at
  `logs/t11/f2-marker-diag-lv4.log:154` shows the marker appearing only at that verbosity. Corollary
  for anyone grepping historical logs: no `ggml_vulkan:` INFO line appears in any prior
  benchmark log of this series — they were never printed, not lost.

## 8. Methodology: the A/B protocol

- **Within-build, always.** Both arms of each experiment run the same docker image; the only
  difference is one environment variable. Cross-build comparisons are forbidden for A/B gates
  on this stack — the char-identity lesson of
  [interleaved-decode-charidentity.md](interleaved-decode-charidentity.md): baselines are
  build-sensitive, char-exact comparisons are valid only within a build. (The two experiments
  used the two preserved images, two commits apart; each A/B is internally clean, and the OFF
  path re-baselined char-exact against the t0 baseline across all three builds involved.)
- **Interleaved boots.** OFF/ON/OFF/ON, never simultaneous, alternating ports 8193/8194,
  container self-cleanup verified (`docker ps -a` showing no leftover containers after the run). Each boot
  runs both cells with warm-up + 5 replicas: 10 replicas per arm per cell, pooled across the
  two boots per arm.
- **Markers before numbers.** For each arm directory: return codes, per-replica failure
  counts, flag-string counts (zero in OFF), constructor/engaged marker counts in ON, warning
  counts — all verified before any throughput number was read.
- **Round counting.** R = 255 − A (T10's identity: 256 tokens = 1 prefill + A accepted drafts
  + R rounds), cross-checked per replica against the log's cumulative draft-call counter —
  40/40 replicas in each experiment, zero mismatches; k = G/R; acceptance per request = A/G.
- **Gates.** tg: Δtg = mean(ON) − mean(OFF) ≥ 2·SE with SE from the pooled sample σ
  (n−1); acceptance: |Δacc| ≤ SE; content control on top (below). Report thresholds for the
  record: F1 det ≥ +1.11 / prose ≥ +0.61; F2 det ≥ +1.07 / prose ≥ +1.08 — against measured
  deltas of −4.34/−4.87 and −23.38/−14.05.
- **Content control.** Fingerprint = reasoning + content of each replica's JSON, aligned by
  index across the four OFF×ON boot pairs. Within-arm cross-boot identity (10/10 in F1, 20/20
  in F2) establishes the machine is deterministic per configuration, so cross-arm divergence
  is caused by the flag. The machine's rep-phase pattern is declared: the first post-warm-up
  replica can phase-diverge with no configuration change at all; the others reproduce
  exactly. Det cells yield a 4/5 char-identical full-window control (declared minimum
  coverage 24.5% F1 / 24.2% F2); prose cells yield none — with ~96 rounds at acceptance 0.85,
  any numeric drift re-composes the verify batches from round one, so the prose content gate
  fails structurally and the prose verdicts rest on the tg/acceptance gates plus within-arm
  determinism.
- **Analysis tooling.** One script (`scripts/t11-f1-analyze.py` in the lab workspace) produced
  both experiments' analyses, built test-first: a synthetic fixture with hand-computed values
  and an independent oracle, self-test 47/47 PASS before first use on real data, plus a
  negative control (perturbing one counter flips 10/47 checks, as expected). Reused unchanged
  on F2 (the only addition was a report-title flag; F1's outputs were regenerated
  byte-identical as a reproducibility check).

## 9. Limitations

- **Single machine, single model, single slot; two payload classes** (deterministic counting,
  free prose), n_predict 256 — the same scope as T10, chosen for comparability.
- **The verify split is a residual, not a direct timer**: total JSON − (b+g+a). The draft
  phases are directly timed; the verify residual inherits any orchestration cost between
  phases. The localization argument (98.7-98.9% of an ON−OFF delta) is differential and does
  not depend on the absolute split.
- **k is a per-request average** (head-calls over rounds), carried over from T10; per-round
  k is not decomposable with these counters. The planned per-k slope mini-sweep (n ∈
  {1,2,4,6,8}) was not executed (declared here) — the plan conditioned it on the primary gate
  passing, and it did not.
- **The switch's mmq mode was never activated**: all ON arms ran mode f32acc. The f32acc
  verdict does not automatically transfer to the q8_1 integer-dot pipeline; given the f32acc
  margin (×2.5-2.6), the burden of proof is on anyone expecting the opposite sign there.
- **Prose content gates fail structurally** (0/5 identical windows in both experiments) — a
  property of any intervention that shifts logits on a 96-round prose generation, not a
  defect of the runs; declared in every table of the primary reports.
- **No GPU tracer exists in any available image** (carried from T10): inside a phase, the
  split between launch gaps and kernel inefficiency is not directly measurable; the phase
  attribution here rests on the host-side timer plus differential A/B.
- **The two experiments ran on two images two commits apart** (the F1 image predates the
  verify-switch commit). Each A/B is within-build; only the F1-vs-F2 *comparison* of absolute
  cells is cross-build, and no conclusion here rests on it.

## 10. Reproducibility

- **Code.** Fork branch `t11` @`1df4db2aa`, a 4-commit series on main `6d55f6c7d`:
  `6312d00ea` (fused draft chain) → `866d92dcb` (marker/guard fixes) → `517afc7cc` (verify
  dispatch switch) → `1df4db2aa` (mirror/constants fixes); pushed to the fork's git server
  (`refs/heads/t11` = `1df4db2aa`). The whole series is carried in this repository as
  [`patches/t11/0001-t11-series.patch`](../../patches/t11/0001-t11-series.patch) (4 commits
  in one file, research-only: both flags failed their gates and default to off).
- **Images, preserved.** `docker-llm-service:t11-vulkan` (`41301f59806a`, @`1df4db2aa`, both
  flags) and `docker-llm-service:t11-vulkan-f1` (`81eb386179ec`, @`866d92dcb`, the F1 flag) —
  the exact binaries of the two experiments.
- **Primary reports.** The four Italian benchmark reports
  `docs/benchmarks/results-2026-08-25-t11-{f0-feasibility,f1-draft-fused,f2-verify-switch,summary}.md`
  (file-date convention: reports are named 2026-08-25; all runs executed 2026-08-24)
  are the primary sources of this note — every number above is cited to them; they are written
  in Italian and live in the lab workspace, as do the logs.
- **Logs** (lab workspace, `logs/t11/`): build logs `build-000{1,2,2b,3,3b}.log`; the t0
  baseline and smoke runs (`t0-baseline/`, `f1-{off,on}-smoke/`, `f2-{off,on}-smoke/`); the
  A/B runs `f{1,2}-ab-run-{1-off,2-on,3-off,4-on}.log` with per-replica JSON/TSV under
  `f{1,2}-ab-{off1,on1,off2,on2}/`; the analyses `f{1,2}-ab-analysis.{md,json}`; and the
  verbosity diagnostic `f2-marker-diag-lv4.log` (marker at line 154).
- **Scripts** (lab workspace): `t11-f1-analyze.py` + its fixture directory (self-test 47/47)
  as described in section 8; the runs themselves used the T10 harness `t10-bench.sh` with the
  single additive `--env` flag introduced for this thread, and the T10 payloads verbatim
  (`t10-payloads.json`).
- **Spec and plan.** `docs/superpowers/specs/2026-08-24-t11-round-software-design.md` and
  `docs/superpowers/plans/2026-08-24-t11-round-software.md` (lab workspace) hold the
  pre-registered expectations, gates and branch rules quoted in sections 1, 2 and 5.
- **Hardware and flags.** AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151, 128
  GB unified LPDDR5X), Vulkan (RADV) build, target `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN`
  (13.8 GiB), temp 0, ctx 16384, `-fa on`, p_min 0.75, p_split 0.10, n_max 6, parallel 1,
  single stream, dedicated GPU window, warm-up discarded.

Pointers: the companion notes are
[speculative-round-budget.md](speculative-round-budget.md) (T10, whose decomposition and
discard-on-EV decision this thread executed to the end),
[per-round-drafter-switching.md](per-round-drafter-switching.md) (T9) and
[dual-drafter-synergy.md](dual-drafter-synergy.md) (T8) — the speculative round has now been
attacked from every direction this lab could justify, and every attack is closed by
measurement.

---

*Nothing shipped and no gate cleared — but the round's last unrecovered 38 milliseconds are now
spoken for twice over: the levers that promised to recover them were built, measured, and both
came back with the opposite sign.*
