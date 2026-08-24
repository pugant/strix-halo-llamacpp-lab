# strix-halo-llamacpp-lab — llama.cpp experiments for AMD Strix Halo (Ryzen AI MAX+ 395)

## TL;DR

- **What** — llama.cpp server experiments focused on one machine class: per-request speculative-decoding **routing** (built-in MTP ⇄ external DFlash2 block-diffusion drafter), a **reasoning ("thinking") budget** with cache-friendly truncation, **spec-boundary cache salvage**, and a complete **ROCmFP4-STRIX_LEAN quantization pipeline** (imatrix → quantize → sanitize → publish).
- **Where it runs** — AMD Strix Halo: Ryzen AI MAX+ 395, Radeon 8060S iGPU (gfx1151, RDNA 3.5), 128 GB unified LPDDR5X, inside ROCm 7.2.4 containers built from [kyuz0](https://github.com/kyuz0)'s [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).
- **Status** — the dual-drafter routing server is live in daily production since the morning of **2026-08-20** (published the same day), on the author's machine. Every performance claim in this README was measured on our hardware; nothing is projected or taken from vendor material.
- **Code** — the **full buildable source of the runtime fork is included in this repo under [`rocmfpx/`](rocmfpx/)** (snapshot of our `drafter-routing` branch, a fork of [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), itself a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). The same work is also carried as `git am`-clean patches, plus the scripts and the raw benchmark notes.
- **Models** (Hugging Face, weights under their own licenses):

| Repo | What it is |
|---|---|
| [`pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN`](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN) | Dense 27B, 13.8 GiB — the canonical model of this lab |
| [`pugant/Qwen3.8-27B-imatrix`](https://huggingface.co/pugant/Qwen3.8-27B-imatrix) | The importance matrix used by the preset above |
| [`pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, reasoning/tool-call finetune |
| [`pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, multimodal |
| [`pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN) | MoE 35B-A3B, multimodal — MTP head measured, NOT recommended (degraded by the finetune) |
| [`pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN) | Mamba-hybrid MoE |
| [`pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX`](https://huggingface.co/pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX) | MoE 35B-A3B at Q6_0 |

**Repository layout** — `rocmfpx/` (**full runtime fork source, buildable — clone this repo and compile**) · `patches/` (all features, `git am`-able) · `scripts/` (download / imatrix / quantize / sanitize / test / bench) · `docker/` (convert container) · `docs/` (raw experiment notes) · [LICENSE](LICENSE) · [NOTICE](NOTICE).

---

## 📖 Featured research note

**[Dual-drafter synergy: a research line closed by measurement](docs/research/dual-drafter-synergy.md)** — the full story of our attempt to make two speculative drafters *cooperate inside a single decode round*. The routing server above already runs both drafters, one per request; the bet was to make them cooperate in the same round (condition the DFlash2 block on the MTP head token) for **3–5× the accepted tokens per round**. Seven gated steps later, the whole same-round design space is closed by data — the last step by pure arithmetic: even under *perfect* complementarity between the two chains, no same-round design can meet any positive throughput criterion. The realized synergy is **alternation**: per-request routing, **+19% agentic**, in production since 2026-08-20.

| Design | Closed by | One-line epitaph |
|---|---|---|
| Concat, k1=1 (condition block on head) | Phase A gate: 0.831 < 0.90 content-controlled; −27.7% tok/s | The head is accepted 98%; the conditioned block is the problem |
| …failure mode | Phase 0: copy-at-period, 103/111 increment-slot proposals, 47/47 categorical stops | Not wiring — the drafter clones the pattern item when conditioned |
| Exclusion (gate the concat off on patterns) | G3: 0.8346 < 0.90 with recovery real (−27.7% → −2.0% tok/s) but ceiling 0.86 | A perfect in-region detector cannot fix windows with non-pattern content |
| Pooled "value" on residual | Deep-dive: same-slots ratio ≤ 1.0 in 8/9 windows vs pooled 1.36 | The metric was gameable by slot selection |
| KV-pollution rescue / clean-context value | 2×2 twin: same-slots ratio 0.81 ≤ 1.00, cost 7% tok/s, on the maximally favorable context | The conditioned draft accepts less than plain, pollution or not |
| Depth (n-max 8/9) | Clamped to 7 by the drafter's trained block size; DF8/DF9 bit-identical to DF7 | Bottleneck is draft length, not acceptance (0.92 conditional at pos 7) |
| Coverage (two-root tree) | Counterfactual: −20.1% det / −17.6% prose; Fréchet max 7.63 < 8.25 needed (det), 2.96 < 3.02 (prose) | No correlation between the chains can meet any positive criterion |

Full narrative with all 26 data tables: **[docs/research/dual-drafter-synergy.md](docs/research/dual-drafter-synergy.md)**.

Also in this series: **[ROCmFP3 (Q3_0_ROCMFPX) on Qwen3.8-27B: not a true 3-bit on hybrid gated-deltanet architectures](docs/research/2026-08-23-rocmfp3-quality-speed.md)** — the T5 thread, closed NO-GO: K-quant-protected attention pushes the "3.50 bpw" preset to 4.44/5.72 effective bpw, so there are no bytes to save and tg only pays the K-quant-path tax.

And closing the dual-drafter line itself: **[Per-round drafter switching: the twin-run measurement that closed the line](docs/research/per-round-drafter-switching.md)** — the T9 thread. Same-round cooperation (featured above) was closed by measurement; the last untested axis was choosing the drafter *per decode round*. A shadow drafter wired into the server's round loop produced the first public same-round paired dataset for an MTP6/DFlash2 pair (1,433 rounds, published with the note). Two findings close it: acceptance correlation ρ = +0.729 pooled (round difficulty dominates drafter complementarity) and a twin cost of ~83 ms/round to keep the second drafter synchronized — more than an entire MTP round itself, so every per-round oracle loses to the best static drafter by 66-74%. The run's own pre-registered timing gate declared the measurement NO-GO, so the numbers are published as exploratory; per-request routing remains the practical boundary of the two-drafter idea.

Follow-up diagnosis: **[Why an interleaved decode breaks char-exact reproducibility on this stack](docs/research/interleaved-decode-charidentity.md)** — the engine-level question the twin-run note left open, closed with three diagnostic modes wired into the server: the cause is the **multi-row interleaved decode on the target context itself** (an 8-row decode with dummy tokens and full rollback still flips the greedy output; the drafter's internal state and even a 1-row decode are bit-identical to baseline; first divergence is the target's committed token). Corollaries: a rollback is not numerically neutral after a >1-row decode, and baselines are build-sensitive — char-exact comparisons only within a build.

And answering the throughput question itself: **[Round budget: the speculative round decomposed to the millisecond, and the 3 bpw door closed](docs/research/speculative-round-budget.md)** — the T10 thread. How much decode throughput is still recoverable on the production quant? Both remaining levers are closed by measurement: ~3 bpw weight compression fails the perplexity gate ~5× (kernels verified healthy — the quantization is simply too aggressive for this dense model), and the one concrete kernel candidate has a best case exactly at the 2σ gate. The central deliverable is the round itself, decomposed component by component — weights/KV/SSM floor, per-token GDN rollback snapshots, per-row verify KV-read, head-call floor, software residual — with the accounting identity closing at 136.43 vs 136.42 ms measured, and a code-level falsification of the "each verify row re-reads the output head" hypothesis (the Vulkan VEC pipeline shares each dequantized weight across all batch columns). Ceiling ladder: 43 tok/s measured → 60.6 with all software recovered → 69.8-79.2 at pure bandwidth. Bonus: the historical "+53% ROCm on dense models" reading is resolved as a protocol artifact — at unified protocol the two backends are at parity on plain decode.

And spending it: **[Round software: the speculative round's ~38 ms residue attacked with both remaining levers, and both falsified](docs/research/speculative-round-software.md)** — the T11 thread. T10 left the software residue (~38 ms/round, 30% of the round) as the only untried lever; T11 built both remaining interventions and measured each in a within-build A/B — a **fused draft chain** collapsing N draft decodes into one graph (−10.2%/−18.6%: ggml has no data-dependent control flow, so the p_min early-stop stays host-side and the fused graph pays the full n_max=6 chain every round — flat ~38 ms/round against a per-token loop that scales with accepted length), and a **verify dispatch switch** moving the verify batch onto the batched MMQ pipelines (−55.3%/−55.6%: the batched path is ~2.5-2.6× *slower* than VEC/DMMV at 2-9 columns — the VEC path was already near-optimal, and T10's "width-dependent excess" is the real cost of the computation, not a dispatch artifact). Conclusion: the residue is structural on Vulkan/RADV for this model; both interventions stay in branch `t11` behind default-off flags (patch series in [`patches/t11/`](patches/t11/)). Bonus: the fork's `dur(b,g,a)` draft timer turns out to work on Vulkan builds too, a non-gated `draft_aborted` counter closes T10's per-k confounder, and the ggml INFO→TRACE log remap is documented (boot markers need `-lv` ≥ 4).

---

## ⚠️ Read this first

Full transparency, before anything else:

- **This is an unofficial, experimental community project.** It is *not* affiliated with, endorsed by, or sponsored by AMD, ggml-org, or any hardware vendor.
- **MIT license, "as is", NO WARRANTY.** Use it only if you accept full responsibility for what it does on your machine.
- **FP4 is software-emulated on RDNA 3.5.** gfx1151 has no FP4 silicon; the ROCmFP4-STRIX_LEAN preset wins on *memory bandwidth* (roughly 4x smaller weights on a bandwidth-starved unified-memory part), not on FP4 compute. Do not expect datacenter-FP4 numbers or semantics.
- **Your mileage WILL vary.** Every number here was measured on one specific machine (ours), one container stack, one set of models — careful point measurements, not statistics. See the [methodology note](#methodology).
- **Back up your data** before running any container or script from the internet, this repo included.
- **Do not blame the toolchain authors.** The containers come from kyuz0's toolboxes, the fork from charlie12345, the foundation from ggml-org. If something here breaks, the fault is ours, not theirs.

---

## Credits 🙏

This lab adds a thin layer on top of giants' work.

- **ggml-org / llama.cpp** — Georgi Gerganov and contributors. The main branch of llama.cpp is the foundation of everything here and is invaluable to the whole local-inference community. Thank you for keeping it open.
- **charlie12345** — maintainer of the ROCmFPX fork of llama.cpp, where the ROCmFP4 / `Q4_0_ROCMFP4_STRIX_LEAN` preset, the HIP kernels and the GGUF extensions live. Most of the code this lab builds upon — and merges, PR #67–#82, including our own #69 — is their work.
- **kyuz0** — the `amd-strix-halo-toolboxes` / `docker-llm-service` containers this lab builds and runs on.
- **The Strix Halo community** — where the ROCmFP4-STRIX_LEAN preset was born and tuned collectively; its benchmarks, feedback and hardware knowledge shaped every decision in this repo.
- **Jian Chen** — author of the upstream DFlash2 support (llama.cpp PR #27342) that we ported onto the fork.

If we forgot anyone: it is an omission, not an intent — open an issue and we will credit you.

---

## What we built

### Per-request drafter routing (MTP ⇄ DFlash2)

**What.** One `llama-server` holding *two* draft contexts at once: the target model's built-in MTP layer (nextn) and an external DFlash2 block-diffusion draft model. Each request is routed to one drafter; the other stays idle for that request (verified: zero parasitic draft work on the inactive drafter).

**Why.** The measured asymmetry is stark: DFlash2 dominates deterministic and agentic content but loses ~26% on free prose; MTP is the reverse. No single drafter wins — so the server picks per request. Details in [Benchmarks](#benchmarks).

**How.**

- Routing policy keys on the `tools` signal: agentic/tool-calling requests → DFlash2, prose → MTP.
- Optional per-request override in the body: `"spec_drafter": "mtp"` or `"dflash"`.
- Prompt-cache entries are tagged with the drafter that produced them, so switching drafters between turns does not force a cold target context (the target KV is drafter-independent).
- Boot fallback: if the draft-model file is missing, the server degrades to mono MTP-nextn with a warning; an explicit `"spec_drafter"` override then returns a clear 400.
- Metrics: Prometheus counter `spec_route_cache_rebuild_total` (per kind), plus `spec-route:` log lines.

**Patch.** [`patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch`](patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch) — this single file contains the full 14-commit series, `git am`-clean (the resulting code is already included in [`rocmfpx/`](rocmfpx/)).

**Status.** In daily production since 2026-08-20; validated by gates T1–T5 (see [Validation gates](#validation-gates-t1t5)).

### DFlash2 block-diffusion drafter porting

**What.** Port of upstream llama.cpp PR #27342 ("spec: add DFlash2 support") adapted to the fork's internals — commits `ba2485545`, `ebf1cc855`, `fdd53c521` (all included in [`rocmfpx/`](rocmfpx/)). Usage notes in [`rocmfpx/docs/dflash2.md`](rocmfpx/docs/dflash2.md).

**Outcome.** NO-GO as a full replacement — free prose pays −26% tg — but it set the deterministic record on our hardware: **57.4 tok/s**. This asymmetry is exactly what motivated the routing work above.

### Reasoning budget (thinking cap)

**What.** A hard cap on reasoning tokens, server-wide or per request, with a forced-end sequence that does not poison the prompt cache.

**How.**

- Server flag: `--reasoning-budget`.
- Per-request body fields: `thinking_budget_tokens` (exact name), `thinking_token_budget` (vLLM-style alias), `reasoning_budget_message` (OpenAI-compat wrap-up message).
- The forced-end sequence aligns the newline + `'\n\n'` coda so that a budget-truncated turn stays **cache-friendly on resend**: the next request reuses the KV instead of re-processing the whole prompt.

**Where it lives.** The cache-resend alignment was merged in the fork as [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69). For the related upstream report see [ggml-org/llama.cpp issue #21831](https://github.com/ggml-org/llama.cpp/issues/21831) ("Server forces full prompt re-processing on subsequent requests"); our prepared upstream patch is archived in [`patches/upstream-llamacpp/`](patches/upstream-llamacpp/README.md).

**Status.** In production (hard cap + soft wrap since 2026-08-18; the dual-routing stack re-validates it via gate T3).

### Spec-boundary cache salvage

**What.** When a speculative-decoding conversation crosses a cache boundary cold, the server recovers instead of re-processing everything: first a bounded trailing rollback (part of PR #69), then a context-checkpoint-based rollback (patch `0007` in [`patches/spec-cache-trailing-rollback/`](patches/spec-cache-trailing-rollback/)).

**Measured effect.** ~91% of prefill tokens salvaged on spec-boundary cold starts, on our 12-request agent workload (design doc: [`docs/design/2026-08-19-t7f2-drafter-routing-design.md`](docs/design/2026-08-19-t7f2-drafter-routing-design.md), §4).

### Device split-buffer checkpoint restore (fix)

**What.** A rare failure (1 task in 16 in production logs) — `device checkpoint restore buffer mismatch` refused the draft-state restore during cache salvage and forced a **full** prompt re-processing (~43k tokens, ~2.5 min) even though the target state had already been restored fine.

**Root cause.** The device-side state write emits one block per contiguous cell range (and per state-row group on recurrent memories), while the read always emits one block per layer tensor — same bytes, different split, rejected by the per-buffer geometry check.

**Fix (in production since 2026-08-20).** When the per-buffer total size matches, the saved blocks are copied serially through cursor views instead of refusing the restore (log marker: `restored split device buffers`); a WARN at save time makes the multi-range condition observable. Patch: [`patches/ckpt-device-split-restore/0001-ckpt-device-split-restore-tool-fix-warn.patch`](patches/ckpt-device-split-restore/0001-ckpt-device-split-restore-tool-fix-warn.patch) — 3 commits, including `llama-state-split-test`, a deterministic repro/verification tool (fill a sequence, punch a mid-hole to force 2 physical cell ranges, then save/wipe/restore on device).

### SPEC_VERIFY_LOG instrumentation

**What.** Optional per-position acceptance logging of the MTP verify batch — [`patches/spec-verify-log/0001-spec-verify-log.patch`](patches/spec-verify-log/0001-spec-verify-log.patch).

**Why it matters.** It made the acceptance analysis possible: the verify batch is strongly **bimodal** — on positions where the draft matched, acceptance ≈ 1.0; where it mismatched, ≈ `p_draft`. Every workload characterization in this README rests on that instrumentation.

### Documented NO-GO experiments

Transparency is part of the method. These were built, measured, and rejected:

- **Typical-acceptance MTP decoding** — typical sampling was *always* worse than exact match (best case Δ −0.24). Closed NO-GO.
- **Reasoning-pressure steering (notice + squeeze)** — [`patches/reasoning-pressure/0010-reasoning-pressure.patch`](patches/reasoning-pressure/0010-reasoning-pressure.patch), design in [`docs/design/2026-08-17-reasoning-pressure-design.md`](docs/design/2026-08-17-reasoning-pressure-design.md). 9/10 runs exhausted the budget anyway, and cache resend broke under pressure. NO-GO, kept for the record.
- **Thinkingcap budget-aware anchor line** — the anchor line caused quality regressions. NO-GO.
- **Dual-drafter same-round synergy (T8)** — conditioning the DFlash2 block on the MTP head token, and every follow-up design (pattern exclusion, deeper drafts, a two-root verify tree). Closed by measurement, not fatigue: the full story with all the numbers is in [`docs/research/dual-drafter-synergy.md`](docs/research/dual-drafter-synergy.md). The realized synergy is the per-request **routing** described above.

Negative results are results. They are kept in the repo so nobody (ourselves included) has to re-run them.

---

## Benchmarks

All numbers: **Qwen3.8-27B ROCmFP4-STRIX_LEAN (13.8 GiB)**, Vulkan RADV build, temp 0, single stream, `p_min 0.75`, ctx 16384 — unless noted otherwise.

### Main table — which drafter, which workload

| Setup | prose tg (tok/s) | deterministic tg (tok/s) | Source |
|---|---|---|---|
| MTP n6 (control) | 19.6 / 20.2 | 45.2 / 26.1 | [`results-2026-08-19-dflash2-vs-mtp.md:15`](docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md) |
| DFlash2 n7 | 14.2 / 15.0 | **57.4 / 36.3** — record | [`:16`](docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md) |
| DFlash2 n5 (best single drafter) | 17.5 / 15.9 | 52.2 / 39.5 | [`:17`](docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md) |
| **Dual routing (production)** | 19.7 / 20.4 (≈ MTP) | 54.7 / 40.7 (+19.2% / +20.8%) | [`results-2026-08-20-drafter-routing-t1-t5.md:30-36`](docs/benchmarks/results-2026-08-20-drafter-routing-t1-t5.md) |

Each cell is two runs. "Deterministic" = counting/alphabet-style prompts. Dual-routing prose values are prompts P1/P2 of the T4 table. The MTP control row and the T4 gate runs were **separate sessions** (e.g. mono prose 19.6/20.2 here vs 19.8/20.4 in the T4 table) — compare deltas, not absolute cells, across sessions. Read the table as: routing keeps MTP-class prose speed *and* captures most of the DFlash2 deterministic win, in one server.

### Why routing works: workload and acceptance

Agentic workload, MTP n6 vs DFlash2 n7 (tok/s) — [`results-2026-08-19-dflash2-vs-mtp.md:73-75`](docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md):

| Prompt | MTP n6 | DFlash2 n7 | Delta |
|---|---|---|---|
| coding (10 functions) | 27.5 | 33.8 | **+23%** |
| JSON (30 objects) | 35.0 | 36.2 | +3% |
| log (20 fixed-format lines) | 28.0 | 35.8 | **+28%** |

Acceptance, per draft position ([`:26-30`, `:77-79`](docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md)):

- On deterministic/structured content, DFlash2 stays **≥ 0.90 through position 7** (0.99, 0.97, 0.96, 0.94, 0.92, 0.91, 0.90).
- On agentic content it decays but holds **≥ 0.50 at position 7** (0.92, 0.77, 0.71, 0.65, 0.56, 0.52, 0.51), where MTP drops to 0.25–0.50.
- MTP acceptance **collapses after position 1** on structured content (0.93 → 0.33): long drafts are wasted there, which is precisely the slack DFlash2 picks up.

### Validation gates T1–T5

The routing feature shipped only after these gates passed ([`results-2026-08-20-drafter-routing-t1-t5.md:19-24`](docs/benchmarks/results-2026-08-20-drafter-routing-t1-t5.md)):

| Gate | What it checks | Result |
|---|---|---|
| T1 smoke dual-load | 14 checks: boot / policy / override / 400 / fallback / cache-switch / metrics | **PASS 14/14** |
| T2 cache round-trip | 4 gates per config (simple + production), drafter switched every turn | **PASS 4/4** per config |
| T3 sacred paths (patches 0005–0009) | budget-forced end, altered resend, trailing rollback, checkpoint restore — in dual mode | **PASS 7/7** |
| T4 routing vs mono | prose ≥ −3%; agentic ≥ +10% (mean of 3 prompts) | **PASS** — prose +1.0% / −0.2%; agentic +19.6% / +18.9% |
| T5 numerical spot-check | dual-MTP ≡ mono ckpt7, dual-DFlash ≡ mono DF7 (greedy) | **PASS** — char-identical, 4/4 |

### Routing per-prompt deltas (T4)

[`results-2026-08-20-drafter-routing-t1-t5.md:30-36`](docs/benchmarks/results-2026-08-20-drafter-routing-t1-t5.md):

| Prompt | Class | Mono MTP6 | Dual | Delta | Routed to |
|---|---|---|---|---|---|
| P1 | prose | 19.8 | 19.7 | −0.5% | mtp |
| P2 | prose | 20.4 | 20.4 | +0.0% | mtp |
| D1 | deterministic | 45.9 | 54.7 | **+19.2%** | dflash |
| D2 | deterministic | 33.7 | 40.7 | **+20.8%** | dflash |
| A1 | agentic | 38.6 | 46.0 | **+19.2%** | dflash |
| A2 | agentic | 42.0 | 51.1 | **+21.7%** | dflash |
| A3 | agentic | 39.9 | 46.2 | **+15.8%** | dflash |

### Backend choice by model class: ROCm vs Vulkan

On dense 27B models the fork's ROCm build beats its Vulkan build on generation; on MoE FP4 the Vulkan build wins ([`results-2026-08-15-qwen38-27b.md:58-60`](docs/benchmarks/results-2026-08-15-qwen38-27b.md)):

| Metric | ROCm | Vulkan RADV | Note |
|---|---|---|---|
| plain tg128 (dense 27B) | **13.8** | 9.0 | ROCm **+53%** |
| plain pp512 (dense 27B) | **354.7** | 319.8 | ROCm +11% |
| tg (MoE 35B FP4 class) | 71.2 | **81.6** | Vulkan fork wins |

Practical rule we follow: dense → ROCm; MoE FP4 → Vulkan fork. (The speculative-decoding tables above are all Vulkan RADV.)

### Methodology

- **One machine** — the author's Strix Halo (Ryzen AI MAX+ 395, 128 GB), dedicated GPU window (production service stopped during runs).
- temp 0, single stream, warm-up discarded, **2–3 runs per number** — these are not statistical means; treat them as careful point measurements.
- Full raw data, logs and per-experiment setup live in [`docs/benchmarks/`](docs/benchmarks/) — they are the raw working notes. This README is the summary and entry point.

---

## Full replication guide (canonical: Qwen3.8-27B)

End-to-end: BF16 GGUF → imatrix → ROCmFP4-STRIX_LEAN quant → sanitized GGUF → fork server build → dual-drafter server. Every script referenced below is in this repo under `scripts/`.

### Step 0 — Prerequisites

- AMD Strix Halo with 128 GB unified memory (Ryzen AI MAX+ 395), a ROCm-capable container stack, and an HF account.
- Free disk: plan for **~75 GB** end-to-end (the BF16 shards alone are ~54.7 GB, plus imatrix and the ~14.8 GB quant output; the quantize script itself pre-checks for 40 GB free at its step).
- Build the **base container first** from [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) (the `docker-llm-service` image). Nothing here works without it.

### Step 1 — Build the convert container

```bash
docker build -t docker-llm-service:convert -f docker/Dockerfile.convert .
```

Note: the base image (`docker-llm-service:latest`) is **not publicly pullable** — it must be built locally from the toolboxes (Step 0). The convert container adds the Python deps needed by the fork's `convert_hf_to_gguf.py` and its custom `gguf-py`.

### Step 2 — Download the target (BF16) and the DFlash2 drafter

```bash
LLMODELS_DIR=~/llmodels HF_TOKEN=... bash scripts/download-qwen38-bf16.sh

# DFlash2 draft model (1.9B; the Q4_K_M file is ~1.1 GB):
huggingface-cli download incoai/Qwen3.8-27B-DFlash2-GGUF \
  --include "*Q4_K_M*" --local-dir "$LLMODELS_DIR/models/QWEN3.8"
```

Env: `LLMODELS_DIR` (default `$HOME/llmodels`), `HF_TOKEN`. The script downloads `unsloth/Qwen3.8-27B-GGUF` BF16 (2 shards, ~54.7 GB) + mmproj; idempotent (`wget -c`). On our high-latency network the `hf`/xet downloader sometimes stalls on files around 1 GB — `wget -c` with a Bearer header is the fallback we use (see the notes in `docs/benchmarks/`).

### Step 3 — Importance matrix

**Canonical path — use the published matrix:** download it from [`pugant/Qwen3.8-27B-imatrix`](https://huggingface.co/pugant/Qwen3.8-27B-imatrix).

**Regenerating it yourself** (what we did): the calibration corpus is a composite of agentic traces + technical prose + real code (~1.7 MB), because the quant must serve both chat and tool-calling:

```bash
python3 scripts/prep-qwen38-calibration.py   # build the calibration corpus
bash scripts/run-imatrix-qwen38.sh           # GPU run: 256 chunks x 512 (probe first)
python3 scripts/check-imatrix-coverage.py    # coverage gate
```

The coverage gate must pass: **all 496 expected tensors covered, layers 0–63, zero NaN**.

### Step 4 — Quantize

```bash
bash scripts/quantize-qwen38-27b.sh
```

Produces `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (≈ 13.8 GiB, 4.34 bpw, MTP layer included; automatic dry-run before the real pass).

### Step 5 — Sanitize GGUF metadata

```bash
# CRITICAL: run this ONLY inside the convert container.
# <config.json> is the model's HF config (tokenizer/architecture metadata).
docker run --rm \
  -v "$LLMODELS_DIR/models:/llmodels" -v "$PWD:/lab:ro" \
  docker-llm-service:convert \
  python3 /lab/scripts/sanitize-gguf-v3.py <input.gguf> <output.gguf> <config.json>
```

The host-side `gguf-py` does **not** know the fork's custom tensor types — running the sanitizer on the host corrupts or rejects the file. This is the single most common way to waste an afternoon here.

### Step 6 — Build the fork's llama-server (Vulkan RADV)

The server binary is **not** distributed: build it from the sources, **already included in this repo under `rocmfpx/`** (no extra clone needed). We build inside Docker with the Vulkan Dockerfile from kyuz0's toolboxes, feeding it our sources instead of its default clone (`$TB` = your checkout of [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)):

```bash
mkdir -p image-build/src
cp -r rocmfpx/. image-build/src/          # the full fork source, included in this repo
sed -e 's|^RUN git clone -b ${BRANCH} --single-branch ${REPO} \.$|COPY src/ .|' \
    -e 's|^RUN git clean -xdf \\$|RUN patch -p1 < /tmp/llama-grammar.patch \\|' \
    -e 's|^  \&\& patch -p1 < /tmp/llama-grammar.patch \\$|  \&\& true \\|' \
    "$TB/toolboxes/Dockerfile.vulkan-rocmfpx" > image-build/Dockerfile
cp "$TB"/toolboxes/{llama-grammar.patch,rocmfpx-vulkan-shader-concurrency.patch,gguf-vram-estimator.py} image-build/
docker build -t docker-llm-service:vulkan-fork-dflash2-route image-build
```

The companion files come from the toolboxes too — both patches (`llama-grammar.patch` and `rocmfpx-vulkan-shader-concurrency.patch`) are required by their base Dockerfile and are their work, not ours. Build time is ~7 min on our toolboxes revision; upstream `main` builds with `--parallel 1` (to keep concurrent shader generation from exhausting memory) and will run longer. **Verified backend:** every `llama-server` invocation and every speculative-decoding benchmark was verified on the fork's **Vulkan (RADV)** build produced here — the ROCm column of the backend-choice table was measured on the fork's ROCm build. The full `docker run` pattern we use (devices, render group, `--entrypoint llama-server`) is in `scripts/test-drafter-routing-t1.sh`.

### Step 7 — Serve with dual drafters

Verified invocation — the flags exactly as we run them in production and in the test suite:

```bash
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 -fa on --jinja -c 16384 \
  --spec-type draft-mtp,draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-draft-ngl all --spec-draft-n-max 7 \
  --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
```

Flag notes, learned the hard way:

- `--spec-type` takes a **comma-separated list** of draft implementations. There is **no** `--spec-draft-type` flag.
- `--spec-draft-p-split` is accepted for CLI compatibility but is a **no-op in `llama-server`** — its only consumer is the upstream `examples/speculative` example.
- The `-k` / `-v` variants floating around in discussions are KV-cache type flags, not drafter selection.
- In dual mode the server clamps MTP to n-max 6 and DFlash2 to 7 (its block size is 8).
- The DFlash2 drafter is [`incoai/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) (1.9B params; the Q4_K_M file is ≈ 1.1 GB).

### Step 8 — Exercise the routing

```bash
# 1) tools present in the request body -> auto-routed to DFlash2
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"List the files in /tmp"}],
       "tools":[{"type":"function","function":{"name":"list_files",
                  "description":"List files in a directory",
                  "parameters":{"type":"object","properties":{}}}}]}'

# 2) explicit override, no tools needed
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 50, one per line"}],
       "spec_drafter":"dflash"}'

# 3) per-request thinking budget
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Solve this step by step"}],
       "thinking_budget_tokens":4096}'
```

### Step 9 — Smoke-test the routing

```bash
bash scripts/test-drafter-routing-t1.sh
```

14 checks: dual boot / default policy / override / invalid-enum 400 / boot fallback / cache switch across drafter changes / metrics endpoint. Companion suites: `test-drafter-routing-t2.sh` (cache round-trip), `test-drafter-routing-t3.sh` (sacred paths), `test-ckpt-rollback-t1t2.sh` (checkpoint rollback), `bench-routing-vs-mono.sh` (the T4 A/B).

### Other models

The same pipeline produced the other published quants. Their quantize scripts are **not** in this repo — the canonical, fully scripted flow is the Qwen3.8-27B one above.

| Model | Class | HF card |
|---|---|---|
| grug-35b-v2 | MoE 35B-A3B, reasoning/tool-call finetune | [`pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN) |
| Ornith-1.0-35B | MoE 35B-A3B, **multimodal** | [`pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN) |
| Ornith-1.5-35B | MoE 35B-A3B, **multimodal**; ships an MTP head whose pos-2 acceptance collapsed to ~0.07 (0.99 pos-1) — speculative decoding loses to plain at every n-max | [`pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN) |
| Nemotron-3.5-Lightning-30B-A3B | **Mamba-hybrid** MoE | [`pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN) |
| Qwen3.6-35B-A3B | MoE 35B-A3B, Q6_0 | [`pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX`](https://huggingface.co/pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX) |

---

## Patch index

| Patch | Series | Purpose | Upstream status |
|---|---|---|---|
| [`patches/spec-cache-trailing-rollback/`](patches/spec-cache-trailing-rollback/) | 9-patch series — breakdown below | Spec-boundary cache salvage + reasoning-budget request surface | partially merged — PR #69 |
| [`patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch`](patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch) | full 14-commit series in one file | Dual draft contexts, per-request routing, drafter-tagged cache, boot fallback, metrics | included in [`rocmfpx/`](rocmfpx/) |
| [`patches/reasoning-pressure/0010-reasoning-pressure.patch`](patches/reasoning-pressure/0010-reasoning-pressure.patch) | single | Reasoning-pressure steering (notice + squeeze) — **NO-GO experiment**, documented, not for production | archived |
| [`patches/spec-verify-log/0001-spec-verify-log.patch`](patches/spec-verify-log/0001-spec-verify-log.patch) | single | Per-position acceptance instrumentation of the verify batch | instrumentation only, not in the runtime build |
| [`patches/upstream-llamacpp/0001-server-reasoning-budget-forced-newline.patch`](patches/upstream-llamacpp/0001-server-reasoning-budget-forced-newline.patch) + [`README.md`](patches/upstream-llamacpp/README.md) | single | The reasoning-budget forced-newline fix as prepared against ggml-org master | **PR archived, never sent** — see the patch dir README for the story |

Breakdown of [`patches/spec-cache-trailing-rollback/`](patches/spec-cache-trailing-rollback/):

- `0001` + `0003` — bounded trailing rollback at the spec boundary (**merged upstream** via [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69))
- `0002` — checkpoint-based rollback for the spec-boundary cache
- `0004` — push all batch/verify rows into the MTP boundary state
- `0005` — forced-end sequence prefixed with a newline, for cache round-trip
- `0006` — accept `thinking_token_budget` (vLLM name) as an alias
- `0007` — checkpoint-based salvage (the ~91% prefill figure)
- `0008` — accept `reasoning_budget_message` per request (OpenAI-compat)
- `0009` — newline between wrap-up message and end tag

All patches apply with `git am` on the appropriate branch of the fork.

---

## Relationship to upstream

```text
ggml-org/llama.cpp (main)
  └── charlie12345/ROCmFPX          (ROCmFP4 preset, HIP kernels, GGUF types)
        └── pugant fork (GitHub, since removed): branch drafter-routing
              = charlie main + upstream merges PR #67–#82 + our work
              (routing, DFlash2 port, reasoning budget, cache salvage)
                    │  snapshot of its final state (34a127168)
                    ▼
  pugant/strix-halo-llamacpp-lab  ← THIS REPO — the full fork source
                                    included in rocmfpx/ (buildable),
                                    plus the same work as git am-able
                                    patches, docs and replication scripts
```

What is merged where:

| Work | Where it lives |
|---|---|
| Spec-boundary cache trailing rollback (+ reasoning-budget resend alignment) | Merged in [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69) |
| Reasoning-budget forced-newline, prepared for ggml-org | Upstream PR **archived, not sent** — story in [`patches/upstream-llamacpp/README.md`](patches/upstream-llamacpp/README.md) |
| Everything else (drafter routing, DFlash2 port, verify-log, remaining cache patches) | **Included in this repo**: full source in [`rocmfpx/`](rocmfpx/), plus the `git am`-able series in `patches/` |

> History note: our pull requests to the fork (incl. #69, merged; #80, open at
> the time) went through the now-removed GitHub fork; the patches and this
> snapshot preserve everything.

---

## License & attribution

- Code and scripts in this repo: **MIT** — see [LICENSE](LICENSE).
- Credits and attribution: [NOTICE](NOTICE).
- **No model weights in this repo.** Weights live on Hugging Face under their own licenses; the base-model licenses carry through to the quants.
