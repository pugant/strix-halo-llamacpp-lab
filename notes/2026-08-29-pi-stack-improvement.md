# Strix Halo pi-stack cycle: hipCUB top_k regresses, KV quantization costs 19-26%, n-gram has no terrain, drafter-state rollback fix — measured results

> Final (29/08): all gates closed, candidate deployed to production as `-rs2` (f16 K/V, drafter-reset fix live). Full internal cycle: spec `2026-08-28-pi-stack-improvement-design.md`, reports `results-2026-08-28-pi-stack-*.md`.

## TL;DR (bozza)

- Ported the (closed, unmerged) upstream PR ggml-org/llama.cpp#27874 to the HIP build (option `GGML_CUDA_USE_CUB` + explicit `-DGGML_CUDA_USE_CUB=ON`, `hipcub-devel` from the Fedora 43 + ROCm 7.2.4 repos). Activation marker verified in-runtime (production build: 4× `does not have support for op TOP_K ... needed for sampler 'top-k'`; enablement build: 0). **Result: pp@131k B/A = 0.58 (−42%, reps stable 133.0 vs 77.2 t/s) — the naive hipCUB enablement REGRESSES deep-context prefill on gfx1151 in llama-bench fresh.** Measure, don't assume. (tg@8k 1.02 non-regression holds; pp@8k cells unreadable — non-stationary reps in both arms, documented.) The patch is preserved in the lab mirror as a measured regression.
- KV q8_0 vs f16 quantified on the real-session regime (MTP on, 131k ctx, prompt-cached reps): **q8_0/f16 = 0.74 pure-cached (9.83 vs 13.29 t/s), 0.81 on means — a 19-26% tg penalty at high context**, while ppl is +0.10% (3.5631 vs 3.5595) and MTP acceptance-length is unchanged (1.97 vs 1.94, n≈680, same-mode greedy). The bottleneck is bandwidth/dequant, not numerics; the classic KV is ~25 KB/token (dense QSA layers) and the real VRAM occupant is the ~7.65 GiB RS ring (GDN state) which cache-type does not touch → on 96 GB VRAM, KV quantization buys nothing. Deploy moves to f16 K/V.
- Per-drafter spec-decoding engagement counters exposed via `/metrics` (`spec_route_ngram_drafts_total`, `spec_route_model_drafts_total`): unconditional scalars from boot — needed to prove both drafters actually fire when combining MTP with n-gram (upstream issue #24507 documents combinations where one drafter silently never runs).
- Reproduced and statistically pinned the post-rollback rejection spike (T19 data, ring-buffer RS + MTP): P(p0-reject | previous round rejected) = 0.233 vs 0.071 baseline (RR 3.3, Fisher p = 1.2e-08); damage concentrated at draft position 0; without RS, post-reject checkpoint replays re-accept the previously rejected position 123/123 times.
- **H1 confirmed, H2 excluded** by a 47-minute GPU A/B/C experiment (gated env-var diagnostic patches on a throwaway branch): resetting ONLY the MTP drafter state after a rollback partial-reject cuts the excess rejection rate 0.172→0.078 (−55%, acc-len conditioned 1.51→1.73); re-asserting the target ring rewind (206 interventions) changes NOTHING (identical to baseline) → the poison is drafter-head state surviving the rollback, not the ring. Production fix: unconditional drafter reset on partial-reject (same primitive as cold-fallbacks). Formal gate partial (two-sided p=0.149 > 0.05, residual 0.35) — documented honestly.
- Root-caused the end-of-session mega-prefill in the *client* (pi coding agent 0.84.3): it is the auto-compaction summary request — by design non-cacheable (`cacheRetention:"none"`, fresh sessionId), triggered at ctx > 245,760 tokens (match 1:1 with the 185,618-token request observed in production logs). Mitigations documented in the cycle report.
- **Integration + deploy**: candidate (counters + drafter-reset fix, f16 K/V) passed the integration bench as full non-regression vs production (identical restore/cold markers across candidate/no-fix/production in the same bench — the bench's own eviction pattern, not a candidate defect) and was deployed. On the rollback-heavy slice the candidate generates at 18.75 t/s vs 15.61 for the previous production build (+20%).

## 1. Why: the deep-context cliff

On Strix Halo the QSA (sparse-attention) index top_k runs on CPU past 1024 KV positions when the CUB paths are compiled out for HIP — in our production build this showed up as the sampler warning above plus `graph splits = 28` (75 MiB host buffer). The community-facing symptom is the well-known pp collapse beyond ~128k. F1 measures what enabling hipCUB actually buys on a 99 GiB ROCmFP4 LEAN model at 8k/131k contexts.

## 2. The enablement (patch `patches/hipcub-enablement/0001-*.patch`)

- Via: apply PR #27874 as-is (`option(GGML_CUDA_USE_CUB ...)` default OFF) and turn it ON explicitly in the build (`-DGGML_CUDA_USE_CUB=ON`). The default-OFF option would otherwise produce a build identical to baseline (silent no-op → false NO-GO). The same option+`add_compile_definitions` mechanism is already used by this fork for `GGML_USE_HIP`, which is the empirical proof the define reaches the target compiling the `.cu` files.
- Builder needs `hipcub-devel` (Fedora 43 + ROCm 7.2.4 repos; header-only; no extra find_package needed — the include path comes with the ROCm toolchain). C++17 confirmed (`CMAKE_CXX_STANDARD 17`).
- Porting notes vs upstream diff: 2 hunks applied by hand (common.cuh capture-alias macro appended at the fork's location; top-k.cu include adapted — the fork's CCCL branch has no `<cuda/iterator>`; on HIP the kernel stays on the "argsort + copy" route with `next_power_of_2` already defined).

## 3. Marker methodology (the bit that generalizes)

The activation warning comes from the *sampling chain of the speculative-verification path*: a probe server without the MTP drafter never emits it, on any build. A correct probe = production flags incl. `-md` + `--spec-type draft-mtp`, `-lv 3` (INFO carries `graph splits`), prompt >1024 tokens, grep only after health-OK. We burned one round discovering this (both arms 0/0) — documented so others don't.

## 4. Perf (gate F1) — NO-GO

pp/tg at 8192/131072 (2-3 reps/cell, MTP off, production flags). Non-regression @8k: tg 1.02 PASS, pp cell unreadable (non-stationary reps, documented). Improvement @131k: **pp 0.58 = −42% → FAIL** (the improvement branch never materialized; tg@131k cells were cut by the global wall guard — verdict robust to their absence). 262k cells skipped for GPU budget.

Caveat we will report: 8k cells showed non-stationary reps in both arms (first-rep warmup ~180 t/s vs steady ~242-244); per-cell medians and the full sample lists go in the report.

## 5. Per-drafter counters (patch `patches/ngram-drafter-instrumentation/0001-*.patch`)

- `common_speculative_round_drafter_type(spec, seq_id)` getter (the per-round drafter identity lives in `spec->impl_last[seq_id]`, set in `common_speculative_draft()`; it survives accept — only `impl_head` is cleared).
- Two scalar `/metrics` counters incremented per-round, OUTSIDE the `if (n_concat_head > 0)` block (ngram rounds never satisfy it), emitted unconditionally from boot so "absent" vs "delta 0" are distinguishable. Cumulative counters → snapshot PRE/POST and diff (delta, not absolute).
- Code facts worth knowing before combining drafters on this codebase: selection is static at load, consumption is per-round first-non-empty-wins with draftless (n-gram) FIRST; `ngram-mod` defaults to `--spec-ngram-mod-n-min 48` (only verbatim continuations ≥ ~48 tokens are ever drafted); n-gram disables the RS rollback ring on hybrid targets (`common.cpp:1277-1296`); on a dual-drafter server (per-request routing) the n-gram impl is silently skipped.

## 6. Post-rollback rejection spike (analysis on production logs)

Method and numbers in the cycle's `h1h2-baseline` report (repro script included). Key figures: 675 spec-decode rounds with RS on; p0-reject 12.0% overall; conditioned on a rejected previous round 23.3% (Fisher p=1.2e-08); next-round mean accepted-length −0.47 (bootstrap 95% CI [−0.73, −0.20]); with RS off, the post-reject replay path re-accepts the same position 123/123 times. ⏳ H1-vs-H2 discrimination and fix land with the F4 phase.

## 7. Client-side mega-prefill (pi coding agent)

Not a server bug: auto-compaction summary request, non-cacheable by design. Mitigations that exist today: compaction settings (reserveTokens/keepRecentTokens), manual `/compact` before the threshold, custom `session_before_compact` extension (summary without an LLM call ⇒ no mega-request at all).

## Artifacts

- Patches: `patches/hipcub-enablement/`, `patches/ngram-drafter-instrumentation/` (+ mirrors here), more land as gates close.
- Bench scripts (workspace): `pi-f1-marker.sh`, `pi-f1-ab.sh`, `pi-f2-quant.sh`, `pi-f3-{payload,replay,driver,classify}.*`, `pi-fint.sh`.
- Full reports (internal workspace): `docs/benchmarks/results-2026-08-28-pi-stack-*.md` ⏳.

*Attribution: GLM by z.ai.*
