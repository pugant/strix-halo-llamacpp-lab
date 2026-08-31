# Agent latency: decomposing a real session (T12-T14)

**Research note — August 2026.** The round bench of our fork reports 43 tok/s on
deterministic content at moderate context; the production agent session logs read
16-20. This thread is the decomposition of that gap on a *real* agent session —
replayed against the production server with production flags and instrumented end to
end — then the root-cause hunt the decomposition triggered, then the validation of the
fix that came out of it. The headline: generation itself was never the problem. The
latency an agent actually feels is dominated by things that are not generation, and
the largest of those turned out to live on the client side of the wire.

Everything below was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395,
Radeon 8060S iGPU, gfx1151, 128 GB unified LPDDR5X), the Vulkan (RADV) build of our
llama.cpp fork, production serving flags (dual-drafter routing with the block
drafter on tool-calling requests, KV q8_0, context to 262k), a long-horizon thinking
agent with tool calls as the load, temp 0 throughout. Instrumentation: a tee-proxy
timestamping every stream chunk, a 2 s hardware sampler, the server log — every
number below traces to a log line or an analysis key, cross-checked by a second
reviewer before the reports were closed.

**TL;DR**

- **Two clocks, one session.** Per-generation throughput S1 = **17.11 tok/s**
  (58.46 ms/tok, 67,370 tokens over 35 generations) — inside the 16-20 band the
  memory budget predicts. Wall-clock session throughput S2 = **8.36 tok/s**. The
  difference, C4 = **61.12 ms/tok**, contains no decode at all: it is prefill,
  waiting, and stalls.
- **The decomposition explained an honest 50.1%.** The additive identity in ms/token
  (bench base + session structure + acceptance + KV physics + clock) leaves a
  residual of +11.34 ms/tok against a gate threshold of 2.27 — a 5× miss, declared
  as a FAIL, not buried.
- **The dominant C4 term was misread once, correctly once.** 74.3% of C4 was full
  cold re-prefills (3,059.9 s of them), first attributed to an abort-to-cascade
  mechanism. The real cause: a client-side memory extension stamps a timestamped
  line into the system prompt at every process start, so the first request of every
  new process misses the cache at exactly 15,447-15,449 tokens. The aborts were
  being recovered all along.
- **The block drafter serves reasoning well.** Per-phase tok/round, recovered by
  triangulation at zero GPU cost: 4.25 on reasoning (95% CI [3.82, 4.64]) vs 2.43 on
  prose and 5.76 on agentic anchors. The belief that it "cannot draft reasoning" is
  refuted; the content+tool phase of the identity closes to +0.47 ms/tok.
- **Fix validated on a real 6 h 45 m session: 0 cold fallbacks** (was 7 in 36),
  121/124 requests served from cache/checkpoint replay, C4 **61.12 → 21.66 ms/tok
  (−64.6%)**, generation-vs-wall gap ratio **2.05 → 1.23**. What remains of latency
  is one client stall (57% of the residual) and prefill physics at 200k+ context —
  plus an end-of-session mega-prefill that is client-side history compaction, due
  by design and not cacheable.

---

## 1. Two clocks, one session

The bench-vs-session gap had been an anecdote ("the logs say 16-20") until it became
a protocol: replay a scripted multi-turn agent session against a fresh container
running the exact production flag table (verified invocation-by-invocation against a
stub, 67/67 tokens identical), with the request stream teed through a proxy that
timestamps every chunk, a sampler polling clocks and temperatures every 2 s, and the
server log captured with timestamps. One session, 36 server tasks, context growing
22.8k → 84k, ~97% of the generated characters inside the reasoning phase (the code
the agent writes travels in tool-call arguments, not in the visible stream).

Two statistics, deliberately kept apart:

| clock | value | what it is |
|---|---|---|
| **S1** | **17.11 tok/s** (58.46 ms/tok) | generation windows only: first delta to stream end, token-weighted over 35 generations |
| **S2** | **8.36 tok/s** (119.58 ms/tok) | whole-session wall: first request to last stream end (8,056 s) |
| **C4 = S2 − S1** | **61.12 ms/tok** | everything that is not generation: re-prefill, replay, waits, stalls |

S1 landing inside the predicted 16-20 band is the quiet first result: the decode path
was already understood, bounded by KV-read bandwidth. The interesting money was
always in C4 — and in whether the per-token model could explain S1.

## 2. The identity, and an honest FAIL

The model is additive in ms/token: observed = bench base (phase-matched, 13.7k
context) + ΔC1 + ΔC2 + ΔC3 + ΔC5 + residual, with a fixed attribution order and each
term carrying a declared mechanism:

| term | ms/tok | mechanism |
|---|---:|---|
| observed (session) | 56.34 | token-weighted, 60,422 of 67,370 tokens covered |
| bench base | 33.64 | round bench at matched phase mix (43.0 det / 26.65 prose tok/s) |
| ΔC1 session-vs-bench | +12.02 | structure: client JSON streaming, per-chunk flush, no discarded warm-up, a different drafter than the base |
| ΔC2 acceptance | −8.74 | the session accepts *more* per round than the matched bench — plays in the session's favor |
| ΔC3 KV read bytes | +8.09 | physics: q8_0 KV at 34,816 B/token, 7 drafts/round measured, 236.6 GB/s; grows +2.3 at ctx 22.8k → +13.5 at 73.5k |
| ΔC5 clock | 0.00 | not attributable: thermal range compressed, slope sign wrong, \|r\| < 0.5 — term zeroed by rule, declared |
| **residual** | **+11.34** | unexplained |

The pre-registered gate demanded ≥90% of the gap explained (residual ≤ 2.27 ms/tok).
The identity delivered **50.1%** and the gate was reported as **FAIL** — per-cell
worse (1 of 32 cells in band). A tokenizer parity gate passed 57/57 phases (max
deviation 1.13%), so the tokens the model stands on are sound. The failure is
informative, and its structure was the lead: the content+tool phase nearly closes
(residual −2.80), and *all* of the residual lives in reasoning (+16.63).

## 3. Where C4 went — and what it really was

C4 decomposed into 4,117.7 s of non-generation wall. Six completed requests paid
full re-prefills measurable from the server's prompt-eval lines (2,109.9 s); two
aborted requests paid full re-prefills visible only in progress lines (394.8 +
555.2 s). Total full re-prefill: **3,059.9 s = 74.3% of C4** (the 51% from
prompt-eval lines alone is a lower bound). Seven `prompt cache cold fallback` markers
with `reason=target-draft-restore-rejected`, each following an aborted request —
hence the first reading: *abort invalidates the slot, the next request re-evaluates
everything*.

That reading was wrong, and the correction is the transferable part. A same-evening
reconnaissance found the actual root cause with a four-legged proof:

1. **Code**: a client-side memory extension appends a caveat line carrying a
   wall-clock timestamp to the system prompt at every process start.
2. **Position**: in all 8 full re-prefill events the cache's longest common prefix
   is truncated at 15,447-15,449 tokens — *inside* the system block.
3. **Signature**: the timestamp's hour pair predicts the exact LCP — same
   requester/creator hour → 15,449, different hours → 15,447, a delta of exactly 2
   BPE tokens. The rule predicts all 40 cache-skip lines of the session.
4. **Arithmetic**: 15,449 < system+tools (~16.2k) < first prompt — the truncation
   cannot be anywhere else.

And the aborts the first reading blamed? **Recovered all along**: all 7 intra-turn
resends were served by the checkpoint-salvage rollback (the cache-boundary fix
series, see [spec-boundary-cache.md](spec-boundary-cache.md)) with 90-98% prefix
replay — 17,404 replayed tokens over 399,886 (95.6% saved). The confounder was
process churn: aborts and new processes co-occurred. The counterfactual was cheap —
make the caveat line static, a 1-3 line client change — and bounded the recovery at
≥95% of the re-prefill quota (S2 ceiling ~13.0-13.5 tok/s in that regime).

## 4. Zero-GPU follow-ups: the drafter is not the residual

Two follow-ups ran on the already-captured logs, no GPU time.

**Per-phase acceptance (K1).** The server only reports rounds per request, so
per-phase tok/round was recovered by triangulation: a duration-split solver over the
two phase windows (3.99), an acceptance regression on per-position rates (4.05) —
two independent data paths agreeing to ±0.06 — and a mono-phase estimate biased high
by the invisible tool-args tail (4.94). Combined: **4.25 on reasoning** (bootstrap
95% CI [3.82, 4.64]; per-request spread 3.25-5.40 — acceptance tracks request
content, it is not a constant of "prose"). This *refutes* the hypothesis that the
reasoning residual hid a weak drafter (~3.3-3.5 would have been needed to close it;
the closing value 2.93 is below every observed request). Recomputed per phase: the
content+tool residual closes to **+0.47** ms/tok (tok/round 5.47 ≈ bench 5.40); the
reasoning residual barely moves (+16.63 → +14.87). The drafter serves the reasoning
phase well; what stays open is elsewhere.

**The ramp (K2).** Excluding the intra-request ramp (steady = last 50% of each
window) removes only ~2.1 ms/tok; the ramp is uniform across phases (1.20 vs 1.23),
identical for cold and replay windows (1.144 vs 1.149), and uncorrelated with clock
at window start (R² = 0.007). The earlier "steady 27-29 tok/s" observation was
mostly phase structure — reasoning steady runs 17.0, content 28.3.

**Per-phase routing (K1.4) — NO-GO, analytically.** If the drafter serves reasoning
at 4.25, would routing reasoning to the MTP head pay? Per-token gain on the phase:
[−7.0, +17.2] ms/tok across the anchor grid, +9.2 realistic (−15% on reasoning
ms/token). But a mid-request drafter switch costs a one-shot rebuild of the block
drafter's state over the whole context — 4.5-15.8 s per request, 7.0-28.0 ms/reasoning-
token weighted, up to 415 ms/token on short-reasoning requests, and growing with
context. Net realistic: **[−18.9, +2.2] — no-go** without spending a single GPU run.
The surviving variant is the free one: requests that are reasoning-only (13 of 35)
could take the MTP head with zero switches, worth +2.8% S1 in the realistic anchor —
parked pending one measurement of the MTP head's acceptance on real reasoning, which
is the one number this line never measured.

## 5. Validation on a real session: the fix works

The client caveat was made static; the question became whether the re-prefill quota
of C4 actually comes back. Validation ran the same scripted agent workload for a
real **6 h 45 m session** (22,940 s of data, 123 generations, 199,419 tokens,
context reaching 224k because the turn timeout had been raised from 900 s to 5,400 s):

| gate | result |
|---|---|
| full-prefix replay at turn starts | **PASS** — 121/124 requests served as replay; checkpoint LCP strictly monotone with context, 31,966 → 246,470 (94.30-99.98% of each request); **zero** occurrences of the 15,447±2 timestamp signature |
| zero cold fallbacks from the timestamp | **PASS** — 0 `cold fallback` markers in the log (previous session: 7 in 36); the boot prefill is the only cold |
| re-prefill ≈ 0 | **mixed but green where it counts** — 99.26% of the 16.69 M served prompt tokens came from cache/replay; prompt-eval time 7.77% of wall; one out-of-window event, §below |
| session throughput in band | **out of band, for a physics reason** — S2 = 8.69 tok/s vs the 13-14.5 target, but the longer timeout let context grow 2.8× (224k vs 80k) and S1 *itself* fell to 10.71 tok/s (the KV term alone adds +27.30 ms/tok at that regime). The honest comparison is the ratio, not the absolute |

That ratio: **C4 61.12 → 21.66 ms/tok (−64.6%)**, generation-vs-wall gap **2.05 →
1.23**. The fix is validated where it had to work; the numeric S2 target was defined
on the old regime and is not testable at 224k.

**Where the residual latency lives now.** C4 decomposes into: one **client stall of
2,462 s (41 min) — 57% of C4** (the server healthy and idle the whole time, zero
requests in flight; the turn had to be killed); **1,813 s of pre-first-delta waits**
— ordinary 3-9k token replays that simply cost real prefill time at 200k+ context;
~44 s of minor idle. Nothing of substance is left on the serving side.

**The end-of-session mega-prefill.** After the last stream of the session, one final
request pre-filled a physical **185,604 tokens in 2,127.8 s** — the only request of
the session off the main slot, the only one routed to the MTP head (no-tools signal),
and its prompt was **64,286 tokens smaller** than the request before it. The reading
(hypothesis, honestly labeled): the client compacts its history at session end and
issues a wrap-up request that is not a prefix of anything cached. That prefill is
*due* — content the cache has never seen — and by design not recoverable. It
contributes zero to C4 (it started after the last stream end) but it caps what any
cache fix can promise at session end.

## 6. What remains open

- The client stall (57% of the remaining C4 in one event) — client-side, the single
  largest recoverable latency item left (~+12% S2 if eliminated).
- MTP-head acceptance on real reasoning content — the one measurement that would
  decide the free reasoning-only routing variant.
- Long-context replay cost: 3-9k token replays at 200k+ context are now a visible
  latency class; the concatenation lever measured earlier at +4.4% wall at 80k needs
  re-evaluation at 200k.

---

## What transfers

1. **Report two clocks or you will argue about one.** Generation throughput and
   wall-clock session throughput differ by everything an agent actually feels; on
   this workload the delta was 2× and almost none of it was decode. Any "the model
   got slower" report about an agent should first be split into S1, S2 and the gap.
2. **Declare gate failures at the top.** The identity explained half the gap and the
   report leads with the FAIL. The residual structure (one phase closes, one does
   not) was worth more than a massaged pass would have been.
3. **A plausible mechanism plus a confounder is not a root cause.** "Abort
   invalidates the slot" fit every cold marker — and was wrong. The proof that
   holds is the one that *predicts*: a 2-BPE-token signature that reproduces all 40
   cache-skip lines, plus the arithmetic showing the truncation point sits inside
   the system block.
4. **Check what the recovery path already did before blaming the failure path.** The
   aborts looked destructive; the checkpoint salvage had been quietly absorbing them
   at 95.6% replay. Instrument the happy path, not only the failure.
5. **Drafter folklore is a measurement question.** "Block drafters can't draft
   reasoning" died to a triangulated 4.25 tok/round. Per-phase quantities that the
   server does not log can often be recovered from durations plus acceptance
   statistics — two independent estimators agreeing is the gate.
6. **Bound the counterfactual before the A/B.** Per-phase routing was closed on
   paper — its best case absorbed by a rebuild cost estimable from GGUF sizes and
   cold-prefill rates — at zero GPU cost.

---

*Thread index: [`README.md`](README.md); related notes:
[spec-boundary-cache.md](spec-boundary-cache.md) (the salvage machinery that was
recovering the aborts all along), [qwen4exp-runtime.md](qwen4exp-runtime.md) (the
production runtime), [2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md)
(the agent-stack cycle this thread belongs to). Numbers transcribed verbatim from
the lab's raw session-replay, per-phase and validation reports of 2026-08-25 →
08-26.*

*Attribution: GLM by z.ai.*
