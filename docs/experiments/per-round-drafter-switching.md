# Per-round drafter switching: the twin-run measurement that closed the line

**Research note — August 2026 (thread T9).** Companion to
[dual-drafter-synergy.md](dual-drafter-synergy.md), which closed *same-round* cooperation between
our two drafters; this one closes the last untested way to use two loaded drafters, namely choosing
between them **round by round inside a single request**.

Everything was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU,
gfx1151, 128 GB unified LPDDR5X), Vulkan (RADV) build of the fork, target model
`Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx 16384, single stream, dedicated GPU
window, warm-up discarded.

**TL;DR**

- **The question.** Two drafters are loaded — the target's own MTP head and an external DFlash2
  block-diffusion model — and routed per request; does choosing the drafter *per decode round* beat
  the best static one?
- **The instrument.** A shadow drafter in the server: after each committed round, draft+verify the
  DFlash2 chain on the exact next-round context, then roll back — same-round (E_M, E_D) pairs and
  timings, served round untouched.
- **Finding 1.** ρ (the same-round correlation between the two drafters' accepted tokens) = +0.729
  pooled [+0.696, +0.760], n = 1,433: round difficulty dominates drafter complementarity — easy
  rounds easy for both, hard rounds hard for both.
- **Finding 2.** Synchronizing the second drafter costs ~83 ms/round (mean 83.1, p10-p90 70-108)
  versus a real MTP round of 26-53 ms; the twin oracle loses to the best static drafter by 66-74%
  in every pre-registered configuration (up to 86% at the extreme sweep corner).
- **Gate verdict.** The run was declared NO-GO by the pre-registered kill criterion: the timing
  control on the identity window — the initial rounds where ON and OFF outputs coincide. All
  statistics below are exploratory, from a technically clean run (0 warnings, 40/40 requests,
  bit-identical replicas).
- **Byproduct.** The first public per-round same-context dataset for an MTP6/DFlash2 pair — 1,433
  rounds, at [data/t9-fasea-same-context.csv](data/t9-fasea-same-context.csv).

## 1. The question and where it came from

The routing server (T7) picks the drafter once per request — agentic traffic to DFlash2, prose to
MTP — and has been in production since 2026-08-20 at +19% agentic throughput
([dual-drafter-synergy.md](dual-drafter-synergy.md)). Routing exploits heterogeneity *between*
requests; the untested axis was *within* a request, where picking the better drafter each round
could beat any static choice. The bounds phase (T9 Phase B, zero-GPU Monte Carlo over measured
marginals) said "worth measuring": at ρ=0 the per-round oracle prize over the best static drafter
is +14.72% (prose), +10.64% (det), +27.74% (agentic) — gate G-B (the bounds-phase go gate) PASS —
with break-even ρ* = 0.85 / 0.69 / 1.00. That authorized the twin run: measure the real same-round
ρ and the real per-round cost of holding both drafters.

## 2. The instrument: a shadow drafter in the round loop

In the `t9-shadow` branch, armed by `SPEC_SHADOW_DRAFT=1`, the served round is a normal MTP round.
After its tokens commit (context now C_{r+1}), the server runs the DFlash2 drafter out of band on
that committed context — draft the 7-token block, verify on the fixed 8-row batch (greedy, temp 0) —
then rolls all three contexts (target, MTP, DFlash2) back to C_{r+1}. Pairing each shadow with the
real MTP round on the same context gives the same-round pair (E_M, E_D) — accepted tokens per round
for the MTP head and for DFlash2 — rounds 2..N per request.

Declared semantics (pre-registered amendment of 2026-08-24): the pairs are same-context *inside* the
ON arm (boots with the shadow armed; OFF = shadow off) — the ON arm is a deterministic alternative
trajectory, not the OFF one, so ON-vs-OFF holds only where an identity window exists (section 3).

## 3. The measurement broke its own first gate — and we measured why

The original pre-registered control was a content control: shadow-on output char-identical to
shadow-off. The smoke runs broke it. Interleaved shadow decode makes the output deterministic but
*different* from the no-shadow arm — greedy near-tie flips, first divergence inside the draft chain.
Five of six rollback/memory variants of the ON arm (every rollback mechanism except the first naive
trim, which flipped EOS at round 4) were bit-identical to each other while OFF stayed bit-identical
to itself across the same six builds: the cause is the interleaving itself, not a failure to restore
state. The amendment replaced content control with A1 internal determinism, A2 an initial identity
window (ON ≡ OFF for ≥ the first 7 rounds, estimated from one smoke payload diverging at rounds
7-12), A3 runtime sanity, and pointed the timing control B1 at that window. The full run showed the
window does not generalize: 4 of 9 payloads keep it ≥7 rounds (divergence at round 7/12/9/11), 5 of
9 diverge at round 1-4 — a greedy flip can land in any round.

## 4. The run and its controls

Protocol: 4 GPU boots of the same image, interleaved on-1/off-1/on-2/off-2, fresh boot per run, port
8193, temp 0, warm-up discarded, 9 payloads (7 verbatim from the earlier T7/T8 experiments + 2 new
mixed prompts M1/M2), `spec_drafter: "mtp"` on every request — 4/4 boots OK, 40/40 requests HTTP
200, 40/40 snapshot triples (server-state snapshots after each request), cleanup verified.

| Control | Result | Numbers |
|---|---|---|
| A1 internal determinism | PASS | fingerprints identical on-1≡on-2 and off-1≡off-2 on 9/9 payloads; ON vs OFF differ on 9/9 (expected — the char-identity finding) |
| A2 identity window ≥ round 7 | FAIL | only 4/9: P1a 7, P2b 12, AG2 9, AG3 11; P1b/P2a/M2 diverge at round 1, M1 at 2, AG1 at 4 |
| A3 runtime sanity | PASS | armed/disabled markers correct on all 4 boots; 0 shadow warnings; 0 shadow lines in OFF boots; 40/40 requests 200; cleanup ok |
| B1 commit ±2% on the window (kill) | FAIL → NO-GO | \|d\|/off: P1a 10.4%, P2b 12.9%, AG1 11.8%, AG2 12.0%, M1 17.9% (W=1); AG3 1.9% PASS (W=10); P1b/P2a/M2 empty window (W=0) |
| B2 total generation time (non-gate) | — | ON slower by +83→131% on 7/9; AG1 +23.0%; M2 −3.4% (only payload with ON faster) |

B1 is the kill criterion, and its failure warrants a nuanced reading. Deviations reach 5-9× the ±2%
tolerance but carry **no direction signature**: in the logged sums the ON arm's real round was
*faster* than OFF on 4 of the 6 measurable windows (P1a −10.4%, P2b −12.9%, AG2 −12.0%, AG3 −1.9%)
and slower on 2 of 6 (AG1 +11.8%, M1 +17.9%) — the gate compares the absolute value, not the sign.
With windows this short (W=1-11 rounds, median 7) and rounds of ~30-50 ms, |10-18%| fits both
interleaving perturbation and small-sample timing noise: the residual window cannot certify ±2%. B2
is expected: the twin runs the real round *plus* the shadow round, ~2× generation time.

The analyzer's sanity gates all passed — G1 within 0.31-1.59% (formula amended post-run, see
limitations), G2 within 0.13-0.85% (no double-counting), G3 validating t_D = draft+accept against
the T8-A cost model at 8.26% agreement (pooled mean 141.4 ms vs α+8β+d_D = 154.15 ms; α intercept,
β per-verify-row cost, d_D DFlash2 draft time) — and parsing determinism held: 2,920 timing-free
lines hash-identical between replicas, 1,433 pairs, 0 malformed, 0 overwritten, 10 trailing shadow
lines = exactly 1 per request → no round lost.

**Verdict (pre-registered rule, applied): B1 FAIL ⇒ NO-GO of the measurement. No G-A verdict (the
go/no-go on per-round switching itself) is issued from this run; the statistics below are
exploratory.**

## 5. What the data still says (exploratory)

Per payload (ON arm, primary replica):

| payload | class | pairs | mean E_M (MTP6) | mean E_D (DF7) | mean t_M (ms) | mean t_D (ms) |
|---|---|---|---|---|---|---|
| P1a | prose | 201 | 0.925 | 1.353 | 29.32 | 138.28 |
| P1b | prose | 322 | 0.851 | 1.335 | 27.07 | 140.44 |
| P2a | det | 93 | 5.376 | 6.570 | 52.88 | 142.00 |
| P2b | det | 75 | 1.800 | 2.960 | 31.29 | 140.75 |
| AG1 | agentic | 93 | 2.774 | 4.398 | 38.50 | 142.35 |
| AG2 | agentic | 152 | 3.553 | 5.112 | 41.80 | 142.98 |
| AG3 | agentic | 118 | 3.983 | 5.203 | 43.37 | 143.33 |
| M1 | mixed | 264 | 1.242 | 2.705 | 26.82 | 142.44 |
| M2 | mixed | 115 | 2.087 | 3.287 | 32.08 | 142.53 |

E_D > E_M on every payload (Δ 0.4-1.6 tokens/round), but t_D ≈ 140 ms — nearly constant, the fixed
8-row verify block — is 2.7-5.3× t_M (26-53 ms). Declared caveat: t_M is the ON arm's commit
segment, perturbed beyond ±2% on short windows (B1) and excluding the ~70 ms process segment of the
real round, which t_D includes via its 8-row batch.

Same-round correlation of the pair (E_M, E_D) — the central number:

| set | n | Spearman ρ | 95% CI ρ | Pearson r | 95% CI r |
|---|---|---|---|---|---|
| prose | 523 | +0.519 | [+0.440, +0.594] | +0.541 | [+0.437, +0.635] |
| det | 168 | +0.879 | [+0.818, +0.929] | +0.862 | [+0.797, +0.918] |
| agentic | 363 | +0.784 | [+0.729, +0.832] | +0.772 | [+0.717, +0.823] |
| mixed | 379 | +0.607 | [+0.531, +0.674] | +0.650 | [+0.579, +0.716] |
| **pooled** | **1433** | **+0.729** | **[+0.696, +0.760]** | +0.790 | [+0.762, +0.816] |

Bootstrap 10k, seed 20260823, tie-averaged ranks; every interval excludes 0. Against the break-even
thresholds ρ* of the bounds phase (which motivated this measurement): prose 0.519 < 0.85 and agentic
0.784 < 1.00 — in a twin-cost-free model there would still be theoretical headroom; det 0.879 > 0.69
— above break-even, complementarity insufficient even in the model; mixed 0.607 has no bounds-phase
counterpart. The pooled +0.729 says: **round difficulty dominates drafter complementarity** — the
condition most adverse to per-round switching.

## 6. The economics that close the line

Pre-registered oracle formula: per round pick argmax over d ∈ {M, D} of (E_d+1)/t_d and pay
t_chosen + mirror_us (+ s(k) at switches); baseline = best single static drafter per prompt,
*without* mirror (the per-round replay that keeps the idle drafter synchronized) — declared
favorable to the baseline. The prize (oracle/baseline − 1) is negative everywhere: −66.14% to
−74.62% across the two pre-registered corners (h=0,s=0 optimistic; h=5,s=full conservative), −66.14%
to −85.97% across the full sweep hysteresis h ∈ {0,5,10} × switch cost s ∈ {0, full}; pooled −71.39%
conservative. The exploratory G-A arithmetic: the mean mixed-payload prize in the conservative case
is −73.48% against the +5% threshold. The sweep is nearly flat — not a tuning problem.

The mirror is the reason. Keeping the second drafter synchronized costs `mirror_us` on *every*
round: mean 83.1 ms/round, p10-p90 70-108 ms (computed from the published CSV; the run report
characterizes it as ≈70→100 ms) — more than an entire real MTP round (t_M 26-53 ms). DFlash2 accepts
more tokens than MTP on every payload, but not enough to pay for a second verify block per round;
the static baselines that beat the twin oracle run at 65.67-120.58 tok/s.

A deployed per-round switcher would not mirror every round — it would pay catch-up at switch time
(re-drafting the k absorbed tokens), the s(k) term. But the two measured facts point the same way:
the mirror price puts per-round synchronization on the order of a full round or more, and ρ = +0.729
pooled leaves little complementarity to harvest. Per-request routing stays the practical boundary.

## 7. The dataset

[data/t9-fasea-same-context.csv](data/t9-fasea-same-context.csv) — a byte-identical copy of the
run's `dataset-same-context.csv`; 1,433 rows, one per paired round. Columns:
`payload, task, round, classe, e_m, e_d, t_m_us, mirror_us, t_d_us, t_d_draft_us, t_d_accept_us`;
`classe` carries the labels prosa/det/agentic/misto (Italian for prose/deterministic/agentic/mixed;
523/168/363/379 rows). Nine prompts: P1a/P1b (prose), P2a/P2b (deterministic), AG1-AG3 (agentic),
M1/M2 (mixed — designed for alternating
phases); 7 verbatim from the earlier T7/T8 experiments plus the 2 new mixed ones; temp 0, ctx 16384.
Timing semantics, declared: `t_m_us` is the ON arm's commit segment and *excludes* the process
segment (~70 ms) of the real round; `t_d_us` = `t_d_draft_us` + `t_d_accept_us` (draft + accept of
the shadow, excluding prep/rollback); `mirror_us` is logged separately, excluded from the commit
timer. Caveat from the run: `t_m_us` comes from the arm that failed B1 — perturbed beyond ±2% on
short windows, without direction signature. To our knowledge this is the first published same-round
paired dataset for an MTP + block-diffusion drafter pair.

## 8. Limitations

- The measurement run was declared NO-GO by the pre-registered timing control (B1); every statistic
  in sections 5-6 is exploratory, not a gate verdict.
- t_M is perturbed beyond ±2% on a short window with no direction signature, and excludes the
  process segment; the static MTP baselines are favored (declared). The prize direction is robust:
  it is dominated by the mirror, not by t_M.
- "ON = alternative deterministic trajectory": the pairs are same-context inside the ON arm;
  ON-vs-OFF holds only where an identity window exists (4/9 payloads — and there B1 measures exactly
  the perturbation); total times are not comparable (B2).
- The switch cost s_full(k) = k·(d_M+8β) is model-derived, not measured (the wiring never switches
  drafters); immaterial to the outcome — the sweep is flat because the mirror dominates.
- The mixed prompts are synthetic, designed for the per-round use case; single machine, single
  model, single slot.
- Analyzer gate G1 was amended post-run (documented in its source): in the real wiring the process
  segment excluded from the commit timer is part of the real MTP round — touching no decision.

## 9. Reproducibility

- **Code**: branch `t9-shadow` of the fork @ `03ab68271`; a single patch file
  (`0001-t9-shadow-shadow-drafter-wiring.patch`, 14 commits) in `ROCmFPX/patches/t9-shadow/`. Image
  `docker-llm-service:vulkan-fork-t9-shadow` (947dd60d1941, build-0013).
- **Switches**: `SPEC_SHADOW_DRAFT=1` arms the shadow (absent → no shadow code in the round loop at
  all); `SPEC_SHADOW_LOG` and `SPEC_VERIFY_LOG` select the per-round logs; `LLAMA_TRACE=1` enables
  tracing.
- **Protocol**: 4-boot interleaved on-1/off-1/on-2/off-2, fresh boot per run, port 8193,
  `spec_drafter: "mtp"` per request; cumulative snapshot triples after each request.
- **Scripts** (preserved with the branch in the lab workspace): `bench-t9-fasea.sh` (selftest on
  CPU, then run + controls) and `t9-fasea-analyze.py` (sanity gates, correlation, oracle, CSV), plus
  the payload generator and the two test suites; ~25 min of GPU time across the 4 boots, artifacts
  under `logs/t9-fasea/`.
- **Hardware and flags**: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151, 128 GB
  unified LPDDR5X), Vulkan (RADV) build of the fork, target model
  `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx 16384, single stream, dedicated GPU
  window, warm-up discarded.

Pointers: the companion story is [dual-drafter-synergy.md](dual-drafter-synergy.md); the bounds
phase (`docs/research/2026-08-24-t9-faseb-bounds.md`) and the full run report (Italian,
`docs/benchmarks/results-2026-08-24-t9-shadow-run.md`) are in the lab workspace.

---

*Negative results are results: the pairs are published so nobody has to wire a shadow drafter again.*
