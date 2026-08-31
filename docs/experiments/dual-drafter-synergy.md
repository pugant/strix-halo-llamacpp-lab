# Dual-drafter synergy: a research line closed by measurement

**Research note — August 2026.** This is the full story of "T8", our attempt to make two
speculative-decoding drafters *cooperate inside a single decode round*, and of how we
closed the whole design space with data instead of shipping it. Every branch ended in
a measured NO; the closure was airtight enough that we never wrote the ~400–700 lines
of engine code the last design required; and the realized win of the line — per-request
drafter routing — has been running in daily production since 2026-08-20.

Everything below was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395,
Radeon 8060S iGPU, gfx1151, 128 GB unified LPDDR5X), Vulkan (RADV) build of the fork,
target model `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx 16384,
single stream, dedicated GPU window, warm-up discarded — careful point measurements
with bit-identical replicas, not statistics.

**TL;DR**

- **The bet** — our server already loads two speculative drafters (the target's own
  MTP head and an external DFlash2 block-diffusion model) and routes between them per
  request. We tried to make them *cooperate in the same round* — conditioning the
  DFlash2 block on the MTP head token — aiming for 3–5× the accepted tokens per round.
- **What we measured** — a seven-step gated campaign (concat → copy-mode diagnosis →
  pattern exclusion → fair-metric deep-dive → 2×2 payload-twin → depth ceiling → tree
  counterfactual) that closed every same-round design with data. The decisive bound is
  arithmetic: even under perfect complementarity between the two chains (the Fréchet
  upper bound), no same-round design can meet any positive throughput criterion.
- **The outcome** — same-round synergy is closed by measurement; the realized synergy
  is *alternation*: per-request drafter routing, in production since 2026-08-20 at
  +19% agentic throughput.

---

## 1. Where we started

Our llama.cpp fork (a fork of the ROCmFPX fork of ggml-org/llama.cpp — see the repo
README for lineage and credits) runs one `llama-server` with **two speculative
drafters loaded at the same time**:

- **MTP** (Multi-Token Prediction) — the target model's own extra "nextn" head, shipped
  inside the model GGUF. It drafts autoregressively, token by token, and stops early
  when its confidence drops below `p_min` (0.75 in all runs here). In dual mode the
  server clamps it to 6 draft tokens.
- **DFlash2** — an external *block-diffusion* drafter: a separate 1.9B-parameter model
  (`Qwen3.8-27B-DFlash2-Q4_K_M`, ~1.1 GB, trained block size 8) that regenerates a
  whole 7-token block in one shot, every round, no autoregressive chain. Ported onto
  the fork from upstream llama.cpp PR #27342.

For a general audience, the vocabulary of this note in one line each:

| Term | Plain-English meaning |
|---|---|
| **Speculative decoding** | A cheap model proposes several upcoming tokens; the big model checks all of them in *one* batched forward pass; every agreed token is generated "for free". At temp 0 the output is the same as normal greedy decoding. |
| **Drafter** | The cheap model that proposes. Here: MTP head or DFlash2 block. |
| **Verify round** | One batched forward of the target over the drafted tokens (plus the boundary token). |
| **Acceptance** | Fraction of drafted positions the target agrees with (greedy exact match). "Per-position acceptance" is that fraction at position k of the draft. |
| **tok/s** | Generated tokens per second (throughput). |
| **n6 / n7 / DF7** | Draft-depth conventions: `n6` = up to 6 draft tokens per round (MTP), `n7` = 7; **DF7** = DFlash2 drafting its full 7-token block. |
| **Arm A / Arm B** | The two configurations of an A/B experiment. Defined per experiment below; always identical payload, identical server flags except the one under test. |

Before this research line, we had already established — and productionized — the
*alternating* use of the two drafters. The T7 A/B ([results-2026-08-19-dflash2-vs-mtp.md](results-2026-08-19-dflash2-vs-mtp.md))
showed a stark asymmetry:

| Arm (single drafter) | prose tg (tok/s), 2 prompts | deterministic tg (tok/s), 2 prompts |
|---|---|---|
| MTP n6 (control) | 19.6 / 20.2 | 45.2 / 26.1 |
| DFlash2 n7 | 14.2 / 15.0 | **57.4 / 36.3** (record) |
| DFlash2 n5 | 17.5 / 15.9 | 52.2 / 39.5 |

DFlash2 dominates deterministic content (+27% / +39% over MTP) but loses ~26% on free
prose; MTP is the mirror image. The acceptance profiles explain it: on deterministic
content DFlash2 holds **≥ 0.90 per-position acceptance through draft position 7**
(0.99, 0.97, 0.96, 0.94, 0.92, 0.91, 0.90) where MTP collapses after position 1
(0.93 → 0.33). On agentic content (coding/JSON/log prompts) DFlash2 wins +23% / +3% /
+28% and holds ≥ 0.50 at position 7 where MTP drops to 0.25–0.50.

Since no single drafter wins everywhere, we built **per-request drafter routing**
(T7): the server keys on the `tools` signal — agentic/tool-calling requests go to
DFlash2, prose goes to MTP — with an optional per-request override and a
drafter-tagged prompt cache. It shipped only after gates T1–T5 passed
([results-2026-08-20-drafter-routing-t1-t5.md](results-2026-08-20-drafter-routing-t1-t5.md)):

| Gate | Checks | Result |
|---|---|---|
| T1 smoke dual-load | boot / policy / override / 400 / fallback / cache-switch / metrics (14 checks) | PASS 14/14 |
| T2 cache round-trip | KV reuse across drafter switches, 2 configs | PASS 4/4 per config |
| T3 sacred paths | pre-existing cache/budget patches in dual mode | PASS 7/7 |
| T4 routing vs mono-MTP | prose ≥ −3%; agentic ≥ +10% (mean of 3 prompts) | PASS — prose +1.0% / −0.2%; **agentic +19.6% / +18.9%** (2 runs) |
| T5 numeric spot-check | dual ≡ mono per drafter, greedy | PASS, char-identical 4/4 |

Per-prompt T4 (mono-MTP6 → dual, tok/s): prose 19.8→19.7 and 20.4→20.4; deterministic
45.9→54.7 (+19.2%) and 33.7→40.7 (+20.8%); agentic 38.6→46.0 (+19.2%), 42.0→51.1
(+21.7%), 39.9→46.2 (+15.8%). This is the routing server that has been in production
since the morning of 2026-08-20.

## 2. The idea (the bet)

Routing is *alternation*: per request, pick one drafter, the other stays idle. The T8
question was **cooperation**: can both drafters contribute to the *same* verify round?

The mechanism we bet on was the cheapest form of it: **condition the DFlash2 block on
the MTP head token** — in one round the MTP head drafts first, its tokens condition the
DFlash2 block, and the block continues the draft: one linear chain per round, verified
in one batched target forward. If the MTP head is right where it is confident (98%
accepted, it turned out) and DFlash2 is right where MTP is weak, the combined chain
could accept far more tokens per round than either drafter alone. The ambition was
**3–5× the tokens per round** (from ~5 to the mid-teens) — a step-change for agentic
throughput. This note is the story of that bet's measured closure: the same-round
design space was unpacked into candidate mechanisms, and every one of them was either
measured to fail or bounded above by a quantity that cannot meet any positive
criterion.

## 3. How we unpacked the problem

The strategy scan that opened the line decomposed "make two drafters cooperate" into
candidate mechanisms of increasing engineering cost. Each got a **pre-registered
gate**: a threshold written into the plan *before* the run, so the result could not be
negotiated afterwards. The final gate of the whole line (never reached, because
intermediate gates failed) was: deterministic ≥ 65 tok/s on the D1-primary workload,
agentic ≥ 58, prose ≥ −3% versus the production router.

| # | Mechanism | Question it answers | Pre-registered gate | Outcome |
|---|---|---|---|---|
| 0 | Ngram overlay (a third, table-lookup drafter for verbatim repetition) | Cheapest roadmap item; not a *model-drafter* synergy | — | Rejected earlier at the roadmap stage; not measured in this line |
| 1 | **Concat**: MTP head token conditions the DFlash2 block (k1=1) | Does conditioning preserve acceptance? | Content-controlled combined acceptance ratio ≥ 0.90 vs plain DFlash2 | **FAIL (0.831)** → diagnose |
| 2 | **Phase 0 diagnosis** (no design) | Is the failure our plumbing (wiring) or the drafter itself? | Pre-registered signature tests for 4 wiring candidates + copy collapse | Copy-mode confirmed, wiring excluded → Branch-2 |
| 3 | **Branch-2 exclusion**: runtime detector gates the concat off on pattern content | Does removing copy-mode restore the gate? | G0 bit-identical inertia; G1 parity 200 fixtures; G2 suite; G3 = phase-A gate re-run with exclusion ON (≥ 0.90); G4 value on production-like residual (pooled ≥ 1.0) | G0–G2 PASS, **G3 FAIL (0.8346)**, G4 formally PASS but downgraded → branch stopped |
| 4 | **Deep-dive** (offline, no GPU) | Why is G3 unreachable, and was G4's PASS real? | Same-slots fair comparison computed for all windows | **Slot-selection bias discovered**; G4 PASS certifies nothing → methodology fixed for the next cycle |
| 5 | **2×2 twin experiment** | Does pattern-in-context (KV pollution) explain the deficit, and what does the concat cost on clean context? | M-a confirmed ≥ +0.10 / refuted ≤ +0.03; M-c cost ≥ 0.08 costly / ≤ 0.03 cheap; M-value row 2: ratio ≤ 1.00 with cost ≥ 0.03 → close the line | M-a GRAY (+0.086), M-c GRAY (0.071), **M-value row 2 fires → closure dominates** |
| 6 | **Depth** (direction C): draft longer, n-max 8/9 | Can the gain come from longer drafts? | n-max bench 7/8/9 | Structurally impossible: clamped to 7 by the drafter itself |
| 7 | **Tree** (direction A): two independent roots in the same verify round | Can the gain come from coverage (union of two chains)? | Feasibility study, then offline counterfactual: det ≥ DF7-mono +10% AND prose ≥ MTP6-mono −3% | Feasible-costly (~400–700 LOC), but **counterfactual says NO under every cost variant, bounded by Fréchet** |

Each step was justified *only* by the previous result. Phase A failed on pattern
content → diagnose, don't redesign. The diagnosis isolated copy-mode → exclude the
trigger, don't retrain. Exclusion recovered the damage but the gate stayed
unreachable → the deep-dive found the metric itself was gameable → the next cycle
pre-registered a fairer metric. The fair metric said the conditioned draft accepts
*less* than the plain one even in the most favorable constructed context → before
closing, measure the only two remaining escape routes (deeper drafts, wider
coverage). Both closed. No step ran because it was interesting; each ran because the
previous one left exactly one hypothesis standing.

Throughout, "arm A" is the experiment arm (concat / conditioned draft, k1=1) and "arm
B" is the control (plain DFlash2 n7, k1=0), same payload, same flags otherwise.

## 4. Step 1 — Stage 2 concat, Phase A: the gate fails on clean data

Phase A ran the first conditioning experiment of **Stage 2** — the second design stage
of the line (the first, an n-gram overlay drafter, was rejected earlier at the roadmap
stage). Arm A = dual boot + `--spec-concat-k1 1` (MTP head token prepended to the
DFlash2 draft in one linear chain); arm B = the identical boot without the flag, i.e.
plain DFlash2 drafting 7 tokens per round (**DF7**). Three fresh-boot runs per arm,
interleaved A/B, warm-up discarded; each run issues the same three requests: D1
(deterministic counting, 600 tokens) and two agentic prompts (700 tokens each).
Instrumentation identical in both arms.

The pre-registered gate metric: acceptance of the DFlash2 segment in arm A's
*composed* rounds (rounds where the MTP head actually drafted), versus acceptance of
all drafted positions in arm B, threshold ratio ≥ 0.90.

**The literal, pooled reading passed.** The clean reading failed. The difference
between the two is the first methodological discovery of the line:

| Subset | Arm A (concat) acceptance | Arm B (plain DF7) acceptance | Ratio | vs 0.90 |
|---|---|---|---|---|
| All 3 requests pooled (literal gate) | 3039/4215 = 0.7210 | 4209/5730 = 0.7346 | **0.9815** | PASS |
| D1 only (content-controlled) | 394/654 = 0.6024 | 518/562 = 0.9217 | **0.6536** | **FAIL (−35%)** |
| D1 + AG1 | 0.6572 | 0.8523 | 0.7710 | FAIL |
| AG1 + AG2 only | 0.8242 | 0.6565 | 1.2554 | not interpretable |

At temp 0, replicas within an arm were bit-identical — but the *arms diverged in
content* on the two agentic prompts (AG1 at character 410, AG2 at character 1034:
near-tie argmax flips caused by the different batch shape, 9 vs 8 columns, a known
f32 property of batched verify). Once the arms generate different text, acceptance
compares "the content each arm happened to produce", not the conditioning. D1 was the
only request whose output was token-identical across arms — the only valid
comparison — and there the conditioning cost 35% relative acceptance.

The follow-up amendment re-measured everything content-controlled: output streams were
rebuilt token-by-token from the verify logs, aligned, and only rounds entirely inside
the longest common prefix (LCP) of the two arms were scored. Result across all
content-controlled windows:

| Window (content-controlled) | Class | cc rounds A/B | Arm A acc | Arm B acc | Ratio |
|---|---|---|---|---|---|
| D1 (counting 1..159) | numeric pattern | 94/81 | 0.6024 | 0.9217 | 0.654 |
| AG1 (one of the T7 agentic prompts; its measured content was free reasoning, hence the "prose" class) | prose | 23/59 | 0.5404 | 0.4770 | **1.133** |
| F1 (alphabet run inside reasoning) | letter pattern | 20/49 | 0.1857 | 0.3703 | **0.502** |
| F3 (fixed-format log) | log pattern | 4/4 | 0.1786 | 0.3214 | 0.556 (n=4) |
| F4 (JSON) | structured | 2/5 | 0.5714 | 0.4571 | 1.250 (n=2) |
| **Combined content-controlled** | | 143/198 | 520/997 = 0.5216 | 867/1381 = 0.6278 | **0.831** |

**Gate: FAIL, 0.831 < 0.90.** The damage is concentrated on pattern content (0.654
numeric, 0.502 letters) and absent on free reasoning (1.133). Throughput on the
content-controlled window makes the cost concrete:

| Request | Arm A (tok/s, mean of 3) | Arm B | Δ |
|---|---|---|---|
| D1 (identical content) | 34.2 ± 0.4 | 47.3 ± 0.2 | **−27.7%** |
| F1 (identical content) | 18.8 | 25.5 | **−26%** |
| AG1 / AG2 (diverged content) | 34.5 / 35.7 | 40.4 / 33.6 | −14.5% / +6.4% |

Two facts localized the failure precisely. First, the per-position profile on D1:

| Draft position (DFlash segment) | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 |
|---|---|---|---|---|---|---|---|
| Arm A acceptance | 0.989 | 0.936 | **0.479** | 0.473 | 0.462 | 0.452 | 0.419 |
| Arm B acceptance | 0.975 | 0.951 | **0.925** | 0.912 | 0.912 | 0.887 | 0.887 |
| Arm A p_dft (target prob of draft) | 0.984 | 0.928 | **0.494** | 0.956 | 0.983 | 0.964 | 0.880 |

Positions 1–2 are at parity (the conditioning per se does not hurt the start of the
block), and the counterfactual target probability at positions 4+ is healthy
(0.95–0.98) — their low acceptance is only the contiguous consequence of dying at
position 3. Second, the **MTP head itself was accepted 98% of the time**
(594/606 composed rounds; segment acceptance conditioned on an accepted head: 0.7357).
The problem was never the quality of the MTP token. It was the DFlash2 block's
behavior *when conditioned on it*.

Verdict per the pre-registered gate: **FAIL — Stage 2 stops here; the k1=6 escalation
never ran.** But a 35% acceptance hole with a clean position-3 wall and categorical
rejections (median target probability of the drafted token at the stop point: 0.000
over 55 D1 stops) is too specific to hand-wave. Before abandoning or redesigning
anything we had to answer: is this our bug or the drafter's nature?

## 5. Step 2 — Phase 0 diagnosis: copy-mode, not wiring

Phase 0 was a zero-GPU forensic pass over the phase-A verify logs. The question:
**wiring or copy?** "Wiring" means our plumbing feeds the DFlash2 block misaligned
positions or the wrong conditioning rows (an engineering bug we must fix). "Copy"
means the conditioned block, on repetitive content, degenerates into proposing the
previous pattern item instead of computing the next one (a property of the drafter's
behavior off its trained distribution).

Four wiring candidates had been pre-registered, each with a predicted signature:

| Candidate | Predicted signature (if real) | Measured | Verdict |
|---|---|---|---|
| C1 — boot on head's top-1 | anomaly concentrated at pos 1, decaying; systematic shift-back ≥ 0.80 | shift-back absent: 0.0588 over N=442; anomaly at pos 3, not pos 1; strong identity with the stream exists | falsified |
| C2 — head row in wrong slot | no systematic relation to the stream; low p_dft spread over all positions; same behavior on non-repetitive content | strong systematic copy-at-period relation; content-dependent (AG1 immune at ratio 1.133) | falsified |
| C3 — position bookkeeping | only on non-textual inputs | text-only workload | refuted a priori |
| C4 — stale KV residue | copy distance = fixed lag m, same in all rounds | no fixed m: the distance chases the content's period (4 and 2); on prose P(proposal = stream[p−4]) = 0.000 | cleared |
| (reference) chain lag by 1 | f[2][k] == f[3][k−1] systematically | 0.069 / 0.020 / 0.059 (D1/F1/AG1 rejected rows) | falsified |

The positive signature — **copy at the period of the pattern** — was confirmed on
every independent cut:

- On D1's 3-digit region (items like `100\n` = 4 tokens, period 4), at *increment
  slots* (the positions where copying the previous item is provably wrong), arm A
  proposed the previous item's same-slot token **103/111 times (0.928)**; all 47
  stop-rows were such copies, all with categorical rejections (p_dft < 1e-4). Arm B
  (plain DFlash2, same content): **0/66 copies**, 66/66 correct.
- On D1's 2-digit region (period 3): arm A made **0 copies in 104 increment-slot
  proposals** (31 consecutive fully-accepted rounds) — then collapsed exactly at the
  2→3-digit boundary. Arm B: 3 copies in 134.
- On F1's letter run (period 2): arm A copied **24/30 (0.800)**, all 9 stop-rows
  categorical copies; arm B: 0/17.

| Region (D1) | Arm A: cc rounds / stops / full-accept | Arm B: cc / stops / full |
|---|---|---|
| prose header (p < 63) | 10 / 7 / 3 | 10 / 4 / 6 |
| prose tail + 1- and 2-digit run (p 63–359) | 31 / **0** / **31** | 41 / 5 / 36 |
| 3-digit run, period 4 (p 360–598) | 53 / **48** / 5 | 30 / **0** / **30** |

The failure is *selective by content geometry*: perfect in one region of the same
window, catastrophically copied in the next, while the same drafter unconditioned is
perfect exactly where the conditioned one collapses. A plumbing offset is a constant
transformation; it cannot adapt to the period of the content. **Wiring excluded,
copy-mode confirmed.**

One example round, verbatim from the logs: context `..., '1', '0', '1', '\n', '1',
'0'` → the conditioned block proposes **'1'** — the units digit of 101, four positions
back — while the target wants **'2'** (of 102); target probability of the draft:
5.6e-07. And the phase-lock detail that seals the mechanism: 47 of the 48 stops in the
3-digit region have the MTP head at round position ≡ 3 (mod 4) — the configuration
where the head sits on `\n`, the block's first drafted token is the constant leading
digit, and the *incrementing* digit lands on the block's position 3. Exactly the wall
seen in the phase-A profile.

Epistemic note, kept honest: the pre-registered copy hypothesis said "copies at
distance 1". That geometry was falsified (distance-1 copies were a small minority, and
the healthy window AG1 showed *more* of them than the sick ones). The confirmed
signature — copy at the content's period — was a data discovery, documented as a
deviation, and it rests on independent concordant cuts (increment slots, stop rows,
regions, the F1 geometry, the AG1 control) with a zero-copy counterproof in arm B.

## 6. Step 3 — Branch-2 exclusion: a working detector that cannot save the design

If copy-mode triggers on pattern geometry, detect the geometry at runtime and route
those rounds to plain DFlash2. Branch-2 — the second branch of the concat stage, a
redesign of the same mechanism rather than a new design — built exactly that: an
offline-calibrated window detector (lookback window of 16 tokens, two signals:
autocorrelation-like repetition "ac" and constant-token "class") wired into the
concat round.

**Calibration first, as a kill-criterion.** The literal separation target (≥ 95% of
collapsed windows detected, on a 146-document, 691,624-token corpus) failed at
153/186 = 82.26%, and no point on the tuning grid fixed it (union ceiling 165/186 =
88.71%; best single point 162/186 with a 5.56% false-positive rate — a protocol stop).
A clean-window amendment (score only windows entirely inside a pattern region) was
ratified before proceeding: **147/147 = 100% detection, 0/54 false gates.** The 33
misses are all prose, short windows, or boundary-straddling windows — invisible to any
window detector, declared as such. The payload estimate said the residual (the
workload the concat would still serve) is **98.69%** of windows (2,263/172,416 gated,
1.31%) — far above the 20% floor, so the branch was worth building.

Then the certification chain, all green:

| Gate | What it certifies | Result |
|---|---|---|
| G0 | inert at k1=0: 4 runs across 2 images, byte-identical outputs (same sha256), zero concat/exclusion markers (10/10 checks) | PASS |
| G1 | Python reference detector ≡ C++ runtime detector: 200/200 parity fixtures, plus ~3,000 unseen corpus windows added by the reviewer | PASS |
| G2 | certified test suites on the exclusion build: concat t1 60/60, t2 20/20, ring-window 10/10 | **PASS 90/90** |

Then G3: the phase-A gate re-run with exclusion ON, same payload, A/B
content-controlled, 3 bit-identical replicas per arm, 18 boots:

| Metric (content-controlled) | Phase A (no exclusion) | G3 (exclusion ON) | Threshold | Verdict |
|---|---|---|---|---|
| Combined | 0.831 | **0.8346** (230/441 vs 968/1549) | ≥ 0.90 | **FAIL** |
| D1 (numeric pattern) | 0.654 | **0.7440** (72/105 vs 518/562) | ≥ 0.90 | **FAIL** |
| F1 (letter pattern) | 0.502 | **0.7097** (26/91 vs 124/308) | ≥ 0.90 | **FAIL** |
| AG1 (reasoning, control) | 1.133 | 1.1329 — unchanged | informational | no harm |

The recovery is real — the in-region copy-mode is gone (D1 +0.090, F1 +0.208) and the
throughput damage was reabsorbed (D1: **−27.7% → −2.0%**; the detector fired on 60 of
79 D1 concat-eligible rounds and where it fired, the rounds were indistinguishable
from arm B: 60/60 fully accepted) — but the threshold stayed out of reach.

And G4, the value gate on six production-like "residual" windows (logs, JSON, shell
code, NDJSON, CSV — content with no phase-A pattern provenance), 6 boots:

| Window | Content | Arm A acc (composed rounds) | Arm B acc | Pooled ratio | tok/s A vs B |
|---|---|---|---|---|---|
| G1 | agent log | 17/28 = 0.6071 | 82/245 = 0.3347 | **1.8140** | −9.8% |
| G2 | real run log | 44/84 = 0.5238 | 101/238 = 0.4244 | **1.2343** | −2.8% |
| G3 | real JSON | 10/21 = 0.4762 | 17/56 = 0.3036 | **1.5686** | +4.3% |
| G4 | shell code | 5/28 = 0.1786 | 8/21 = 0.3810 | 0.4688 | −5.2% |
| G5 | generated NDJSON | 5/7 = 0.7143 | 13/63 = 0.2063 | **3.4615** | −17.9% |
| G6 | generated CSV | 5/7 = 0.7143 | 6/7 = 0.8571 | 0.8333 | −8.6% |
| **pooled** | | **86/175 = 0.4914** | **227/630 = 0.3603** | **1.3639** | 5 of 6 negative |

Formal verdict: **G3 FAIL + G4 PASS** (criterion: ≥ 3 windows > 1.0 and pooled ≥ 1.0;
measured 4/6 and 1.36) → branch stopped, no deploy. The final gate of the line
(deterministic ≥ 65, agentic ≥ 58) was never even reachable.

But the G4 PASS smelled wrong — a mechanism whose acceptance ratio is 1.36 pooled yet
slower in tok/s on 5 of 6 windows. That tension is what the deep-dive resolved, and it
produced the most transferable finding of the whole line.

## 7. Step 4 — the deep-dive discovery: slot-selection bias in A/B acceptance metrics

The pooled G4 metric compares arm A's *composed* rounds against *all* of arm B's
rounds. But composed rounds exist exactly where the MTP head was confident enough to
draft — the easy slots, easy for both drafters. Arm B's denominator includes all the
hard slots. The metric can therefore be beaten by *selection alone*, with zero value
added. The fair comparison: arm B's acceptance on **exactly the positions arm A's
composed rounds drafted** (the DFlash2 segment span, head slot excluded) —
"same-slots".

Computed for all nine windows measured in the branch:

| Window | Arm A (composed) | Arm B (same slots) | **Same-slots ratio** | Pooled ratio (the gate's metric) |
|---|---|---|---|---|
| D1 | 0.6857 | 0.7917 | 0.866 | 0.7440 |
| F1 | 0.2857 | 0.5233 | 0.546 | 0.7097 |
| AG1 (control) | 0.5404 | 0.6481 | 0.834 | **1.1329** |
| G1 | 0.6071 | 0.6452 | 0.941 | **1.8140** |
| G2 | 0.5238 | 0.8030 | 0.652 | **1.2343** |
| G3 | 0.4762 | 0.4483 | **1.062** | **1.5686** |
| G4 | 0.1786 | 0.3500 | 0.510 | 0.4688 |
| G5 | 0.7143 | 0.7143 | 1.000 | **3.4615** |
| G6 | 0.7143 | 0.8333 | 0.857 | 0.8333 |

**In 8 of 9 windows the same-slots ratio is ≤ 1.0** (seven below, one equal, only G3
above at 1.062) — while the pooled criterion said "PASS, 1.36". The conditioned
DFlash2 block never decisively accepts more than the plain one on the same positions;
where the pooled ratio screams value (AG1: 1.13 pooled vs 0.83 same-slots; G4
windows: 1.36 pooled vs 0.51–0.94 same-slots), the same-slots ratio says the
opposite. G4's PASS certified that the
pooled criterion is gameable, not that the concat adds value. From this point on,
every criterion in the line was pre-registered as same-slots and/or tok/s.

Why G3 is structurally unreachable also fell out of the deep-dive. Arm A's metric only
counts its residual non-gated rounds — free thinking and region transitions, whose
intrinsic acceptance is 0.67–0.79 *even for arm B* — while arm B's denominator is the
whole window (0.9217, dominated by ~1.0-acceptance pattern rounds the detector just
removed from A). **The better the detector, the lower arm A's average.** On D1's
anatomy:

| Round class (D1) | Rounds | Tokens (% of 599-token stream) | Acceptance | tok/round |
|---|---|---|---|---|
| Gated → plain DFlash2 | 60 | 480 (80.1%) | 420/420 = 1.0000 | 8.0 |
| Composed (conditioned) | 15 | 102 (17.0%) | 72/105 = 0.6857 | 6.8 |
| Head-empty → plain | 4 | 17 (2.8%) | 13/28 = 0.4643 | 4.3 |
| Arm B (all rounds) | 81 | 599 (100%) | 518/562 = 0.9217 | 7.4 |

Counterfactual removals on D1: drop the two transition collapses → 0.7473, pooled
0.810, still FAIL; drop *all* categorical stops → 0.8052, pooled 0.874, still FAIL;
give arm A the best observed same-slots level (0.79) → pooled ratio 0.86. The
**practical ceiling of the mechanism on this data is 0.86 < 0.90**. Not a logical
impossibility (the conditioned block *can* beat plain on the same slots — G3 measured
1.062 once) but the observed limit across every window of the run.

The deep-dive also left three measured gaps as the input for whatever came next: (a)
entrance transitions are invisible to any memoryless window detector (the two 3-digit
transition rounds: arm A 4/14 vs arm B 10/10); (b) on prose the conditioning neither
collapses nor pays (D1 thinking same-slots 0.951), with a possible *context-pollution*
confound left open (in F1, the MTP head was rejected in 2 of 6 prose rounds — weak
signal, n=6); (c) **mass head-empty**: the MTP head declines to draft (p < p_min) in
84% (F1), 73% (G4), 65% (AG1) of concat-eligible rounds, degrading them to plain
rounds with a wasted head draft. Gap (b) with its pollution confound is exactly what
the next experiment was built to close.

## 8. Step 5 — the 2×2 twin experiment: pollution is not the explanation

The cleanest experiment of the line. One factor is the **payload**: P1 = the phase-A
F1 payload, whose *context* contains the alphabet pattern (the suspect pollutant);
P2 = a **twin** payload, built deterministically to have the identical assistant-turn
content in a *clean* context — token-count difference ±0, verified by server-side
tokenize/detokenize round-trip. The other factor is the **treatment**: A = k1=1 (MTP
head injected), B = k1=0 (plain). P2 cells are 6 fresh GPU runs, interleaved A/B, 3
replicas per arm, all bit-identical; P1 cells reuse the certified G3 runs (md5-pinned
before and re-verified after). If KV pollution explains the F1 deficit, the
same-slots ratio should jump toward (or above) 1.0 on the clean-context twin.

| Cell | Rounds (taxonomy) | cc compounds | Acceptance (A: composed / B: same-slots) | tok/s (mean of 3) | Head-empty | LCP coverage |
|---|---|---|---|---|---|---|
| P1-A (pattern in context) | 96 = 13 comp + 81 head-empty + 2 gated | 13 | 26/91 = **0.2857** | 18.97 | 84.4% | 50/174 = 28.7% |
| P1-B (plain) | 75 plain | — | 45/86 = **0.5233** | 25.67 | — | 168/174 = 96.6% |
| P2-A (clean-context twin) | 67 = 17 comp + 47 head-empty + 3 gated | 17 | 74/119 = **0.6218** | 26.77 | 70.1% | 108/299 = 36.1% |
| P2-B (plain) | 66 plain | — | 80/104 = **0.7692** | 28.80 | — | 299/299 = 100% |

Derived, per payload (same-slots ratio — internally `ratio_equo` — = A/B on the same
slots; cost = 1 − tok/s A/B):

| Payload | Same-slots ratio | tok/s cost |
|---|---|---|
| P1 (known baseline) | 0.5460 | 0.2610 (sanity: matches the known −26.1%) |
| **P2 (the favorable case)** | **0.8084** | **0.0706** |

The clean context helps — but not enough, and not in the way the pollution hypothesis
predicted. Against the pre-registered thresholds:

| Criterion | Value | Thresholds (pre-registered) | Verdict |
|---|---|---|---|
| M-a (pollution confirmed?) | Δ = 0.8084 − 0.722 = **+0.0864** | confirmed ≥ +0.10 / refuted ≤ +0.03 | **GRAY** (0.014 from confirm, 0.056 from refute) |
| M-c (cost on clean context) | **0.0706** | costly ≥ 0.08 / cheap ≤ 0.03 | **GRAY** |
| **M-value row 2** | same-slots ratio 0.8084 ≤ 1.00 **and** cost 0.0706 ≥ 0.03 | fires → close the line | **FIRES** |

(The 0.722 baseline is F1's post-pattern subset — the prose-comparable rounds —
recomputed as 12/42 vs 19/48 = 0.7218, matching the deep-dive.)

The position-level check confirms the reading is not a slot-mix artifact: P2-A
dominates the P1 baseline at *every* position (0.82 vs 0.67 at pos 1 … 0.59 vs 0.17
at pos 5, 0.35 vs 0.00 at pos 7) — a genuine level shift from the clean context —
yet it still sits below its own plain twin at every position.

| Draft position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| P2-A (conditioned, clean context) | 0.8235 | 0.7647 | 0.7059 | 0.6471 | 0.5882 | 0.4706 | 0.3529 |
| P1-A post-pattern baseline | 0.6667 | 0.5000 | 0.3333 | 0.1667 | 0.1667 | 0.1667 | 0.0000 |

**Reading: even on a context built to be maximally favorable, the conditioned draft
accepts less than the plain draft on the same positions (0.81 ≤ 1.00) and still costs
~7% tok/s.** The KV-pollution hypothesis does not explain the phase-A deficit, and the
same-slots ≤ 1.0 regularity (8/9 windows) now holds even in the window constructed to
break it. With M-value row 2 firing alongside two GRAYs, the pre-registered decision
rule said closure dominates: a context-aware design is pointless if the conditioning
does not pay even in the ideal case. The recommendation went to close the line —
with two escape routes left to measure first, to make the closure cover the whole
space rather than just the concat mechanism.

## 9. Step 6 — the depth ceiling: n-max 8 and 9 do not exist

Could the gain come from drafting *longer*? The DFlash2 drafter declares its trained
block size (8) in the GGUF metadata, and the fork clamps
`n_draft_max = block_size − 1`. So n-max 8 and 9 are clamped to 7 at boot, with an
explicit warning. The bench proves the clamp is total rather than partial: DF8 and DF9
produced **bit-identical verify logs to DF7** — identical speclog (the per-round
verify log) md5 across all three arms: three separate boots ran the same sequence of
computations.

| Arm | det-1 counting (tok/s) | det-2 alphabet (tok/s) | prose (tok/s, 2 prompts) | Mean det acceptance | Acceptance at pos 8 / 9 |
|---|---|---|---|---|---|
| DF7 (n-max 7) | 45.8 | 30.9 | 14.4 / 15.7 | 0.784 (674/860), mean length 6.06 | n/a |
| DF8 (n-max 8) | 46.7 | 31.4 | 14.2 / 16.0 | 0.784 — identical to DF7 | 0.000 (clamp padding, not a measurement) |
| DF9 (n-max 9) | 46.5 | 31.4 | 14.3 / 15.9 | 0.784 — identical to DF7 | 0.000 / 0.000 (clamp) |

And the per-position profile says the constraint is *length*, not acceptance. The
conditional acceptance (computed only in rounds that survive to each position — 588
rounds pooled) **rises** with position:

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Conditional acceptance (pooled) | 0.730 | 0.625 | 0.682 | 0.709 | 0.837 | 0.889 | **0.916** (87/95) |

The rounds that survive deep are the deterministic ones, and on them the drafter is
still excellent at position 7 (on the pure counting prompt, unconditional acceptance
at pos 6/7 is 0.866/0.841). The drafter would be *ready* to go deeper; its training
(block size 8) is what stops it. **A dual-drafter gain cannot come from depth with
this drafter** — going beyond 7 requires a retrained drafter with a larger block,
which is a different project. The only remaining escape route was coverage: two
independent roots in the same round.

## 10. Step 7 — the tree counterfactual: where the Fréchet bound closed the door

The final candidate: a **dual tree** — the MTP chain and the DFlash2 chain drafted as
two independent roots in the same verify round, both verified in one batched target
forward, longest accepted prefix wins. This is the design the feasibility study called
*feasible-costly*:

- Today's server verifies exactly one linear chain per round for the request being
  decoded — but the limit is the harness, not the engine. The KQ-attention mask is
  already a per-row 2D tensor, the KV cache already supports multi-sequence rows, and
  prefix sharing/cleanup (`seq_cp`/`seq_rm`) is metadata-only on a unified cache.
- Tree verification *already works* in the upstream `examples/speculative` CLI
  example (branch forking, coupled batch rows, winner selection) — on the same
  engine layer our server uses, same Vulkan/ROCmFP4 path.
- Porting that pattern into the dual-drafter server was estimated at **~400–700
  touched lines**, with the risk concentrated in per-position MTP boundary
  synchronization and multi-sequence checkpoint state. The verify-batch budget fits:
  1 boundary + 6 (MTP) + 7 (DFlash) = 14 rows, within the fork's 16-row fast-path
  (a patch had raised the Vulkan fast path from 8 to 16 columns; beyond that the
  kernel falls back to the tiled path at ~+15–30% per-iteration cost).

So the engineering was plausible. The question that decided everything: **what
throughput would the tree actually deliver?** Answering it required no engine work —
only per-position survival rates measured from verify logs, a round-time cost model,
and one honest bound.

**Survival rates** (probability that a chain accepts ≥ p tokens in a round, measured
on *total* rounds — see the denominator caveat in §12 — from a dedicated mini-run,
2 boots, single image, T7 protocol verbatim; the tree column is the independent-union
probability p(A∪B) = pA + pB − pA·pB):

Deterministic content (pooled, N_MTP=187, N_DF=123 rounds):

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| pi MTP6 | 0.754 | 0.668 | 0.615 | 0.513 | 0.497 | 0.428 | — |
| pi DFlash7 | 0.951 | 0.894 | 0.837 | 0.748 | 0.740 | 0.683 | 0.626 |
| pi TREE (indep. union) | 0.988 | 0.965 | 0.937 | 0.877 | 0.869 | 0.819 | 0.626 |

Free prose (pooled, N_MTP=568, N_DF=457 rounds):

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| pi MTP6 | 0.532 | 0.206 | 0.065 | 0.026 | 0.012 | 0.009 | — |
| pi DFlash7 | 0.670 | 0.337 | 0.164 | 0.072 | 0.028 | 0.020 | 0.018 |
| pi TREE (indep. union) | 0.845 | 0.474 | 0.219 | 0.097 | 0.040 | 0.028 | 0.018 |

The tree's token gain is real but front-loaded: nearly all of it lands at position 1
(prose: 0.53/0.67 → 0.85) and none of it exists at positions ≥ 6–7, where only
DFlash2 can reach. Expected tokens per round: deterministic MTP 4.48 / DFlash 6.48 /
**tree 7.08** (+9.3% over the better mono); prose MTP 1.85 / DFlash 2.31 / **tree
2.72** (+47%, from a low base).

**The cost model.** Fitting measured round times (t_round minus per-call draft time,
8 points across the four prompts × two drafters):

> **t_round = 67.7 ms + 9.0 ms × (verify rows), R² = 0.997.**

The marginal cost of a verify row on this hardware is real and about 9 ms. Validated
by construction: the model reproduces all eight mono arms' measured tok/s within
±0.7% before any tree is projected. A tree round carries both drafts' rows: expected
11.83 rows (deterministic), 9.01 rows (prose), versus 1.98–6.38 for MTP6 and ~8.0
for DFlash7 (the two monos).

Projected throughput (same run, same image — the mono references are this run's own,
not T7's):

| Configuration | Deterministic (tok/s) | Prose (tok/s) |
|---|---|---|
| MTP6 mono | 33.42 | **19.02** |
| DFlash7 mono | **41.61** | 15.11 |
| **Routing (T7, in production)** | **41.61** | **19.02** |
| Tree, `opt` variant (14-row verify *free* — a physically impossible lower bound) | 39.30 | 16.62 |
| **Tree, empirical fit (primary)** | **33.25** | **15.68** |
| Tree, pessimistic (kernel-curve cost applied to all non-draft time) | 31.76 | 15.56 |

Against the pre-registered criterion (deterministic ≥ DFlash7-mono +10% AND prose ≥
MTP6-mono −3%): the tree is **−20.1% on deterministic and −17.6% on prose** — and
even the impossible free-verify bound stays negative (−5.6% / −12.6%). The tree loses
on every individual prompt too, not just on averages.

**And then the bound that cut the legs out from under us.** All of the above assumes independent chains.
What if they were *perfectly complementary* — every time one chain dies, the other
survives? That is the Fréchet upper bound (correlation −1), a physical ceiling no
pair of real drafters can reach (on deterministic content the chains are in fact
*positively* correlated — they survive together on the easy stretches, so the
independent-union numbers above are already optimistic). Compute what the tree would
need versus what perfect complementarity could at most deliver:

| Class | Tokens/round the tree NEEDS to meet the criterion | Theoretical max under PERFECT complementarity (Fréchet upper bound) | Verdict |
|---|---|---|---|
| Deterministic (≥ +10% vs DFlash7-mono 41.61) | ≥ 8.25 (up to 9.75 depending on the cost variant) | **7.63** | unreachable |
| Prose (≥ −3% vs MTP6-mono 19.02) | ≥ 3.02 (up to 3.23) | **2.96** | unreachable — below even with free verify |

**No correlation between the two chains can save the design.** The needed per-round
yield exceeds the arithmetic ceiling of the two chains' combined length — 7.63 < 8.25
and 2.96 < 3.02 — and the ceiling itself assumes zero row cost and correlation −1,
both physically excluded (the fit says 9 ms/row, and the measured chains co-move on
easy content). The one corner that approaches break-even (deterministic, free verify,
correlation −1: needs 7.50 vs max 7.63) requires both exclusions at once. The
~400–700 lines of engine work had no throughput target to aim at. The tree was
rejected before a line of it was written — which is the only cheap way to reject a
design.

Declared limits, for completeness: the independent-union assumption overestimates the
tree on deterministic content (positive correlation — direction favors the verdict);
only 5 co-starting rounds existed to measure actual chain correlation (declared
unusable); the projections use the four T7 protocol prompts (2 deterministic, 2
prose; agentic was not re-sampled — T7 had already shown DFlash2 dominant there).

## 11. Conclusion

**The realized synergy is alternation.** Per-request drafter routing has been the
production configuration since 2026-08-20: prose stays on MTP (19.7 / 20.4 tok/s,
parity with mono), deterministic and agentic traffic goes to DFlash2 (+19.2% / +20.8%
deterministic; **+19% agentic** mean, gates T1–T5 green).

**Same-round cooperation is closed by measurement, across the design space** — each
link forced by the previous one:

| Design | Closed by | One-line epitaph |
|---|---|---|
| Concat, k1=1 (condition block on head) | Phase A gate: 0.831 < 0.90 content-controlled; −27.7% tok/s | The head is accepted 98%; the conditioned block is the problem |
| …failure mode | Phase 0: copy-at-period, 103/111 increment-slot proposals, 47/47 categorical stops | Not wiring — the drafter clones the pattern item when conditioned |
| Exclusion (gate the concat off on patterns) | G3: 0.8346 < 0.90 with recovery real (−27.7% → −2.0% tok/s) but ceiling 0.86 | A perfect in-region detector cannot fix windows with non-pattern content |
| Pooled "value" on residual | Deep-dive: same-slots ratio ≤ 1.0 in 8/9 windows vs pooled 1.36 | The metric was gameable by slot selection |
| KV-pollution rescue / clean-context value | 2×2 twin: same-slots ratio 0.81 ≤ 1.00, cost 7% tok/s, on the maximally favorable context | The conditioned draft accepts less than plain, pollution or not |
| Depth (n-max 8/9) | Clamped to 7 by the drafter's trained block size; DF8/DF9 bit-identical to DF7 | Bottleneck is draft length, not acceptance (0.92 conditional at pos 7) |
| Coverage (two-root tree) | Counterfactual: −20.1% det / −17.6% prose; Fréchet max 7.63 < 8.25 needed (det), 2.96 < 3.02 (prose) | No correlation between the chains can meet any positive criterion |

The bet was 3–5× tokens per round. The measured answer: with these two drafters on
this hardware, same-round combination tops out *below* the better single drafter per
class, and even the best-case combination (perfect complementarity, free verification)
cannot reach the criteria. The physics of p^k survival plus a real 9 ms marginal cost
per verify row does not negotiate.

Two data discoveries from the closure work outlived the design they were measured
for, and both are worth more than the design itself:

1. **MTP's declined rounds are invisible in verify logs.** With p_min = 0.75, the MTP
   head declines to draft whenever its confidence is low; those rounds generate one
   token with no draft and leave no log row. On prose this is the *majority* of rounds
   — in the counterfactual run's prose prompt P1b: 319 drafter calls, only 189 with a
   draft, **130 declined (41%)**. Anyone computing MTP survival curves from verify
   logs (as one legitimately does for DFlash2, which never declines) would overstate
   MTP survival by **~1.7×** on prose (position 1: 0.931 from the log vs 0.552
   against the true denominator of total calls). The correct denominator is the delta
   of drafter call counters, not logged rounds.
2. **The two drafters' token streams diverge early.** MTP and DFlash2, run
   independently on the same prompts, produce token streams with longest common
   prefixes of **24 / 5 / 5 / 3 tokens** (the four protocol prompts). On one prompt
   the *text* is identical but the token boundaries are not — the DFlash2 rounds
   accept BPE re-tokenization boundaries (space merges) that the MTP stream never
   emits. The lesson generalizes: "greedy outputs are identical" (which our
   drafter-vs-concat certification had legitimately established) does not transfer to
   drafter-vs-drafter comparisons, and any token-level joint analysis between two
   decoders must first measure its alignment budget — ours had 1–2 usable
   co-starting rounds per prompt.

## 12. What transfers

For anyone running gated A/B experiments on speculative decoding (or on any
two-configuration inference comparison):

1. **Pre-register the gate.** Write the threshold and the exact metric into the plan
   before the run. Twice it saved us from ourselves: the phase-A pooled metric passed
   (0.9815) while the clean reading failed (0.831), and the pre-registered reading
   won; and when the data refuted the *letter* of a pre-registered signature
   (copy-at-distance-1) in favor of a stronger one (copy-at-period), the deviation
   was declared and argued, not silently absorbed.
2. **Compare acceptance on the same slots.** Any metric that scores the treatment
   arm's *selected* rounds against the control's *all* rounds is gameable by
   selection: our pooled criterion said 1.36 while the same-slots ratio was ≤ 1.0 in
   8 of 9 windows. If the treatment decides *when* it activates, the control must be
   scored exactly where the treatment activated — and the throughput (tok/s) should
   back the acceptance story, because a ratio > 1 with negative tok/s is a selection
   artifact until proven otherwise.
3. **Control the content, or you measured nothing.** At temp 0, two arms with a
   different batch shape can still diverge on near-tie tokens (f32 batched-verify
   flips). Once arms generate different text, acceptance ratios compare contents, not
   mechanisms. Rebuild both streams, align, and score only inside the longest common
   prefix. Bit-identical replicas within an arm are the instrument that makes the
   divergence visible rather than noise.
4. **Beware gate no-ops.** A criterion that cannot fail is not a gate. Concretely
   from this line: a pooled acceptance metric passable by slot selection; a flag that
   is a documented no-op in the server (`--spec-draft-p-split` is only consumed by
   the upstream CLI example — it does nothing in `llama-server`); a bench "arm" that
   is silently clamped into being another arm (n-max 8/9 → 7 — caught only because
   the verify logs were md5-compared).
5. **Use bit-identity as an instrument.** md5-comparing verify logs across boots
   certified that DF8/DF9 were literally DF7 re-run, that replicas were deterministic,
   and that the exclusion build at k1=0 was inert — each in one cheap check.
6. **Count tokens on the actual concatenation.** BPE space-merging means two streams
   with identical text can have different token boundaries; any payload-twin or
   token-parity check must tokenize the real serialized content (round-trip through
   the server's tokenizer), not count characters or assume counts.
7. **Mind the denominator in acceptance logs.** Drafters that can decline to draft
   (p_min gating) leave no log row for declined rounds; survival curves computed from
   logs overestimate such a drafter (~1.7× here). Track call counters.
8. **Bound before you build.** The tree was rejected by a survival-rate measurement
   plus a two-parameter cost model (67.7 ms + 9.0 ms/row, R² = 0.997) plus an
   arithmetic ceiling (Fréchet) — roughly half a day, versus ~400–700 lines of
   subtle engine work whose best case was proven insufficient. When a design's
   benefit can be expressed as a probability union, compute the union's upper bound
   first; if the upper bound cannot meet the target, no implementation can.

Negative results are results. The artifacts of this line — the exclusion patch, the
detector, the analysis scripts, the raw logs behind every table above — are kept in
our lab records so that nobody, ourselves included, has to re-run them; the
routing-baseline notes referenced in this document are in this repo under
[`docs/benchmarks/`](README.md).

---

*Raw benchmark notes for the routing baseline: [`docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md`](results-2026-08-19-dflash2-vs-mtp.md), [`docs/benchmarks/results-2026-08-20-drafter-routing-t1-t5.md`](results-2026-08-20-drafter-routing-t1-t5.md). The T8 line's internal reports (phase A, phase 0, branch-2, 2×2, n-max, tree feasibility and counterfactual) are dated lab notes from 2026-08-21 → 2026-08-23; the numbers in this document are transcribed from them verbatim.*
