# Drafter routing (MTP ↔ DFlash2 per-request) — T1-T5 Outcomes

**Date:** 19-20/08/2026 · **Branch:** `drafter-routing` (14 commits beyond `dflash2`, durable patch
`patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch`) ·
**Image:** `docker-llm-service:vulkan-fork-dflash2-route` ·
**Spec:** `docs/design/2026-08-19-t7f2-drafter-routing-design.md` (amended 19/08 evening: kind
`dflash-prefix-miss`) · **Plan:** internal implementation/experiment plans `2026-08-19-t7f2-drafter-routing-impl.md` +
`2026-08-20-rs-rollback-dflash-experiment.md` (not included in this repo).

**Dual config:** `--spec-type draft-mtp,draft-dflash --spec-draft-model
<drafter> --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75
--spec-draft-p-split 0.10` (MTP→6 clamped in dual, DFlash→7). Model
Qwen3.8-27B STRIX_LEAN, Vulkan RADV, temp 0, c 16384.

## Gate summary

| Test | Gate | Outcome |
|---|---|---|
| T1 smoke dual-load | 14 checks (boot/policy/override/400/fallback/cache-switch/metrics) | **PASS 14/14** |
| T2 cache round-trip | R3.3 (4 gates × config simple+prod, + post-RS re-cert) | **PASS 4/4** per config |
| T3 sacred paths 0005-0009 | S1-S4 + G-final (simple+prod) | **PASS 7/7** (post-RS) |
| T4 A/B routing vs mono | **R3.1** prose ≥ −3% | **PASS** (+1.0% / −0.2%, 2 runs) |
| | **R3.2** agentic ≥ +10% (mean of 3) | **PASS** (+19.6% / +18.9%, 2 runs) |
| T5 numeric spot-check | (a)≡(b), (c)≡(d) char-identical | **PASS** (4/4 identical arms, ct=305) |

## T4 — table (run rs2; rs1 consistent)

| prompt | class | MONO-MTP6 | DUAL | Δ% | routing |
|---|---|---|---|---|---|
| P1 | prose | 19.8 | 19.7 | −0.5% | mtp |
| P2 | prose | 20.4 | 20.4 | +0.0% | mtp |
| D1 | det | 45.9 | 54.7 | +19.2% | dflash |
| D2 | det | 33.7 | 40.7 | +20.8% | dflash |
| A1 | agentic | 38.6 | 46.0 | +19.2% | dflash |
| A2 | agentic | 42.0 | 51.1 | +21.7% | dflash |
| A3 | agentic | 39.9 | 46.2 | +15.8% | dflash |

Invariants verified: prose routed to mtp with **dflash gen-drafts delta = 0**
(no parasitic work); tools → dflash with mtp delta = 0; `replay stalled` 0.

## The RS cycle (user decision 20/08)

Before the RS flip, T4 failed R3.1 (prose −7.1/−7.5/−15.0%): the blanket
condition `[TAG_RS_STATE_ROLLBACK_SUPPORT]` (common/common.cpp, motivated upstream by
the ngram methods ALONE) zeroed `n_rs_seq` with any non-MTP → target on
seq_rm-FULL → every round with a partial reject paid snapshot+restore+replay
(~150 MB; `replay stalled` 3×/run). Flip `38483cc64` (RS kept with
draft-dflash): tax eliminated, **S3 trailing-rollback reachable again in the
dual** (original gate restored), S1-a solved. All G1-G6 gates of the
experiment PASS; T5 confirms that the flip does NOT alter the numerics.

## Pre-existing bugs found and fixed by the tests (all reviewed)

| Commit | Bug |
|---|---|
| `4a2ec491c` | fatal crash on dflash-routed tasks in the partial-reject/replay path (untrimmed MTP ctx → assert) |
| `db2d81c5d` | reasoning-budget never exhausted on every dflash config (sampler clone reloaded `remaining`) — relevant also for mono-dflash in production |
| `a303242e2` + `c7ad19bff` | missing `\n\n` trailing part in the forced-end sequence (caveat 15/08; server+cli) |
| `bad0be701` | MTP n_max=7 in dual (chain_heads clamp inert on mono-head nextn) |
| `8b35a795f`, `9d0ed2380` | observability (counter pre-registration; MTP resync warning demoted to DBG for seqs routed elsewhere — 107/run, all benign) |

## Documented limitations (not routing regressions)

1. **Template + tools:** adding `tools` to the body re-renders the template's system
   block → ~40 token prefix reuse (lcp=40). A property of the Qwen3.8
   template, identical in mono on ckpt7. Real mixed conversations (prose →
   agentic with tools) do not reuse the system prompt KV.
2. **After a drafter switch with cache hit:** the just-routed drafter starts with
   a missing draft prefix (`dflash-prefix-miss (<P> tok)`) → first drafts
   degraded, correct output, target KV fully reused (T2: ~99.4-100% reuse at
   every switch).
3. **Boot fallback:** drafter file absent → mono MTP-nextn with WARNING;
   override `spec_drafter:"dflash"` → explicit 400 (`not loaded`).

## Artifacts

- Tests: `logs/test-drafter-routing/` (t1-t5, console runs + server log + json),
  `logs/test-drafter-routing/t3-rs/`, `t2-rs/`, scripts in `scripts/` + snapshot.
- Bench: `logs/bench-routing-vs-mono/` (pre-RS and rs1/rs2 runs, TSV + response).
- Build: `rebuild-image-{1..7}.log` (same image, single tag).

**Next step:** Task 12 production rollout — 🚫 GATED on explicit user
approval (llm-service image+flag switch, config backup, 24h observation
on `spec-route:` + counters).
