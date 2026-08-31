# Why an interleaved decode breaks char-exact reproducibility on this stack

**Research note — August 2026 (thread T9).** Companion to
[per-round-drafter-switching.md](per-round-drafter-switching.md): that note reported the
char-identity discovery — interleaving a shadow decode into the round loop makes the output
deterministic but *different* — worked around it with amended gates, and left the mechanism open.
This note closes it: the cause is the multi-row interleaved decode on the target context itself,
not the restore mechanism, not the memory type, not the second drafter's state.

Everything was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU,
gfx1151, 128 GB unified LPDDR5X), Vulkan (RADV) build of the fork, target model
`Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx 16384, single stream, dedicated GPU
window, warm-up discarded.

**TL;DR**

- **The phenomenon.** An extra decode interleaved into the target context makes the next round's
  output deterministic but *different* from the arm without it: a greedy near-tie flips.
- **The question.** Which layer causes it — the interleaved decode itself, or the internal state of
  the always-on second drafter (the DFlash2 implementation, whose draft call advances samplers,
  selector RNG and backend chains every round)?
- **The instrument.** Three diagnostic modes wired into the server, each defined by subtraction
  from the full shadow round: a noise decode with dummy tokens, a draft-only mode, and a row-count
  parameter for the noise batch.
- **The answers.** An 8-row interleaved decode with dummy content and full rollback still flips the
  output; the drafter's internal state alone is bit-identical to baseline; a 1-row interleaved
  decode is also bit-identical — the flip needs a multi-row batch.
- **The locus.** The first divergence is the target's *committed* token (a greedy flip), tens of
  tokens before any visible text difference.

## 1. The phenomenon and where it surfaced

The twin-run measurement of the companion note hit this first: its ON arm (shadow drafter armed)
was deterministic but produced a different fingerprint from OFF. The session-1 smoke matrix
(2026-08-23/24; six rollback/memory variants of the ON arm) had already excluded the obvious suspects:
five of the six variants were bit-identical to each other (`6d2d722e…`) while OFF stayed
bit-identical to itself (`9b64c227…`) across the same six builds — the cause was the interleaving,
not a failure to restore state; the sixth variant, an initial naive trim, flipped EOS earlier (round
4). What remained open was the *layer*: with the mirror gate armed, the MTP head never ingests the
shadow rows and the committed tokens are identical up to the textual flip, yet the proposed MTP
chain diverges — and the chain nominally depends only on head state and input tokens, both
identical. The cause therefore had to sit upstream: in the numerics of the real decode on the
target context, or in the state of the always-on drafter. Those are the two hypotheses we
pre-registered before building anything (H1 = the decode's numerics, H2 = the drafter's state).

## 2. The instrument: three diagnostic modes

All modes hook the same site as the shadow round and are mutually exclusive with it. Each arm is
defined relative to the full shadow round (ON: draft the DFlash2 block, verify on the fixed 8-row
batch, accept nothing, roll all three contexts back):

- **E1 — noise-decode** (`SPEC_T9_NOISE_DECODE=1`): instead of the shadow round, decode a batch of
  8 rows on the target context — the anchor (the served round's committed token at the next
  position) plus 7 copies of that same token — then process it and roll back exactly like the
  shadow (checkpoint + the three trims + implementation-state restore). Semantics: **ON minus
  {DFlash2 draft, accept, variable row count}** — fixed dummy content, KV rolled back.
- **E1r1 — single row** (`SPEC_T9_NOISE_DECODE=1` + `SPEC_T9_NOISE_ROWS=1`, pre-registered as
  E3a): E1 with a 1-row batch (anchor only). Isolates the row count.
- **E2 — draft-only** (`SPEC_T9_DRAFT_ONLY=1`): only the DFlash2 draft call, with the same
  conditional drafter rollback as the shadow (only when it actually drafted rows), and no decode on
  the target context at all. Semantics: **ON minus {target decode, process, accept, target/head
  trims}** — isolates the drafter implementation's internal state.

Boot behavior is shared: fail-stop if more than one mode is set, one `spec-t9ci: mode=…` marker per
boot, diagnostic N/D rows in the spec log; with no env set the code is inert — certified by the OFF
boots being deterministic with 0/0/0 diagnostic rows.

The pre-registered plan review verified the hook points against the source and forced three fixes
before implementation — without them the experiment would not measure what it claims:

- **B1, processing guard.** The noise decode runs inside the shadow-processing guard, so the MTP
  head does not ingest the garbage rows (plus the same implementation-state snapshot/restore as the
  shadow).
- **B2, mirror arming.** The DF mirror is armed in every active-mode boot; without it the DFlash2
  context of an MTP-routed sequence would be empty and E2 would draft from a state production never
  sees.
- **B3, conditional rollback.** The drafter rollback fires only on a non-zero draft (boundary
  position initialized to 0, the shadow's own pattern); an unconditional trim from position 1 would
  wipe the mirrored prefix and contaminate the very arm meant to isolate it.

## 3. Protocol

Port 8194 (8193 was still owned by the Phase A run), one dedicated boot per arm, P1a verbatim —
`Scrivi un paragrafo dettagliato sulla storia di Roma.` — max_tokens 100, temp 0,
`spec_drafter: "mtp"` per request, warm-up discarded, fingerprint = `reasoning_content` + `content`
concatenated, then md5. Three runs across two builds of the branch: runs 1-2 on build-0014 (image
`fb7cded0ab68`, commit `fdb4a9735`), run 3 (the single-row arm) on build-0015 (`f0067e1587f`,
commit `5e2744459`). Run 1 (off/on/e1/e2, no verify log) agrees with run 2 on all four md5s.
Per-arm determinism is checked by an off2 boot in every run that carries one. Script:
`scripts/bench-t9-ci.sh`.

## 4. Results

| run | build | off | off2 | on | e1 (8 rows) | e1r1 (1 row) | e2 |
|---|---|---|---|---|---|---|---|
| 1 | fb7cded0 | 67f834cf | — | d1dad0a5 | f2379c41 | — | 67f834cf |
| 2 | fb7cded0 | 67f834cf | 67f834cf | d1dad0a5 | f2379c41 | — | 67f834cf |
| 3 | f0067e15 | 67f834cf | 67f834cf | — | f2379c41 | **67f834cf** | — |

First 8 hex of each digest; the full digests (off `67f834cfbff6a760b730df78faade859`, on
`d1dad0a5b2de3e67813572d1de693ef8`, e1 `f2379c4111f967d788476ef69c02cca4`) are in the run logs.

Readings (pre-registered decision rules of the plan, applied):

1. **Per-arm determinism is absolute**: every arm's md5 reproduced across runs and builds — e1 =
   `f2379c41` on build-0014 *and* build-0015, off identical on all three runs.
2. **e2 ≡ off → H2 excluded**: the DFlash2 draft with its advancing internal state does not
   contaminate the output.
3. **e1 ≠ off → H1 supported**: the 8-row interleaved decode contaminates even with irrelevant
   content and a full rollback.
4. **e1r1 ≡ off**: a single row is harmless — the flip needs the multi-row batch; a 1-row
   interleaved decode (anchor at the next position) is bit-neutral.
5. **on ≠ e1 ≠ off** (each a distinct digest): the content/shape of the interleaved batch picks
   different trajectories — expected in the plan (E1's fixed 8 rows vs ON's variable 1+n_df, the
   drafted rows).
6. tok/s, informational only: off 21.5-22.4; e2 18.8-19.3; e1r1 11.0; e1 9.2-9.4; on 8.1-8.2.

## 5. The baseline lesson

The plan imposed a stop-rule: if the re-checked off did not reproduce the session-1 baseline
(`9b64c227…`, build `947dd60d`), stop and re-derive before any conclusion. It fired: off =
`67f834cf` ≠ `9b64c227`. We re-derived the baseline within-build — off replicated bit-identically
by off2 in every new build — so every comparison in section 4 is within-build and valid. The
cross-build shift itself is declared as a finding, not swept aside: OFF differs between builds
whose off-path code is functionally unchanged (certified in review). Two hypotheses, stated and not
resolved: inlining differences between builds of the same translation unit, or a warm-up protocol
difference versus the original smokes. Neither is needed for the thesis; the operational
consequence is point 3 of section 8.

## 6. Characterization: where the flip lands

Round-by-round comparison of the verify log's P rows (run 2; sequences include the discarded
warm-up at the head):

- **e1 vs off**: first divergence at R-row #10 (a spec round). Proposed-chain sizes identical
  (dl/na = 1/1), but the *committed* token — the P row at position 1, the anchor with `dft=-1` —
  differs: e1 310 vs off 5707, a complete greedy flip at the first committed position.
  Committed-token longest common prefix (LCP): 40 of 118 (off) / 124 (e1).
- **on vs off**: first divergence at R-row #15, same pattern (anchor sample 6970 vs 5790). LCP 66
  of 118/133.
- Accepted-count rows: first divergence at #15 (e1) and #24 (on) — the number of accepted tokens
  stays equal for some rounds past the first flip: a near-tie with equal acceptance length.

Reading: the first observable effect is in the target's greedy (the committed token), not in the
drafter's proposed chain; the visible text stays identical for tens of tokens past the first flip
(LCP 40-66 of 118-133 committed tokens).

## 7. A mechanistic conjecture (declared as conjecture, not kernel-proven)

The rollbacks in this stack — the RS (recurrent-state memory) ring-index rewind plus a partial
checkpoint, or a full
checkpoint — restore the *logical* state at the boundary, but they do not guarantee that the next
decode is numerically identical after a multi-row batch has written and "removed" cells. On RS
memory the physical ring keeps the content written by the extra decode: the rewind only moves the
index, and the ring beyond the boundary is never serialized by the partial checkpoint. The f32
Vulkan kernels are not invariant to that write history. This is consistent with all three outcomes:
a multi-row batch contaminates, a single row does not (that one cell is rewritten identically by
the real round itself), and the drafter's internal state does not. Kernel-level verification is
future work, out of scope here.

## 8. Practical implications

1. A/B tests that interleave multi-row decodes on the measured context — this stack: the Vulkan
   fork, Qwen3.8-27B, f32 numerics — cannot use char-exact matching. Valid gates are per-arm
   internal determinism, an identity window where one exists, and zero warnings; this is exactly
   what the Phase A amendment adopted.
2. A rollback is not a numerically neutral operation after a decode of more than one row: any
   instrument that interleaves decodes on the very context it measures records an alternative
   trajectory, not a passive observation.
3. Baselines are build-sensitive: OFF changes between builds with functionally unchanged off-path
   code, so char-exact comparisons are only meaningful within a build.

## 9. Limitations

- Single machine, single model, single slot.
- The cross-build baseline shift is unexplained (two hypotheses stated, not resolved).
- The kernel-level mechanism is a conjecture.
- Smoke-scale evidence: a 100-token P1a plus the session-1 six-variant matrix, not a full benchmark.

## 10. Reproducibility

- **Code**: branch `t9-shadow` @ `5e2744459` — two diagnostic commits (`fdb4a9735`, `5e2744459`) on
  top of the Phase A branch; patch series 2,278 lines in a single file
  (`ROCmFPX/patches/t9-shadow/0001-t9-shadow-shadow-drafter-wiring.patch`, 16 commits).
- **Switches**: `SPEC_T9_NOISE_DECODE=1` (noise decode), `SPEC_T9_NOISE_ROWS=N` (batch rows),
  `SPEC_T9_DRAFT_ONLY=1` (draft-only); absent → inert.
- **Images**: `vulkan-fork-t9-shadow` `fb7cded0ab68` (build-0014) and `f0067e1587f` (build-0015).
- **Script and artifacts**: `scripts/bench-t9-ci.sh`; per-arm artifacts (fingerprint, spec log with
  C/N/D/S rows, verify log with R/P rows, server log, response, meta) under `logs/t9-ci{,-run1,
  -run2}/`. The pre-registered plan and the Italian run report are preserved in the lab workspace.
- **Hardware and flags**: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151, 128 GB
  unified LPDDR5X), Vulkan (RADV) build of the fork, target model
  `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN` (13.8 GiB), temp 0, ctx 16384, single stream, dedicated GPU
  window, warm-up discarded.

Pointers: [per-round-drafter-switching.md](per-round-drafter-switching.md) is the companion whose
section 3 this note resolves; [dual-drafter-synergy.md](dual-drafter-synergy.md) closes the
background story.

---

*Diagnosis is a result: the cause is pinned to the multi-row interleaved decode, so nobody has to
re-run an exclusion matrix to find out their rollback was never the suspect.*
