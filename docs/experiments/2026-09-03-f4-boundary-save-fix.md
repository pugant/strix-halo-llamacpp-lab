# The F4 reset and the speculative boundary it erased

2026-09-03, the lab's production server (same machine as every note here). A cluster of
full re-prefills in the pi-stack workload turned out to be the interaction of two of our
own fixes: the F4 drafter reset of 08-29
([`patches/f4-rollback-fix/`](../../patches/f4-rollback-fix/)) and the persistent
prompt-cache save path of 09-01
([`patches/t23-kv-disk-persist/`](../../patches/t23-kv-disk-persist/)). One reset cleared
more state than it meant to, and a task that ended on the wrong kind of round lost its
restart cache. Two commits close it; this note is the record.

## 1. Symptom — run7 of 2026-09-03

The run's pi workload (120 tasks, hours of agent traffic) came back with a full-re-prefill
cluster: **5 of 5 full re-prefills (~82k–139k tokens each) correlated with a partial-reject
final round** on the task before the save. Two counters sat next to it in the same logs:
**30 checkpoint deferrals**, and `spec_state_resets_total` **6577 over 120 tasks** — the F4
reset window is not a rare corner, it opens every few rounds of every speculative task.

The shape of the failure, read from the server logs: the periodic `prompt_save` logs
`required speculative state unavailable` and skips the save — so the session history lives
only in the live slot. The next session then finds nothing restorable on disk, falls back
to a cold slot, and that fallback destroys the live history too: the full ~82k–139k-token
re-prefill is the bill.

## 2. Root cause — a reset that cleared too much

The F4 fix of 08-29 resets the MTP drafter state when a verify round ends in a partial
reject (`ids.size() < n_draft + 1`): the memory rollback of a partial reject does not
clean the drafter's own state, and the next round would draft from ghost tokens
(p0-reject 0.272 → 0.161 with the reset — see
[2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md)). The reset was
one call: `set_state({})`.

But `set_state` is the *full* invalidation primitive, and the drafter's state serves two
masters:

- the **draft-sync pair** (`pending_*` / `verify_*` in eagle3 terms) — the ghost tokens
  live here, and this is what the reset must clear;
- the **cache-facing boundary** — the row `get_state` must hand to `prompt_save` /
  checkpoint creation, which the disk library stores alongside the prompt state
  (`state_required` makes the save conditional on it).

Clearing both at a partial reject means: a task whose *final* verify round is a partial
reject goes idle with an empty boundary, stays empty through idle, and the next
`prompt_save` skips. The 08-29 fix was correct about what it targeted and collateral about
what it swept.

## 3. The fix — split the reset

Two commits, one shape, one per drafter implementation
([`patches/f4-boundary-save/`](../../patches/f4-boundary-save/), base `6144779cc`, 122
insertions in `common/speculative.cpp`, 4 in `common/speculative.h`, 11 changed lines in
`tools/server/server-context.cpp`):

- A new `draft_sync_reset` primitive clears the draft-sync bookkeeping **without**
  invalidating the cache-facing boundary. The base-class default is the old
  `set_state({})` — every other implementation (dflash, future ones) behaves exactly as
  before.
- **eagle3** (`0001`): the state splits into `boundary_*` (mirrors the accepted row,
  survives the reset; `get_state` reads it) and the `pending_*`/`verify_*` pair (still
  cleared — drafting behavior byte-identical).
- **`draft-mtp`** (`0002`) — the production drafter for the qwen4exp MTP setup
  (`--spec-type draft-mtp`): its `get_state` gates on `pending_h_valid`, which the F4
  reset cleared the same way. Its ring-based state takes the same fix as a blob:
  `draft_sync_reset()` snapshots the accepted-boundary blob (coherent right after
  `accept()`) before clearing; `get_state` serves the snapshot while the live state is in
  the F4 window; `set_state` and `shift_state` drop it.
- The one call site (`server-context.cpp`, the F4 window) moves from
  `set_state(slot, {})` to `draft_sync_reset(slot)`.

The blob format is unchanged: existing disk entries written by the pre-fix build stay
restorable.

## 4. Post-deploy verification — evening of 2026-09-03

Deployed on the production image `qwen4exp-mtp-vk-optim2` and gated on the smoke before
the server went back to work:

| Check | Result |
|---|---|
| save-skip / deferral over the smoke (with **479 F4 resets exercised** — the window the fix guards was hit 479 times) | **0 skipped saves, 0 checkpoint deferrals** |
| persist entries written by the smoke | 4 |
| draft acceptance across the smoke | 0.52–0.97 (normal spread, no post-reset dip) |
| reuse: verbatim replay + new delta | restored via checkpoint rollback |
| reuse: truncated replay | restored via trailing rollback, lcp 1535 |
| boot adopt after restart | 14 entries / 9.07 GB |

## 5. Scope, stated plainly

- What survives the F4 window now is the boundary and nothing else: the draft-sync pair is
  still cleared (that was the 08-29 fix, and its p0-reject numbers still stand), and every
  implementation that did not opt in keeps the old full-reset behavior.
- **Known pre-existing, out of scope here:** the F4 reset also zeroes the
  trailing-rollback ring — unchanged by this series, declared rather than discovered
  later.

## 6. Artifacts

- **Patches** — [`patches/f4-boundary-save/`](../../patches/f4-boundary-save/): `0001`
  (eagle3 boundary/pending split, fork `5db15ccdb`) and `0002` (`draft-mtp` boundary
  snapshot, fork `7d54c4bd2`), base `6144779cc`; the series reproduces the tree of
  `7d54c4bd2`, which is the state of the [`rocmfpx/`](../../rocmfpx/) snapshot in this
  repo.
- **Related series** — [`patches/f4-rollback-fix/`](../../patches/f4-rollback-fix/) (the
  08-29 reset this refines) and
  [`patches/t23-kv-disk-persist/`](../../patches/t23-kv-disk-persist/) (the save path this
  unblocks); the restore semantics the smoke re-verified are in the
  [prompt-cache guide](../guide/qwen38-flash-next-prompt-cache-disk.md).
- **Production** — running on the lab's production server since 2026-09-03 (evening).

---

*Thread index: [`README.md`](README.md) — this note joins
[Infrastructure fixes](README.md#infrastructure-fixes); related:
[spec-boundary-cache.md](spec-boundary-cache.md) (the boundary contract and its salvage
machinery), [2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md) (the
F4 reset itself), [2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md](2026-09-02-qwen4exp-graph-reuse-and-dense-decode.md)
(the campaign already in the deployed image). Numbers transcribed verbatim from the lab's
run7 logs and post-deploy smoke notes of 2026-09-03.*

*Attribution: GLM by z.ai.*
