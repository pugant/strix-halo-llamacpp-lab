# Speculative-boundary cache: forced-end, alias resend, checkpoint salvage

**Research note — August 2026.** Our production agent workload resends the whole
conversation every turn and expects the prompt cache to absorb it — that is what
makes a multi-hour, 40k-token agent session affordable. It did not. Conversations
that should have hit the cache exactly were falling cold and re-processing
everything, and the misses clustered in one place: the speculative-decoding
boundary. This note is the bug class and the three fixes that closed it — a
one-newline fix, a field-name alias, and a checkpoint-salvage path that saved 91% of
the prefill work — plus the honest classification of what looked like residual colds
afterwards.

Context: the serving stack at the time was the 27B dense model with its MTP head as
drafter and a server-side reasoning budget, on our llama.cpp fork (Vulkan/RADV
build, Strix Halo, 128 GB unified memory). Everything below shipped behind gates and
has run in production since 2026-08-18; the patch series is
[`patches/spec-cache-trailing-rollback/`](../../patches/spec-cache-trailing-rollback/)
(9 patches, per-patch breakdown in [`PATCHES.md`](../../PATCHES.md)).

**TL;DR**

- **The bug class**: with speculative decoding active, a cache hit needs a three-way
  byte match — what the server *emitted*, what the chat template *re-renders* when
  the client sends the turn back, and what the cache actually *holds*. All three of
  our misses broke one leg of that contract at the boundary where a verify round or
  a budget-forced end had stopped, and the server logged them as
  `spec-boundary-mismatch` cascading into full cold reprocesses.
- **Forced-end (`0005`)** — the reasoning-budget forced end was missing the newline
  the template inserts before the closing tag, so every budget-truncated turn
  diverged on resend *at the reasoning boundary*. The fix is one leading `\n` in the
  forced sequence.
- **Alias (`0006`)** — the client sends its per-request thinking budget under a
  different field name than the fork read; the server silently applied its default,
  so the hard cap fired where the client's re-render did not expect it. The fix
  accepts the alias before the default.
- **Checkpoint salvage (`0007`)** — when a resend diverges further back than the
  rollback ring can rewind, restore the newest context checkpoint at or below the
  divergence point and replay from there instead of going cold:
  **91% of prefill work saved** on spec-boundary cold starts of a 12-request agent
  workload.
- **The residual "colds" were not colds**: after the fixes, remaining skip events
  were slot selection (a request landing on a different LRU slot), not cache
  failures — **exactly 1 real cold in 11 h** of production, from the initial-load
  path that does not consult checkpoints at all.

---

## 1. The contract at a speculative boundary

A turn ends in one of two ways under speculation: the final verify round accepts its
prefix and the round closes, or the reasoning budget runs out and the server forces
an end sequence. Either way the *next* request re-sends that assistant turn as part
of the prompt, and the cache lookup has to reach all the way through it for the hit
to be worth anything — on a 44k-token conversation, a miss is minutes of prefill.

The contract is stricter than "the client resends what it got". The client resends
what it got *as re-rendered by the chat template*: a returned assistant turn with
reasoning comes back as `<think>\n` + reasoning + `\n</think>\n\n` + content, whether
or not those bytes are what streamed out. And under speculative decoding the cache
holds exactly the tokens the target processed, boundary token included. Three
parties must agree byte-for-byte; each of our three bugs made one of them disagree
by a couple of tokens, always at the boundary.

The failure signature in the log was consistent: a cache skip with
`spec-boundary-mismatch`, then a full cold reprocess of the whole conversation — for
a divergence of a few hundred tokens at the tail.

## 2. Fix one: the missing newline (patch 0005)

When the reasoning budget is exhausted mid-generation, the server stops the model
and force-emits an end sequence so the turn terminates cleanly and the client sees a
complete message. The bug: the forced sequence was `message + end_tag`, while the
Qwen3.8-style template re-renders a returned assistant turn as `<think>\n` +
reasoning + `\n</think>` — the newline *before* the closing tag is part of the
template, not part of the content.

So every budget-truncated turn came back on resend with one extra byte the cache had
never seen, at exactly the reasoning boundary. The longest common prefix stopped
there; the skip fired; the whole conversation re-processed. The fix prefixes the
forced sequence with `\n` at both the server and the CLI call sites, so the emitted
text matches the re-rendered template. One character, and an entire class of
per-turn cold starts disappears.

The general statement is worth more than the character: **anything the server forces
into the output stream must be byte-exact against what the template will produce
when that turn comes back** — including whitespace the template owns.

## 3. Fix two: the alias (patch 0006)

The second miss did not corrupt any byte — it moved the boundary. The production
agent client sends its per-request thinking budget under the vLLM-style field name
`thinking_token_budget`; the fork parsed only `thinking_budget_tokens`. With the
field unrecognized, the server silently fell back to its own default budget, so the
hard cap fired at a different token position than the client planned for — and the
truncated turn the client re-rendered (with its own budget accounting) diverged from
what the cache held, again at the boundary.

The fix falls back to the alias before the server default, leaving the canonical
name and the default behavior untouched. The lesson generalizes badly enough to be
interesting: **a silently-ignored request field is a cache bug whenever that field
governs where generation stops**. The parameter did not change any weight or kernel;
it moved the exact byte where the turn ended, and that is precisely the byte the
cache cares about.

## 4. Fix three: checkpoint salvage (patch 0007)

The first two fixes make the *common* resend exact. The third handles the honest
case where the resend legitimately differs — a turn edited, a message dropped, a
prefix that changed further back than the speculative boundary ring can rewind. The
trailing-rollback path (patches 0001/0003, the part that merged upstream as
[ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69)) could rewind only
within the ring window; beyond that it fell back to a **full cold reprocess** — on a
44k-token agent conversation, minutes of prefill for a few hundred tokens of
divergence.

The observation that unlocks it: by the time the fallback is reached, the target and
draft *memories* have already been truncated successfully — the cheap part of the
rewind (the KV truncation) is already done; what a replay cannot recompute on a
recurrent architecture is the memory state itself, and only a saved snapshot
can restore that. The server already had context checkpoints for another path (the
sliding-window attention save/restore). Salvage reuses the primitive: restore the
newest checkpoint at or below the divergence point, replay the tokens after it,
reset and recapture the boundary bookkeeping from the replayed batches. Cold
reprocessing remains the fallback when no usable checkpoint exists or the restore
fails — the fix removes the *needless* colds, never the honest ones.

Measured on spec-boundary cold starts of our 12-request agent workload: **91% of the
prefill work saved** — the reprocess shrinks from "the whole conversation" to "the
tokens after the nearest checkpoint".

## 5. What looked like residual colds

After the three fixes, the cache metrics still showed events a careless reading
would call cold starts. Classification over 11 h of production: almost all were
**slot selection** — a request landing on a different LRU slot performs a lookup
that skips against that slot's state and the counters tick, but the cache itself is
intact and the conversation continues warm where it lives. Not cache failures, not
recoverable by cache work.

Real colds in that window: **exactly one**, from the `load` path — the initial
context load that does not consult checkpoints at all. That is by design and low
priority: it fires once per context lifetime, not per turn.

The later session-decomposition work
([agent-latency.md](agent-latency.md)) measured what these fixes do under real
client behavior, not just in gates: intra-turn aborts recovered by the salvage
path — only 17,404 of 399,886 prompt tokens re-evaluated, 95.6% saved — and a
6 h 45 m validation session with 121 of 124 requests served from
cache/checkpoint replay — 99.26% of
the 16.69 M served prompt tokens never re-evaluated.

## 6. What shipped

The 9-patch series [`patches/spec-cache-trailing-rollback/`](../../patches/spec-cache-trailing-rollback/):
the bounded trailing rollback at the spec boundary (0001+0003, merged upstream via
[ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69)), the checkpoint
rollback machinery (0002), full batch/verify rows into the MTP boundary state
(0004), the three fixes above (0005-0007), and two reasoning-budget surface
additions that ride the same contract (0008-0009). The series became part of the
"sacred paths": every later feature that touches the cache or the drafter stack —
dual-drafter routing first among them — re-runs these paths in its gate suite
(routing's T3 gate: 7/7 sacred-path checks in dual mode).

---

## What transfers

1. **State the cache contract before debugging it.** Under speculation the contract
   is three-way (emitted = re-rendered = cached), and each leg has a different
   owner. Naming the leg that broke turned "flaky cold starts" into three bounded
   fixes.
2. **Forced emissions must be template-exact.** A forced end sequence that differs
   by one newline from the template's re-render breaks every truncated turn, and
   the break points exactly where the money is: the boundary.
3. **An ignored request field can be a latency bug.** If a parameter decides where
   generation stops, an unrecognized alias silently moves the cache boundary. Parse
   the alias, or refuse loudly.
4. **Snapshots beat recompute when rewind cannot reach.** The salvage path exists
   because a primitive already did — checkpoint restore for sliding-window
   attention — and the only new idea was pointing it at the divergence point.
   Audit what restore primitives the codebase already has before building new ones.
5. **Classify residual failures before fixing them.** "Still some colds" dissolved
   into slot-selection counters plus one honest load-path cold in 11 h. The
   classification decided that there was nothing left worth fixing — the cheapest
   closure of the whole thread.

---

*Thread index: [`README.md`](README.md); related notes:
[agent-latency.md](agent-latency.md) (what these fixes do under real client
behavior), [results-2026-08-20-drafter-routing-t1-t5.md](results-2026-08-20-drafter-routing-t1-t5.md)
(the sacred-path gate this series became part of),
[qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md) (the later hybrid the same machinery
serves). Numbers transcribed verbatim from the lab's cache-incident and gate notes
of 2026-08-15 → 08-20.*

*Attribution: GLM by z.ai.*
