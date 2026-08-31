# pi-stack follow-ups: disk-KV surveyed, real-agent quality matrix (T23, T24)

**Research note — August 2026.** Two follow-ups to the agent-stack improvement cycle
([2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md)): a survey of
disk-KV offload as a RAM-relief direction (T23), and a quality matrix run on real agent
sessions instead of on benches (T24). The first closed as *right question, different
answer* — the RAM pressure it anticipated did arrive, and was relieved by moving a
different tensor out of RAM entirely. The second produced the lab's cleanest attribution
result to date: a defect class that looks like a serving-stack problem (out-of-format
tool calls) proved independent of quant and speculative decoding, while the degeneration
family that had been read as model quality stayed tied to one specific arm.

Everything below comes from one machine: AMD Strix Halo (Ryzen AI MAX+ 395, Radeon
8060S iGPU, gfx1151, 128 GB unified LPDDR5X), our llama.cpp fork, production serving
flags. The quality side is not a bench: the tasks are our production agent workload,
run as real sessions, back-to-back across arms.

**TL;DR**

- **T23, disk-KV survey** — catalogued the community disk-KV pattern (KV pages on NVMe,
  restore on prefix hit) as the candidate RAM relief; deliberately parked at low
  priority because the RAM margin was not yet blocking. When the pressure became real,
  the answer came from another direction: the qwen4exp PLE (per-layer embedding)
  disk-offload ([qwen4exp-runtime.md](qwen4exp-runtime.md)). The KV cache itself never
  moved to disk.
- **T24, the matrix** — same tasks, real agent sessions, arms across quant, drafter and
  sampling. The LEAN + speculative + greedy arm showed broken thinking segments,
  zero-degeneration episodes and truncated writes; the plain-quant arm ran the same work
  incident-free.
- **The decisive isolation** — out-of-format tool calls are not tied to quant or
  speculation (they occurred on the plain arm too — once, on a 28k-token write, with a
  clean server log): a model-or-parser issue, fixed with a defensive parse client-side.
- **The speed side** — plain-quant decode on a real session fell from 15.7 tok/s at
  19k context to 6.3 at 50k, while the speculative LEAN still held ~16 tok/s at 40k:
  speculation, not quant, is what holds the decode floor as context grows.
- The production quality verdict these sessions fed is recorded in
  [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md).

---

## 1. Where these two sit

The improvement cycle of 2026-08-28/29 closed its levers by measurement: hipCUB
enablement regressed deep-context prefill by −42% pp (prompt processing) at 131k, KV
quantization cost 19–26% tg (token generation) on the real-session regime, the n-gram
drafter had no terrain, and the drafter-state rollback fix shipped. Two questions were
left open on purpose.

The first was **RAM**. The cycle had established that on this machine KV compression
buys no speed — the decode bottleneck is the bytes of the weights, not of the KV — so
any future relief had to come from capacity: fewer bytes resident, or bytes resident
somewhere else. The second was **quality**. The T22 quant comparison had ended with a
one-line observation that the LEAN arm (drafter on) produced output-integrity incidents
in a back-to-back session while the plain arm did not — with the honest caveat that the
arms differed in quant *and* in speculation at once. T23 and T24 are those two loose
ends, each closed enough to record.

## 2. T23 — the disk-KV survey: the right question, answered from elsewhere

Community inference stacks were shipping disk-KV layers: page the KV of cold prefixes
out to NVMe, restore it on a prefix hit, and buy back RAM at the price of disk latency
on the restore path. We surveyed the pattern as our RAM-relief candidate — what is
resident today (weights, KV, the rollback ring at ~7.2 GiB, the external drafter at
3.85 GiB), what a disk tier would sit in front of, and what the restore path would cost
on a session that re-enters old context.

The catalog was written and then **deliberately parked at low priority**: the RAM margin
was not yet blocking. Sessions completed, the resident set fit, and a disk tier is the
kind of machinery you do not want to build ahead of the need — its failure modes
(restore latency spikes on cache hits that miss, double-booked pages) only pay to
discover under real pressure. The survey's durable output was not a design but the
budget arithmetic: what is resident, what is hot, and the trigger for revisiting
("when the margin starts blocking the workload").

The trigger fired with the 98.5 GiB Flash-Next deployment and the switch to
`--no-mmap` (see [vulkan-nommap-backend.md](vulkan-nommap-backend.md)): with the
weights fully resident in anonymous memory, RAM became the binding constraint again.
The relief, when it came, did not touch the KV:

- the **PLE disk-offload** (T25) moved the model's biggest tensor — the 35.76 GiB
  read-only n-gram table — out of RAM, reading blocks on demand from the GGUF file
  itself: **+36 GB RAM back, output char-identical**, −4/−9% kernel-warm;
- the **KV cache stayed in RAM**, where it always was.

The retrospective is worth stating plainly. The KV is written per position and read per
token at decode — random-access and latency-critical, the worst shape for a disk tier.
The PLE table is a read-only gather whose reuse is shared across content (common
n-grams recur in all prose), the best shape for block caching — and on this APU the
kernel's page cache already behaves as the L2 that a disk tier would have had to
reimplement. When the pressure arrived, the bytes that *wanted* to be moved were not
the bytes the survey catalogued. The survey still paid for itself: the budget
arithmetic it produced is what made the PLE decision a one-day closure instead of a
fresh investigation.

## 3. T24 — the quality matrix on real agent sessions

The matrix was designed to do what the T22 post-scriptum could not: attribute defects
to a layer. **Same tasks** — the daily coding-and-tool-use workload of our production
agent, thinking mode on, ~40-minute sessions — run as **real sessions**, one arm at a
time, same engine build and flags except the arm variable, all arms on
Qwen3.8-Flash-Next. The arms:

| Arm | Quant | Drafter | Sampling |
|---|---|---|---|
| A | our STRIX_LEAN | MTP on | greedy |
| B | plain quant (Unsloth UD) | none | default |
| C | plain quant (Unsloth UD) | none | temp 1.0 |
| D | plain quant (Unsloth UD) | MTP on | — |

What was recorded was not a score but an **incident taxonomy**: broken thinking segments
(reasoning stopping mid-sentence), degeneration (runs of repeated zeros), truncated
writes (file writes cut short), and malformed / out-of-format tool calls (call syntax
the client's parser cannot consume).

## 4. Findings — which incidents belong to which arm

| Incident class | A: LEAN + spec + greedy | B: plain | C: plain, temp 1.0 |
|---|---|---|---|
| broken thinking segments | **yes** | no | no |
| zero-degeneration episodes | **yes** | no | no |
| truncated writes | **yes** | no | no |
| out-of-format tool calls | yes | **yes** | (not re-tested) |

Three classes cluster on arm A and only arm A; arm B completed the same workloads
incident-free. The degeneration family did not recur at temp 1.0 without LEAN —
consistent with a mechanism that needs the greedy + speculative combination (near-tie
flips in batched verify, or rollback-adjacent state), not with the model's intrinsic
quality. The standing caveat is inherited from T22 and kept honest: in the
production-shaped arms the quant and the drafter move together, so this is a strong
observational isolation, not a factorial one — the one cross arm probed (D: UD +
speculation, a brief run) was clean server-side; a LEAN-plain control remains the open
thread, exactly as recorded there.

The decisive finding is the fourth row, and it is the reason the matrix was worth
running. **Out-of-format tool calls crossed the arms**: they appeared on the plain
quant too — on a 28k-token write, with a server log clean of errors, rollbacks and
decode warnings. Two conclusions follow. The defect is **independent of quant and of
speculative decoding**: whatever mixture of model output and client parsing produces
it, it is a model-or-parser issue, not a serving issue — the clean server log rules out
the transport. And the fix therefore belongs **on the client**: a defensive parse
around the tool-call delimiters, tolerant of format drift, deployed there. Had the
attribution not been done, the natural reflex would have been to "fix" it server-side
or to blame the quant — both wrong layers.

One caution from earlier in this same stack is why we resisted that reflex: the
conv/PLE ring-slot writer bug had produced exactly this kind of incident picture
(spliced fragments, premature stops) and had been misread for a while as a bad quant.
On a speculative arm, quality incidents earn attribution to a mechanism before they
get attributed to the model.

## 5. Findings — the speed side of the same sessions

The matrix also yielded the plain-quant decode curve on a real session, sampled across
its life:

| context | plain quant, real session (tok/s) |
|---|---|
| ~19k | 15.7 |
| ~50k | 6.3 |

For contrast, the speculative LEAN arm in its own session still held **~16 tok/s at
40k** context. The two arms are sampled at different depths of different sessions —
read them as curves, not as a paired cell — but the shape is unambiguous: without
speculation, decode sinks as the per-token byte traffic grows with context; with it,
the floor holds. The mechanism is the one measured in
[results-2026-08-17-ckpt-ring-tg-24k.md](results-2026-08-17-ckpt-ring-tg-24k.md): the
MTP boost is **constant with context** (~3.09× at 1k and at 24k), because each verify
round amortizes the weight read over every accepted token. Removing the drafter for
quality reasons does not just divide speed by the boost — it gives back the flat part
of the curve, which on long agent sessions is the larger loss.

## 6. The verdict

The quality verdict these sessions fed — daily agent use moved to the plain quant — is
recorded where it belongs, in [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md),
together with its caveat. What this note adds on top is the attribution: the incident
family that motivated the switch is tied to the LEAN + speculative + greedy
combination, the tool-call defects that looked like part of the same story are not tied
to any of it, and the speed price of the switch is the context slope above, not a
constant.

---

## What transfers

1. **When you park a direction for priority rather than falsify it, write the revisit
   trigger down.** Ours was "RAM margin starts blocking". When it fired, the survey's
   budget arithmetic turned a fresh investigation into a one-day decision.
2. **Move the bytes that want to be moved.** Per-position-written, per-token-read KV is
   the worst candidate for a disk tier; a read-only gather with content-shared reuse is
   the best. Catalog your resident bytes by access shape before choosing which to evict.
3. **Attribute a defect across arms before fixing it in any layer.** Recurrence of
   out-of-format tool calls on the plain arm, plus a clean server log, moved the fix
   from the server to a client-side parse — a fix at the wrong layer tests clean and
   solves nothing.
4. **On a speculative arm, quality incidents are not evidence against the quant until
   the mechanism is checked.** The ring-slot writer bug in this same stack produced a
   "bad quant" picture while being a plain writer bug.
5. **Compare decode speed at matched context, or you are comparing sessions.** 15.7 vs
   6.3 tok/s within one session is context, not configuration.

---

*Thread index: [`README.md`](README.md); parent cycle:
[2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md);
the offload that answered the survey's question:
[qwen4exp-runtime.md](qwen4exp-runtime.md) (guide:
[`../guide/qwen4exp-ple-disk.md`](../guide/qwen4exp-ple-disk.md));
the production verdict: [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md).
Numbers transcribed verbatim from the lab's raw notes of August 2026.*

*Attribution: GLM by z.ai.*
