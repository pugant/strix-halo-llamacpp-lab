# The "bad quant" that was a bug: conv/PLE ring-slot rollback fix

**Research note — August 2026.** A week of production on our flagship hybrid model
started producing corrupted text: digits dropped inside hexadecimal runs, spliced
fragments mid-line, generations stopping early, tool calls nested inside tool calls.
The obvious suspect was the newest thing on the quality axis — a fresh 4.78-bpw
ROCmFP4 quantization of a 180B-class hybrid — and for a while the incident read as
"the quant is too aggressive for this model". It was not. The corruption was a
correctness bug in the rollback machinery that speculative decoding had just started
exercising: the ring that snapshots the hybrid's recurrent state had only ever been
writing one of its slots, so every partial-reject rollback restored the
convolution and n-gram-table histories from zeros. This note is the symptom, the
misread, the root cause, and the fix that took acceptance from ~0.74 to 0.91–0.95
and throughput from 24 to 41–44 tok/s.

Context: Qwen3.8-Flash-Next (`qwen4exp`) on our llama.cpp fork, Vulkan (RADV) build,
Strix Halo — a gated-deltanet hybrid where 36 of 48 layers are linear attention with
a short convolution history, an indexer-attention block with its own KV stream, and
a 51B-parameter PLE (per-layer embedding) n-gram table consulted early in the stack.
Speculation runs an external MTP head; partial rejects are rewound through a snapshot
ring (rollback salvage, ~7.2 GiB of buffers) instead of paying a full checkpoint
restore. The runtime story up to that point is
[qwen4exp-runtime.md](qwen4exp-runtime.md) §6.

**TL;DR**

- **Symptom**: token-level corruption immediately after speculative rollback rounds
  — lost digits in hex runs, spliced fragments, early stops, nested tool calls —
  with target and drafter both confident and in agreement. First read as a
  quantization-quality problem.
- **Root cause**: the ring keeps K = `n_rs_seq` + 1 snapshot slots so a partial
  reject can rewind to any of the last K positions, but `build_conv_state_at` wrote
  only ring group 0. The rollback gather restores group `rs_idx` (the number of
  rolled-back rows), so any reject with `rs_idx ≥ 1` restored the conv (GDN) and PLE
  rows from never-written, zero-filled slots — while the SSM state rewound
  correctly. The hybrid recurrent state came back incoherent.
- **The fix**: write one conv-history snapshot per slot (slot s = state after the
  token s positions before the ubatch end), ported from the existing K-slot loop in
  `delta-net-base.cpp` — fork commit `c1d31353e`, 2026-08-30, patch `0012` of
  [`patches/qwen4exp-mtp/`](../../patches/qwen4exp-mtp/).
- **Measured**: zero anomalies across three post-fix runs (about 24 per five runs
  before); acceptance **~0.74 → 0.91–0.95**; tg (token generation) **24 → 41–44
  tok/s** at n=6 on code workloads; and the "inherent post-rollback dip" earlier
  analyses had measured simply vanished.
- **The lesson, promoted to a rule**: a quality regression that appears after a
  rollback-path change is a rollback-bug candidate *before* it is a quant suspect.

---

## 1. The symptom, and why the quant looked guilty

The corruption had a shape. It was not random noise across the output: it hit
exactly the places where a hybrid model leans on its short-history state — runs of
hexadecimal digits with members missing, fragments of two generations spliced at a
seam, turns ending mid-thought, a tool call emitted inside another tool call's
arguments. And it was rare enough at first to hide in "the model has bad days".

The timing made the quant the natural suspect: the deployment had just switched to
our own 4.78-bpw STRIX_LEAN quant of this model, the most aggressive compression the
lab had shipped. Corrupted text on a freshly quantized 180B hybrid is a textbook
quantization complaint. For a while the incident lived under that framing.

Two observations broke it open:

1. **Position, not prevalence**: the corruption clustered *immediately after
   rollback rounds* (partial rejects of the speculative draft). A quantization
   error has no reason to know where the previous verify round was rejected; a
   state-restore bug does.
2. **Confidence, not confusion**: target and drafter were both confident and in
   agreement across the corrupted spans. Degraded weights make a model genuinely
   uncertain. A model with freshly zeroed history computes happily and wrongly.

That pair — *corruption that follows the rollback boundary, produced by a confident
model* — is the fingerprint of a state bug, and it redirected the hunt from the
quantizer to the rollback path.

## 2. Root cause: one writer, K slots

The rollback design (see [qwen4exp-runtime.md](qwen4exp-runtime.md) §6) is a ring of
K = `n_rs_seq` + 1 snapshot slots of the recurrent state. When a verify round
partially rejects — the target keeps the first tokens of the draft and discards R
rows — the gather (`build_rs`) restores slot `rs_idx` = R: the snapshot taken R
tokens before the end of the batch, i.e. the state as it stood after the last token
that survived. That is what makes rollback cheap: no full checkpoint restore, just a
gather from the ring.

The qwen4exp writer for the convolution history never got the memo.
`build_conv_state_at` kept the last `state_cols` columns of the conv input and copied
them into the ring **at group 0 only** — one destination offset, no loop. Groups 1
through K−1 of the ring buffer were allocated, zero-filled at allocation, and never
touched.

So the rollback contract was silently broken on one side:

| state component | on a partial reject (R ≥ 1) |
|---|---|
| SSM (gated-deltanet) state | **rewound correctly** |
| conv history (`r_l`, GDN convolution state) | restored from **zeros** (group R never written) |
| PLE rows (`p_l`, n-gram table history) | restored from **zeros** (same group) |

The hybrid stack then continued with a recurrent state that mixed a correct SSM with
a zeroed convolution history and a zeroed n-gram context. Nothing asserts. The PLE
lookup still hashes and reads a row; the conv still computes over whatever it was
handed; the model keeps generating fluent text whose conditioning is wrong — which
is precisely why it presented as *quality* and not as *crash*. A corrupted KV would
have shown up as attention garbage immediately; a zeroed short-history state
produces locally-plausible, globally-wrong output, the hardest kind to notice.

## 3. The fix

One commit, `c1d31353e` (2026-08-30): port the K-slot snapshot loop from
`delta-net-base.cpp`'s `build_conv_state` into the qwen4exp graph builder. The loop
writes one conv-history snapshot per ring slot — slot s must hold the state after
the token s positions before the end of the ubatch — so whatever group the gather
restores holds the conv/PLE history *after the last kept token*, consistent with the
SSM that rewinds alongside it.

Two details worth keeping:

- **The correct implementation already existed in the codebase.** The same model
  family's delta-net base had the K-slot loop; the hybrid's copy had drifted into a
  single-slot write. Porting a graph builder means porting its invariants, not just
  its tensor shapes.
- **One assumption travels with the loop** (tagged in the code): the last
  `n_rs_seq` + 1 tokens of a sequence must land in the same ubatch, otherwise slot
  s cannot see "the token s positions before the end". The delta-net base makes the
  same assumption; making it explicit in the port is what keeps the next porter
  honest.

The patch ships as `0012` of the qwen4exp MTP series
([`patches/qwen4exp-mtp/`](../../patches/qwen4exp-mtp/)), inside the fork snapshot
in [`rocmfpx/`](../../rocmfpx/).

## 4. Measured

| metric | before | after |
|---|---|---|
| corruption anomalies | ~24 per five code-generation runs | **0 across three runs** |
| draft acceptance (code, n=6) | ~0.74 | **0.91–0.95** |
| tg, code workloads, n=6 | 24 tok/s | **41–44 tok/s** |
| "post-rollback dip" | measured, analyzed, partly attributed to drafter state | **gone** |

The last row deserves honesty about the analysis history. Before this fix, the
post-rollback acceptance dip was real and had been *measured and worked on*:
one-round poisoning after a reject (P(p0-reject | previous round rejected) 0.233
with the ring vs 0.071–0.075 after a full accept — the base rate, identical across
arms), and a drafter-state reset (H1) that shaved the excess (0.172 → 0.078). Those
measurements were correct on their own terms — but a
meaningful share of what they were explaining was the zeros. Once every ring slot
was written, the "inherent" cost of the ring collapsed, and with it the dip. The
moral is not that the earlier analysis was wrong to look at the drafter; it is that
a known-broken path upstream of your measurement will happily explain your
measurement.

## 5. What transfers

1. **Rollback-path change first, quant second.** The rule as we now run it: any
   quality regression that appears after a change to rollback/restore machinery is a
   rollback-bug candidate before it is a compression suspect — no matter how
   tempting the new quant is. The correlation with reject rounds is a free
   instrument; log it before re-quantizing anything.
2. **Zero-filled snapshot buffers fail silent.** The compute graph does not know the
   difference between a written slot and an allocated one; the model stays fluent,
   the logs stay clean, and the damage reads as model quality. Any multi-slot
   snapshot ring should either assert all-slots-written under a debug flag or fill
   new buffers with a poison value that fails loudly the first time it is gathered.
3. **A confident model producing locally-plausible garbage is a state bug.** Weight
   degradation and state corruption both look like "worse output", but only one of
   them keeps target and drafter in agreement. Check confidence before blaming the
   weights.
4. **Port invariants, not shapes.** The bug was a faithful-looking port of a graph
   builder that dropped the loop the original needed. When copying an
   implementation, list what it assumes (here: same-ubatch locality of the last K
   tokens, one snapshot per rewind position) and port the list too.
5. **Cheap rollback has two halves.** The ring's gather side had been verified; the
   write side had not. A fast path that reads state nobody wrote is worse than the
   slow path it replaced — it was both slower in effect (24 vs 41–44 tok/s) and
   wrong.

---

*Thread index: [`README.md`](README.md); the parent thread:
[qwen4exp-runtime.md](qwen4exp-runtime.md) §6 (rollback correctness — the ring
itself, the stall that preceded this bug, and the poisoning analysis this fix
re-contextualizes). Anomaly counts, acceptance and throughput transcribed verbatim
from the lab's fix-validation runs of 2026-08-30.*

*Attribution: GLM by z.ai.*
