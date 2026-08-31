# qwen4exp runtime: from FP8 release to the PLE table on disk

**Research note — August 2026.** Our flagship thread: Qwen3.8-Flash-Next (`qwen4exp` in
the fork), from a day-one FP8 release to the lab's daily production model in six gated
steps (2026-08-27 → 08-31) — quant pipeline, runtime port, external MTP drafter, vision
× speculation, rollback correctness, and the model's biggest tensor leaving RAM. Mostly a
chain of YESes, with two instructive detours and a close we believe is underused: serving
a huge read-only lookup table straight from the model file.

Everything below was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395,
Radeon 8060S iGPU, gfx1151, 128 GB unified LPDDR5X), our llama.cpp fork (a fork of the
ROCmFPX fork of ggml-org/llama.cpp — lineage in the repo README). Production runs the
Vulkan (RADV) build with `--no-mmap`; the drafter benches ran on HIP, dedicated GPU
window, temp 0, ctx 8192, medians of 2–3 runs, warm-up discarded — point measurements.

**TL;DR**

- **The model** — a 180B-class hybrid: gated-deltanet linear attention (GDN), indexer
  attention (QSA) with its own KV stream, a 51.2B-parameter PLE (per-layer embedding)
  n-gram table over ~320M rows, 6B active per token; our ROCmFP4-STRIX_LEAN quant with
  imatrix: 98.5 GiB @ 4.78 bpw.
- **The thread** — pipeline (T15) → runtime port (T16) → external MTP drafter (T17) →
  vision × MTP (T18) → rollback correctness (T19) → PLE disk-offload (T25).
- **The outcome** — in production since 2026-08-31: +108/+127% (n=3/n=5) deterministic
  decode with the drafter, vision and speculation coexisting, and `--ple-disk` keeping
  the 35.76 GiB PLE table on disk — char-identical, +36 GB RAM back, ~3% warm cost.

---

## 1. The model

| Component | What it is | Why it matters on 128 GB |
|---|---|---|
| GDN | 36 of 48 layers are gated-deltanet linear attention (sigmoid gate, the one delta from the qwen35 GDN) | fixed-size recurrent state, no growing KV |
| QSA | the other 12 layers: full attention with a learned indexer (top-k selection, compressed keys, a third cache stream) | long-context KV stays small |
| PLE | a 51.2B-parameter n-gram embedding table: ~320M rows × 160, hashed lookup early in the stack | **35.76 GiB of the 98.5 GiB GGUF** — more than a third of the file |
| MoE + hyper-connections | 512 experts, top-10 + 1 shared, on every layer; 4 low-rank hyper-connection streams | 6B active parameters per token |
| MTP head | a 4B next-token prediction head, published by the community as a separate GGUF | the drafter of §4 |

Vocabulary 248,320, native context 262,144. The PLE table is a read-only gather — hash
the recent n-gram, read one row — which is why the model fits at all, why pp (prompt
processing) is content-dependent (§3), and why it can live on disk (§7).

## 2. The pipeline (T15, 08-27)

| Stage | Output | Notes |
|---|---|---|
| FP8 release | 186 GB, 131 shards | source, deleted after the next gate |
| convert → Q8_0 | 188.2 GB, 1224 tensors | quantizer dry-run gate; conversion stages tensors on disk (`--use-temp-file`), not in RAM |
| Q4_K → imatrix | 119.1 GB; imatrix 926 entries / 256 chunks | computed on the intermediate quant (community practice at this size), stored as a GGUF the fork's quantizer reads natively |
| **ROCmFP4-STRIX_LEAN** | **98.47 GiB @ 4.78 bpw** | quantize run: **16 minutes**; perplexity sanity on the Italian holdout; published (3 shards + card) |

Two converter fixes on the way, both PLE-specific: the table's `weight_scale` (a
per-tensor 0.0002 in the source) is applied once when the table rows are finalized, not
carried per-tensor; and the 205 GB PLE temp is unlinked at the end of prepare, not of the
whole conversion — otherwise the ~396 GB disk peak ends in ENOSPC. Published card:
[`pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF`](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF);
quant quality vs the Unsloth UD family:
[2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md).

## 3. The runtime port (T16)

Full `qwen4exp` support on the fork, in one session: the architecture graph (~1100
lines), the indexer's third KV stream (`idx_`-tagged memory), the hyper-connection
plumbing, and type/mask details where the fork's idioms differ from the upstream port
(PR #27742). Gates were parity gates: trunk perplexity identical to the vanilla-build
reference, our LEAN reading 1.095 vs 1.156 for the vanilla-quant build. First bench
(ROCm/HIP, fa1, 3 reps) vs the reference community quant (kingjones777's, no imatrix):

| Metric | ours | reference quant |
|---|---|---|
| tg128 (token generation) | **22.03 ± 0.23 tok/s** | 22.6 |
| pp2048 | 131 tok/s | 345 (standard llama-bench prompt) |

Generation parity with the imatrix in the file is the meaningful cell. The pp gap is
mostly protocol: the PLE lookup repeats (cached rows) or explores (fresh rows), so prompt
processing is a function of content — the same server later measured 131 tok/s on real
agent traffic vs 242 on repeated corpus.

## 4. The external MTP drafter (T17, 08-28)

The published LEAN is backbone-only (1224 tensors, no MTP head), so speculation needed
the community head: agentionai's `Qwen3.8-Flash-Next-MTP-Q8_0` (3.85 GiB), driven via
`-md` by a port of upstream PR #27836 adapted to the fork's idioms — 8 commits, one
day, running unchanged on HIP and Vulkan builds (acceptance nearly identical: 3.24 vs 3.18).

Main result (HIP, dedicated GPU, ctx 8192, temp 0, median of 3, 600 generated tokens):

| Workload | plain | +MTP n=3 | +MTP n=5 |
|---|---|---|---|
| deterministic (counting) | 22.1 | 46.0 (**+108%**) | **50.2 (+127%)** |
| deterministic (alphabet) | 20.9 | 32.0 (+53%) | 32.6 (+56%) |
| open prose | 22.6 | 22.8–25.4 (+1–12%) | — |

Acceptance explains the split. Per-position acceptance at n=5: 1.000, 0.857, 0.811,
0.776, **0.755** — above 0.5 at position 5; token acceptance 95.7%, mean accepted length
3.24 at n=3, 5.20 of 6 at n=5 on deterministic text. Open prose decays after position 1
and returns ~2.2–2.5 tokens per round, hence the single-digit boost. Why n scales so
well: drafting costs ~16 ms/round (~5% of wall); the bottleneck is the batched verify
over the hybrid trunk (KV + GDN + indexer + PLE gather), close enough to free on GPU
that every accepted position converts to throughput (37 → 46 → 50 tok/s for n=2/3/5).

Two porting bugs are worth recording, both silent:

- the wide pre-norm export (`n_hc × n_embd` = 10240, not `n_embd`) — without it the
  drafter receives a mis-sliced hidden state and acceptance collapses **with no error
  anywhere**; it would have read as "weak drafter";
- the PLE conv-state block was never written by the state serializer, though the reader
  had expected it since the runtime port — it surfaced only when a drafter first
  exercised checkpoint restore.

## 5. Vision × MTP (T18)

With `--mmproj` and the drafter loaded, the first image request aborted the server:
`GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input")` inside the MTP
graph, on the backtrace of the image-chunk decode. The root cause, verified in code and
by git blame, was not accidental routing: our drafter-routing work had added a
**deliberate replay of every image chunk on the active drafter context**, written for
token-based drafters that accept embedded inputs. The MTP graph's input is dual — token
ids plus the trunk's wide hidden state — and demands tokens, so the ViT embeddings
tripped its assert. The target-side decode was already correct; neither reference
implementation supported the combination either (upstream skips there, a TODO; PR
#27836 carries the identical assert).

The fix is a gate, not a port: when the active drafter is the MTP head, skip the image
replay on the drafter (with a warning). The head still sees the image through the trunk
hidden state at draft time, and the target's verify pass guarantees the output. Gates:
vision request 200 with the correct answer, zero asserts, the skip marker present; text
acceptance unchanged at **98.5%** (mean accepted length 4.51 at n=6); the cumulative
statistics row that includes the vision request reads **851/866 = 98.3%** — drafting
after an image does not degrade. Production has run vision and speculation together since.

## 6. Rollback correctness (T19, 08-28)

Hours after deploy, in about an hour of our production agent workload the server logged 28
`speculative replay stalled … dropping draft` warnings and generation fell to **~13 tok/s**.
The cause was a one-line omission in the runtime port: `qwen4exp` was missing from
`llm_arch_supports_rs_rollback`. The rollback-salvage ring had shipped with the port (hybrid
sequence removal over recurrent+indexer+attention state, widened to include the PLE rows) — but
with the architecture absent from the list, the ring count clamps to zero and **every partial
reject paid a full checkpoint restore** (~114–148 MiB) instead: thrash, then the livelock guard
dropped speculation. The fix is one case in the list; the ring costs ~7.2 GiB of buffers.

The causal question after the fix — does the ring degrade acceptance? — needed care: a
first A/B at temp 0.7 (91.5% vs 76.9%) was **invalid**, because with rollback enabled
the verify takes the probabilistic acceptance branch and without it exact match.
Re-measured greedy, same mode both arms, 10 prompts per arm:

| Metric | checkpoints (off) | ring (on) |
|---|---|---|
| p0-reject | 6.0% (47/789) | 12.0% (81/675) |
| mean accepted length | 2.34 | 1.97 (−16%) |
| P(p0-reject \| previous round rejected) | 0.006 | **0.233** |
| P(p0-reject \| previous round full-accept) | 0.075 | 0.071 |

The base rate is identical across arms — the ring does not degrade normal rounds; the
entire excess is **one-round poisoning after a rollback** (3.3× base). We deployed
anyway: the problem was the 13 tok/s stall regime, the ring is stall-free by
construction (16 slots > 7-token max round), tg at parity in bench, pp −5.9% tolerated.

Two later fixes closed the file. The poisoning was the **drafter's own state** not being rewound:
H1 (reset the external head's state on rollback) was confirmed, though the formal gate was partial
(two-sided p = 0.149 > 0.05) — the excess rejection fell 0.172 → 0.078 (p0-reject 0.272 → 0.161;
agent-stack note, [2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md)); H2
(re-assert the target ring) was excluded — 206 forced re-asserts, zero effect. And the ring's
conv/PLE writer had been filling only group 0, so rollbacks restored the GDN and PLE histories from
never-written slots: that fix ([conv-ple-ring-slots.md](conv-ple-ring-slots.md)) took acceptance
~0.74 → 0.91–0.95 and tg 24 → 41–44 tok/s at n=6 on code, and the "inherent" post-rollback dip
vanished with it.

## 7. The PLE table leaves RAM (T25, 08-31)

The arithmetic that closed the thread: the PLE table is 35.76 GiB of the 98.5 GiB file, and the
production combo is Vulkan + `--no-mmap` (mmap collapses this build's prompt processing ~3× —
[vulkan-nommap-backend.md](vulkan-nommap-backend.md)) — those 35.76 GiB are real resident RAM
beside the other weights, the KV cache, a 3.85 GiB drafter and a 7.2 GiB ring.

`--ple-disk` changes where the table lives, not what the model computes. The direct
inspiration is **Salvatore Sanfilippo (antirez)'s dwarstar** — disk-resident model
state, pay only for what a generation touches — here applied to one sparse n-gram
table. The tensor is never created at load; rows are read **on demand from the GGUF
file itself** via `pread`, through a fixed-size LRU of whole 128-row blocks (16
shards, `--ple-cache-mib`, default 4096). No sidecar files, no preprocessing, the
published GGUF used verbatim — the row dequantization is unit-tested bit-exact
against the in-memory gather path.

| Gate (v1) | Result |
|---|---|
| Correctness: char-identical greedy output vs in-RAM (`cmp`) | **PASS** |
| RAM: +36.07 GiB available (5.10 → 41.16; the table's bytes now page cache) | **PASS** |
| Speculation unchanged: acceptance 0.9207 / mean length 4.51, identical both arms | **PASS** |
| Bench ±3%: pp512 **−58.0%**, pp2048 **−51.6%**, tg128 −9.0% | **FAIL → no deploy** |

The cold-cache failure had two causes. First, the fetch path was single-flight serial: ~1.3 ms
per 15 KB `pread`, 13× the disk's QD1 latency — the synchronous queue, not the disk. Second, a
structural miss on generative content: prose explores a new trigram at essentially every token
across 320M rows, so the hit rate never converges (a second pass over the same prose still
missed 35.6%): measured, 400 tokens of prose = 86.8% misses, 6,887 blocks, 9.13 s of a 23.5 s
wall (39%) → 17.0 tok/s; deterministic content, whose trigrams repeat, was already at parity.

The v2 insight reframed the problem: **the kernel page cache is the real L2.** Kernel-warm, a
"miss" of our own LRU costs ~5 µs (dequantization is 0.25% of wall); the 1.3 ms of v1 was almost
entirely the *first physical encounter* with each page — per-block cost fell 634/257 µs (truly
cold) → 41–46 (semi-warm) → ~5 (kernel-warm). And because common-language trigrams are shared
across all prose, a long-lived server's working set stays kernel-warm: true cold is rare and is a
warm-up cost, not a steady state. Two levers were kept and merged: per-block deduplicated
lookups, and `POSIX_FADV_WILLNEED` readahead submitted for missing blocks before the reads. Two
were rejected by data: a 16 GiB block cache (hit rate 19.7% vs ~50%, tok/s identical — the kernel
dominates) and a decompressed-row cache (dequant is no bottleneck when warm).

Result: the kernel-warm gap to the in-RAM table is **−4/−9%**; the fully cold gap stays
−52/−58% on pp. Deployed to production on 2026-08-31: warm tg **30.45 tok/s** (vs ~31.5
with the table in RAM, inside the expected band), MTP acceptance unchanged from the A/B
(0.92 / 4.51), 43 GiB available at boot, read-through counters live on `/metrics`.
Rollback is one flag and a restart. Guide: [`../guide/qwen4exp-ple-disk.md`](../guide/qwen4exp-ple-disk.md);
implementation and A/B harness in [`../../patches/t25-ple-disk/`](../../patches/t25-ple-disk/)
(15 patches: 12 base + 3 v2), already in the [`rocmfpx/`](../../rocmfpx/) snapshot.

## 8. What remains open

- **The warm-cache gap** (−4/−9% tg; cold pp −52/−58%): the deferred lever is a `pread`
  thread-pool, and it is *conditional* — worth building only if a `drop_caches` A/B
  shows the readahead submission insufficient, or once the store has parallel consumers.
- **The upstream drafter path**: official llama.cpp builds still cannot load the
  external-drafter setup — the `-md` MTP path lives in our fork (the upstream PR
  remains unmerged). Running this model with speculation requires this fork.
- **PLE-only mmap** as an alternative offload: sketched, unmeasured, needs its own spec.

---

## What transfers

1. **A missing name in a capability list is a silent performance cliff.** The rollback
   omission raised no error — a clamped-to-zero counter and a guard firing an hour later.
   Audit every capability switch a port touches.
2. **Gate acceptance, not "it runs".** Both T17 bugs were silent; one would have been
   misread forever as drafter quality. Only a parity gate against a known-good path
   catches a mis-sliced tensor.
3. **Measure where the time is before building concurrency.** The obvious fix for slow
   reads is a thread-pool; the measurements said the queue was 13× the disk, and that
   the disk barely matters once the kernel holds the pages.
4. **On a sparse-lookup architecture, pp is content-dependent** — 131 vs 242 tok/s on
   the same server, same model, different corpus. Cross-source pp numbers need matching
   prompt distributions.

---

*Thread index: [`README.md`](README.md); related notes:
[2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md) (quant quality),
[2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md) (agent stack).
Numbers transcribed verbatim from the lab's raw notes of 2026-08-27 → 08-31.*

*Attribution: GLM by z.ai.*
