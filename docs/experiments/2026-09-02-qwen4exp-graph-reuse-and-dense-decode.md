# Graph reuse and dense decode: two `qwen4exp` runtime speedups

2026-09-02 → 09-03, same machine as every note here (AMD Strix Halo, Ryzen AI MAX+ 395,
Radeon 8060S iGPU, 128 GB unified LPDDR5X), both builds of our fork (HIP and
Vulkan/RADV). A one-day sweep of runtime-optimization candidates on Qwen3.8-Flash-Next
(`qwen4exp`, the model of [qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md))
produced four falsified candidates (§5) and two patches that cleared their gates. The two
patches are the subject of this note. Neither touches a kernel: both live in the
model-specific graph-building code, both are gated on shapes rather than types, and
together they take plain tg512 from 28.10 to 33.33 tok/s on HIP (**+18.6%**) and from
25.79 to 34.18 tok/s on Vulkan (**+32.5%**), with perplexity identical to the fourth
decimal and bit-identical greedy fingerprints. Prefill does not move. Both have been
running on the lab's production server since 2026-09-03.

The two mechanisms, one line each:

- **Graph reuse** — the two `qwen4exp`-specific graph inputs never answered
  `can_reuse`, so every decode round rebuilt the whole compute graph from scratch;
  teaching them the shape check they were missing turns 1025 rebuilds into 6.
- **Dense decode** — while the learned indexer's selection budget covers the entire
  cache (the bench regime lies entirely below ~2k tokens), the QSA layer cannot
  prune anything and dense attention over the same cells is an identity; the indexer
  chain, ~372 nodes per round, exits the graph.

Bench model of the campaign: the 2-bit FP2MIX build of
[2026-09-02-flashnext-fp2-64gb-and-lean-requant.md](2026-09-02-flashnext-fp2-64gb-and-lean-requant.md)
(57,038,416,256 B on disk; llama-bench's `size` column reads 53.11 GiB because it
counts loaded tensors, not the file). The production FP4 LEAN build was re-measured
separately as a transfer check (§4).

The backdrop is the ~38 ms/round software residue the speculative-round thread
measured on this Vulkan build and could not remove from the decoding side
([speculative-round-software.md](speculative-round-software.md)) — round after round
of work that was neither drafting nor verifying. This note found two pieces of that
residue in the model graph itself, where they cost one and two patches respectively.

## 1. The mechanisms

Both patches are small and local: 102 insertions in the model file for the first,
104 across the model file and its header for the second. No backend code, no new
types, no arithmetic changes — which is also why the correctness gates of §3 could be
bit-level.

| | Graph reuse | Dense decode |
|---|---|---|
| Symptom | `graphs reused = 0`, `rebuilds = 1025` — full rebuild every round, both backends | ~372 nodes per round computing a selection the graph then ignores |
| Root cause | `llm_graph_input_qsa` and `llm_graph_input_ple` implement no `can_reuse`; the base class answers false | with `width == n_kv` the top-k mask surgery unmasks every row and returns the plain causal mask |
| Fix | `can_reuse` for both inputs, gated on the shapes `set_input` writes (`c7c523ee1`) | take the standard dense attention path in decode when the budget covers the cache (`6144779cc`) |
| Result | `reused = 1019/1025`, the six rebuilds are the 256-bucket steps | same output by construction; indexer chain out of the decode graph |
| tg512 | HIP +8.15%, Vulkan +27.0% (alone) | HIP +9.7%, Vulkan +4.3% (on top of reuse) |

### 1.1 A model that rebuilt its graph every round

llama.cpp reuses the decode graph across rounds by design: `n_kv` is padded to buckets
of 256 precisely so that the topology stays constant between rounds, and the reuse
decision is the AND of `can_reuse()` over every graph input. The base class returns
false — its own comment reads "returning false here by default will prevent from
reusing the graph if the check for the input type has not been implemented yet" — and
every standard input implements the check. The two inputs `qwen4exp` adds on top of
the standard set did not: `llm_graph_input_qsa` (the indexer attention, one per
full-attention layer) and `llm_graph_input_ple` (the PLE n-gram gather). Both carried
only `set_input`. One input answering false is enough: on this model every round of
every graph paid a full rebuild — decode and prefill batches alike, and the drafter
passes of the server too.

The counters, read before any patch with instrumentation the fork already ships
(`LLAMA_GRAPH_BUILD_TIMING=1` and verbose logging): `graphs reused = 0`,
`graph rebuilds = 1025`, reset+build **1.56 ms per round on HIP and 1.74 ms on
Vulkan** — 4.4-4.5% of the 35.59/38.77 ms round, before counting what a fresh graph
invalidates downstream. On HIP the new graph id also breaks the CUDA-graph fast path
every round: disabling CUDA graphs entirely on the baseline moves tg512 by +0.1%
(28.13 ± 0.30 vs 28.10 ± 0.09), so the captured graphs were contributing nothing.
On Vulkan every rebuild walks the per-node fusion cascade again and re-records the
command buffer from scratch.

Patch `c7c523ee1` implements `can_reuse` for both inputs, modeled on the
existing `llm_graph_input_attn_k_dsa` check: QSA compares the index, cell, position
and bias tensor shapes against the current batch's indexer-cache context; PLE compares
the row/gathered shapes and the disk flag. Both refresh the stored memory context to
the current batch's, like every other `can_reuse`, and fall back to a rebuild on
anything they cannot verify — `dynamic_cast` and null guards first — so the failure
mode is the old behavior, never a stale graph. With the patch in: `graphs reused =
1019`, `rebuilds = 6`; the six are the bucket-256 steps, which are legitimate topology
changes (1019 + 6 = 1025, the same rounds as before). The HIP backend logs the
mechanism directly from then on — `ROCm graph id N reused` lines appear round after
round, where the baseline printed a `ROCm graph warmup reset` per round instead.

Two facts about the size of the win. On HIP the round got 2.68 ms shorter
(35.59 → 32.91 ms) — more than the 1.56 ms the build itself costs — because the
reused uid keeps the CUDA-graph launches alive, so the patch collects both effects.
On Vulkan the same patch buys 8.25 ms of the 38.77 ms round: the 1.74 ms of build
plus ~6.5 ms consistent with command-buffer re-registration and allocator churn —
not individually attributed. That asymmetry is why one patch is worth +8.15% on one
backend and +27.0% on the other, and it inverts the backend ranking (§6).

### 1.2 Dense attention when the budget covers the cache

The 12 full-attention layers of this model are QSA layers: a learned indexer scores
the cached cells, keeps the top-k, and attention runs over the selected cells with a
compressed key stream — the mechanism that keeps long-context KV small (flagship note
[qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md) §1). The width of the
selection is `min(n_kv, budget)` with
`budget = top_k + compress_ratio − 1 = 2048 + 4 − 1 = 2051` on this model. In
single-stream decode below that boundary, `width == n_kv`: the selection is asked to
keep every visible cell, the mask surgery that carves the top-k mask unmasks every
row, and the mask that comes out is the plain causal `self_kq_mask` again. The indexer
chain that feeds it — pooling, rope, relu, the top-k itself, the mask surgery —
computes an answer the graph then ignores: **(27 + 4) nodes per full-attention layer,
12 layers, ~372 nodes per round**, for nothing. Reading that regime as dense attention
is a mathematical identity, not an approximation.

Patch `6144779cc` builds the standard dense attention path when the condition holds —
a decode round, one stream, `n_kv` at or below the budget. What changes in the
graph, per round:

| Piece | Nodes | Inside the budget |
|---|---|---|
| indexer chain (gather, pooling, rope, relu, top-k) | 27 × 12 layers | out of the graph |
| top-k mask surgery | 4 × 12 layers | out — it returned the causal mask unchanged |
| attention itself | — | dense over the same cells, same mask, same result |
| indexer-cache writes | 2 | **stay**, so the selection returns intact |

The cache writes are the load-bearing detail: when `n_kv` outgrows the budget and
the selection comes back, it reads a cache that was maintained all along — no key is
missing at the regime boundary. And the padded-`n_kv` step already forces a graph
rebuild there, so the reuse gate of §1.1 never carries a graph across it; the
shortcut condition is re-checked in `can_reuse` on the same terms.

Scope, stated plainly: the shortcut covers single-stream decode rounds inside the
budget — tg512 lives entirely inside it; a bench like tg128 or a server decoding one
token at a time does too. Prefill batches (`n_seq_tokens > 1`) never take it, and
beyond ~2k context the indexer chain returns. The verify rounds of a speculative
server run multi-row batches, which the condition excludes.

## 2. Measurements

Protocol, identical for every cell: frozen baseline builds and patched builds of the
same fork; FP2MIX model; KV q8_0 on both streams, `-fa 1`, full offload, `--no-mmap`
(`-mmp 0` — required on this Vulkan build,
[vulkan-nommap-backend.md](vulkan-nommap-backend.md)); plain llama-bench, no drafter;
5 repetitions after a discarded warm-up, mean ± sample standard deviation; go/no-go
thresholds fixed before each run (≥ +4% tg512 for the reuse patch, ≥ +3% for the
dense-decode patch, quality gates per §3); the production service stopped, model in
page cache, clocks at their sustained level.

Labels, so the cells are comparable across this repo's notes: tgN is N tokens
generated from an empty prompt (`-p 0 -n N`); ppN is the prefill of an N-token
prompt, and every pp cell below carries its context in the label. The ± values are
standard deviations over the 5 repetitions, not ranges. Perplexity runs on two
corpora — the Italian holdout (dante) and the wikitext-2 test set, 16 chunks each at
ctx 4096 — always on the same backend before and after a change.

### tg512 — 512 generated tokens, empty prompt

| Build | HIP (tok/s) | Vulkan (tok/s) |
|---|---|---|
| baseline (`dadc23e44`) | 28.10 ± 0.09 | 25.79 ± 0.63 |
| + graph reuse (`c7c523ee1`) | 30.39 ± 0.23 (+8.15%) | 32.76 ± 0.13 (+27.0%) |
| + dense decode (`6144779cc`) | **33.33 ± 0.14 (+18.6%)** | **34.18 ± 0.40 (+32.5%)** |

The step deltas are +9.7% (HIP) and +4.3% (Vulkan) for the dense-decode patch on top
of reuse. Three of the four steps are ≥9σ (reuse on HIP 9.3σ, dense decode on HIP
10.9σ, reuse on Vulkan 10.8σ); the fourth — dense decode on Vulkan,
32.76 ± 0.13 → 34.18 ± 0.40 — is ~3.4σ, with 3σ bands that barely overlap. That is
the weakest link of the chain, stated as measured.

Drift was controlled, not assumed away. The Vulkan reuse-only cell comes from a
control arm re-measured hours after the baseline with its pp cells at parity
(365.05 / 326.40 / 241.28 vs 365.27 / 327.66 / 240.51 tok/s on pp2048 / pp8192 /
pp32768), so the tg gain is not a machine-wide speedup; the final Vulkan build was
measured against that same control, minutes apart. On HIP, re-running the baseline
binary mid-campaign reproduced 28.13 ± 0.30 — no time-of-day effect on the tg side
either.

Run conditions were logged around every measurement: model in page cache (boot to a
healthy server in ~11 s), sustained clocks 2108-2135 MHz and package power 68-85 W,
consistent between baseline and patched builds — the post-patch stdevs (±0.13 /
±0.14 on tg512) are among the tightest of the campaign and incompatible with
throttling.
Warm-up runs were discarded on both arms of every comparison.

### Prefill — unchanged, as the mechanisms predict

Both patches act on decode rounds; prefill rebuilds once per ubatch either way and
never takes the dense shortcut. Measured on Vulkan:

| Cell (ctx = prompt length) | baseline | both patches | Δ |
|---|---|---|---|
| pp2048 (ctx 2048) | 365.27 ± 4.18 | 368.31 ± 2.84 | +0.8%, inside the baseline's ±4.2 |
| pp8192 (ctx 8192) | 327.66 ± 6.37 | 324.10 ± 1.81 | −1.1% |
| pp32768 (ctx 32768) | 240.51 ± 1.81 | 239.82 ± 1.35 | −0.3% |

The HIP baselines for the same cells: pp2048 354.77 ± 0.84, pp8192 313.43 ± 1.20.

## 3. Correctness

| Gate | Result |
|---|---|
| ppl, Italian holdout (dante), 16 chunks, ctx 4096, same backend | **33.7892 ± 0.583** at the HIP baseline; 33.7892 at the reuse step; 33.7892 with both patches — identical to the fourth decimal, and all 16 per-chunk values identical, not just the final estimate |
| Greedy fingerprint — 50 tokens, temp 0, seed 42, pinned prompt | sha256 `8790370738f5abdb60d88cac75d91bcd9c385d17caa00d946e97641c86a8df75` (prefix `879037…df75`) bit-identical on every build measured: both baselines, each patched step, the final Vulkan build — one prompt per backend per build |
| Mechanism counters | `rebuilds` 1025 → 6 with the reuse patch, and only with it; `graphs reused` 0 → 1019 |

The fingerprinted output opens with `<think>` — this model reasons at temp 0 — so the
hash covers reasoning tokens as well as content. One property makes it the
portable gate of the two: the two backends' arithmetic diverges enough to move ppl
−3.4% on the hard corpus, yet the greedy outputs converge to the same 50 tokens —
so a bit-identical fingerprint is meaningful across builds where a ppl comparison
would not be.

Coverage, stated as measured rather than generalized:

- **ppl gates are same-backend by construction.** The two backends read −3.4% apart on
  this corpus at baseline (34.9777 Vulkan vs 33.7892 HIP) and within 0.1% on
  wikitext-2 (17.2724 / 17.2847); cross-backend ppl is never a valid comparison on
  this stack. Every ppl number in the table above is HIP, before/after on the same
  build chain. On Vulkan, the reuse step gated wikitext-2 ppl (17.2724, identical to
  baseline) plus the fingerprint.
- **ppl does not exercise the dense-decode shortcut.** Perplexity is a prefill
  measurement (`n_seq_tokens > 1`); the shortcut fires in decode only. The equivalence
  evidence for that identity is the greedy fingerprint — 50 decode tokens per backend
  per build, bit-identical — which is the actual scope of the proof, not corpus scale.
- **The final cumulative Vulkan build carried the fingerprint gate only** (no ppl run
  on it); its decode path comes from the same patch set as the HIP build's, and the
  fingerprint is bit-identical there too.

## 4. Transfer check: the production FP4 build

The campaign benched the 2-bit FP2MIX; production serves the FP4 LEAN. Same protocol,
LEAN build — 105,753,531,776 B = **98.491 GiB** on disk (the bench `size` column
reads 98.48 GiB, again tensors-loaded accounting):

| Cell (HIP) | baseline | both patches | Δ |
|---|---|---|---|
| tg512 | 23.42 ± 0.35 | 27.52 ± 0.08 | **+17.5%** |
| tg128 | 23.29 ± 0.08 | 26.54 ± 0.16 | +14.0% |
| pp8192 (ctx 8192) | 353.49 ± 1.23 | 348.79 | −1.3%, within noise |

No Vulkan cell was measured on the LEAN build; the table is HIP-only, and the ppl and
fingerprint rows of §3 were not re-run for it. Two notes on reading it: the first
pass of the pp8192 cell read 313.50 ± 37.50 — one outlying repetition — and the
verification re-run read 348.79, which is the number quoted. And +17.5% on FP4 vs
+18.6% on FP2MIX is the transfer holding within ~1 pt, which is what the patches'
shape-gated (format-agnostic) design predicts.

## 5. What didn't work

Four candidates failed their gates and were not merged. Numbers first, cause after:

| Candidate | Numbers | Cause |
|---|---|---|
| RS-VIEW — elide the recurrent-state gather (view instead of `get_rows`), −244 MB/round (226.5 state + 17.7 conv) | tg512 32.84 ± 0.18 vs 33.33 = **−1.5%** (gate ≥ +2%) | removing 244 MB per round made the round *slower*: the round is not byte-bound (§6), and the contiguous temp the gather produces is worth more than the bytes it costs |
| SLIM-GRAPH — fuse hyper-connection and scatter nodes (−1135 nodes/round verified) | tg512 33.59 ± 0.20 = +0.78% (gate ≥ +3%); pp8192 295.63 ± 3.85 = **−5.7%** vs 313.43 ± 1.20 | the `cont` after the permute materializes the transposed buffer; in prefill that costs more than the fusion saves. Any fusion that materializes a transpose must be measured on pp before trusting its tg |
| spirv-opt on the coopmat shader family (Vulkan) | pp2048 −2.9%, pp8192 −3.6%, pp32768 −2.4%; tg at parity (32.77 ± 0.18 vs 32.76 ± 0.13); ppl 17.2724 and fingerprint identical | quality intact, the shaders are ~47% slower — the pass inlines past the register budget. First experimental confirmation, on RADV, of the upstream workaround that excludes coopmat from `-O` |
| UB-1024 — physical batch 1024, flag only | pp8192 Vulkan 330.28 ± 5.36 = +0.8% (gate ≥ +3%), HIP 307.33 ± 6.22 = −1.9%; ppl wiki 17.2522 (Δ −0.12%) | expert-tile coverage is already full at the default 512 on this model; the bigger batch had nothing left to amortize |

The corresponding patch branches (`ea15d2fbb`, `40fd518d8`, `f4d7058f4`) are archived
as series, not applied. Method note on the spirv row: its A/B ran interleaved on the
same pair of builds (one pp pass and one tg pass), warm-up discarded on both arms.
The RS-VIEW row doubles as an independent
falsification of "decode is memory-bound on this model" — the cheapest byte elision
available made things worse.

## 6. Structural findings

These outlive the patches; they set the priority order for the next levers.

1. **The decode round is not byte-bound.** A tg512 round moves ~3.0-3.56 GB; at the
   236.6 GB/s read ceiling that is a 12.9-15.05 ms floor against 35.59 ms measured —
   tg runs at **36-42% of the ceiling**. The bottleneck is nodes, latency and kernel
   efficiency. Two falsifications now agree: eliding bytes regressed (RS-VIEW),
   removing nodes paid off.

   | Baseline, tg512 | HIP | Vulkan |
   |---|---|---|
   | round | 35.59 ms | 38.77 ms |
   | bandwidth reached | 100.1 GB/s | 91.9 GB/s |
   | % of the 236.6 GB/s read ceiling | **42%** | **39%** |
   | byte floor for the same 3.56 GB | 15.05 ms | 15.05 ms |
2. **Residual node cost is ~0.2 µs — for the elementwise class only.** SLIM measured
   it directly: −1135 mostly-elementwise nodes = +0.78%. Node cost is heterogeneous
   by ~40× across classes: the indexer chain removed here cost ~8 µs per node (372
   nodes, 2.90 ms) because those are structural nodes — mask surgery, degenerate
   shapes. A "cost per node" is a class property, not a constant; do not reuse the
   0.2 µs figure outside its class.
3. **The LM head is paid every round, even in llama-bench.** The Q6_K output head
   ([2560 × 248,320], 521.5 MB) reads at 2211-2275 µs per call (2.21-2.27 ms) —
   231.8 GB/s, **98% of the read ceiling** — and a 64-token perf-logged run calls it
   65 times. It is the single largest line item in the round and the one place where
   byte→time is linear, so a head requant (Q6_K → FP2, 198.7 MB) has a measured bound
   of ~+4.5% tg. That is format-specific work, outside these two patches.
4. **The backend ranking is a function of the patches, not of the hardware.** At
   baseline HIP won tg512 by +9.0% (28.10 ± 0.09 vs 25.79 ± 0.63) and Vulkan won
   every pp cell. With both patches in, Vulkan wins tg512 too (34.18 ± 0.40 vs
   33.33 ± 0.14) and keeps pp (368.31 ± 2.84 vs the 354.77 ± 0.84 HIP baseline on
   pp2048) — the reuse patch removed a cost Vulkan was paying more of. A
   command-buffer-replay candidate had been planned to chase exactly the pre-patch
   Vulkan gap; with the gap closed by reuse instead, it is left with ~0-3% of
   headroom and sits at the bottom of the queue. Production has been on Vulkan since
   08-29; it inherits both patches directly.

## 7. Run it

- **The code.** Build from the [`rocmfpx/`](../../rocmfpx/) snapshot in this repo —
  it matches fork `6144779cc`, both patches in. Or apply the two patches of
  [`patches/optim-camp/`](../../patches/optim-camp/) onto `dadc23e44` (ple-store v2,
  itself `t27-ple-store-v2` on `f629365da`); that reproduces the tree of `6144779cc`.
- **No flags.** Both mechanisms are on whenever their shapes hold. The reuse gate
  falls back to a rebuild on anything it cannot verify; the dense path exists only
  for single-stream decode rounds inside the budget. There is nothing to enable.
- **The bench** behind every table here (plain, no drafter; the pp32768 cell came
  from a separate pass at the same flags):

```console
$ llama-bench -m <gguf> -p 0 -n 512 -r 5 -ngl 99 -fa 1 -mmp 0 -ctk q8_0 -ctv q8_0
$ llama-bench -m <gguf> -p 2048,8192 -n 128 -r 5 -ngl 99 -fa 1 -mmp 0 -ctk q8_0 -ctv q8_0
```

- **To watch the mechanism fire:** `LLAMA_GRAPH_BUILD_TIMING=1` plus verbose logging
  prints `graphs reused` / `graph rebuilds` per run — 0/1025 before the patch,
  1019/6 after. `LLAMA_GRAPH_REUSE_DISABLE=1` A/Bs the reuse on the same binary.
- **Quality gates:** `llama-perplexity` at ctx 4096, 16 chunks, same KV/no-mmap flags,
  compared same-backend only; the greedy fingerprint is a 50-token completion at
  temp 0 / seed 42 on a pinned prompt, sha256 over the content:

```console
$ llama-perplexity -m <gguf> -f <corpus> -c 4096 --chunks 16 \
    -ngl 99 -fa 1 --no-mmap -ctk q8_0 -ctv q8_0
```

## 8. Artifacts

- **Model files** — HF repo
  [`pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF`](https://huggingface.co/pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF):
  the FP2MIX bench model as
  `ROCmFP2-STRIX_LEAN/Q2_0_ROCMFP2_STRIX_LEAN-00001-of-00002.gguf` and
  `ROCmFP2-STRIX_LEAN/Q2_0_ROCMFP2_STRIX_LEAN-00002-of-00002.gguf` (2 shards), the
  production FP4 LEAN under `ROCmFP4-STRIX_LEAN/` (105,753,531,776 B = 98.491 GiB).
- **Patches** — [`patches/optim-camp/`](../../patches/optim-camp/): `0001`
  (`can_reuse` for QSA and PLE graph inputs, fork `c7c523ee1`) and `0002` (dense
  attention in decode when the QSA budget covers the cache, fork `6144779cc`), base
  `dadc23e44`.
- **Production** — running on the lab's production server since 2026-09-03.

---

*Thread index: [`README.md`](README.md); related notes:
[qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md) (the model and its
runtime thread),
[2026-09-02-flashnext-fp2-64gb-and-lean-requant.md](2026-09-02-flashnext-fp2-64gb-and-lean-requant.md)
(the FP2MIX build benched here),
[speculative-round-software.md](speculative-round-software.md) (the ~38 ms/round
structural residue on Vulkan that this thread attacks from the model side),
[vulkan-nommap-backend.md](vulkan-nommap-backend.md) (the no-mmap requirement behind
the bench flags).
Numbers transcribed verbatim from the lab's raw notes and logs of 2026-09-02 → 09-03.*

---

*Attribution: GLM by z.ai.*
