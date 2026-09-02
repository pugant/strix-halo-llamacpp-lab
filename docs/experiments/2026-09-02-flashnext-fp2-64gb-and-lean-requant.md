# T27 — Flash-Next at 2.58 bpw: the 64 GB FP2/FP4 mix, and the LEAN requantized from BF16

2026-09-01 → 09-02, same machine as every note here (AMD Strix Halo, Ryzen AI MAX+ 395,
Radeon 8060S iGPU, 128 GB unified LPDDR5X). Two builds of Qwen3.8-Flash-Next (`qwen4exp`,
the model of [qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md)) came out of one
plan. **F2** re-quantizes the production 98.5 GiB STRIX_LEAN from the native BF16 release
with the unsloth importance matrix: same per-tensor format recipe, better source weights,
perplexity down on both corpora — it now serves under the same filename. **F1** answers a
different question — how small can the model get and still *run* on a 64 GB Strix Halo
machine: an imatrix-guided FP2/FP4 mix at 2.58 bpw, 57.04 GB, 97.1% of parameters in the
2-bit ROCmFPX format.

F1's speed and RAM numbers are good: +3.6 to +18.5% over the LEAN on every tg/pp cell, and
a peak of 55 GiB of host memory with the drafter and the vision projector loaded, inside a
60 GiB boundary. F1's quality numbers are not: perplexity 37.9 on the Italian holdout (32×
the LEAN) and 7.29 on the English calibration corpus (2.17×). Most of this note is about
being precise as to why — the 2-bit dual-UE4M3 codebook sits at its theoretical error floor,
and on memorized text the damage explodes. The diagnosis closed only after chasing a
phantom "wrong tensor" through three test rounds; the root cause of that detour (§6) is
worth more than the bench tables. All phases B-F measurements ran on the Vulkan build of
the `t27-ple2-vk` image (§4), dedicated GPU window, greedy.

## 1. The two builds

| | F2 (LEAN requant) | F1 (FP2MIX-64GB) |
|---|---|---|
| Question | same model, better source | same model, 64 GB machine |
| Source | BF16 unsloth release (354 GB, 8 shards) + unsloth imatrix | same |
| Format recipe | identical to the previous LEAN build (byte-identical type multiset) | per-tensor mix, imatrix-guided |
| Result | 98.491 GiB @ 4.78 bpw | 57.04 GB @ 2.58 bpw |
| Quality | ppl −0.6% / −4.9% vs previous LEAN | ppl 32× / 2.17× — structural, §5 |
| Disposition | swapped in place, same filename | an honest entry point for 64 GB, §5 |

(Every cell of this overview is measured and sourced in §2-§3 and §5-§7 below.)

## 2. F2 — the LEAN requant from native BF16

The previous LEAN was built the hard way: FP8 release → converter (lossy FP8→F32 dequant)
→ Q8_0 intermediate → quantize with `--allow-requantize`. The 354 GB BF16 release (8
shards) plus the full unsloth importance matrix (580 MB, 926 entries, exact tensor-name
matching) allow a direct path: `llama-quantize --imatrix --output-tensor-type Q6_K` at
preset 106, 32 threads, ~21 minutes of wall time. Format recipe unchanged on purpose —
output Q6_K, token embedding Q5_K, the PLE table falling back from the preset Q5_K to
Q5_1 (its 160 columns are not divisible by 256), everything else on the STRIX_LEAN preset.

Gates (all from the quantize log and a stdlib-only GGUF reader, `logs/t27-quant-f2.log`):

| Gate | Expected | Measured |
|---|---|---|
| imatrix coverage of quantizable 2D+3D tensors | ≥ 95% | 878/915 = **96.0%** |
| tensor type multiset vs previous LEAN build | identical | **1224/1224, zero diff** (type+shape+bytes) |
| file size | ~98.5 GiB | 105,753,531,776 B = **98.491 GiB** (Δ 800 B, header) |
| `did not find weights` lines | ≤ 6 | **6** = 5 never-calibrated embeddings + 1 `blk.1.ple_conv1d` (all benign: the message is the imatrix lookup, not a load error) |

Perplexity, full corpora, Vulkan build, `-fa on` (Fase A protocol; logs `t27-faseA-*` for
LEAN and F1, `t27-gpu/ppl-F2-*` for F2 — `Final estimate` lines):

| GGUF | Italian holdout (dante) | EN calibration |
|---|---|---|
| previous LEAN | 1.1843 | 3.3601 |
| **F2** | **1.1776 (−0.6%)** | **3.1961 (−4.9%)** |
| F1 (next section) | 37.9027 | 7.2928 |

Same bytes per tensor, better weights inside them. F2 was not re-benched for speed:
identical format and size mean unchanged kernels by construction (the tg/pp tables of §7
use the previous LEAN as the baseline for exactly this reason). After the Fase A numbers
above, the build was swapped in place: the production LEAN filename now carries it.

## 3. F1 — the imatrix-guided FP2/FP4 mix

Format constants of the fork's types, verified against `ggml` type traits: FP2
(`Q2_0_ROCMFPX`, type 107) = 10 B per 32-element block = **2.5 bpw**; ROCmFP4 = 18/32 =
4.5 bpw; ROCmFP4_FAST = 17/32 = 4.25 bpw. Two structural facts shape the whole build:

- The PLE table (51.2 B of the model's 176.94 B parameters) **must** go to FP2 — at Q5_1
  it is 35.763 GiB of the file and the 57 GB target is unreachable without it. Its 160
  columns are block-aligned, so the per-tensor override is legal.
- The 48 `ffn_gate_inp` routers and 36 `ssm_conv1d` convolutions are **not quantizable at
  all** (`tensor_allows_quantization` gate in the quantizer): they stay F32, as in the
  LEAN — the protection is structural, not a mapping choice.

The mapping generator (`scripts/t27-fp2-mapping.py`, self-test of 25 asserts, dry-run on
the real LEAN) reads the imatrix importance per tensor, sorts quantizable tensors by
ascending importance, assigns FP2 to the whole tail, and promotes the importance head to
FP4 until the estimated size fits the 57 GB target. It emits a tensor-type file for the
fork's `--tensor-type-file`: **847 rules**, anchored regexes with escaped dots,
first-match-wins, with the PLE override pinned as rule 1. One launch detail worth
recording: without `--output-tensor-type Q6_K` the output head drifts to FAST (322 MiB
≠ recipe) — caught at tensor [4/1224] and relaunched.

Measured composition (stdlib GGUF reader over the final file, `logs/t27-f1-tensors.tsv`):

| Type | Tensors | Bytes | Notes |
|---|---|---|---|
| Q2_0_ROCMFPX (2.5 bpw) | 607 | 50.026 GiB | 171.89 B params = **97.1%** |
| Q4_0_ROCMFP4 (4.5 bpw) | 224 | 1.945 GiB | the 192 hyper-connection projections, all 24 attn_k/v, the last two blocks' expert matrices (blk.45-46) |
| Q4_0_ROCMFP4_FAST | 2 | 0.003 GiB | output hc heads (never calibrated) |
| F32 | 388 | 0.244 GiB | routers, conv1d, norms — as in the LEAN |
| F16 / Q5_K / Q6_K | 1 each | 0.000 / 0.407 / 0.486 GiB | `ple_conv1d` fallback; embeddings; output |

File: `FLASHNEXT-FP2MIX-64GB.gguf`, **57,038,416,256 B = 57.04 GB decimal (53.1 GiB)**,
2.58 bpw effective, quantize run ~19 minutes at 32 threads. The all-FP2 floor is 55.86 GB,
so the FP4 head costs ~1.2 GB — the mix buys attention and hyper-connection precision at
the cheapest possible price. On disk the FP2 PLE is 16,000,076,800 B = 14.90 GiB (50 B per
row vs 120 B at Q5_1).

## 4. The enabler: ple-store v2

F1 initially could not be *served* within 64 GB: the T25 on-disk PLE store
([qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md) §7,
[guide](../guide/qwen38-flash-next-ple-disk.md)) supported Q5_1 tables only — a hard
assert at load — and without `--ple-disk` the 14.90 GiB FP2 PLE would sit in RAM on top
of the remaining weights, drafter, KV and buffers, past a 60 GiB budget.

ple-store v2 (branch `t27-ple-fp2-store`, commit `dadc23e44` on the fork) generalizes the
type-specific assumptions instead of adding a second hardcoded layout: `row_bytes` derives
from `ggml_blck_size`/`ggml_type_size`, the head-dimension and file-parity checks derive
from it, and the type allowlist becomes {Q5_1, Q2_0_ROCMFPX}. The bit-exact test battery
(`tests/test-ple-store.cpp`) now runs twice — Q5_1 as the v1 regression, FP2 with a
dedicated fixture — which is what let both arms of every §5-§7 comparison run on the same
patched image (`t27-ple2-vk`): the fix is inert on Q5_1 tables, verified by the battery.

In production measurements the FP2 store behaves like the Q5_1 one. Boot marker, verbatim:
`load_arch_tensors: PLE table disk-resident: 14.90 GiB externalized (q2_0_rocmfpx,
320001536 rows), cache 4.00 GiB`. On-disk vs in-RAM generation: **zero bytes of
divergence at equal cache state** (cold==cold, warm==warm, and a cross-boot disk run —
all byte-identical greedy outputs), tg gap −1.3% cold (28.02 vs 28.38 tok/s, mean of 3
prompts) and ≤ ~4% warm. One methodological point that cost a detour: the naive
first-run-disk vs first-run-ram comparison *diverges* — but the control matrix shows the
divergence is cold-vs-warm (full prefill vs prefix-cached), not disk-vs-RAM; on this
Vulkan stack greedy tokens are only char-exact within the same compute path (the same
lesson as [interleaved-decode-charidentity.md](interleaved-decode-charidentity.md), the
twin-run note of 08-24).

## 5. Quality: the honest numbers

### Perplexity

Quoted again for focus (Fase A, full corpora):

| GGUF | dante | EN calibration | vs LEAN |
|---|---|---|---|
| LEAN (previous) | 1.1843 | 3.3601 | — |
| F2 | 1.1776 | 3.1961 | **−0.6% / −4.9%** |
| F1 | 37.9027 | 7.2928 | **32× / 2.17×** |

A CPU run (no GPU, `-ngl 0`, dequant-and-matmul) reads ppl 24 on the first dante chunk —
the explosion is in the weights, not in a backend kernel.

### The error is at the codebook floor

Relative RMSE vs the BF16 source, per family (`scripts/t27-fp2-rmse.py`,
`logs/t27-fp2-rmse-report.txt`; decoder validated against sanity fixtures, §6):

| Family (tensor) | F1 type | RMSE F1 | RMSE LEAN |
|---|---|---|---|
| PLE (`per_layer_token_embd`) | FP2 | **0.3169** | n/d (Q5_1) |
| MoE experts 3D (`blk.1.ffn_gate/up/down_exps`) | FP2 | 0.3277 / 0.3291 / 0.3294 | ~0.106 (FAST) |
| dense 2D (`blk.1.attn_qkv`, `ffn_*_shexp`) | FP2 | 0.3500 / 0.3536 / 0.3593 | ~0.092-0.113 |
| hyper-connections (`hc_*`) | FP4 | 0.0867 / 0.0898 / 0.0917 | ~0.102-0.106 |
| attention (`blk.3.attn_k`) | FP4 | 0.0919 | 0.0922 |

Reference: for a Gaussian source and the FP2 codebook {-4, −1, +1, +4} × block scale, the
optimal relative RMSE (scale s = 0.377σ) is **0.347**. Every FP2 family measures
0.317-0.359, bracketing that floor; the FP2 aggregate is 0.3380; the dense projections,
whose sources run at kurtosis 4.8-10.3 against the Gaussian's 3, are the ones above it. The
quantizer is doing its job — the format itself is the cost.

### Why 32× on Italian but 2.17× on English

The Italian holdout is memorized text: the LEAN's 1.18 there is recall, not prediction.
FP2 noise on the PLE (cos 0.949 with the source table — pure quantization noise, §6)
breaks that fine-grained recall and the loss compounds non-linearly; the English
calibration corpus carries the "true" 2-bit damage, 2.17×. What transfers from the T10
finding is the conclusion, not the ratio: its ~3 bpw candidate failed a perplexity gate
~5× — nearly symmetric across corpora (EN ~5×, IT ~4.5×)
([speculative-round-budget.md](speculative-round-budget.md)) — while the same nominal step
here reads 32× on one corpus and 2.17× on the other. Nominal-bpw damage is not comparable
across corpora, and nominal bpw does not port across architectures, as ROCmFP3 already
showed ([2026-08-23-rocmfp3-quality-speed.md](2026-08-23-rocmfp3-quality-speed.md)).

### Agentic and vision smoke (Fase F)

| Smoke | LEAN | F1 |
|---|---|---|
| 3-round tool calling | 3/3 well-formed (get_weather ×2, calculate 25−22), acc-len 4.67 | **1/3** — round 1 fine; round 2 answered directly with hallucinated data; no malformed calls, thinking intact; acc-len 3.65 (0.78× LEAN) |
| Vision (screenshot description) | detailed, accurate (filters, counts, dates match ground truth) | reasoning **sees the image correctly** ("a car rental or listing site", the filter labels, "Auto disponibile: 132" — the true count) but never answers: 660 tokens of thinking degenerating into a repetition loop, `finish=stop`, content length 0 |

No `speculative replay stalled` in any of the 10 faseF logs (`faseF-agentic-*`,
`faseF-vision-*`, `faseF-smoke-*-server.log`), no mmproj×drafter regression;
vision prompts were 1,111 tokens on both arms. The vision acceptance row (F1 0.868-0.916,
LEAN 0.712 — F1 *higher*, plausibly an artifact of its own repetitive text) missed the
≥0.95 threshold on **both** arms: that threshold is not calibrated for this protocol and
needs re-anchoring before reuse.

### Positioning

F1 is published with these numbers in the open: an **entry point for 64 GB Strix Halo
machines** — it runs the full production stack (external MTP drafter, vision, disk PLE,
SSD prompt cache) inside 60 GiB at LEAN-beating speed — with **clearly degraded
quality**, most visibly on memorized content, long agentic loops and the vision
reasoning-to-answer handoff. It is not a drop-in LEAN replacement, and the card says so.

## 6. The debugging story: two PLE families that never existed

The ppl table above was expensive to trust. This is the full arc, kept because the
failure mode is generic.

**The alarm.** F1's Italian ppl at 32× the LEAN, with F2 — built from the *same* BF16
shards and imatrix, same quantizer — *better* than the LEAN. If the pipeline were broken,
F2 should have paid too.

**Front 1 — the backend.** The CPU run (ppl 24, above) rules the Vulkan kernels out.

**Front 2 — the quantizer.** Per-family RMSE says FP2 sits at its theoretical floor (§5).
The quantizer is optimal-for-format; the format is the problem.

**Front 3 — the phantom.** A Python RMSE/provenance probe (Test B) then reported that the
F1 PLE and the LEAN PLE were nearly orthogonal (cos ≈ 0.017), with LEAN-vs-source cos
≈ 0.018 and row norms ×1.28 — reading for all the world like **two different tensors**:
the LEAN built from a "wrong-family" source, F1 faithful (cos 0.949). Test C reproduced
the numbers; Test D deepened them (F2-vs-LEAN PLE cos 0.816, unimodal, byte-level
d/m correlations 0.995 — read as two independent renderings of a hidden common content)
and concluded F2's baseline was invalid pending a PLE rebuild. Three test rounds, one
consistent story — and one shared, unvalidated assumption: the Python Q5_1 decoder.

**The root cause.** Block-forensics on F2's PLE bytes against the source shard showed the
block scales and mins matched the source *exactly* (d correlation 1.0000, max |Δm| = 0)
while 97% of quant codes differed, uniform-random — and a byte-permutation search closed
it: bytes 4-8 of the block are the high-bit array `qh`, bytes 8-24 the nibbles `qs`
(match 0.995-0.998). The Q5_1 block layout in our fork (`ggml-common.h:247-258`) and in
current upstream ggml / llama.cpp is:

```c
typedef struct {
    ggml_half d, m;
    uint8_t qh[4];          // 5th bit of quants  <-- FIRST
    uint8_t qs[QK5_1 / 2];  // nibbles / quants   <-- AFTER
} block_q5_1;
```

The classic Python decoders assume `{d, m, qs, qh}` — the pre-reorder order. Reading
`qh` as `qs` turns the tensor into noise: cos ≈ 0.018, norms ×1.28, exactly the
"two families" signature. Quantizer and runtime address the struct *fields*, so they were
self-consistent all along — which is also why the LEAN and F2 perplexities were fine while
the Python probes disagreed.

**The resolution.** With the corrected decoder (`scripts/t27-fp2-rmse.py`,
`deq_q5_1` fixed to the fork order): cos(F2-PLE, source) = **0.99930**, cos(LEAN-PLE,
source) = **0.99894**, homogeneous across 12 sparse windows of the 320M rows; F1's PLE
reads 0.9487-0.9493 — pure FP2 noise. **Every PLE was faithful all along.** The small
real LEAN-vs-F2 residual (cos 0.99829) is lineage, not corruption: the LEAN passed through
FP8 → Q8_0 before its Q5_1, F2 went BF16 → Q5_1 directly. File-level forensics completed
the exoneration: the PLE lives entirely in shard 3 (one tensor, 95.37 GiB), the converter
merges shards by plain concatenation, the runtime indexes the fused row space — no
interleaving to get wrong — and the same inode fed both quantize runs. No fix was needed
in loader, converter, quantizer or runtime.

**The lesson, twice over.** The plan's own first GGUF reader had already failed the same
way earlier in the thread (inverted header fields, wrong KV skip sizes → an 88 GiB read);
its replacement validates `data_start + Σ block_bytes = filesize` exactly. The Q5_1
episode adds the sharper rule: **validate any bit-level decoder against the C struct of
record before drawing conclusions from bit-level comparisons** — the corrupted comparison
*reproduced* across three independent tests, which is precisely what made it convincing.

## 7. Speed and the 60 GiB boundary (Fase B/C/E)

tg, card protocol: 4 prompts × 3 reps, greedy, external drafter Q8_0 at n3, KV q5_1,
`--ple-disk` (logs `card-LEAN.log`, `card-F1.log`):

| Prompt (tok/s) | LEAN | F1 | Δ |
|---|---|---|---|
| prose (Roma) | 31.8 | 34.5 | +8.5% |
| prose (Rinascimento) | 31.3 | 37.1 | +18.5% |
| counting 1-200 | 48.0 | 50.8 | +5.8% |
| alphabet | 37.0 | 39.2 | +5.9% |
| mean accepted length (n3) | 3.02 | 3.09 | — |

pp (llama-bench, median of 3, and a real-corpus prefill; logs `lb-*.md`,
`faseC-pp-reale-*`):

| Metric | LEAN | F1 | Δ |
|---|---|---|---|
| llama-bench pp2048 | 325.15 ±1.98 | 338.21 ±3.19 | +4.0% |
| llama-bench tg128 (plain) | 26.89 ±0.14 | 27.86 ±0.13 | +3.6% |
| real prefill, 6k-char dante prompt (1,943 tokens) | 288.4 | 310.3 | +7.6% |

RAM boundary, F1 alone, `--memory=60g --memory-swap=60g` around the server with
`--ple-disk --cache-disk --no-mmap` — no `--cache-disk-persist` in these runs
(logs `faseE-*`):

| Step | Config | OOM | Host peak | Session tg (n6) | acc-len |
|---|---|---|---|---|---|
| E0 | trunk, ctx 32k | no | 46 GiB | 25.8-27.5 (plain) | — |
| E1 | + drafter n6 | no | 54 GiB | 29.4 / 41.1 / 58.9 / 39.8 | 3.70 |
| E2 | + mmproj (vision) | no | 55 GiB | 31.0 / 38.4 / 59.0 / 42.4 | 3.54 |

Eight-turn sessions completed 8/8 on every step, no stalls. Host baseline is 3 GiB, so the
full E2 stack sits ~52 GiB above it — **F1 serves inside the 60 GiB budget with ~5-7 GiB of
headroom**. One measurement caveat that generalizes: the cgroup's `docker stats` RSS is
meaningless for this backend — VK/GTT weights are not counted there (it shows 1.5-2.0 GiB
while ~40-50 GiB are physically resident), so the boundary must be gated on host-level
memory, as done here.

## 8. A new server bug (not quantization-specific)

During boundary rev1 the server aborted on **byte-identical resubmission of a long
prompt** (`n_tokens == n_past`, i.e., no new tokens to process) under cache-eviction
pressure (`--cache-ram 2048`):

```
common.cpp:1526: failed to remove sequence 0 with p0=1934, p1=-1
```

The failing path is prompt-cache sequence removal, upstream machinery independent of the
quantized types; the LEAN would hit it under the same pressure. Reproduction context in
`faseE-bnd0-crash-context.log`; logged as an open issue on the fork. The measurement
protocol worked around it with a unique per-iteration prefix on long prefills.

## 9. Artifacts

- **GGUFs** — `FLASHNEXT-FP2MIX-64GB.gguf` (57,038,416,256 B; HF home
  `pugant/Qwen3.8-Flash-Next-FP2MIX-ROCmFPX-64GB-GGUF`) and the F2 requant, swapped in
  place as `Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (98.491 GiB, same filename
  and card as before).
- **Workspace scripts** (source workspace, `scripts/`) — `t27-fp2-mapping.py` (importance
  mapping + tensor-type-file generator, 25-assert self-test), `t27-gguf-reader.py`
  (stdlib-only GGUF reader; validates `data_start + Σ bytes = filesize`), `t27-fp2-rmse.py`
  (per-family RMSE/cos with the corrected `deq_q5_1`).
- **Fork** — branch `t27-ple-fp2-store` @ `dadc23e44` (ple-store v2), test image
  `t27-ple2-vk`; the extended `test-ple-store` battery runs Q5_1 + FP2.
- **Logs** — `t27-quant-f2.log`, `t27-quant-f1.log`, `t27-faseA-*`, `t27-gpu/`
  (faseB-F), `t27-fp2-rmse-report.txt`, `t27-testd-report.txt`, `t27-testef-report.txt`.

## 10. What transfers

1. **A requant from the native release beats a requant of a requant, cheaply.** Twenty-one
   minutes of BF16 + imatrix quantization bought −0.6/−4.9% ppl at identical format, size
   and speed — the cheapest quality win this thread has measured.
2. **2.5 bpw on this architecture is a floor you can hear.** The codebook error is at its
   theoretical bound, speed improves (fewer bytes to move), and quality breaks where the
   task is recall-like. Below ~3 bpw the question stops being "how much precision do we
   lose" and becomes "which tasks survive" — answer, here: single-round tool calling yes,
   multi-round agentic loops and vision answers no.
3. **Memorized corpora are quality amplifiers, in both directions.** The same weights read
   2.17× (calibration corpus) and 32× (memorized holdout). Perplexity deltas across
   corpora of different memorization content are not comparable.
4. **Validate every bit-level decoder against the C struct of record.** The phantom "two
   PLE families" reproduced across three tests because the bug was in the shared reader,
   not in any single comparison — reproduction is not correctness.
5. **Cgroup RSS does not see VK/GTT weights.** RAM-boundary gates for GPU-resident
   servers must read host memory; the container metric can be off by an order of
   magnitude.
6. **Structural protections belong to the tool, not the recipe.** Routers and conv1d
   staying F32 is enforced by the quantizer's own allowlist; the mapping's defensive
   rules for them are belt-and-suspenders, and knowing which is which is what makes an
   847-rule file auditable.

---

*Thread index: [`README.md`](README.md); related notes:
[qwen38-flash-next-runtime.md](qwen38-flash-next-runtime.md) (the flagship thread this
extends), [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md)
(quant quality on this model),
[speculative-round-budget.md](speculative-round-budget.md) (the ~3 bpw door, closed on
ppl), [2026-08-23-rocmfp3-quality-speed.md](2026-08-23-rocmfp3-quality-speed.md)
(nominal bpw does not port across architectures).
Numbers transcribed verbatim from the lab's raw notes and logs of 2026-09-01 → 09-02.*

---

*Attribution: GLM by z.ai.*
