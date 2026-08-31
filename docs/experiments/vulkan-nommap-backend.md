# Vulkan, no-mmap, and the 8060S physics (T21)

**Research note — August 2026.** Which backend should the daily driver run — the ROCm
build of our fork, or its Vulkan (RADV, Mesa) build? The lab had asked this twice
before and provisionally answered "ROCm for dense, Vulkan for MoE FP4"; the Flash-Next
deployment re-asked it, and the answer arrived welded to a second discovery: on this
APU the backend comparison is inseparable from a memory-mapping flag, because **mmap
collapses Vulkan prompt processing by 3×**. The same thread closed the hardware
questions — what the Radeon 8060S is actually bounded by, where the long-context
prefill cost lives, and whether long sessions degrade thermally.

Everything below was measured on one machine: AMD Strix Halo (Ryzen AI MAX+ 395,
Radeon 8060S iGPU, gfx1151, 128 GB unified LPDDR5X), our llama.cpp fork, one model
family (Qwen3.8-Flash-Next, our 98.5 GiB ROCmFP4-STRIX_LEAN quant unless stated),
dedicated GPU windows, medians, warm-up discarded — point measurements, not statistics.

**TL;DR**

- **Cross-backend, same protocol, medians: Vulkan ahead.** Open prose **+22%**
  (26.0 → 31.8 tok/s), alphabet **+21%** (30.4 → 36.7), counting +7%.
- **The flag discovery**: mmap collapses Vulkan prompt processing **3× (244 → 70
  tok/s)**. With mmap, the weights sit in pageable GTT memory and paging stalls eat
  the prefill; `--no-mmap` pins anonymous memory instead. Backend + flag together took
  run-to-run variance from **±10% to ±0.6%**.
- **Production switch (2026-08-29)**: Vulkan + `--no-mmap` + KV q8_0 — the KV quant
  funds the RAM that resident weights need, at a measured cost the prose win pays for.
- **The physics, measured**: tg (token generation) is memory-bound (closed);
  flash-attention is **44.8% of GPU time at 80k context** at an MFU (model-FLOPs
  utilization) of ~13% vs ~40% on plain matmul; clocks hold a **2220 MHz** plateau
  with no thermal degradation.

---

## 1. The question, and its history

The backend question has a paper trail worth three lines, because each entry changed
the operating rule:

- `2026-08-14` — first in-house validation on Qwen3.6-35B
  ([results-2026-08-14-vulkan-vs-rocm.md](results-2026-08-14-vulkan-vs-rocm.md)): at
  equal quant RADV beat ROCm on tg by +14.1%, but the ROCmFP4 stack won both fronts on
  the strength of the lighter format; ROCm kept pp (prompt processing) at +34.8%.
- `2026-08-15` — on the dense 27B, ROCm won tg by +53% and pp by +11%
  ([results-2026-08-15-qwen38-27b.md](results-2026-08-15-qwen38-27b.md)) — the number
  that set the "dense → ROCm" rule.
- `2026-08-17` — at unified protocol, tg was **equivalent** at 1k and 24k (+1.5% /
  +1.2%, one to two sigma): the +53% did not reproduce and is resolved as a protocol
  artifact, not physics ([results-2026-08-17-rocm-vs-vulkan-tg.md](results-2026-08-17-rocm-vs-vulkan-tg.md)).

Meanwhile the MoE FP4 class had already gone the other way — on the 35B MoE the Vulkan
fork of the same ROCmFP4 types won tg outright (81.6 vs 71.2 tok/s) — so the operating
rule entering Flash-Next was: dense → ROCm, MoE FP4 → Vulkan, backend neutral for tg
at matched protocol. Flash-Next is a 180B-class MoE, which made the re-run mandatory
rather than optional.

## 2. Cross-backend on Flash-Next, same protocol

Same four-prompt protocol as the T22 quant table (greedy, external MTP drafter,
medians; [2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md)),
ROCm build vs Vulkan build of the same fork, same GGUF:

| Workload | ROCm build | Vulkan (RADV) | Δ |
|---|---|---|---|
| open prose | 26.0 | **31.8** | **+22%** |
| alphabet | 30.4 | **36.7** | **+21%** |
| counting | — | — | +7% |

The counting cells are empty because the absolutes for that pair were never
published — only the +7% delta.

Prose — the workload that dominates agent sessions — is where the margin lives, at
+22%; the alphabet prompt matches it at +21%; on the counting prompt the ordering holds
but the margin narrows to +7%. Same asymmetry the drafter threads kept measuring:
deterministic content spends its time in round shape, prose in raw decode, and raw
decode is where the backend differs. Decision: the daily driver goes Vulkan.

## 3. The flag discovery — mmap collapses Vulkan prefill

The first Vulkan runs did not look like the table above. Prompt processing came in at
~70 tok/s where ~244 was expected, and run-to-run variance was ±10% — on a greedy,
single-stream protocol that should repeat to a fraction of a percent. The isolation
took one flag:

| Config (same server, same prompt) | pp (tok/s) |
|---|---|
| Vulkan, mmap (default) | 70 |
| Vulkan, `--no-mmap` | **244** |

**mmap collapses Vulkan prompt processing 3×.** The mechanism, on this APU: with the
default mmap path the weights are file-backed pages landed in **GTT pageable memory**
(GTT is the iGPU's pageable graphics aperture), and the prefill pays a paging stall
every time a batch first touches a page — the paging path runs mid-kernel,
serializing the prefill. `--no-mmap` instead reads the weights into **anonymous
memory** at load time, which stays resident. The price is load time (a cold start of
the 98.5 GiB file runs ~14 minutes; page-cache-warm, a couple of minutes) and RAM:
resident is resident, which is what forces the next decision (§4).

Two operational readings, both learned the honest way:

- **Variance was the symptom.** With backend and flag pinned together, run-to-run
  variance collapsed from ±10% to **±0.6%** — the swings had been the paging
  pathology, not the GPU or the driver. A machine that will not repeat is telling you
  the config is wrong before it tells you anything else.
- **Memory flags are per-backend and per-config.** A flag that is a no-op (or worse,
  a cliff) on one backend says nothing about the next; on this stack the same
  comparison run under mmap would have scored Vulkan's prefill at a third of its
  value. Every backend number in this repo is now pinned with its memory flags.

## 4. The production switch

The deployed combo since 2026-08-29 is **Vulkan (RADV) + `--no-mmap` + KV q8_0**.

The KV quantization is not a speed choice — it is the RAM that resident weights need.
With `--no-mmap`, the full weight file is anonymous resident memory beside the KV
cache, the external drafter (3.85 GiB) and the rollback ring (~7.2 GiB); shrinking the
KV is what makes the arithmetic close on 128 GB of unified memory shared with the OS.

The cost is measured, not assumed, and the same measurement had earlier produced the
opposite decision. In the agent-stack cycle
([2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md)), on the
real-session regime at 131k context, KV q8_0 vs f16 K/V cost **−19–26% tg**
(q8_0/f16 = 0.74 pure-cached, 0.81 on means) with perplexity at parity — a speed-bound
cycle therefore deployed f16. The Vulkan switch re-spent exactly that margin to fund
`--no-mmap`: a capacity-bound decision overrides a speed-bound one, and the +22% prose
win pays the bill. The lesson we keep from the pair: record which constraint decided,
so a reversal reads as new information rather than a flip-flop. (The tg-only NO-GO on
the dense 27B, where q8_0 KV worsened or tied everywhere, is
[results-2026-08-17-kv-quant-tg-context.md](results-2026-08-17-kv-quant-tg-context.md);
the RAM consequence that followed this switch is told in
[pi-stack-followups.md](pi-stack-followups.md).)

## 5. The 8060S physics, measured

Four questions about the silicon itself, closed by measurement across this thread:

**Is tg memory-bound? Yes — closed.** Three independent lines: the backend is
equivalent on tg at matched protocol (§1); halving the KV bytes gained nothing (the
08-17 NO-GO above — if KV read bytes were the bottleneck, half the bytes would have
shown); and the decode floor tracks the bytes that must move per token, weights plus
per-position KV. The only levers that ever beat the floor are fewer bytes (a lighter
quant) and more tokens per weight-read (speculative decoding — the MTP boost measured
constant with context, ~3.09× at both 1k and 24k,
[results-2026-08-17-ckpt-ring-tg-24k.md](results-2026-08-17-ckpt-ring-tg-24k.md)).
Everything else is closed, and the closed levers stay documented: KV quantization for
tg, checkpoint/ring tuning, hipCUB top-k enablement (−42% pp at 131k, the agent-stack
cycle).

**Where does the long-context prefill cost live? In attention.** At 80k context,
flash-attention accounts for **44.8% of GPU time**, running at an MFU of **13%**
versus ~40% on the plain matmuls — the attention kernels are the inefficient consumer,
and they grow with context while the matmuls do not. The time-crossover sits at
~15–25k tokens, far before the FLOP crossover (~238k), which is why long-context pp
degrades long before the arithmetic says it must. Any long-context prefill work is
attention work; nothing else is worth touching.

**Is the hybrid scan expensive? No.** The gated-deltanet recurrent scan measures at
**0.1–0.9% of GPU time** — not the double-digit share occasionally suggested for hybrid
models' scan overhead. The architecture's long-context cost is in its attention
layers, like everyone else's.

**Does the APU throttle over long sessions? No.** Clocks hold a **2220 MHz plateau**
across long production sessions, with no thermal degradation in the measurements.
Sustained decode speed reflects the memory floor and the context, not the thermals.

## 6. Where this leaves the machine

The serving config the thread produced — Vulkan/RADV + `--no-mmap` + KV q8_0, cache
sized for the agent — has been the production baseline since 2026-08-29, and the
speculative tables since the switch run on it. What remains genuinely open on
this hardware is short: the attention-side long-context prefill cost above, and the
~38 ms/round structural residue on Vulkan measured in
[speculative-round-software.md](speculative-round-software.md). The physics questions
are closed; the engineering ones have numbers attached.

---

## What transfers

1. **A backend benchmark that does not pin memory flags measures the flag, not the
   backend.** Here the default put one configuration at a third of its prompt-
   processing value; the cross-backend table is only meaningful because both sides
   carry their flags in the caption.
2. **Run-to-run variance is a diagnostic, not noise to average over.** ±10% on a
   deterministic protocol was the fingerprint of paging stalls; the same machine at
   ±0.6% once pinned. When the machine will not repeat, find the config fault before
   publishing the mean.
3. **The same measurement can justify opposite configs when the binding constraint
   changes.** KV q8_0 was rejected in a speed-bound cycle and adopted in a
   capacity-bound one; write down which constraint decided, or the reversal will look
   like error.
4. **Close the physics questions explicitly and keep the closures.** "tg is
   memory-bound" retires a family of proposed optimizations at once; "attention is
   44.8% of GPU time at 80k" says exactly where long-context work pays and where it
   does not.

---

*Thread index: [`README.md`](README.md); related notes:
[results-2026-08-14-vulkan-vs-rocm.md](results-2026-08-14-vulkan-vs-rocm.md),
[results-2026-08-15-qwen38-27b.md](results-2026-08-15-qwen38-27b.md),
[results-2026-08-17-rocm-vs-vulkan-tg.md](results-2026-08-17-rocm-vs-vulkan-tg.md),
[results-2026-08-17-kv-quant-tg-context.md](results-2026-08-17-kv-quant-tg-context.md),
[2026-08-29-t22-lean-vs-ud-quant.md](2026-08-29-t22-lean-vs-ud-quant.md) (the quant
table this protocol shares),
[2026-08-29-pi-stack-improvement.md](2026-08-29-pi-stack-improvement.md) (the KV
cost), [qwen4exp-runtime.md](qwen4exp-runtime.md) (the production runtime). Cross-backend
medians and the flag isolation are transcribed from the Backend section of the public
LEAN model card; historical cells from the linked notes.*

*Attribution: GLM by z.ai.*
